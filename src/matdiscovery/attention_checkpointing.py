"""Optional recomputation of native Qwen attention/GDN during MLP-cut attribution.

Only the original attention module's forward is checkpointed. A decoder layer or
an MLP must never be wrapped: its recomputation could run after the native MLP
cut hooks have been removed. This changes activation storage, not the forward
function, and does not establish a backend or transcoder fidelity pass.
"""
from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import copy
import functools
import hashlib
import inspect
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.checkpoint import checkpoint
from torch.utils._pytree import tree_flatten, tree_unflatten


CONTRACT = "native_attention_only_nonreentrant_frozen_parameter_recompute_v1"
_MARKER = "_matdiscovery_attention_checkpoint_owner"
_ABSENT = object()
_CACHE_ARGUMENTS = {"cache_params", "past_key_values", "past_key_value", "cache", "inference_params"}


class AttentionCheckpointError(RuntimeError):
    pass


def _parameter_identity(parameters):
    # requires_grad may deliberately change after capture; numerical weights,
    # placement, dtype, and parameter identity may not change before replay.
    return tuple((id(p), p._version, tuple(p.shape), str(p.dtype), str(p.device)) for p in parameters)


class _RecomputeContext:
    """Reusable: retain_graph permits entering the same context more than once."""

    def __init__(self, module, parameters, identity, counters):
        self.module, self.parameters, self.identity, self.counters = module, parameters, identity, counters
        self.saved_flags = []

    def __enter__(self):
        if self.module.training or _parameter_identity(self.parameters) != self.identity:
            raise AttentionCheckpointError("Attention weights, device, precision, or mode changed before recomputation")
        flags = tuple(p.requires_grad for p in self.parameters)
        self.saved_flags.append(flags)
        try:
            for parameter in self.parameters:
                parameter.requires_grad_(False)
        except BaseException:
            self.saved_flags.pop()
            for parameter, flag in zip(self.parameters, flags):
                parameter.requires_grad_(flag)
            raise
        self.counters["recomputations"] += 1
        return self

    def __exit__(self, exception_type, exception, traceback):
        # This also runs for checkpoint early-stop exceptions and failed VJPs.
        for parameter, flag in zip(self.parameters, self.saved_flags.pop()):
            parameter.requires_grad_(flag)
        return False


class AttentionCheckpointHandle:
    """Installed forward wrappers plus stable metadata and separate diagnostics."""

    def __init__(self, metadata):
        self._metadata = metadata
        self._installed = []
        self.counters = Counter()

    def metadata(self) -> dict[str, Any]:
        """Stable configuration only; counters and requires_grad are excluded."""
        return copy.deepcopy(self._metadata)

    def remove(self) -> None:
        """Restore original forwards after all outstanding attribution VJPs finish."""
        for module, old_instance_forward, wrapper in self._installed:
            if module.forward is not wrapper or getattr(module, _MARKER, None) is not self:
                raise AttentionCheckpointError("Attention forward changed externally; refusing an ambiguous restore")
        for module, old_instance_forward, _ in reversed(self._installed):
            if old_instance_forward is _ABSENT:
                delattr(module, "forward")
            else:
                module.forward = old_instance_forward
            delattr(module, _MARKER)
        self._installed.clear()


def install_attention_checkpointing(
    model: torch.nn.Module, *, attention_paths: Sequence[str] | None = None,
) -> AttentionCheckpointHandle:
    """Wrap all native text attention/GDN modules, leaving MLPs untouched.

    Call before constructing a policy configuration fingerprint or attributor.
    Add handle.metadata() to that immutable runtime identity. Only eval forwards
    with input gradients, frozen attention parameters, and no cache use the
    non-reentrant checkpoint. Generation and cache updates execute the original
    forward directly. Concurrent mutation/use of this policy is unsupported.
    """
    import transformers
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5GatedDeltaNet

    supported = {Qwen3_5Attention: "self_attn", Qwen3_5GatedDeltaNet: "linear_attn"}
    discovered = {path: module for path, module in model.named_modules() if type(module) in supported}
    paths = list(discovered) if attention_paths is None else list(attention_paths)
    if not paths or len(set(paths)) != len(paths) or set(paths) != set(discovered):
        raise AttentionCheckpointError("Declare every native Qwen text attention/GDN module exactly once")
    selected = []
    for path in paths:
        module = discovered[path]
        if path.rsplit(".", 1)[-1] != supported[type(module)] or any(name == "mlp" for name, _ in module.named_modules()):
            raise AttentionCheckpointError("Only native self_attn/linear_attn boundaries are supported; never decoder or MLP")
        if getattr(module, _MARKER, None) is not None:
            raise AttentionCheckpointError("Attention checkpointing is already installed")
        selected.append((path, module, module.forward, module.__dict__.get("forward", _ABSENT)))
    handle = AttentionCheckpointHandle({
        "contract": CONTRACT, "enabled": True, "use_reentrant": False,
        "preserve_rng_state": True, "determinism_check": "default",
        "scope": "original native text attention/GDN forwards only; excludes decoder, MLP, norm and residual boundaries",
        "activation_condition": "eval, grad enabled, hidden_states requires grad, all attention parameters frozen, no cache",
        "cache_and_no_grad_behavior": "original forward without checkpointing",
        "recompute_parameter_flags": "restore first-forward all-false flags; finally restore flags present at each recomputation",
        "weight_semantics": "same original resident parameters; no proxy, cast, detach, or weight update",
        "validation": "required separately; no transcoder fidelity claim",
        "attention_paths": paths,
        "module_classes": {path: type(module).__module__ + "." + type(module).__name__ for path, module, _, _ in selected},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
    })
    try:
        for path, module, original_forward, old_instance_forward in selected:
            signature = inspect.signature(original_forward)
            parameters = tuple(module.parameters())

            def build_wrapper(module, original_forward, signature, parameters):
                @functools.wraps(original_forward)
                def wrapped(*args, **kwargs):
                    handle.counters["forward_calls"] += 1
                    if not torch.is_grad_enabled():
                        handle.counters["bypassed_no_grad"] += 1
                        return original_forward(*args, **kwargs)
                    bound = signature.bind(*args, **kwargs).arguments
                    forwarded = {**bound, **bound.get("kwargs", {})}
                    if any(forwarded.get(key) is not None for key in _CACHE_ARGUMENTS) or forwarded.get("use_cache", False):
                        handle.counters["bypassed_cache"] += 1
                        return original_forward(*args, **kwargs)
                    hidden = forwarded.get("hidden_states")
                    if not isinstance(hidden, torch.Tensor) or not hidden.requires_grad:
                        handle.counters["bypassed_no_input_gradient"] += 1
                        return original_forward(*args, **kwargs)
                    if module.training or any(parameter.requires_grad for parameter in parameters):
                        handle.counters["bypassed_trainable_or_training"] += 1
                        return original_forward(*args, **kwargs)
                    identity = _parameter_identity(parameters)
                    flattened, specification = tree_flatten((args, kwargs))

                    def original_with_flat_inputs(*values):
                        call_args, call_kwargs = tree_unflatten(values, specification)
                        return original_forward(*call_args, **call_kwargs)

                    context = _RecomputeContext(module, parameters, identity, handle.counters)
                    handle.counters["checkpointed_forwards"] += 1
                    return checkpoint(original_with_flat_inputs, *flattened, use_reentrant=False,
                        context_fn=lambda: (nullcontext(), context), preserve_rng_state=True,
                        determinism_check="default")
                return wrapped

            wrapper = build_wrapper(module, original_forward, signature, parameters)
            module.forward = wrapper
            setattr(module, _MARKER, handle)
            handle._installed.append((module, old_instance_forward, wrapper))
    except BaseException:
        handle.remove()
        raise
    return handle
