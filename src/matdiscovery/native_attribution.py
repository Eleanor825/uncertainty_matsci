"""Native-HF CRV-inspired LOCAL JACOBIAN attribution, not exact original CRV.

Main contract: native_local_jacobian_action_logprob_mlp_cut_v2
The explicit after_step_next_token_v1 variant retains the old post-prefix target.
* Forward values are the original policy's values on one COMPLETE prefix.
* Every declared MLP output is replaced by a same-valued detached leaf.
  Intervening MLP Jacobians are therefore CUT, avoiding total-derivative edges.
* Attention/GDN, normalization scales, PLE gates, and residual paths retain their
  native local Jacobians. Attention patterns and norm denominators are NOT frozen.
* V2 has one mean unwarped emitted-action-value log-probability sink, with exact
  token t -> prediction position t-1 alignment and no future-position sources.
* Edges are directional derivatives of later active transcoder features or scores
  along earlier feature decoder*activation, reconstruction-error, omitted-feature,
  decoder-bias, or token-embedding directions.

This is a reduced graph of a prompt-local cut model. Finite differences freeze
ALL declared MLP outputs to the captured baseline before injecting a source.
It does not establish interventions in the unconstrained original network.
Architecture/backend support is fail-closed until validate_backend passes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .esopt import tensor_state_hash
from .action_targets import ActionTargets
from .graph_features import GraphPayload
from .transcoders import TopKTranscoder


CONTRACT = "native_local_jacobian_action_logprob_mlp_cut_v2"
LEGACY_CONTRACT = "native_local_jacobian_mlp_cut_v1"
SOURCE_SELECTION_RULES = ("global_activation", "qwen_final_mlp_causal_activation_v1")
QWEN_TOKENWISE_TAIL_SOURCE_SHA256 = "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"


class UnsupportedAttribution(RuntimeError):
    """No proxy graph is substituted when the declared contract cannot be met."""


@dataclass(frozen=True)
class PolicyStamp:
    state_id: str
    checkpoint_hash: str
    model_id: str
    generation: int
    perturbation_seed: int | None = None
    perturbation_sigma: float | None = None


@dataclass
class TranscoderBinding:
    module_path: str
    transcoder: TopKTranscoder
    training_metadata: Mapping[str, Any]


@dataclass
class LocalTrace:
    stamp: PolicyStamp
    input_ids: torch.Tensor
    model_kwargs: dict[str, Any]
    prefix_hash: str
    embedding_leaves: dict[str, torch.Tensor]
    mlp_inputs: dict[str, torch.Tensor]
    mlp_leaves: dict[str, torch.Tensor]
    logits: torch.Tensor
    reference_logits: torch.Tensor
    parity_max_abs: float
    activations: dict[str, torch.Tensor]
    reconstructions: dict[str, torch.Tensor]
    fidelity: dict[str, dict[str, float]]
    action_targets: ActionTargets | None = None


@dataclass
class NativeAttributionResult:
    graph: GraphPayload
    metadata: dict[str, Any]
    node_records: list[dict[str, Any]]


def _tensor_input(args: tuple, kwargs: dict) -> torch.Tensor:
    candidate = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
    if not isinstance(candidate, torch.Tensor) or candidate.ndim != 3 or candidate.shape[0] != 1:
        raise UnsupportedAttribution("Declared MLP must take a [1, complete_prefix, hidden] tensor.")
    return candidate


def _module_runtime(module: torch.nn.Module) -> list[dict[str, Any]]:
    """Describe actual resident tensors without reading/copying their contents."""
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for kind, tensors in (("parameter", module.parameters()), ("buffer", module.buffers())):
        for tensor in tensors:
            key = (kind, str(tensor.device), str(tensor.dtype))
            group = groups.setdefault(key, {"kind": kind, "device": key[1], "dtype": key[2],
                                           "tensors": 0, "numel": 0})
            group["tensors"] += 1
            group["numel"] += tensor.numel()
    return [groups[key] for key in sorted(groups)]


def _qwen_final_mlp_source_proof(policy, bindings, review) -> dict[str, Any]:
    """Admit only the reviewed official complete text decoder and tokenwise tail.

    The final MLP output cannot affect another position: its only successors
    are the residual addition, final RMSNorm, position selection and LM head.
    This is an architecture proof, independent of the current weight values.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    except ImportError as exc:
        raise UnsupportedAttribution("The Qwen source-selection rule requires reviewed official HF classes") from exc
    paths = tuple(f"model.language_model.layers.{i}.mlp" for i in range(32))
    def require(condition):
        if not condition:
            raise UnsupportedAttribution("Qwen source-selection rule requires the reviewed complete 32-layer Qwen3_5 tokenwise tail")
    source = Path(inspect.getsourcefile(hf.Qwen3_5ForConditionalGeneration))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    require(source_hash == QWEN_TOKENWISE_TAIL_SOURCE_SHA256)
    require(type(policy) is hf.Qwen3_5ForConditionalGeneration
            and tuple(binding.module_path for binding in bindings) == paths
            and review.get("model_type") == "qwen3_5" and review.get("layers") == 32
            and tuple(review.get("mlp_paths", ())) == paths)
    require(type(policy.model) is hf.Qwen3_5Model
            and type(policy.model.language_model) is hf.Qwen3_5TextModel)
    text = policy.model.language_model
    require(len(text.layers) == text.config.num_hidden_layers == 32
            and type(text.norm) is hf.Qwen3_5RMSNorm
            and type(policy.lm_head) is torch.nn.Linear
            and policy.lm_head.in_features == text.config.hidden_size
            and policy.lm_head.out_features == text.config.vocab_size)
    modules = [(policy, hf.Qwen3_5ForConditionalGeneration), (policy.model, hf.Qwen3_5Model),
               (text, hf.Qwen3_5TextModel), (text.norm, hf.Qwen3_5RMSNorm),
               (policy.lm_head, torch.nn.Linear)]
    for index, layer in enumerate(text.layers):
        require(type(layer) is hf.Qwen3_5DecoderLayer and type(layer.mlp) is hf.Qwen3_5MLP
                and policy.get_submodule(paths[index]) is layer.mlp)
        modules.extend(((layer, hf.Qwen3_5DecoderLayer), (layer.mlp, hf.Qwen3_5MLP)))
    require(all(getattr(module.forward, "__func__", None) is cls.forward for module, cls in modules))
    return {"schema": "qwen_final_mlp_tokenwise_tail_source_proof_v1",
            "model_class": type(policy).__module__ + "." + type(policy).__name__,
            "source_module": hf.__name__, "source_sha256": source_hash,
            "canonical_mlp_paths": list(paths), "final_mlp_path": paths[-1],
            "tail": ["residual_add", "Qwen3_5RMSNorm_per_token", "logits_to_keep_positions", "Linear_lm_head_per_token"],
            "scope": "complete text-only prefix; no unreviewed trailing token mixing",
            "action_positions": "exact teacher-forced prediction_positions",
            "legacy_positions": "last source position"}


class NativeAttributor:
    """Hooks a canonical HF policy without registering transcoders on that policy.

    state_id_getter MUST change immediately for every weight update, perturbation,
    restore or reload. The caller computes hashes at checkpoint/update boundaries;
    this class checks the immutable stamp before/after work, not a 10GB hash/step.
    No concurrent weight mutation or sharing of a live policy between rollouts is
    permitted. Bind every MLP boundary explicitly after architecture source review.
    """

    def __init__(
        self, policy: torch.nn.Module, bindings: Sequence[TranscoderBinding], *,
        state_id_getter: Callable[[], str],
        architecture_review: Mapping[str, Any],
        auxiliary_embedding_paths: Sequence[str] = (),
        max_nodes: int = 4096,
        max_feature_nodes: int = 4096,
        max_backward_targets: int = 4096,
        max_logits: int = 10,
        desired_logit_mass: float = 0.95,
        node_influence_mass: float = 0.8,
        edge_row_mass: float = 0.98,
        parity_atol: float = 1e-6,
        parity_rtol: float = 1e-5,
        target_mode: str = "action_logprob_v2",
        constant_storage_device: str | torch.device | None = None,
        source_selection_rule: str = "global_activation",
    ) -> None:
        if target_mode not in {"action_logprob_v2", "after_step_next_token_v1"}:
            raise ValueError("Select the explicit action-logprob or legacy after-step target mode")
        self.target_mode = target_mode
        if source_selection_rule not in SOURCE_SELECTION_RULES:
            raise ValueError("Unknown attribution source-selection rule")
        self.source_selection_rule = source_selection_rule
        self.constant_storage_device = torch.device(constant_storage_device) if constant_storage_device is not None else None
        if self.constant_storage_device is not None and self.constant_storage_device.type not in {"cpu", "cuda"}:
            raise ValueError("Attribution constants require a resident CPU or CUDA device")
        self.contract = CONTRACT if target_mode == "action_logprob_v2" else LEGACY_CONTRACT
        if not bindings or len({binding.module_path for binding in bindings}) != len(bindings):
            raise ValueError("Supply unique, ordered, reviewed MLP boundaries.")
        if not architecture_review.get("verified") or not architecture_review.get("source"):
            raise UnsupportedAttribution("An explicit architecture source review is required.")
        if min(max_nodes, max_feature_nodes, max_backward_targets, max_logits) < 1:
            raise ValueError("Graph/target budgets must be positive.")
        if not all(0 < x <= 1 for x in (desired_logit_mass, node_influence_mass, edge_row_mass)):
            raise ValueError("Mass thresholds must be in (0,1].")
        if policy.training:
            raise UnsupportedAttribution("Attribution requires policy.eval() and deterministic forward values.")
        self.policy, self.bindings = policy, list(bindings)
        self.state_id_getter = state_id_getter
        self.architecture_review = dict(architecture_review)
        self.source_selection_proof = (_qwen_final_mlp_source_proof(policy, self.bindings, self.architecture_review)
                                       if source_selection_rule == "qwen_final_mlp_causal_activation_v1" else None)
        self.max_nodes, self.max_feature_nodes = max_nodes, max_feature_nodes
        self.max_backward_targets, self.max_logits = max_backward_targets, max_logits
        self.desired_logit_mass, self.node_influence_mass, self.edge_row_mass = desired_logit_mass, node_influence_mass, edge_row_mass
        self.parity_atol, self.parity_rtol = parity_atol, parity_rtol
        self.modules = {binding.module_path: policy.get_submodule(binding.module_path) for binding in bindings}
        primary = policy.get_input_embeddings()
        paths = [name for name, module in policy.named_modules() if module is primary]
        if len(paths) != 1:
            raise UnsupportedAttribution("Main input embedding module must have one unambiguous path.")
        self.embedding_modules = {paths[0]: primary}
        for path in auxiliary_embedding_paths:
            self.embedding_modules[path] = policy.get_submodule(path)
        # Gemma's separate token-ID -> PLE branch cannot silently disappear.
        for path, module in policy.named_modules():
            if path.endswith("embed_tokens_per_layer") and path not in self.embedding_modules:
                raise UnsupportedAttribution("PLE embeddings require an explicit auxiliary_embedding_paths entry.")
            if getattr(module, "enable_moe_block", False):
                raise UnsupportedAttribution("Parallel MoE branch needs a reviewed composite cut boundary; standalone .mlp is insufficient.")
        policy_parameter_ids = {id(parameter) for parameter in policy.parameters()}
        for binding in self.bindings:
            metadata = binding.training_metadata
            if not metadata.get("fidelity_gate_passed") or metadata.get("max_dev_fvu") is None:
                raise UnsupportedAttribution("Every transcoder needs a passed held-out reconstruction gate.")
            if not np.isfinite(float(metadata["max_dev_fvu"])) or float(metadata["max_dev_fvu"]) < 0:
                raise UnsupportedAttribution("Transcoder fidelity thresholds must be finite and nonnegative.")
            if metadata.get("provenance", {}).get("layer_path") != binding.module_path:
                raise UnsupportedAttribution("Transcoder training layer does not match its MLP boundary.")
            if policy_parameter_ids & {id(parameter) for parameter in binding.transcoder.parameters()}:
                raise UnsupportedAttribution("Transcoders must be independent of policy parameters.")
            parameters = list(binding.transcoder.parameters())
            if len({(parameter.device, parameter.dtype) for parameter in parameters}) != 1:
                raise UnsupportedAttribution("Each transcoder must reside on one device in one precision.")
            if parameters[0].device.type == "meta":
                raise UnsupportedAttribution("Transcoders require resident tensor values, never meta placeholders.")
        self._validation: dict[str, Any] | None = None
        self._backend_identity = self._backend_signature()

    def runtime_metadata(self) -> dict[str, Any]:
        """Actual policy/TC placement and arithmetic settings, also bound by the gate.

        A CPU-resident output head and/or TC bank is supported. Feature targets
        are computed on the TC device through a differentiable input copy;
        decoder directions are copied to the original policy source device.
        Neither copy detaches the policy's gradient path.
        """
        output_getter = getattr(self.policy, "get_output_embeddings", None)
        output = output_getter() if callable(output_getter) else None
        matmul = torch.backends.cuda.matmul
        return {
            "schema": "native_attribution_runtime_precision_devices_v1",
            "policy_tensors": _module_runtime(self.policy),
            "embedding_tensors": {path: _module_runtime(module) for path, module in self.embedding_modules.items()},
            "mlp_tensors": {path: _module_runtime(module) for path, module in self.modules.items()},
            "output_head_tensors": _module_runtime(output) if output is not None else None,
            "policy_placement_contract": dict(getattr(self.policy, "_matdiscovery_runtime_config", {})),
            "constant_storage_policy": str(self.constant_storage_device) if self.constant_storage_device is not None else "cpu_when_tc_on_cpu_else_policy_cut_device",
            "residual_projection_policy": "one detached float32 gradient transfer per connected layer/target to its constant device",
            "transcoder_tensors": {binding.module_path: _module_runtime(binding.transcoder) for binding in self.bindings},
            "arithmetic": {
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "allow_bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
                "allow_fp16_reduced_precision_reduction": matmul.allow_fp16_reduced_precision_reduction,
                "autocast_cpu": torch.is_autocast_enabled("cpu"),
                "autocast_cuda": torch.is_autocast_enabled("cuda"),
                "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                "sdpa_memory_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
                "sdpa_cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled() if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") else None,
                "torch_cpu_threads": torch.get_num_threads(),
            },
        }

    def _constant_device(self, binding: TranscoderBinding, policy_tensor: torch.Tensor) -> torch.device:
        if self.constant_storage_device is not None:
            return self.constant_storage_device
        return torch.device("cpu") if binding.transcoder.encoder.weight.device.type == "cpu" else policy_tensor.device

    def _backend_signature(self) -> str:
        config = getattr(self.policy, "config", None)
        record = {"class": type(self.policy).__module__ + "." + type(self.policy).__name__, "torch_version": str(torch.__version__),
                  "target_contract": self.contract,
                  "source_selection_rule": self.source_selection_rule,
                  "source_selection_proof": self.source_selection_proof,
                  "runtime": self.runtime_metadata(),
                  "attention_implementation": repr(getattr(config, "_attn_implementation", None)),
                  "boundaries": [(name, type(module).__module__ + "." + type(module).__name__) for name, module in self.modules.items()],
                  "embedding_paths": list(self.embedding_modules)}
        return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()

    def _guard(self, stamp: PolicyStamp) -> None:
        if not stamp.state_id or not stamp.checkpoint_hash or not stamp.model_id:
            raise ValueError("Provide immutable policy/checkpoint identities.")
        if self.state_id_getter() != stamp.state_id:
            raise UnsupportedAttribution("Policy version changed; stale prefix/graph state is forbidden.")
        if self.policy.training or self._backend_signature() != self._backend_identity:
            raise UnsupportedAttribution("Backend or training mode changed; construct and validate a new attributor.")

    def capture(self, input_ids: torch.Tensor, stamp: PolicyStamp, *, model_kwargs: Mapping[str, Any] | None = None,
                action_targets: ActionTargets | None = None) -> LocalTrace:
        """Capture full-prefix leaves and compare every selected raw prediction logit."""
        self._guard(stamp)
        if torch.is_inference_mode_enabled():
            raise UnsupportedAttribution("Native Jacobian extraction cannot run inside torch.inference_mode().")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("Provide one nonempty, complete token prefix.")
        kwargs = dict(model_kwargs or {})
        for key in ("past_key_values", "inputs_embeds", "pixel_values", "input_features", "per_layer_inputs"):
            if kwargs.get(key) is not None:
                raise UnsupportedAttribution(f"{key} bypasses the declared full-text-prefix source contract.")
        if kwargs.get("use_cache", False):
            raise UnsupportedAttribution("KV/GDN cache reuse is forbidden during attribution.")
        kwargs["use_cache"] = False
        mask = kwargs.setdefault("attention_mask", torch.ones_like(input_ids))
        # Output-position selectors are not source tokens. Preserve the canonical
        # corpus/dedup hash and bind the objective through its separate target hash.
        prefix_hash = tensor_state_hash({"input_ids": input_ids, "attention_mask": mask})
        if self.target_mode == "action_logprob_v2":
            if not isinstance(action_targets, ActionTargets):
                raise UnsupportedAttribution("ActionTargets are required for the v2 material-action graph")
            action_targets.validate(input_ids, mask)
            kwargs["logits_to_keep"] = torch.tensor(action_targets.prediction_positions, device=input_ids.device, dtype=torch.long)
        elif action_targets is not None:
            raise UnsupportedAttribution("Legacy after-step graphs cannot accept action targets")
        def selected_logits(output):
            logits = output.logits
            if action_targets is None:
                return logits[:, -1]
            if logits.ndim != 3 or logits.shape[1] != len(action_targets.token_ids):
                raise UnsupportedAttribution("Backend did not honor the exact teacher-forced prediction positions")
            return logits
        with torch.no_grad():
            reference = selected_logits(self.policy(input_ids=input_ids, **kwargs)).detach().clone()
        embeddings, mlp_inputs, outputs = {}, {}, {}
        handles = []
        original_flags = [(parameter, parameter.requires_grad) for parameter in self.policy.parameters()]
        try:
            for parameter, _ in original_flags:
                parameter.requires_grad_(False)
            for path, module in self.embedding_modules.items():
                def embedding_hook(module, args, result, path=path):
                    if path in embeddings or not isinstance(result, torch.Tensor) or result.shape[:2] != input_ids.shape:
                        raise UnsupportedAttribution("Embedding source must run once and align to the complete prefix.")
                    leaf = result.detach().requires_grad_(True)
                    embeddings[path] = leaf
                    return leaf
                handles.append(module.register_forward_hook(embedding_hook))
            for path, module in self.modules.items():
                def mlp_hook(module, args, kw, result, path=path):
                    if path in outputs or not isinstance(result, torch.Tensor) or result.ndim != 3 or result.shape[:2] != input_ids.shape:
                        raise UnsupportedAttribution("MLP boundary must execute once and return aligned tensor output.")
                    mlp_inputs[path] = _tensor_input(args, kw)
                    leaf = result.detach().requires_grad_(True)
                    outputs[path] = leaf
                    return leaf
                handles.append(module.register_forward_hook(mlp_hook, with_kwargs=True))
            with torch.enable_grad():
                logits = selected_logits(self.policy(input_ids=input_ids, **kwargs))
            if set(outputs) != set(self.modules) or set(embeddings) != set(self.embedding_modules):
                raise UnsupportedAttribution("Some declared MLP/embedding boundaries did not execute.")
            if list(outputs) != list(self.modules):
                raise UnsupportedAttribution("MLP binding order does not match execution order.")
            parity = float((logits.detach().float() - reference.float()).abs().max())
            if not torch.allclose(logits.detach().float(), reference.float(), atol=self.parity_atol, rtol=self.parity_rtol):
                raise UnsupportedAttribution(f"Cut model changed forward logits (max abs {parity}).")
            activations, reconstructions, fidelity = {}, {}, {}
            with torch.no_grad():
                for binding in self.bindings:
                    path, transcoder = binding.module_path, binding.transcoder
                    x = mlp_inputs[path].detach().to(transcoder.encoder.weight.device, transcoder.encoder.weight.dtype)
                    if outputs[path].shape[-1] != transcoder.config.output_dim:
                        raise UnsupportedAttribution("Transcoder output dimension does not match the MLP.")
                    activation = transcoder.encode(x)
                    reconstruction = transcoder.decoder(activation).to(
                        device=self._constant_device(binding, outputs[path]), dtype=outputs[path].dtype)
                    horizon = input_ids.shape[1] - 1 if action_targets is None else action_targets.causal_horizon
                    y = outputs[path][:, :horizon + 1].detach().float()
                    # Keep the original fidelity reduction on the policy device;
                    # only one layer's temporary reconstruction is copied there.
                    fidelity_reconstruction = reconstruction[:, :horizon + 1].to(y)
                    error = y - fidelity_reconstruction
                    centered = (y - y.mean(dim=1, keepdim=True)).square().sum()
                    fvu_defined = bool(centered > 1e-12)
                    fvu = float(error.square().sum() / centered) if fvu_defined else 0.0
                    relative = float(error.square().sum() / y.square().sum().clamp_min(1e-12))
                    fidelity[path] = {"output_fvu": fvu, "fvu_undefined": float(not fvu_defined),
                                      "relative_squared_error": relative, "output_mse": float(error.square().mean()),
                                      "evaluated_positions": horizon + 1}
                    # Undefined prefix FVU cannot certify fidelity; short prefixes fail closed.
                    threshold = float(binding.training_metadata["max_dev_fvu"])
                    if not fvu_defined or not np.isfinite(fvu) or fvu > threshold:
                        raise UnsupportedAttribution(f"Current-policy transcoder fidelity gate failed at {path}: FVU={fvu}, defined={fvu_defined}.")
                    activations[path], reconstructions[path] = activation.detach(), reconstruction.detach()
                    del x, y, error, centered, fidelity_reconstruction
            self._guard(stamp)
            return LocalTrace(stamp, input_ids.detach().clone(), kwargs, prefix_hash, embeddings, mlp_inputs, outputs,
                              logits, reference, parity, activations, reconstructions, fidelity, action_targets)
        except (RuntimeError, NotImplementedError) as exc:
            if isinstance(exc, UnsupportedAttribution):
                raise
            raise UnsupportedAttribution(f"Native forward/Jacobian backend unsupported: {exc}") from exc
        finally:
            for handle in handles:
                handle.remove()
            for parameter, flag in original_flags:
                parameter.requires_grad_(flag)

    @torch.enable_grad()
    def _assemble(self, trace: LocalTrace, *, feature_cap: int | None = None) -> NativeAttributionResult:
        self._guard(trace.stamp)
        action_targets = trace.action_targets
        if action_targets is None:
            probabilities = trace.reference_logits[0].float().softmax(-1)
            top_prob, top_id = probabilities.topk(min(self.max_logits, probabilities.numel()))
            k = min(top_id.numel(), int(torch.searchsorted(top_prob.cumsum(0), self.desired_logit_mass)) + 1)
            top_prob, top_id = top_prob[:k], top_id[:k]
        else:
            k, top_prob, top_id = 1, None, None
        positions, layers = trace.input_ids.shape[1], len(self.bindings)
        horizon = positions - 1 if action_targets is None else action_targets.causal_horizon
        # Three aggregated residual directions per layer preserve reconstruction
        # error, omitted features, and decoder bias as distinct causal sources.
        room = min(self.max_nodes - (3 * layers + horizon + 1 + k), self.max_backward_targets - k,
                   self.max_feature_nodes, self.max_feature_nodes if feature_cap is None else feature_cap)
        if room < 1:
            raise UnsupportedAttribution("Complete prefix/residual nodes exceed graph budget; no silent prefix truncation.")
        candidates = []
        total_active, total_activation_mass, full_prefix_active = 0, 0.0, 0
        excluded_active, excluded_activation_mass = 0, 0.0
        for layer, binding in enumerate(self.bindings):
            full_z = trace.activations[binding.module_path][0]
            full_prefix_active += int((full_z > 0).sum())
            z = full_z[:horizon + 1]
            active = (z > 0).nonzero()
            total_active += active.shape[0]
            total_activation_mass += float(z.sum())
            if self.source_selection_rule == "qwen_final_mlp_causal_activation_v1" and layer == layers - 1:
                allowed = torch.tensor(action_targets.prediction_positions if action_targets is not None else (positions - 1,),
                                       dtype=torch.long, device=active.device)
                eligible = (active[:, 0, None] == allowed[None, :]).any(dim=1)
                excluded = active[~eligible]
                excluded_active += len(excluded)
                excluded_activation_mass += float(z[excluded[:, 0], excluded[:, 1]].sum())
                active = active[eligible]
            if active.numel():
                values = z[active[:, 0], active[:, 1]]
                count = min(room, values.numel())
                best = torch.topk(values, count).indices
                for index in best.tolist():
                    pos, feature = active[index].tolist()
                    candidates.append((float(values[index]), layer, pos, feature))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        chosen = candidates[:room]
        records: list[dict[str, Any]] = [dict(kind="feature", layer=layer, position=pos, feature=feature, activation=value) for value, layer, pos, feature in chosen]
        for layer in range(layers):
            for kind in ("reconstruction_error", "omitted_features", "decoder_bias"):
                records.append(dict(kind=kind, layer=layer))
        records.extend(dict(kind="token", position=pos, layer=-1) for pos in range(horizon + 1))
        logit_start = len(records)
        if action_targets is None:
            records.extend(dict(kind="logit", token_id=int(token), layer=layers) for token in top_id.tolist())
        else:
            records.append(dict(kind="score", score_kind="emitted_action_value_mean_logprob", layer=layers,
                                target_spec_hash=action_targets.target_spec_hash, token_count=len(action_targets.token_ids)))
        n = len(records)
        selected_contributions = {binding.module_path: torch.zeros_like(trace.reconstructions[binding.module_path]) for binding in self.bindings}
        with torch.no_grad():
            for record in records[:len(chosen)]:
                binding = self.bindings[record["layer"]]
                direction = binding.transcoder.decoder.weight[:, record["feature"]].detach().to(selected_contributions[binding.module_path])
                selected_contributions[binding.module_path][0, record["position"]] += direction * record["activation"]
        residual_vectors = {}
        for layer, binding in enumerate(self.bindings):
            path = binding.module_path
            reconstruction = trace.reconstructions[path]
            bias = binding.transcoder.decoder.bias.detach().to(reconstruction).view(1, 1, -1).expand_as(reconstruction)
            residual_vectors[(layer, "reconstruction_error")] = trace.mlp_leaves[path].detach().to(reconstruction) - reconstruction
            residual_vectors[(layer, "omitted_features")] = trace.reconstructions[path] - bias - selected_contributions[path]
            residual_vectors[(layer, "decoder_bias")] = bias
        feature_sources = {}
        for layer, binding in enumerate(self.bindings):
            indices = [index for index, record in enumerate(records[:len(chosen)]) if record["layer"] == layer]
            output = trace.mlp_leaves[binding.module_path]
            positions_tensor = torch.tensor([records[index]["position"] for index in indices], device=output.device, dtype=torch.long)
            feature_ids = torch.tensor([records[index]["feature"] for index in indices], device=binding.transcoder.decoder.weight.device, dtype=torch.long)
            activation_tensor = torch.tensor([records[index]["activation"] for index in indices], device=output.device, dtype=output.dtype)
            vectors = binding.transcoder.feature_directions(feature_ids).detach().to(output) * activation_tensor[:, None]
            feature_sources[layer] = (np.array(indices, dtype=np.int64), positions_tensor, vectors.float())
        token_nodes = np.array([index for index, record in enumerate(records) if record["kind"] == "token"], dtype=np.int64)
        source_tensors = list(trace.embedding_leaves.values()) + list(trace.mlp_leaves.values())
        num_embeddings = len(trace.embedding_leaves)
        edge_sources, edge_targets, edge_values, edge_keep = [], [], [], []
        row_ranges: dict[int, tuple[int, int]] = {}
        target_nodes = list(range(len(chosen))) + list(range(logit_start, n))
        all_gradients_present = False
        future_gradient_max_abs = 0.0
        residual_projection_counts = {binding.module_path: {"projections": 0, "device_transfers": 0} for binding in self.bindings}
        try:
            for target_index in target_nodes:
                target = records[target_index]
                if target["kind"] == "logit":
                    scalar = trace.logits[0, target["token_id"]]
                elif target["kind"] == "score":
                    scalar = action_targets.mean_logprob(trace.logits)
                else:
                    binding = self.bindings[target["layer"]]
                    encoder_row = binding.transcoder.encoder.weight[target["feature"]].detach()
                    # Selected feature is positive & TopK-active at baseline: its
                    # local derivative is exactly this encoder row (gate fixed).
                    # This .to() must stay differentiable when the TC is on CPU
                    # and the original MLP input is on GPU. Keep TC precision.
                    scalar = (trace.mlp_inputs[binding.module_path][0, target["position"]].to(encoder_row) * encoder_row).sum()
                if not scalar.requires_grad:
                    raise UnsupportedAttribution("Target lacks autograd connectivity; no proxy fallback.")
                gradients = torch.autograd.grad(scalar, source_tensors, allow_unused=True, retain_graph=True)
                if target["kind"] == "score" and horizon + 1 < positions:
                    for gradient in gradients:
                        if gradient is not None:
                            future = gradient[:, horizon + 1:]
                            if future.numel():
                                future_gradient_max_abs = max(future_gradient_max_abs, float(future.detach().abs().max()))
                    if future_gradient_max_abs > 1e-7:
                        raise UnsupportedAttribution("Action score depends on future token/feature positions; causal backward failed")
                row = np.zeros(n, dtype=np.float64)
                # Batch source projections by layer: no GPU -> Python scalar sync
                # for every feature edge (which is prohibitive at 4096 nodes).
                for embedding_index, embedding in enumerate(trace.embedding_leaves.values()):
                    gradient = gradients[embedding_index]
                    if gradient is not None:
                        effects = (gradient[0].float().reshape(positions, -1) * embedding[0].detach().float().reshape(positions, -1)).sum(dim=-1)
                        row[token_nodes] += effects[:horizon + 1].detach().cpu().numpy()
                for layer in range(target["layer"]):
                    gradient = gradients[num_embeddings + layer]
                    if gradient is None:
                        continue
                    all_gradients_present = True
                    indices, source_positions, vectors = feature_sources[layer]
                    if indices.size:
                        effects = (gradient[0, source_positions].float() * vectors).sum(dim=-1)
                        row[indices] = effects.detach().cpu().numpy()
                    constant_device = residual_vectors[(layer, "reconstruction_error")].device
                    projection_gradient = gradient.detach().to(device=constant_device, dtype=torch.float32)
                    counts = residual_projection_counts[self.bindings[layer].module_path]
                    counts["projections"] += 1
                    counts["device_transfers"] += int(gradient.device != constant_device)
                    residual_effects = torch.stack([
                        (projection_gradient * residual_vectors[(layer, kind)].float()).sum()
                        for kind in ("reconstruction_error", "omitted_features", "decoder_bias")
                    ])
                    offset = len(chosen) + 3 * layer
                    row[offset:offset + 3] = residual_effects.detach().cpu().numpy()
                    del projection_gradient
                if not np.isfinite(row).all():
                    raise UnsupportedAttribution("Nonfinite local Jacobian contributions.")
                present = np.flatnonzero(row != 0)
                keep = np.zeros(n, dtype=bool)
                if present.size:
                    ordered = present[np.argsort(-np.abs(row[present]), kind="stable")]
                    mass = np.cumsum(np.abs(row[ordered]))
                    cutoff = min(ordered.size, int(np.searchsorted(mass, self.edge_row_mass * mass[-1])) + 1)
                    keep[ordered[:cutoff]] = True
                row_start = len(edge_values)
                for source in present:
                    edge_sources.append(int(source)); edge_targets.append(target_index)
                    edge_values.append(float(row[source])); edge_keep.append(bool(keep[source]))
                row_ranges[target_index] = (row_start, len(edge_values))
                # Drop the previous complete gradient tuple before the next VJP
                # allocates another one; all contributions above are copied.
                del gradients
                # The loop local and the future-position view must not keep a
                # final full layer gradient alive across the next VJP either.
                gradient = future = None
        except (RuntimeError, NotImplementedError) as exc:
            if isinstance(exc, UnsupportedAttribution):
                raise
            raise UnsupportedAttribution(f"Native backward unavailable for this attention/GDN backend: {exc}") from exc
        if not all_gradients_present:
            raise UnsupportedAttribution("No connected MLP-output -> target Jacobian was found.")
        source_array, target_array, weight_array = np.array(edge_sources, dtype=np.int64), np.array(edge_targets, dtype=np.int64), np.array(edge_values)
        influence = np.zeros(n, dtype=np.float64)
        influence[logit_start:] = top_prob.cpu().numpy() if top_prob is not None else 1.0
        for target in sorted(target_nodes, key=lambda index: records[index]["layer"], reverse=True):
            start, stop = row_ranges[target]
            total = np.abs(weight_array[start:stop]).sum()
            if total:
                influence[source_array[start:stop]] += influence[target] * np.abs(weight_array[start:stop]) / total
        node_mask = np.ones(n, dtype=bool)
        if action_targets is not None:
            for index, record in enumerate(records):
                if record["kind"] == "token" and record["position"] > horizon:
                    node_mask[index] = False
        if chosen:
            ordered = np.argsort(-influence[:len(chosen)], kind="stable")
            total = influence[ordered].sum()
            if total:
                cutoff = int(np.searchsorted(np.cumsum(influence[ordered]), self.node_influence_mass * total)) + 1
                node_mask[:len(chosen)] = False
                node_mask[ordered[:cutoff]] = True
        types = np.array([record["kind"] if record["kind"] in {"feature", "token", "logit", "score"} else "error" for record in records])
        activations = np.array([record.get("activation", 0.0) for record in records])
        node_layers = np.array([record["layer"] for record in records], dtype=np.int64)
        final_edge_mask = np.array(edge_keep, dtype=bool) & node_mask[source_array] & node_mask[target_array]
        edge_mass = float(np.abs(weight_array).sum())
        metadata = {"contract": self.contract, "original_crv_exact_reproduction": False, "status": "computed_native_local_jacobian",
                    "policy": trace.stamp.__dict__, "full_prefix_hash": trace.prefix_hash, "full_prefix_tokens": positions,
                    "source_prefix_hash": trace.prefix_hash,
                    "logit_target": "legacy_after_step_raw_next_token_logit" if action_targets is None else "teacher_forced_action_value_mean_unwarped_logprob",
                    "selected_logit_probability_mass": float(top_prob.sum()) if top_prob is not None else None,
                    "forward_parity_max_abs": trace.parity_max_abs, "backend_signature": self._backend_identity,
                    "runtime": self.runtime_metadata(),
                    "trace_runtime": {"logits": {"device": str(trace.logits.device), "dtype": str(trace.logits.dtype)},
                                      "mlp_leaves": {path: {"device": str(value.device), "dtype": str(value.dtype)} for path, value in trace.mlp_leaves.items()},
                                      "reconstructions": {path: {"device": str(value.device), "dtype": str(value.dtype)} for path, value in trace.reconstructions.items()},
                                      "transcoder_activations": {path: {"device": str(value.device), "dtype": str(value.dtype)} for path, value in trace.activations.items()}},
                    "constant_runtime": {binding.module_path: {
                        "reconstruction_device": str(trace.reconstructions[binding.module_path].device),
                        "selected_contributions_device": str(selected_contributions[binding.module_path].device),
                        "residual_devices": {kind: str(residual_vectors[(layer, kind)].device) for kind in ("reconstruction_error", "omitted_features", "decoder_bias")},
                        "residual_projection_device": str(residual_vectors[(layer, "reconstruction_error")].device),
                        "residual_projection_dtype": "torch.float32",
                        "persistent_dense_constant_bytes": 4 * trace.reconstructions[binding.module_path].numel() * trace.reconstructions[binding.module_path].element_size(),
                        "decoder_bias_storage": "broadcast view",
                        **residual_projection_counts[binding.module_path],
                    } for layer, binding in enumerate(self.bindings)},
                    "gradient_rules": {"declared_mlp_outputs": "same-valued detached leaves", "attention_gdn": "native local Jacobian; not frozen", "norm_scales": "native local Jacobian; not frozen", "other_residual_paths": "native local Jacobian"},
                    "architecture_review": self.architecture_review, "fidelity": trace.fidelity,
                    "fidelity_scope": "full_prefix" if action_targets is None else "action_causal_horizon",
                    "feature_selection": "global positive activation magnitude; NOT original CRV iterative influence selection",
                    "source_selection_rule": self.source_selection_rule,
                    "source_selection_proof": self.source_selection_proof,
                    "structurally_excluded_active_features": excluded_active,
                    "structurally_excluded_activation_mass": excluded_activation_mass,
                    "eligible_active_features": total_active - excluded_active,
                    "coverage_denominator": "all positive activations within original causal horizon, including structurally excluded final-MLP positions",
                    "total_active_features": total_active, "selected_features": len(chosen),
                    "feature_count_coverage": len(chosen) / total_active if total_active else 1.0,
                    "full_prefix_active_features": full_prefix_active, "causal_horizon": horizon,
                    "eligible_causal_positions": horizon + 1, "excluded_future_positions": positions - horizon - 1,
                    "future_source_gradient_max_abs": future_gradient_max_abs,
                    "activation_mass_coverage": sum(item[0] for item in chosen) / total_activation_mass if total_activation_mass else 1.0,
                    "residual_granularity": "three separate sources per layer aggregated across full prefix",
                    "node_influence_mass": self.node_influence_mass, "edge_row_mass": self.edge_row_mass,
                    "raw_nonzero_edges": len(edge_values), "retained_edges": int(final_edge_mask.sum()),
                    "edge_absolute_mass_coverage": float(np.abs(weight_array[final_edge_mask]).sum() / edge_mass) if edge_mass else 1.0,
                    "residual_node_indices": {kind: [index for index, record in enumerate(records) if record["kind"] == kind] for kind in ("reconstruction_error", "omitted_features", "decoder_bias")},
                    "backward_targets": len(target_nodes), "backend_validation": self._validation,
                    "transcoders": [{"module_path": binding.module_path, "checkpoint_hash": binding.training_metadata.get("transcoder_hash"),
                                      "training_policy": binding.training_metadata.get("provenance", {}).get("policy_fingerprint")} for binding in self.bindings]}
        if action_targets is not None:
            metadata.update(action_targets=action_targets.to_dict(), target_spec_hash=action_targets.target_spec_hash,
                            objective_value=float(action_targets.mean_logprob(trace.reference_logits)), sink_count=1,
                            sink_influence_seed=1.0, sink_influence_seed_is_probability=False)
        payload = GraphPayload(types, source_array, target_array, weight_array, node_mask, np.array(edge_keep, dtype=bool),
                               layers, node_layers, activations, influence, top_prob.cpu().numpy() if top_prob is not None else None,
                               metadata, graph_kind=self.contract)
        payload.validate()
        self._guard(trace.stamp)
        return NativeAttributionResult(payload, metadata, records)

    def attribute(self, input_ids: torch.Tensor, stamp: PolicyStamp, *, model_kwargs: Mapping[str, Any] | None = None,
                  action_targets: ActionTargets | None = None) -> NativeAttributionResult:
        if self._validation is None or not self._validation["passed"]:
            raise UnsupportedAttribution("Run validate_backend on this architecture/backend before emitting research graphs.")
        return self._assemble(self.capture(input_ids, stamp, model_kwargs=model_kwargs, action_targets=action_targets))

    @torch.no_grad()
    def intervention_value(self, trace: LocalTrace, result: NativeAttributionResult,
                           source_index: int, target_index: int, scale: float) -> float:
        """Actual cut-model intervention. Intermediate MLP outputs stay fixed."""
        self._guard(trace.stamp)
        source, target = result.node_records[source_index], result.node_records[target_index]
        if target["kind"] not in {"feature", "logit", "score"} or source["kind"] in {"logit", "score"}:
            raise ValueError("Intervention requires a feature/logit/action-score target and a non-sink source.")
        handles, captured_inputs = [], {}
        directions: dict[str, torch.Tensor] = {}
        if source["kind"] != "token":
            binding = self.bindings[source["layer"]]
            path = binding.module_path
            original = trace.mlp_leaves[path].detach()
            if source["kind"] == "feature":
                vector = torch.zeros_like(original)
                vector[0, source["position"]] = binding.transcoder.decoder.weight[:, source["feature"]].detach().to(original) * source["activation"]
            elif source["kind"] == "reconstruction_error":
                reconstruction = trace.reconstructions[path]
                vector = (original.to(reconstruction) - reconstruction).to(original)
            elif source["kind"] == "decoder_bias":
                vector = binding.transcoder.decoder.bias.detach().to(original).view(1, 1, -1).expand_as(original)
            else:
                reconstruction = trace.reconstructions[path]
                selected = torch.zeros_like(reconstruction)
                for record in result.node_records:
                    if record["kind"] == "feature" and record["layer"] == source["layer"]:
                        selected[0, record["position"]] += binding.transcoder.decoder.weight[:, record["feature"]].detach().to(reconstruction) * record["activation"]
                vector = (reconstruction - binding.transcoder.decoder.bias.detach().to(reconstruction).view(1, 1, -1) - selected).to(original)
            directions[path] = vector
        try:
            for path, module in self.embedding_modules.items():
                def embedding_hook(module, args, output, path=path):
                    baseline = trace.embedding_leaves[path].detach().clone()
                    if source["kind"] == "token":
                        baseline[0, source["position"]] *= 1 + scale
                    return baseline
                handles.append(module.register_forward_hook(embedding_hook))
            for path, module in self.modules.items():
                def mlp_hook(module, args, kw, output, path=path):
                    captured_inputs[path] = _tensor_input(args, kw).detach()
                    baseline = trace.mlp_leaves[path].detach()
                    return baseline + scale * directions[path] if path in directions else baseline
                handles.append(module.register_forward_hook(mlp_hook, with_kwargs=True))
            all_logits = self.policy(input_ids=trace.input_ids, **trace.model_kwargs).logits
            logits = all_logits[:, -1] if trace.action_targets is None else all_logits
            if target["kind"] == "logit":
                value = float(logits[0, target["token_id"]])
            elif target["kind"] == "score":
                value = float(trace.action_targets.mean_logprob(logits))
            else:
                binding = self.bindings[target["layer"]]
                x = captured_inputs[binding.module_path].to(binding.transcoder.encoder.weight)
                value = float(binding.transcoder.encode(x)[0, target["position"], target["feature"]])
            self._guard(trace.stamp)
            return value
        finally:
            for handle in handles:
                handle.remove()

    def validate_backend(self, input_ids: torch.Tensor, stamp: PolicyStamp, *,
                         model_kwargs: Mapping[str, Any] | None = None, epsilon: float = 1e-3,
                         rtol: float = 0.05, atol: float = 1e-4, max_edges: int = 4,
                         action_targets: ActionTargets | None = None) -> dict[str, Any]:
        """Fail-closed finite-difference gate, including token and feature sources."""
        if epsilon <= 0 or max_edges < 2:
            raise ValueError("Positive epsilon and at least two validation edges are required.")
        self._validation = None
        trace = self.capture(input_ids, stamp, model_kwargs=model_kwargs, action_targets=action_targets)
        result = self._assemble(trace, feature_cap=min(8, self.max_feature_nodes))
        graph = result.graph
        ordered = np.argsort(-np.abs(graph.weights))
        chosen, kinds = [], set()
        for required_kind in ("token", "feature"):
            candidates = [int(index) for index in ordered
                          if result.node_records[int(graph.sources[index])]["kind"] == required_kind
                          and (action_targets is None or result.node_records[int(graph.targets[index])]["kind"] == "score")]
            if not candidates:
                raise UnsupportedAttribution(f"Preflight has no nonzero {required_kind} edge to validate.")
            chosen.append(candidates[0])
            kinds.add(required_kind)
        chosen.extend(int(index) for index in ordered if int(index) not in chosen)
        checks = []
        for index in chosen[:max_edges]:
            source, target = int(graph.sources[index]), int(graph.targets[index])
            positive = self.intervention_value(trace, result, source, target, epsilon)
            negative = self.intervention_value(trace, result, source, target, -epsilon)
            numerical = (positive - negative) / (2 * epsilon)
            analytical = float(graph.weights[index])
            passed = bool(np.isclose(numerical, analytical, rtol=rtol, atol=atol))
            checks.append({"source": result.node_records[source], "target": result.node_records[target],
                           "analytical": analytical, "finite_difference": numerical, "passed": passed})
        report = {"passed": all(check["passed"] for check in checks), "contract": self.contract,
                  "source_selection_rule": self.source_selection_rule,
                  "source_selection_proof": self.source_selection_proof,
                  "backend_signature": self._backend_identity, "policy_state_id": stamp.state_id,
                  "runtime": self.runtime_metadata(),
                  "prefix_hash": trace.prefix_hash, "epsilon": epsilon, "rtol": rtol, "atol": atol,
                  "forward_parity_max_abs": trace.parity_max_abs, "checks": checks}
        if action_targets is not None:
            report.update(target_spec_hash=action_targets.target_spec_hash,
                          objective_value=float(action_targets.mean_logprob(trace.reference_logits)),
                          required_direct_action_sink_source_kinds=["token", "feature"])
        if not report["passed"]:
            raise UnsupportedAttribution("Cut-model finite differences did not match direct edges: " + json.dumps(report))
        self._validation = report
        return report
