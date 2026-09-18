"""Verified native-HF Qwen3.5 policy, separate from all interpretability models.

Only the audited 4B/9B post-trained checkpoints are loadable. Generation records
the exact returned token sequence; capture replays one complete prefix without a
KV/GDN cache, labels, truncation, or a modified MLP forward. Generation-score
statistics describe HF's *processed sampling distribution*, not raw LM logits.

The controller must serialize access and call ``mark_state`` immediately after
EVERY ES perturbation, update, restore, or reload, before any new observation.
``model`` is the original HF ConditionalGeneration model including its lm_head.
No transcoder, uncertainty head, or surrogate model is registered on it.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import threading
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING
import uuid

import torch

if TYPE_CHECKING:
    from .native_attribution import PolicyStamp


TRANSFORMERS_VERSION = "5.17.0"
ARCHITECTURE_SOURCE = (
    "https://github.com/huggingface/transformers/blob/v5.17.0/"
    "src/transformers/models/qwen3_5/modeling_qwen3_5.py"
)
SUPPORTED_CHECKPOINTS = {
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}


class PolicyIntegrityError(RuntimeError):
    """Checkpoint, receipt, model structure, or native-forward contract mismatch."""


class PolicyStateError(RuntimeError):
    """An observation would use an unannounced or stale policy state."""


class PolicyContextError(ValueError):
    """The caller must shorten/reorganize its messages explicitly."""


def declared_runtime_options(protocol: Mapping[str, Any], *, device: str | None = None,
                             cuda_memory_fraction: float | None = None) -> dict[str, Any]:
    """Resolve the locked study runtime; explicit CLI overrides must agree.

    Missing runtime records preserve historical defaults for old technical
    fixtures. New scientific protocols must provide an explicit runtime record.
    This helper does not itself claim backend validation or allocate a model.
    """
    declared = dict(protocol.get("policy_runtime", {}))
    options = {"dtype": "bfloat16", "float32_matmul_precision": "highest", "attn_implementation": "eager",
               "sdpa_backend": "auto", "cpu_embedding_and_lm_head": False, "attention_checkpointing": False, "device": "cuda:0",
               "cuda_memory_fraction": .40, "torch_cpu_threads": 2}
    options.update({key: value for key, value in declared.items() if key in options})
    for key, explicit in (("device", device), ("cuda_memory_fraction", cuda_memory_fraction)):
        if explicit is not None:
            if key in declared and explicit != declared[key]:
                raise PolicyIntegrityError(f"CLI {key} differs from the locked policy runtime.")
            options[key] = explicit
    if (options["dtype"] not in {"bfloat16", "float32"} or options["attn_implementation"] not in {"eager", "sdpa"}
            or options["sdpa_backend"] not in {"auto", "math", "efficient"}
            or type(options["cpu_embedding_and_lm_head"]) is not bool
            or type(options["attention_checkpointing"]) is not bool
            or type(options["torch_cpu_threads"]) is not int or options["torch_cpu_threads"] < 1
            or type(options["cuda_memory_fraction"]) not in (int, float)
            or not 0 < options["cuda_memory_fraction"] <= 1):
        raise PolicyIntegrityError("Unsupported declared numerical policy runtime.")
    if options["dtype"] == "float32" and options["float32_matmul_precision"] != "highest":
        raise PolicyIntegrityError("The FP32 policy requires highest matmul precision with TF32 disabled.")
    if options["sdpa_backend"] != "auto" and options["attn_implementation"] != "sdpa":
        raise PolicyIntegrityError("An explicit SDPA kernel requires the SDPA attention implementation.")
    return options


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _tensor_hash(tensors: Mapping[str, torch.Tensor]) -> str:
    from .esopt import tensor_state_hash
    return tensor_state_hash(dict(tensors))


@dataclass(frozen=True)
class VerifiedCheckpoint:
    directory: Path
    manifest_json: str
    checkpoint_hash: str
    metadata_hash: str
    receipt_hash: str

    @property
    def entry(self) -> dict[str, Any]:
        # A fresh copy prevents accidental mutation of the verified identity.
        return json.loads(self.manifest_json)


def verify_checkpoint(manifest: str | Path | Mapping[str, Any], model_key: str,
                      directory: str | Path) -> VerifiedCheckpoint:
    """Check the completed download receipt AND rehash files before model loading.

    The one-time read prevents a stale receipt from certifying changed weights.
    A metadata-only receipt, absent file, wrong revision, or extra/unverified
    receipt entry fails closed. No network requests or automatic downloads occur.
    """
    payload = json.loads(Path(manifest).read_text()) if isinstance(manifest, (str, Path)) else dict(manifest)
    matches = [m for m in payload.get("models", []) if m.get("key") == model_key]
    if len(matches) != 1:
        raise PolicyIntegrityError("Select exactly one current manifest model, not a deferred candidate.")
    entry = json.loads(json.dumps(matches[0]))
    if SUPPORTED_CHECKPOINTS.get(entry.get("model_id")) != entry.get("revision"):
        raise PolicyIntegrityError("Only the fixed audited Qwen3.5-4B/9B revisions are supported.")
    if entry.get("model_class") != "Qwen3_5ForConditionalGeneration":
        raise PolicyIntegrityError("Expected the complete native ConditionalGeneration checkpoint.")
    if entry.get("metadata_integrity_status") != "fixed_revision_official_hashes_verified":
        raise PolicyIntegrityError("Manifest metadata checksums are not verified.")
    directory = Path(directory).resolve()
    try:
        receipt = json.loads((directory / "download_receipt.json").read_text())
    except (OSError, ValueError) as exc:
        raise PolicyIntegrityError(f"A readable completed download receipt is required: {exc}") from exc
    if (receipt.get("status") != "complete_verified" or receipt.get("scope") != "complete_checkpoint"
            or receipt.get("model_id") != entry["model_id"] or receipt.get("revision") != entry["revision"]):
        raise PolicyIntegrityError("Receipt must certify this complete checkpoint at this exact revision.")
    expected = entry["metadata_files"] + entry["weight_files"]
    names = [item["path"] for item in expected]
    records = receipt.get("files", [])
    if (len(names) != len(set(names)) or len(records) != len(names)
            or {r.get("path") for r in records} != set(names)):
        raise PolicyIntegrityError("Receipt files do not exactly match the pinned manifest.")
    by_name = {r["path"]: r for r in records}
    accepted_statuses = {"verified_existing", "verified_resumed", "downloaded_verified", "canonical_inline_verified"}
    for item in expected:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in item["path"]:
            raise PolicyIntegrityError("Unsafe manifest path.")
        expected_hash = item.get("sha256", item.get("git_blob_sha1"))
        if (not isinstance(expected_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}" if "sha256" in item else r"[0-9a-f]{40}", expected_hash)):
            raise PolicyIntegrityError(f"Missing official checksum: {item['path']}")
        record = by_name[item["path"]]
        if record.get("status") not in accepted_statuses or record.get("hash") != expected_hash:
            raise PolicyIntegrityError(f"Receipt does not verify {item['path']}.")
        target = directory / relative
        if not target.is_file() or target.stat().st_size != item["size_bytes"]:
            raise PolicyIntegrityError(f"Missing or incorrectly sized checkpoint file: {item['path']}")
        digest = hashlib.sha256() if "sha256" in item else hashlib.sha1()
        if "sha256" not in item:
            digest.update(f"blob {item['size_bytes']}\0".encode())
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise PolicyIntegrityError(f"Checkpoint file content changed: {item['path']}")
    def identity(items):
        return [{k: f[k] for k in ("path", "size_bytes", "sha256", "git_blob_sha1") if k in f}
                for f in sorted(items, key=lambda f: f["path"])]
    checkpoint_hash = _digest({"model_id": entry["model_id"], "revision": entry["revision"],
                               "weights": identity(entry["weight_files"])})
    return VerifiedCheckpoint(directory, json.dumps(entry, sort_keys=True), checkpoint_hash,
                              _digest(identity(entry["metadata_files"])), _digest(receipt))


@dataclass(frozen=True)
class DecodingConfig:
    max_new_tokens: int = 384
    do_sample: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    repetition_penalty: float = 1.0
    use_cache: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.do_sample, bool) or not isinstance(self.use_cache, bool):
            raise ValueError("do_sample and use_cache must be explicit booleans.")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        if not 0 < self.top_p <= 1 or isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_p must be in (0,1] and top_k nonnegative.")
        if not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be finite and positive.")

    def generation_kwargs(self) -> dict[str, Any]:
        result = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample,
                  "repetition_penalty": self.repetition_penalty, "use_cache": self.use_cache,
                  "num_beams": 1, "num_return_sequences": 1, "return_dict_in_generate": True,
                  "output_scores": True, "renormalize_logits": True}
        if self.do_sample:
            result.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        return result


@dataclass
class ActionGeneration:
    success: bool
    raw_text: str
    parsed_action: Any
    input_ids_with_completion: torch.Tensor
    prompt_token_count: int
    completion_count: int
    logprobs: tuple[float, ...]
    entropy: tuple[float, ...]
    model_stamp: PolicyStamp
    configuration_fingerprint: str
    seed: int
    prefix_hash: str
    finish_reason: str
    failure_code: str | None = None
    error: str | None = None
    score_semantics: str = "hf_processed_generation_distribution"
    schema_fingerprint: str | None = None
    policy_runtime: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        record = {key: value for key, value in vars(self).items()
                  if key not in {"input_ids_with_completion", "model_stamp"}}
        record["input_ids_with_completion"] = self.input_ids_with_completion.tolist()
        record["model_stamp"] = asdict(self.model_stamp)
        return record


@dataclass
class PrefixCapture:
    stamp: PolicyStamp
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prefix_hash: str
    mlp_inputs: dict[str, torch.Tensor]
    mlp_outputs: dict[str, torch.Tensor]
    hidden_summaries: dict[str, dict[str, Any]]
    last_logits: torch.Tensor
    parity_max_abs: float
    parity_passed: bool
    contract: str = "unmodified_native_mlp_full_prefix_capture_v1"


def _strict_json(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f"Non-JSON constant: {value}")
    result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    # Reject overflowing JSON numbers (e.g. 1e999), too.
    json.dumps(result, allow_nan=False)
    return result


def parse_last_json(raw_text: str) -> Any:
    """Extract the last complete JSON value outside Qwen thinking content.

    Objects/arrays may be surrounded by prose or fences. A scalar is accepted
    only when the entire final content is JSON. Nested objects inside an
    unfinished outer object are not salvaged as independent actions.
    """
    text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
    text = text.split("<think>", 1)[0].strip()
    if not text:
        raise ValueError("No final action content outside thinking tokens.")
    try:
        return _strict_json(text)
    except ValueError:
        pass
    candidates = []
    start = None
    stack: list[str] = []
    quoted = escaped = False
    for index, character in enumerate(text):
        if start is None:
            if character in "{[":
                start, stack, quoted, escaped = index, [character], False, False
            continue
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in "{[":
            stack.append(character)
        elif character in "}]":
            if not stack or (stack[-1], character) not in {( "{", "}"), ("[", "]")}:
                start, stack = None, []
                continue
            stack.pop()
            if not stack:
                try:
                    candidates.append(_strict_json(text[start:index + 1]))
                except ValueError:
                    pass
                start = None
    if not candidates:
        raise ValueError("No complete valid JSON action was generated.")
    return candidates[-1]


_SCHEMA_KEYS = {"type", "enum", "const", "properties", "required", "additionalProperties", "items",
                "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems",
                "minLength", "maxLength", "pattern", "uniqueItems", "anyOf", "oneOf", "allOf",
                "title", "description", "$schema", "default"}


def _check_schema(schema: Mapping[str, Any] | bool) -> None:
    """Fail on unsupported JSON Schema features rather than silently ignoring them."""
    if isinstance(schema, bool):
        return
    if not isinstance(schema, Mapping) or set(schema) - _SCHEMA_KEYS:
        raise ValueError("Unsupported legal schema; use the documented subset or an action_validator.")
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    if not isinstance(types, list) or set(types) - {"object", "array", "string", "number", "integer", "boolean", "null"}:
        raise ValueError("Invalid schema type.")
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            if not isinstance(schema[key], list) or not schema[key]:
                raise ValueError(f"{key} must be a nonempty schema list.")
            for child in schema[key]:
                _check_schema(child)
    if not isinstance(schema.get("properties", {}), Mapping):
        raise ValueError("Schema properties must be a mapping.")
    if "required" in schema and (not isinstance(schema["required"], list)
                                  or not all(isinstance(key, str) for key in schema["required"])):
        raise ValueError("Schema required must be a list of property names.")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError("Schema enum must be a nonempty list.")
    for child in schema.get("properties", {}).values():
        _check_schema(child)
    for key in ("additionalProperties", "items"):
        if key in schema:
            _check_schema(schema[key])
    if "pattern" in schema:
        re.compile(schema["pattern"])


def _schema_error(value: Any, schema: Mapping[str, Any] | bool, path: str = "$") -> str | None:
    if isinstance(schema, bool):
        return None if schema else f"{path}: disallowed by schema"
    checks = {"object": isinstance(value, dict), "array": isinstance(value, list),
              "string": isinstance(value, str), "number": isinstance(value, (int, float)) and not isinstance(value, bool),
              "integer": isinstance(value, int) and not isinstance(value, bool),
              "boolean": isinstance(value, bool), "null": value is None}
    types = schema.get("type")
    if types and not any(checks[t] for t in ([types] if isinstance(types, str) else types)):
        return f"{path}: expected type {types}"
    if "const" in schema and _digest(value) != _digest(schema["const"]):
        return f"{path}: value differs from const"
    if "enum" in schema and not any(_digest(value) == _digest(v) for v in schema["enum"]):
        return f"{path}: value is not a legal enum member"
    for key, required_count in (("anyOf", None), ("oneOf", 1), ("allOf", -1)):
        if key in schema:
            count = sum(_schema_error(value, child, path) is None for child in schema[key])
            if (key == "anyOf" and count == 0) or (required_count == 1 and count != 1) or (required_count == -1 and count != len(schema[key])):
                return f"{path}: {key} failed"
    if checks["number"]:
        for key, failed in (("minimum", lambda n: value < n), ("maximum", lambda n: value > n),
                            ("exclusiveMinimum", lambda n: value <= n), ("exclusiveMaximum", lambda n: value >= n)):
            if key in schema and failed(schema[key]):
                return f"{path}: {key} failed"
    if isinstance(value, (str, list)):
        low, high = ("minLength", "maxLength") if isinstance(value, str) else ("minItems", "maxItems")
        if (low in schema and len(value) < schema[low]) or (high in schema and len(value) > schema[high]):
            return f"{path}: length constraint failed"
    if isinstance(value, str) and "pattern" in schema and re.search(schema["pattern"], value) is None:
        return f"{path}: pattern failed"
    if isinstance(value, list):
        if schema.get("uniqueItems") and len({_digest(v) for v in value}) != len(value):
            return f"{path}: duplicate array item"
        for index, child in enumerate(value):
            error = _schema_error(child, schema.get("items", True), f"{path}[{index}]")
            if error:
                return error
    if isinstance(value, dict):
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            return f"{path}: missing required keys {sorted(missing)}"
        properties = schema.get("properties", {})
        for key, child in value.items():
            error = _schema_error(child, properties.get(key, schema.get("additionalProperties", True)), f"{path}.{key}")
            if error:
                return error
    return None


class QwenPolicyAdapter:
    def __init__(self, model: torch.nn.Module, tokenizer: Any, checkpoint: VerifiedCheckpoint, *,
                 decoding: DecodingConfig | None = None, enable_thinking: bool = False,
                 chat_template_kwargs: Mapping[str, Any] | None = None,
                 context_limit_tokens: int | None = None, max_input_tokens: int | None = None,
                 freeze_unused_vision: bool = True, cpu_embedding_and_lm_head: bool = False,
                 execution_device: str | torch.device | None = None, sdpa_backend: str = "auto",
                 attention_checkpointing: bool = False) -> None:
        self.model, self.tokenizer, self.checkpoint = model, tokenizer, checkpoint
        self.decoding = decoding or DecodingConfig()
        self.entry = checkpoint.entry
        self.model_id = self.entry["model_id"]
        self.revision = self.entry["revision"]
        if SUPPORTED_CHECKPOINTS.get(self.model_id) != self.revision:
            raise PolicyIntegrityError("Unsupported checkpoint identity.")
        self.checkpoint_hash = checkpoint.checkpoint_hash
        if type(cpu_embedding_and_lm_head) is not bool:
            raise ValueError("cpu_embedding_and_lm_head must be an explicit boolean.")
        self.execution_input_device = torch.device(execution_device) if execution_device is not None else model.get_input_embeddings().weight.device
        self.cpu_embedding_and_lm_head = bool(cpu_embedding_and_lm_head)
        self.sdpa_backend = sdpa_backend
        self._placement_handles = []
        self._attention_checkpoint_handle = None
        if sdpa_backend not in {"auto", "math", "efficient"}:
            raise ValueError("sdpa_backend must be auto, math, or efficient.")
        attention = getattr(model.config, "_attn_implementation", None)
        if sdpa_backend != "auto" and attention != "sdpa":
            raise ValueError("An explicit SDPA kernel requires attn_implementation='sdpa'.")
        if sdpa_backend != "auto":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(sdpa_backend == "efficient")
            torch.backends.cuda.enable_math_sdp(sdpa_backend == "math")
            if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(False)
        if self.cpu_embedding_and_lm_head:
            self._install_cpu_embedding_and_head()
        if type(attention_checkpointing) is not bool:
            raise ValueError("attention_checkpointing must be an explicit boolean.")
        if attention_checkpointing:
            from .attention_checkpointing import install_attention_checkpointing
            self._attention_checkpoint_handle = install_attention_checkpointing(self.model)
            self.model._matdiscovery_runtime_config = {**getattr(self.model, "_matdiscovery_runtime_config", {}),
                "attention_checkpointing": self._attention_checkpoint_handle.metadata()}
        self._chat_kwargs = json.loads(json.dumps(dict(chat_template_kwargs or {}), allow_nan=False))
        forbidden = {"tokenize", "return_tensors", "return_dict", "truncation", "max_length", "add_generation_prompt"}
        if forbidden & set(self._chat_kwargs):
            raise ValueError("Chat kwargs cannot override tokenization, generation boundary, or truncation policy.")
        if "enable_thinking" in self._chat_kwargs and self._chat_kwargs["enable_thinking"] != enable_thinking:
            raise ValueError("Conflicting enable_thinking settings.")
        self._chat_kwargs["enable_thinking"] = bool(enable_thinking)
        self.model.eval()
        self._validate_structure()
        if freeze_unused_vision:
            for parameter in self.model.get_submodule("model.visual").parameters():
                parameter.requires_grad_(False)
        self.vision_handling = "retained_frozen_for_autograd" if freeze_unused_vision else "retained_unused_in_text_forward"
        native_limit = int(self.model.config.text_config.max_position_embeddings)
        if context_limit_tokens is not None and (isinstance(context_limit_tokens, bool)
                                                 or not isinstance(context_limit_tokens, int)
                                                 or not 0 < context_limit_tokens <= native_limit):
            raise ValueError("context_limit_tokens must be positive and no larger than native context.")
        self.context_limit_tokens = context_limit_tokens or native_limit
        if max_input_tokens is not None and (isinstance(max_input_tokens, bool)
                                             or not isinstance(max_input_tokens, int)
                                             or not 0 < max_input_tokens <= self.context_limit_tokens):
            raise ValueError("max_input_tokens must be a positive integer within the context limit.")
        self.max_input_tokens = max_input_tokens
        self.configuration_fingerprint = _digest(self._configuration())
        self._lock = threading.RLock()
        self._busy = False
        self._session = uuid.uuid4().hex
        self._state_counter = self._generation = 0
        self._perturbation_seed: int | None = None
        self._perturbation_sigma: float | None = None
        self._versions = self._parameter_versions()
        self._state_id = self._new_state_id("verified_initial_load")

    @classmethod
    def from_verified_checkpoint(cls, manifest: str | Path | Mapping[str, Any], model_key: str,
                                 directory: str | Path, *, device: str | torch.device = "cuda:0",
                                 attn_implementation: str = "eager", dtype: str | torch.dtype = "bfloat16",
                                 cpu_embedding_and_lm_head: bool = False, sdpa_backend: str = "auto",
                                 attention_checkpointing: bool = False,
                                 **kwargs) -> QwenPolicyAdapter:
        selected_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32,
                          torch.bfloat16: torch.bfloat16, torch.float32: torch.float32}.get(dtype)
        if selected_dtype is None or attn_implementation not in {"eager", "sdpa"}:
            raise ValueError("Use explicit BF16/FP32 native weights and an eager/SDPA attention backend.")
        if selected_dtype is torch.float32:
            # These process-global settings are recorded and guarded. A caller
            # must not run a differently configured policy in the same process.
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        checkpoint = verify_checkpoint(manifest, model_key, directory)
        import transformers
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise PolicyIntegrityError(f"Expected transformers=={TRANSFORMERS_VERSION}, got {transformers.__version__}.")
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(checkpoint.directory), local_files_only=True,
                                                               trust_remote_code=False)
        # Official v5.17 AutoModelForImageTextToText mapping selects
        # Qwen3_5ForConditionalGeneration, preserving the complete LM head.
        model = transformers.AutoModelForImageTextToText.from_pretrained(
            str(checkpoint.directory), local_files_only=True, trust_remote_code=False,
            dtype=selected_dtype, attn_implementation=attn_implementation,
        )
        if type(model).__name__ != "Qwen3_5ForConditionalGeneration":
            raise PolicyIntegrityError("The native ConditionalGeneration loader returned an unexpected class.")
        if cpu_embedding_and_lm_head:
            # Keep the vocabulary matrices on CPU from the outset. Moving the
            # whole model first would needlessly allocate both large matrices on
            # GPU, and 4B's shared embedding/head parameter must stay shared.
            embedding = model.get_input_embeddings()
            def place(module):
                if module is embedding or module is model.lm_head:
                    module.to("cpu")
                    return
                if not list(module.children()):
                    module.to(torch.device(device))
                    return
                for child in module.children():
                    place(child)
                # Native Qwen containers also own direct parameters/buffers
                # (e.g. visual positional embeddings and rotary frequencies).
                for parameter in module.parameters(recurse=False):
                    parameter.data = parameter.data.to(torch.device(device))
                for name, buffer in module.named_buffers(recurse=False):
                    setattr(module, name, buffer.to(torch.device(device)))
            place(model)
        else:
            model.to(torch.device(device))
        if selected_dtype is torch.float32 and any(p.is_floating_point() and p.dtype is not torch.float32 for p in model.parameters()):
            raise PolicyIntegrityError("FP32 runtime requires every retained floating-point policy parameter to be FP32.")
        return cls(model, tokenizer, checkpoint, execution_device=device,
                   cpu_embedding_and_lm_head=cpu_embedding_and_lm_head, sdpa_backend=sdpa_backend,
                   attention_checkpointing=attention_checkpointing, **kwargs)

    def _install_cpu_embedding_and_head(self) -> None:
        """Materialized CPU vocabulary parameters with differentiable transfers.

        Hooks are installed before observation/attribution hooks. Thus an
        embedding cut sees the real GPU hidden states used by the decoder.
        There is no parameter proxy, detachment, quantization or untied clone.
        """
        embedding, head = self.model.get_input_embeddings(), self.model.get_output_embeddings()
        tied = embedding.weight is head.weight
        embedding.to("cpu")
        head.to("cpu")
        if tied and embedding.weight is not head.weight:
            raise PolicyIntegrityError("CPU placement changed the native tied embedding/head parameter.")
        def to_cpu(module, args, kwargs):
            if args:
                if not isinstance(args[0], torch.Tensor):
                    raise PolicyIntegrityError("Vocabulary module input must be a tensor.")
                return (args[0].to("cpu"), *args[1:]), kwargs
            if "input" not in kwargs or not isinstance(kwargs["input"], torch.Tensor):
                raise PolicyIntegrityError("Unreviewed vocabulary module call signature.")
            return args, {**kwargs, "input": kwargs["input"].to("cpu")}
        def to_execution(module, args, kwargs, result):
            if not isinstance(result, torch.Tensor):
                raise PolicyIntegrityError("Vocabulary module must return a tensor.")
            return result.to(self.execution_input_device)
        for module in (embedding, head):
            self._placement_handles.append(module.register_forward_pre_hook(to_cpu, with_kwargs=True, prepend=True))
            self._placement_handles.append(module.register_forward_hook(to_execution, with_kwargs=True, prepend=True))
        self.model._matdiscovery_runtime_config = {"cpu_embedding_and_lm_head": True,
            "execution_input_device": str(self.execution_input_device), "embedding_head_tied": tied,
            "transfer_contract": "differentiable_cpu_vocab_gpu_decoder_no_parameter_proxy_v1"}

    def runtime_precision_record(self) -> dict[str, Any]:
        """JSON-ready numerical runtime identity, independent of prompt length.

        This is a declared runtime configuration, not a VJP/fidelity pass.
        max_input_tokens and decoding limits belong to the full configuration.
        """
        inventory = {}
        for parameter in self.model.parameters():
            key = str(parameter.device) + ":" + str(parameter.dtype)
            item = inventory.setdefault(key, {"parameters": 0, "bytes": 0})
            item["parameters"] += parameter.numel()
            item["bytes"] += parameter.numel() * parameter.element_size()
        embedding, head = self.model.get_input_embeddings(), self.model.get_output_embeddings()
        return {"schema": "native_qwen_runtime_precision_v1", "dtype": str(embedding.weight.dtype),
                "attention_implementation": getattr(self.model.config, "_attn_implementation", None),
                "sdpa_backend": self.sdpa_backend, "cpu_embedding_and_lm_head": self.cpu_embedding_and_lm_head,
                "attention_checkpointing": self._attention_checkpoint_handle.metadata() if self._attention_checkpoint_handle else {"enabled": False},
                "execution_input_device": str(self.execution_input_device), "embedding_device": str(embedding.weight.device),
                "lm_head_device": str(head.weight.device), "embedding_head_tied": embedding.weight is head.weight,
                "materialized_parameter_inventory": inventory, "all_parameters_materialized": all(p.device.type != "meta" for p in self.model.parameters()),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "sdpa_flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
                "sdpa_memory_efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                "sdpa_math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
                "sdpa_cudnn_enabled": bool(torch.backends.cuda.cudnn_sdp_enabled()) if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") else None,
                "torch_cpu_threads": torch.get_num_threads(), "torch_version": str(torch.__version__),
                "native_gradient_validation": "required_separately_for_this_runtime"}

    def _validate_structure(self) -> None:
        config = getattr(self.model, "config", None)
        if getattr(config, "model_type", None) != "qwen3_5":
            raise PolicyIntegrityError("Only native dense Qwen3.5 is reviewed for this adapter.")
        text = config.text_config
        architecture = self.entry["architecture"]
        if text.num_hidden_layers != 32 or architecture["num_hidden_layers"] != 32:
            raise PolicyIntegrityError("The selected small Qwen checkpoints require 32 text layers.")
        if text.hidden_size != architecture["hidden_size"]:
            raise PolicyIntegrityError("Runtime hidden size differs from the verified manifest.")
        layers = self.model.get_submodule("model.language_model.layers")
        expected_types = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(32)]
        if len(layers) != 32 or list(text.layer_types) != expected_types:
            raise PolicyIntegrityError("Expected 32 native layers with the reviewed 3:1 GDN/attention schedule.")
        self.mlp_paths = tuple(f"model.language_model.layers.{i}.mlp" for i in range(32))
        modules = dict(self.model.named_modules())
        for index, path in enumerate(self.mlp_paths):
            if path not in modules:
                raise PolicyIntegrityError(f"Missing native MLP boundary: {path}")
            mlp = modules[path]
            expected = {"gate_proj": (text.intermediate_size, text.hidden_size),
                        "up_proj": (text.intermediate_size, text.hidden_size),
                        "down_proj": (text.hidden_size, text.intermediate_size)}
            for name, shape in expected.items():
                if tuple(getattr(getattr(mlp, name, None), "weight", torch.empty(0)).shape) != shape:
                    raise PolicyIntegrityError(f"Unexpected projection shape: {path}.{name}")
            attention_name = "self_attn" if expected_types[index] == "full_attention" else "linear_attn"
            if not hasattr(layers[index], attention_name):
                raise PolicyIntegrityError(f"Missing {attention_name} in layer {index}.")
        if self.model.get_output_embeddings() is not self.model.get_submodule("lm_head"):
            raise PolicyIntegrityError("The original lm_head must remain attached.")
        if self.model.get_output_embeddings().weight.shape[-1] != text.hidden_size:
            raise PolicyIntegrityError("Output embedding dimension does not match the text model.")
        self.model.get_submodule("model.visual")

    def _configuration(self) -> dict[str, Any]:
        generation = getattr(self.model, "generation_config", None)
        defaults = generation.to_dict() if hasattr(generation, "to_dict") else vars(generation) if generation is not None else {}
        defaults = {key: value for key, value in defaults.items() if not key.startswith("_")}
        return {"checkpoint_hash": self.checkpoint_hash, "metadata_hash": self.checkpoint.metadata_hash,
                "decoding": asdict(self.decoding), "chat_template_kwargs": self._chat_kwargs,
                "chat_template": self.tokenizer.chat_template, "generation_defaults": defaults,
                "context_limit_tokens": self.context_limit_tokens,
                "max_input_tokens": self.max_input_tokens,
                "attention_implementation": getattr(self.model.config, "_attn_implementation", None),
                "input_embedding_dtype": str(self.model.get_input_embeddings().weight.dtype),
                "input_embedding_device": str(self.model.get_input_embeddings().weight.device),
                "policy_runtime": self.runtime_precision_record(),
                "vision_handling": self.vision_handling, "torch_version": str(torch.__version__)}

    def _parameter_versions(self) -> tuple:
        return tuple((name, id(p), p._version, tuple(p.shape), str(p.dtype), str(p.device))
                     for name, p in self.model.named_parameters())

    def _new_state_id(self, reason: str) -> str:
        return _digest({"session": self._session, "checkpoint": self.checkpoint_hash,
                        "configuration": self.configuration_fingerprint, "counter": self._state_counter,
                        "generation": self._generation, "reason": reason,
                        "perturbation_seed": self._perturbation_seed, "perturbation_sigma": self._perturbation_sigma})

    def get_state_id(self) -> str:
        return self._state_id

    @property
    def state_id(self) -> str:
        return self._state_id

    @property
    def model_stamp(self) -> PolicyStamp:
        from .native_attribution import PolicyStamp
        return PolicyStamp(self._state_id, self.checkpoint_hash, self.model_id, self._generation,
                           self._perturbation_seed, self._perturbation_sigma)

    def mark_state(self, reason: str, *, generation: int | None = None,
                   perturbation_seed: int | None = None, perturbation_sigma: float | None = None) -> PolicyStamp:
        """Call after an atomic mutation and before generation/capture/graph reuse.

        Even an exact restore receives a new state_id. ``requires_grad=False``
        does not exclude a parameter from a forward-only ES optimizer: its scope
        must still be explicit in that optimizer's own parameter manifest.
        """
        if not reason.strip():
            raise ValueError("Record a state transition reason.")
        if generation is not None and (isinstance(generation, bool) or generation < 0):
            raise ValueError("generation must be nonnegative.")
        if perturbation_sigma is not None and (not math.isfinite(perturbation_sigma) or perturbation_sigma < 0):
            raise ValueError("perturbation_sigma must be finite and nonnegative.")
        with self._lock:
            if self._busy:
                raise PolicyStateError("Cannot mutate or mark a policy during an active observation.")
            if _digest(self._configuration()) != self.configuration_fingerprint:
                raise PolicyStateError("Settings/backend changed; construct a new adapter with a new fingerprint.")
            self._validate_structure()
            self._state_counter += 1
            if generation is not None:
                self._generation = generation
            self._perturbation_seed, self._perturbation_sigma = perturbation_seed, perturbation_sigma
            self._versions = self._parameter_versions()
            self._state_id = self._new_state_id(reason)
            return self.model_stamp

    def _guard(self, stamp: PolicyStamp | None = None) -> None:
        if self.model.training:
            raise PolicyStateError("Policy observations require model.eval().")
        if stamp is not None and stamp != self.model_stamp:
            raise PolicyStateError("The requested prefix belongs to a stale policy stamp.")
        if _digest(self._configuration()) != self.configuration_fingerprint:
            raise PolicyStateError("Chat template or generation/backend settings changed.")
        if self._parameter_versions() != self._versions:
            raise PolicyStateError("Parameters changed without mark_state; previous observations are invalid.")

    @contextmanager
    def _observation(self, stamp: PolicyStamp | None = None):
        with self._lock:
            self._guard(stamp)
            if self._busy:
                raise PolicyStateError("Nested policy observations are forbidden.")
            self._busy = True
            try:
                yield self.model_stamp
                self._guard(stamp)
            finally:
                self._busy = False

    def parameter_report(self) -> dict[str, Any]:
        text_ids = {id(p) for p in self.model.get_submodule("model.language_model").parameters()}
        text_ids.update(id(p) for p in self.model.get_output_embeddings().parameters())
        visual_ids = {id(p) for p in self.model.get_submodule("model.visual").parameters()}
        named = list(self.model.named_parameters())
        return {"runtime_total_parameters": sum(p.numel() for _, p in named),
                "checkpoint_file_parameter_count": self.entry["safetensors_parameter_count"],
                "text_including_lm_head_parameters": sum(p.numel() for _, p in named if id(p) in text_ids),
                "gradient_trainable_text_parameters": sum(p.numel() for _, p in named if id(p) in text_ids and p.requires_grad),
                "unused_visual_parameters": sum(p.numel() for _, p in named if id(p) in visual_ids),
                "other_non_text_parameters": sum(p.numel() for _, p in named if id(p) not in text_ids | visual_ids),
                "text_parameter_names": [n for n, p in named if id(p) in text_ids],
                "unused_visual_parameter_names": [n for n, p in named if id(p) in visual_ids],
                "vision_handling": self.vision_handling,
                "gradient_trainable_visual_parameters": sum(p.numel() for _, p in named if id(p) in visual_ids and p.requires_grad),
                "es_scope_must_be_declared_separately": True,
                "file_vs_runtime_note": "HF may omit unused MTP tensors; file totals and runtime parameter counts are distinct."}

    @property
    def architecture_review(self) -> dict[str, Any]:
        return {"verified": True, "source": ARCHITECTURE_SOURCE, "model_type": "qwen3_5",
                "mlp_paths": list(self.mlp_paths), "layers": 32,
                "hidden_size": self.model.config.text_config.hidden_size,
                "backend_gradient_verified": False,
                "note": "Runtime module layout checked; NativeAttributor.validate_backend remains mandatory."}

    def _input_device(self) -> torch.device:
        device = self.execution_input_device
        if device.type == "meta":
            raise PolicyIntegrityError("The execution input device is not materialized; disk-offload proxies are unsupported.")
        return device

    def _render(self, messages: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        if not messages:
            raise ValueError("messages must be nonempty.")
        for message in messages:
            if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
                raise ValueError("This text-only adapter requires string message content; no visual inputs are silently dropped.")
        encoded = self.tokenizer.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, truncation=False, **self._chat_kwargs,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise PolicyIntegrityError("Tokenizer must return the native tokenized chat input_ids.")
        if set(encoded) - {"input_ids", "attention_mask"}:
            raise PolicyIntegrityError("Unexpected tokenizer inputs; the full-prefix text contract needs review.")
        ids = encoded["input_ids"]
        mask = encoded.get("attention_mask", torch.ones_like(ids))
        self._check_prefix(ids, mask)
        if self.max_input_tokens is not None and ids.shape[1] > self.max_input_tokens:
            raise PolicyContextError("Prompt exceeds max_input_tokens; caller must revise messages explicitly.")
        if ids.shape[1] + self.decoding.max_new_tokens > self.context_limit_tokens:
            raise PolicyContextError("Prompt plus configured completion budget exceeds context; caller must revise messages explicitly.")
        return ids.to(self._input_device()), mask.to(self._input_device())

    def _check_prefix(self, ids: torch.Tensor, mask: torch.Tensor) -> None:
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0 or ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError("Provide one nonempty complete integer token prefix.")
        if mask.shape != ids.shape or not bool(torch.all(mask == 1)):
            raise ValueError("Single complete prefixes must be unpadded with an all-one attention mask.")
        if ids.shape[1] > self.context_limit_tokens:
            raise PolicyContextError("Complete prefix exceeds context; no silent truncation is performed.")
        vocabulary_size = self.model.get_input_embeddings().weight.shape[0]
        if bool(torch.any(ids < 0)) or bool(torch.any(ids >= vocabulary_size)):
            raise ValueError("Prefix contains out-of-vocabulary token IDs.")

    def generate_action(self, messages: Sequence[Mapping[str, Any]], *, seed: int,
                        legal_schema: Mapping[str, Any] | bool | None = None,
                        action_validator: Callable[[Any], bool | str | None] | None = None) -> ActionGeneration:
        """Generate once, then validate; format guidance belongs in caller messages.

        No retry, schema coercion, truncation, or fallback action is hidden here.
        A failed result preserves the raw generated text and token sequence.
        """
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63).")
        if legal_schema is not None:
            _check_schema(legal_schema)
        with self._observation() as stamp:
            ids, mask = self._render(messages)
            prompt_count = ids.shape[1]
            complete = ids.detach().cpu().clone()
            raw, logprobs, entropies = "", [], []
            error = failure = None
            action = None
            finish = "generation_failed"
            devices = sorted({p.device.index for p in self.model.parameters() if p.device.type == "cuda"})
            try:
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    torch.random.default_generator.manual_seed(seed)
                    for device in devices:
                        torch.cuda.default_generators[device].manual_seed(seed)
                    generated = self.model.generate(input_ids=ids, attention_mask=mask, **self.decoding.generation_kwargs())
                sequence = generated.sequences
                if (sequence.ndim != 2 or sequence.shape[0] != 1
                        or not torch.equal(sequence[:, :prompt_count], complete.to(sequence.device))):
                    raise PolicyIntegrityError("Generation did not preserve the exact rendered prefix.")
                complete = sequence.detach().cpu().clone()
                tokens = sequence[0, prompt_count:]
                raw = self.tokenizer.decode(tokens.detach().cpu().tolist(), skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)
                if tokens.numel() > self.decoding.max_new_tokens:
                    raise PolicyIntegrityError("Generation exceeded the declared completion budget.")
                if len(generated.scores) != tokens.numel() or tokens.numel() == 0:
                    raise PolicyIntegrityError("Generation scores must align to every actual completion token.")
                for token, score in zip(tokens, generated.scores):
                    if score.ndim != 2 or score.shape[0] != 1 or bool(torch.isnan(score).any()) or bool(torch.isposinf(score).any()):
                        raise PolicyIntegrityError("Invalid generation score tensor.")
                    distribution = torch.log_softmax(score[0].detach().float(), dim=-1)
                    lp = float(distribution[int(token)])
                    probabilities = distribution.exp()
                    entropy = float(-torch.where(probabilities > 0, probabilities * distribution,
                                                 torch.zeros_like(probabilities)).sum())
                    if not math.isfinite(lp) or not math.isfinite(entropy):
                        raise PolicyIntegrityError("Non-finite probability for an actual generated token.")
                    logprobs.append(lp)
                    entropies.append(entropy)
                eos = getattr(self.model.generation_config, "eos_token_id", None)
                eos = [eos] if isinstance(eos, int) else list(eos or [])
                finish = "eos" if int(tokens[-1]) in eos else "length" if tokens.numel() >= self.decoding.max_new_tokens else "stopped"
            except (RuntimeError, ValueError, AttributeError, IndexError) as exc:
                failure, error = "generation_contract_failed", f"{type(exc).__name__}: {exc}"
            if failure is None:
                try:
                    action = parse_last_json(raw)
                except ValueError as exc:
                    failure, error = "invalid_json", str(exc)
            if failure is None and legal_schema is not None:
                error = _schema_error(action, legal_schema)
                if error:
                    failure = "illegal_action"
            if failure is None and action_validator is not None:
                try:
                    verdict = action_validator(action)
                    if verdict is False or isinstance(verdict, str):
                        failure, error = "illegal_action", verdict if isinstance(verdict, str) else "Action validator rejected the action."
                    elif verdict is not None and verdict is not True:
                        failure, error = "validator_contract_failed", "Validator must return True, False, None, or an error string."
                except Exception as exc:
                    failure, error = "validator_failed", f"{type(exc).__name__}: {exc}"
            return ActionGeneration(failure is None, raw, action, complete, prompt_count,
                                    complete.shape[1] - prompt_count, tuple(logprobs), tuple(entropies), stamp,
                                    self.configuration_fingerprint, seed, _tensor_hash({"input_ids": complete}), finish,
                                    failure, error, schema_fingerprint=_digest(legal_schema) if legal_schema is not None else None,
                                    policy_runtime=self.runtime_precision_record())

    def prepare_trace_inputs(self, input_ids: torch.Tensor, *, stamp: PolicyStamp | None = None,
                             attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return the complete original generated sequence for native attribution.

        No tokens are sliced, appended, or re-tokenized. The v2 action-target
        contract selects each emitted token t at prediction position t-1 during
        a causal teacher-forced forward pass; the attributor selects those
        positions and excludes future-token influence from the decision graph.
        """
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        with self._lock:
            self._guard(stamp)
            self._check_prefix(input_ids, mask)
            ids = input_ids.detach().clone().to(self._input_device())
            mask = mask.detach().clone().to(self._input_device())
            return ids, {"attention_mask": mask, "use_cache": False, "logits_to_keep": 1}

    def capture_prefix(self, input_ids: torch.Tensor, *, stamp: PolicyStamp | None = None,
                       attention_mask: torch.Tensor | None = None, parity_atol: float = 1e-6,
                       parity_rtol: float = 1e-5) -> PrefixCapture:
        """Observe original MLP inputs/outputs, with a no-hook parity reference.

        There is deliberately no labels/target argument. The provided prefix is
        replayed in full; later tokens, cached trajectories and completion labels
        are neither looked up nor appended. Capture itself uses no gradients;
        NativeAttributor performs its separate validated Jacobian extraction.
        """
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        self._check_prefix(input_ids, mask)
        with self._observation(stamp) as current_stamp:
            ids, mask = input_ids.detach().clone().to(self._input_device()), mask.detach().clone().to(self._input_device())
            kwargs = {"input_ids": ids, "attention_mask": mask, "use_cache": False, "logits_to_keep": 1}
            with torch.no_grad():
                reference = self.model(**kwargs).logits[:, -1].detach().float().cpu().clone()
            inputs, outputs, summaries = {}, {}, {}
            handles = []
            try:
                for path in self.mlp_paths:
                    def hook(module, args, kw, output, path=path):
                        value = args[0] if args else kw.get("hidden_states", kw.get("x"))
                        expected = (1, ids.shape[1], self.model.config.text_config.hidden_size)
                        if path in inputs or not isinstance(value, torch.Tensor) or not isinstance(output, torch.Tensor):
                            raise PolicyIntegrityError("An MLP boundary did not execute once with tensor input/output.")
                        if tuple(value.shape) != expected or tuple(output.shape) != expected:
                            raise PolicyIntegrityError(f"MLP input/output does not cover the complete prefix: {path}")
                        inputs[path] = value.detach().cpu().clone()
                        outputs[path] = output.detach().cpu().clone()
                        summaries[path] = {"input_shape": list(value.shape), "output_shape": list(output.shape),
                                           "input_mean": float(value.detach().float().mean()),
                                           "output_mean": float(output.detach().float().mean()),
                                           "input_rms": float(value.detach().float().square().mean().sqrt()),
                                           "output_rms": float(output.detach().float().square().mean().sqrt())}
                        # None is critical: observe without replacing the forward value.
                        return None
                    handles.append(self.model.get_submodule(path).register_forward_hook(hook, with_kwargs=True))
                with torch.no_grad():
                    observed = self.model(**kwargs).logits[:, -1].detach().float().cpu().clone()
            finally:
                for handle in handles:
                    handle.remove()
            if list(inputs) != list(self.mlp_paths) or list(outputs) != list(self.mlp_paths):
                raise PolicyIntegrityError("All 32 MLP boundaries must execute in the reviewed order.")
            parity = float((observed - reference).abs().max())
            passed = bool(torch.allclose(observed, reference, atol=parity_atol, rtol=parity_rtol))
            if not passed:
                raise PolicyIntegrityError(f"Observation hooks changed native logits: max_abs={parity}.")
            cpu_ids, cpu_mask = ids.detach().cpu().clone(), mask.detach().cpu().clone()
            return PrefixCapture(current_stamp, cpu_ids, cpu_mask,
                                 _tensor_hash({"input_ids": cpu_ids, "attention_mask": cpu_mask}),
                                 inputs, outputs, summaries, observed, parity, passed)
