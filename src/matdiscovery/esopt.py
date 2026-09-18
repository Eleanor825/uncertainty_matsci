"""Auditable, forward-only Agentic ESOpt for a canonical Hugging Face policy.

The update follows Agentic ESOpt (arXiv:2608.17310v2): one-sided Gaussian
perturbations, population z-scores, and alpha/G * sum(z_i * epsilon_i), WITHOUT
an additional 1/sigma. Noise is regenerated on CPU with an explicit algorithm
version; this is a portability variant of the official device-local sampler.
It is not bitwise compatible with histories produced by the upstream server.

Candidate restoration uses an exact CPU copy, never floating-point subtraction.
Keep ``with optimizer.perturbation(...)`` open for the complete trajectory.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Sequence

import torch


NOISE_ALGORITHM = "blake2b-name-xor-seed/cpu-torch-float32/chunked-v1"
FORMAT_VERSION = 1


@dataclass(frozen=True)
class ParameterRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    aliases: tuple[str, ...]


def stable_parameter_seed(seed: int, name: str) -> int:
    """Stable across Python processes (unlike Python's built-in hash)."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64).")
    tensor_id = int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "little")
    return (seed ^ tensor_id) & ((1 << 64) - 1)


def iter_noise(seed: int, name: str, numel: int, chunk_size: int) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield a reproducible CPU float32 stream; chunk size is part of its identity."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_parameter_seed(seed, name))
    for start in range(0, numel, chunk_size):
        yield start, torch.randn(min(chunk_size, numel - start), generator=generator, dtype=torch.float32)


def population_zscores(rewards: Sequence[float], epsilon: float = 1e-8) -> list[float]:
    if len(rewards) == 0 or not all(math.isfinite(float(x)) for x in rewards):
        raise ValueError("rewards must be a nonempty finite sequence.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    values = torch.tensor(rewards, dtype=torch.float32)
    if not torch.isfinite(values).all():
        raise ValueError("rewards overflow the float32 normalization contract.")
    # Population, not sample, standard deviation; constant rewards yield zero.
    deviation = values.std(unbiased=False)
    if not torch.isfinite(deviation):
        raise ValueError("reward variance overflows float32 normalization.")
    return ((values - values.mean()) / (deviation + epsilon)).tolist()


def cosine_sigma(start: float, end: float, generation: int, generations: int) -> float:
    """Endpoint-inclusive schedule. A one-generation run uses ``start``."""
    if generations < 1 or not 0 <= generation < generations:
        raise ValueError("generation must index the declared schedule.")
    if not all(math.isfinite(x) and x >= 0 for x in (start, end)):
        raise ValueError("sigma endpoints must be finite and nonnegative.")
    if generations == 1:
        return float(start)
    return float(end + (start - end) * (1 + math.cos(math.pi * generation / (generations - 1))) / 2)


def tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor CONTENT, names, shapes and dtypes, rather than a pickle file."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        metadata = json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":"))
        digest.update(metadata.encode() + b"\0")
        raw = tensor.reshape(-1).view(torch.uint8).numpy()
        for start in range(0, raw.size, 8 * 1024 * 1024):
            digest.update(memoryview(raw[start:start + 8 * 1024 * 1024]))
    return digest.hexdigest()


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class AgenticESOpt:
    """ES over the original policy only, with explicit scope and parameter manifest.

    ``policy_model`` must expose the HF causal-LM contract (``config.model_type``
    and input/output embeddings). Known interpretability wrappers and auxiliary
    parameters are rejected rather than silently included. A canonical HF model
    should be passed BEFORE adding any graph extractor or uncertainty head.
    """

    def __init__(
        self,
        policy_model: torch.nn.Module,
        *,
        policy_model_id: str,
        parameter_scope: str = "full",
        target_modules: Sequence[str] | None = None,
        noise_chunk_size: int = 1_048_576,
        normalization_epsilon: float = 1e-8,
    ) -> None:
        if not policy_model_id.strip():
            raise ValueError("Provide a pinned policy model ID/revision.")
        if parameter_scope not in {"full", "all_linear", "lora"}:
            raise ValueError("parameter_scope must be full, all_linear, or lora.")
        if noise_chunk_size < 1:
            raise ValueError("noise_chunk_size must be positive.")
        if normalization_epsilon <= 0 or not math.isfinite(normalization_epsilon):
            raise ValueError("normalization_epsilon must be finite and positive.")
        if target_modules is not None and parameter_scope != "all_linear":
            raise ValueError("target_modules is valid only for all_linear scope.")
        self._validate_policy(policy_model)
        self.model = policy_model
        self.policy_model_id = policy_model_id
        self.parameter_scope = parameter_scope
        self.noise_chunk_size = int(noise_chunk_size)
        self.normalization_epsilon = float(normalization_epsilon)
        self.target_modules = tuple(target_modules) if target_modules is not None else None
        aliases: dict[int, list[str]] = {}
        for name, parameter in self.model.named_parameters(remove_duplicate=False):
            aliases.setdefault(id(parameter), []).append(name)
        embedding_parameter_ids = {id(p) for module in (self.model.get_input_embeddings(), self.model.get_output_embeddings()) if module is not None for p in module.parameters()}
        selected: list[tuple[str, torch.nn.Parameter]] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.is_floating_point():
                if parameter_scope == "full":
                    raise ValueError(f"Full ES cannot update non-floating parameter {name}.")
                continue
            if parameter.device.type == "meta":
                raise ValueError(f"Parameter {name} is not materialized.")
            if not parameter.is_contiguous():
                raise ValueError(f"Parameter {name} must be contiguous for audited chunk updates.")
            if parameter_scope == "lora" and "lora_" not in name:
                continue
            if parameter_scope == "all_linear":
                if id(parameter) in embedding_parameter_ids:
                    continue
                if parameter.ndim != 2 or not name.endswith(".weight"):
                    continue
                if any(part in name for part in ("embed_tokens", "lm_head", "norm")):
                    continue
                module_name = name.rsplit(".", 1)[0].rsplit(".", 1)[-1]
                if self.target_modules is not None and module_name not in self.target_modules:
                    continue
            selected.append((name, parameter))
        if not selected:
            raise ValueError("The selected scope contains no parameters.")
        self._parameters = selected
        self.manifest = [ParameterRecord(n, tuple(p.shape), str(p.dtype), p.numel(), tuple(aliases[id(p)]))
                         for n, p in selected]
        self.history: list[dict[str, Any]] = []
        self.active_perturbation: dict[str, Any] | None = None
        self._snapshot: dict[str, torch.Tensor] | None = None
        self.base_state_hash = self.model_state_hash()

    @staticmethod
    def _validate_policy(model: torch.nn.Module) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("policy_model must be a canonical HF torch module.")
        if not getattr(getattr(model, "config", None), "model_type", None):
            raise ValueError("Expected canonical HF policy with config.model_type.")
        if not callable(getattr(model, "get_input_embeddings", None)) or not callable(getattr(model, "get_output_embeddings", None)):
            raise ValueError("Expected the HF causal-LM embedding interface.")
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            raise ValueError("Quantized policies are not supported by floating-point full ES.")
        for name, module in model.named_modules():
            marker = (name + "." + type(module).__module__ + "." + type(module).__name__).lower()
            parts = marker.replace("_", ".").split(".")
            if "replacementmodel" in marker or any(word in marker for word in ("transcoder", "uncertainty", "interpreter", "circuit_tracer", "transformer_lens")) or "sae" in parts or "probe" in parts:
                raise ValueError(f"Auxiliary/interpretability module is forbidden in policy: {name or type(module).__name__}")
        for name, _ in model.named_parameters():
            components = name.lower().replace("_", ".").split(".")
            if any(word in name.lower() for word in ("transcoder", "uncertainty", "interpreter")) or "sae" in components or "probe" in components:
                raise ValueError(f"Auxiliary parameter is forbidden: {name}")

    @property
    def total_parameters(self) -> int:
        return sum(record.numel for record in self.manifest)

    def model_state_hash(self) -> str:
        return tensor_state_hash(dict(self.model.state_dict()))

    def parameter_manifest(self) -> list[dict[str, Any]]:
        # JSON conversion makes tuples/lists consistent after serialization.
        return json.loads(json.dumps([asdict(record) for record in self.manifest]))

    def _identity(self) -> dict[str, Any]:
        return {"format_version": FORMAT_VERSION, "policy_model_id": self.policy_model_id,
                "parameter_scope": self.parameter_scope, "parameter_manifest": self.parameter_manifest(),
                "noise_algorithm": NOISE_ALGORITHM, "noise_chunk_size": self.noise_chunk_size,
                "torch_version": str(torch.__version__), "normalization_epsilon": self.normalization_epsilon,
                "candidate_restore": "exact_cpu_snapshot_copy",
                "base_state_hash": self.base_state_hash}

    def _require_idle(self) -> None:
        if self.active_perturbation is not None:
            raise RuntimeError("Complete and restore the current trajectory perturbation first.")

    @torch.no_grad()
    def begin_perturbation(self, seed: int, sigma: float) -> None:
        self._require_idle()
        stable_parameter_seed(seed, "validate")
        if not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative.")
        # clone() is required even on CPU: .cpu() alone aliases CPU policy weights.
        self._snapshot = {name: p.detach().cpu().clone() for name, p in self._parameters}
        self.active_perturbation = {"seed": seed, "sigma": float(sigma)}
        try:
            for name, parameter in self._parameters:
                flat = parameter.view(-1)
                for start, noise in iter_noise(seed, name, flat.numel(), self.noise_chunk_size):
                    flat[start:start + noise.numel()].add_(noise.to(parameter.device, parameter.dtype), alpha=float(sigma))
                    if not torch.isfinite(flat[start:start + noise.numel()]).all():
                        raise ValueError("Perturbation overflowed parameters; exact snapshot restored.")
        except BaseException:
            self.end_perturbation()
            raise

    @torch.no_grad()
    def end_perturbation(self) -> None:
        if self.active_perturbation is None or self._snapshot is None:
            raise RuntimeError("No active perturbation to restore.")
        for name, parameter in self._parameters:
            parameter.copy_(self._snapshot[name], non_blocking=False)
        self._snapshot = None
        self.active_perturbation = None

    @contextmanager
    def perturbation(self, seed: int, sigma: float) -> Iterator[torch.nn.Module]:
        """Hold one coherent perturbation across every turn; restore on exceptions."""
        self.begin_perturbation(seed, sigma)
        try:
            yield self.model
        finally:
            self.end_perturbation()

    @torch.no_grad()
    def step(self, seeds: Sequence[int], rewards: Sequence[float], *, alpha: float,
             sigma: float, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        self._require_idle()
        if len(seeds) == 0 or len(seeds) != len(rewards):
            raise ValueError("seeds and rewards must have equal nonzero length.")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Population seeds must be distinct.")
        for seed in seeds:
            stable_parameter_seed(seed, "validate")
        if not math.isfinite(alpha) or alpha < 0 or not math.isfinite(sigma) or sigma < 0:
            raise ValueError("alpha and sigma must be finite and nonnegative.")
        # Validate metadata BEFORE any mutation so persistence cannot fail on NaN/etc.
        metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
        weights = population_zscores(rewards, self.normalization_epsilon)
        before = self.model_state_hash()
        expected_before = self.history[-1]["state_hash_after"] if self.history else self.base_state_hash
        if before != expected_before:
            raise ValueError("Policy weights changed outside the recorded ES history.")
        baseline = {name: p.detach().cpu().clone() for name, p in self._parameters}
        delta_norms: dict[str, float] = {}
        try:
            for name, parameter in self._parameters:
                generators = [iter_noise(seed, name, parameter.numel(), self.noise_chunk_size) for seed in seeds]
                flat = parameter.view(-1)
                norm_squared = 0.0
                for start in range(0, flat.numel(), self.noise_chunk_size):
                    length = min(self.noise_chunk_size, flat.numel() - start)
                    delta = torch.zeros(length, dtype=torch.float32)
                    for weight, generator in zip(weights, generators):
                        noise_start, noise = next(generator)
                        assert noise_start == start
                        delta.add_(noise, alpha=float(alpha) * weight / len(seeds))
                    previous = flat[start:start + length].detach().cpu().clone()
                    flat[start:start + length].add_(delta.to(parameter.device, parameter.dtype))
                    if not torch.isfinite(flat[start:start + length]).all():
                        raise ValueError("ES update overflowed policy parameters; original weights restored.")
                    actual = flat[start:start + length].detach().cpu().float() - previous.float()
                    norm_squared += actual.double().square().sum().item()
                delta_norms[name] = math.sqrt(norm_squared)
        except BaseException:
            for name, parameter in self._parameters:
                parameter.copy_(baseline[name], non_blocking=False)
            raise
        record = {"generation": len(self.history), "seeds": list(seeds), "rewards": [float(x) for x in rewards],
                  "normalized_rewards": weights, "alpha": float(alpha), "sigma": float(sigma),
                  "reward_normalization": "population_zscore", "ddof": 0,
                  "normalization_epsilon": self.normalization_epsilon,
                  "state_hash_before": before, "state_hash_after": self.model_state_hash(),
                  "parameter_delta_l2": delta_norms, "metadata": metadata}
        self.history.append(record)
        return record

    def save_history(self, path: str | Path) -> None:
        self._require_idle()
        current = self.model_state_hash()
        expected = self.history[-1]["state_hash_after"] if self.history else self.base_state_hash
        if current != expected:
            raise ValueError("Policy weights changed outside the recorded ES history.")
        _atomic_json(path, {**self._identity(), "history": self.history,
                            "current_state_hash": current})

    def _validate_identity(self, payload: dict[str, Any]) -> None:
        identity = self._identity()
        for key in ("format_version", "policy_model_id", "parameter_scope", "parameter_manifest", "noise_algorithm", "noise_chunk_size", "torch_version", "normalization_epsilon"):
            if payload.get(key) != identity[key]:
                raise ValueError(f"Incompatible ES history/checkpoint: {key}.")

    def replay_history(self, path: str | Path) -> None:
        """Replay updates only from the exact base state; verify every generation."""
        self._require_idle()
        if self.history:
            raise ValueError("Replay requires a fresh optimizer and exact base weights.")
        payload = json.loads(Path(path).read_text())
        self._validate_identity(payload)
        if self.model_state_hash() != payload["base_state_hash"]:
            raise ValueError("Replay base checkpoint hash mismatch.")
        for index, record in enumerate(payload["history"]):
            if record["generation"] != index or self.model_state_hash() != record["state_hash_before"]:
                raise ValueError("History generation sequence/state hash mismatch.")
            actual = self.step(record["seeds"], record["rewards"], alpha=record["alpha"], sigma=record["sigma"], metadata=record.get("metadata"))
            if actual["state_hash_after"] != record["state_hash_after"]:
                raise ValueError("History replay produced different model weights.")
        if self.model_state_hash() != payload["current_state_hash"]:
            raise ValueError("History final checkpoint hash mismatch.")

    def save_checkpoint(self, path: str | Path) -> dict[str, str]:
        self._require_idle()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
        state_hash = tensor_state_hash(state)
        expected = self.history[-1]["state_hash_after"] if self.history else self.base_state_hash
        if state_hash != expected:
            raise ValueError("Policy weights changed outside the recorded ES history.")
        payload = {**self._identity(), "history": self.history, "state_dict": state, "model_state_hash": state_hash}
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        os.close(fd)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result = {"model_state_hash": state_hash, "file_sha256": digest.hexdigest()}
        _atomic_json(str(path) + ".json", {**self._identity(), **result, "generations": len(self.history)})
        return result

    def load_checkpoint(self, path: str | Path) -> None:
        self._require_idle()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self._validate_identity(payload)
        if tensor_state_hash(payload["state_dict"]) != payload["model_state_hash"]:
            raise ValueError("Checkpoint content hash mismatch.")
        history = payload["history"]
        previous_hash = payload["base_state_hash"]
        for index, record in enumerate(history):
            if record["generation"] != index or record["state_hash_before"] != previous_hash:
                raise ValueError("Checkpoint history chain is inconsistent.")
            previous_hash = record["state_hash_after"]
        if previous_hash != payload["model_state_hash"]:
            raise ValueError("Checkpoint does not match its ES history.")
        self.model.load_state_dict(payload["state_dict"], strict=True)
        if self.model_state_hash() != payload["model_state_hash"]:
            raise ValueError("Loaded checkpoint did not preserve tensor values.")
        self.history = history
        self.base_state_hash = payload["base_state_hash"]
