"""Serial, resumable full-parameter Agentic ES training with explicit evidence.

The defaults are this study's plan, not claimed official optimal hyperparameters.
Callbacks must finish an entire physical trajectory before returning. A durable
``started`` record precedes even job construction. Unknown outcomes are NEVER
automatically rerun: supply a read-only recovery callback or reconcile a complete
verified result. Resuming restores only committed checkpoints; replaying an ES
arithmetic update from verified results does not repeat physical actions.

The policy stamp's checkpoint_hash identifies the INITIAL downloaded checkpoint.
Every generation and candidate separately records the actual tensor-state hash.
Full ES includes retained visual parameters, although text rollouts do not use
them. Transcoders, interpreters and risk networks must remain outside the model.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import uuid
import weakref

import torch

from .esopt import AgenticESOpt, cosine_sigma, tensor_state_hash


FORMAT_VERSION = 1
REWARDS = {
    "made": ("AUDC", "made_official_orb_audc"),
    "crystalgym": ("reward", "crystalgym_quantum_espresso_official_reward"),
}
_POLICY_LOCKS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_POLICY_LOCK_GUARD = threading.Lock()


class TrainingContractError(ValueError):
    """Invalid split, reward evidence, run identity, or checkpoint contract."""


class TrainingHalted(RuntimeError):
    """Progress is durable; unknown physical actions require reconciliation."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("w") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ESTrainingConfig:
    benchmark: str = "made"
    method: str = "es_only"
    property_name: str | None = None
    seed: int = 0
    generations: int = 16
    population: int = 8
    cases_per_generation: int = 1
    alpha: float = 0.0005
    sigma_start: float = 0.001
    sigma_end: float = 0.0002
    dev_every: int = 4
    noise_chunk_size: int = 1_048_576
    normalization_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.benchmark not in REWARDS or not self.method.strip():
            raise TrainingContractError("Choose a supported benchmark and an explicit independent method.")
        if self.benchmark == "crystalgym" and not self.property_name:
            raise TrainingContractError("CrystalGym training must target one explicit property.")
        if self.benchmark == "made" and self.property_name is not None:
            raise TrainingContractError("MADE fitness is AUDC, not a mixed property objective.")
        for name in ("generations", "population", "cases_per_generation", "dev_every", "noise_chunk_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise TrainingContractError(f"{name} must be a positive integer.")
        if self.population < 2 or type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise TrainingContractError("Population must be at least two and seed an integer in [0,2**63).")
        for name in ("alpha", "sigma_start", "sigma_end", "normalization_epsilon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise TrainingContractError(f"{name} must be finite and positive.")


@dataclass(frozen=True)
class TrainingCase:
    case_id: str
    split: str
    benchmark: str
    property_name: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRequest:
    request_id: str
    run_fingerprint: str
    phase: str
    generation: int  # one-based candidate generation; zero is the saved initial state
    candidate_index: int | None
    case: TrainingCase
    method: str
    environment_seed: int
    perturbation_seed: int | None
    perturbation_sigma: float | None
    actual_model_state_hash: str
    initial_checkpoint_manifest_hash: str
    policy_state_id: str

    @property
    def metric_name(self) -> str:
        return REWARDS[self.case.benchmark][0]

    @property
    def reward_source(self) -> str:
        return REWARDS[self.case.benchmark][1]

    def to_dict(self) -> dict[str, Any]:
        return _json_copy(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EvaluationRequest:
        values = dict(payload)
        values["case"] = TrainingCase(**values["case"])
        return cls(**values)

    def semantic_identity(self) -> dict[str, Any]:
        # Reloading identical weights deliberately gives a fresh runtime state_id.
        return {k: v for k, v in self.to_dict().items() if k not in {"request_id", "policy_state_id"}}

    def slot_identity(self) -> dict[str, Any]:
        # A changed replay hash must collide with the original slot and HALT,
        # never receive a new job ID that could duplicate physical execution.
        return {k: v for k, v in self.semantic_identity().items() if k != "actual_model_state_hash"}


@dataclass(frozen=True)
class EvaluationResult:
    request_id: str
    metric_name: str
    metric_value: float
    reward_source: str
    costs: Mapping[str, float | int]
    result_paths: Sequence[str | Path]
    complete: bool = True


def _unwrap_model(policy: Any) -> torch.nn.Module:
    return policy if isinstance(policy, torch.nn.Module) else policy.model


def verify_clean_reload_checkpoint(
    reference_policy: Any, checkpoint_path: str | Path, expected_state_hash: str, *,
    fresh_policy_factory: Callable[[], Any], input_ids: torch.Tensor,
    model_kwargs: Mapping[str, Any] | None = None, atol: float = 1e-6, rtol: float = 1e-5,
) -> dict[str, Any]:
    """Load a separate model and compare logits, never benchmark rewards.

    The factory must construct a clean canonical model, not reuse a live model or
    its parameter storage. Probe tokens are fixed technical inputs, not held-out
    labels. Use the same backend/dtype for a meaningful equality tolerance.
    """
    reference = _unwrap_model(reference_policy)
    if reference.training or tensor_state_hash(dict(reference.state_dict())) != expected_state_hash:
        raise TrainingContractError("Clean-reload reference must be eval() at the expected final weights.")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise TrainingContractError("Reload verification needs one fixed complete token prefix.")
    kwargs = dict(model_kwargs or {})
    if any(kwargs.get(key) is not None for key in ("labels", "past_key_values", "inputs_embeds", "pixel_values")):
        raise TrainingContractError("Reload checks cannot use labels, supplied activations, images, or caches.")
    kwargs["use_cache"] = False
    checkpoint_path = Path(checkpoint_path)
    file_hash = _file_hash(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (payload["model_state_hash"] != expected_state_hash
            or tensor_state_hash(payload["state_dict"]) != expected_state_hash):
        raise TrainingContractError("Reload checkpoint content does not match the expected tensor hash.")
    fresh = _unwrap_model(fresh_policy_factory())
    if fresh is reference:
        raise TrainingContractError("The clean-reload model must be a different object.")
    storage = {(str(p.device), p.data_ptr()) for p in reference.parameters()}
    if any((str(p.device), p.data_ptr()) in storage for p in fresh.parameters()):
        raise TrainingContractError("Clean reload cannot share parameter storage with the reference.")
    fresh.load_state_dict(payload["state_dict"], strict=True)
    fresh.eval()
    loaded_hash = tensor_state_hash(dict(fresh.state_dict()))
    if loaded_hash != expected_state_hash:
        raise TrainingContractError("Cleanly loaded model has a different tensor-state hash.")
    def logits(model):
        device = model.get_input_embeddings().weight.device
        arguments = {k: v.detach().clone().to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        with torch.no_grad():
            return model(input_ids=input_ids.detach().clone().to(device), **arguments).logits[:, -1].detach().float().cpu()
    original_logits, loaded_logits = logits(reference), logits(fresh)
    equal = bool(torch.allclose(original_logits, loaded_logits, atol=atol, rtol=rtol))
    maximum = float((original_logits - loaded_logits).abs().max())
    if not equal or not math.isfinite(maximum):
        raise TrainingContractError(f"Clean reload changed forward logits (max_abs={maximum}).")
    if _file_hash(checkpoint_path) != file_hash:
        raise TrainingContractError("Checkpoint changed during clean-reload validation.")
    return {"verified": True, "fresh_instance": True, "allclose": equal,
            "expected_state_hash": expected_state_hash, "loaded_state_hash": loaded_hash,
            "checkpoint_file_sha256": file_hash, "max_abs_logit_difference": maximum,
            "atol": atol, "rtol": rtol,
            "probe_input_hash": tensor_state_hash({"input_ids": input_ids}),
            "verification_kind": "fresh_model_tensor_hash_and_forward_logits"}


class ESTrainingDriver:
    """A single training task: one benchmark/property, one method, one seed.

    ``evaluate(policy, job, request)`` returns EvaluationResult after its COMPLETE
    trajectory. ``result_verifier(request, result)`` is read-only and returns
    ``verified=True, evidence=<str>, metric_name, metric_value, costs`` derived
    from official result artifacts, not policy self-ratings. The verifier also
    checks job identity, property/units, seeds, evaluator and complete budgets.

    ``recover_result(request, durable_record)`` may only look up an already
    completed result; it must never execute actions. Returning None halts.
    ``clean_reload_validator(policy, checkpoint_path, expected_state_hash)`` must
    prove fresh-instance forward equality, e.g. with the helper above.
    Job factories must initialize independent seeded trajectories. Auxiliary
    controller state must be immutable/versioned or explicitly restored by the
    caller; it must not drift invisibly between population candidates.
    """

    def __init__(self, policy: Any, config: ESTrainingConfig, *,
                 train_cases: Sequence[TrainingCase], dev_cases: Sequence[TrainingCase],
                 job_factory: Callable[[EvaluationRequest], Any],
                 evaluate: Callable[[Any, Any, EvaluationRequest], EvaluationResult],
                 result_verifier: Callable[[EvaluationRequest, EvaluationResult], Mapping[str, Any]],
                 clean_reload_validator: Callable[[Any, Path, str], Mapping[str, Any]],
                 output_dir: str | Path, callback_fingerprint: str,
                 recover_result: Callable[[EvaluationRequest, Mapping[str, Any]], EvaluationResult | None] | None = None) -> None:
        if not callback_fingerprint.strip():
            raise TrainingContractError("Pin the controller/evaluator/risk-artifact callback identity.")
        if not all(callable(fn) for fn in (job_factory, evaluate, result_verifier, clean_reload_validator)):
            raise TrainingContractError("Job, evaluation, evidence, and clean-reload callbacks are required.")
        self.policy, self.config = policy, config
        self.train_cases = tuple(TrainingCase(**_json_copy(asdict(case))) for case in train_cases)
        self.dev_cases = tuple(TrainingCase(**_json_copy(asdict(case))) for case in dev_cases)
        if not self.dev_cases or len(self.train_cases) < config.cases_per_generation:
            raise TrainingContractError("Declare sufficient train cases and ALL agreed development cases.")
        if len({case.case_id for case in self.train_cases + self.dev_cases}) != len(self.train_cases + self.dev_cases):
            raise TrainingContractError("Case IDs must be unique and train/dev identities disjoint.")
        for split, cases in (("train", self.train_cases), ("dev", self.dev_cases)):
            for case in cases:
                if not case.case_id or case.split != split:
                    raise TrainingContractError("Only train cases train the policy; development is dev-only and test is forbidden.")
                if case.benchmark != config.benchmark or case.property_name != config.property_name:
                    raise TrainingContractError("A training task cannot mix benchmarks, properties, or reward units.")
        self.job_factory, self.evaluate, self.result_verifier = job_factory, evaluate, result_verifier
        self.clean_reload_validator, self.recover_result = clean_reload_validator, recover_result
        self.output_dir = Path(output_dir).resolve()
        self.optimizer: AgenticESOpt | None = None
        self._completed: list[dict[str, Any]] = []
        self._best_generation: int | None = None
        self._best_metric: float | None = None
        self._initial_actual_hash: str | None = None
        self._session_stats = {"evaluations_executed": 0, "results_reused": 0}
        self._running = False
        self.plan = self._make_plan()
        self.identity = {
            "format_version": FORMAT_VERSION, "config": asdict(config),
            "train_cases": [asdict(case) for case in self.train_cases],
            "dev_cases": [asdict(case) for case in self.dev_cases], "plan": self.plan,
            "model_id": policy.model_id, "model_revision": policy.revision,
            "initial_checkpoint_manifest_hash": policy.checkpoint_hash,
            "policy_configuration_fingerprint": policy.configuration_fingerprint,
            "callback_fingerprint": callback_fingerprint,
            "parameter_scope": "full", "population_design": "one_sided_gaussian",
            "reward_aggregation": "equal_mean_over_same_cases_and_environment_seeds",
            "metric_name": REWARDS[config.benchmark][0], "reward_source": REWARDS[config.benchmark][1],
            "hyperparameter_provenance": "study_plan_not_claimed_official_optimum",
        }
        self.run_fingerprint = _fingerprint(self.identity)

    def _make_plan(self) -> list[dict[str, Any]]:
        def rng(label):
            seed = int.from_bytes(hashlib.sha256(f"{self.config.seed}:{label}".encode()).digest()[:8], "little")
            return random.Random(seed)
        population_rng, case_rng, environment_rng, dev_rng = [rng(k) for k in ("population", "case", "environment", "dev")]
        dev_seeds = [dev_rng.randrange(2**32) for _ in self.dev_cases]
        used, plan = set(), []
        for index in range(self.config.generations):
            seeds = []
            while len(seeds) < self.config.population:
                seed = population_rng.randrange(2**63)
                if seed not in used:
                    used.add(seed)
                    seeds.append(seed)
            selected = case_rng.sample(range(len(self.train_cases)), self.config.cases_per_generation)
            generation = index + 1
            plan.append({"generation": generation, "population_seeds": seeds,
                         "sigma": cosine_sigma(self.config.sigma_start, self.config.sigma_end, index, self.config.generations),
                         "train_case_indices": selected,
                         "environment_seeds": [environment_rng.randrange(2**32) for _ in selected],
                         "run_dev": generation % self.config.dev_every == 0 or generation == self.config.generations,
                         "dev_environment_seeds": dev_seeds})
        return plan

    @contextmanager
    def _directory_lock(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / ".training.lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise TrainingHalted("Another process holds this training run; no concurrent candidate execution.") from exc
            yield

    def _event(self, kind: str, **details) -> None:
        record = json.dumps({"time": _now(), "kind": kind, **details}, sort_keys=True, allow_nan=False) + "\n"
        with (self.output_dir / "events.jsonl").open("a") as stream:
            stream.write(record)
            stream.flush()
            os.fsync(stream.fileno())

    def _checkpoint_dir(self, generation: int) -> Path:
        return self.output_dir / "checkpoints" / f"generation_{generation:04d}"

    def _request(self, *, phase: str, generation: int, candidate_index: int | None,
                 case: TrainingCase, environment_seed: int, seed: int | None, sigma: float | None,
                 actual_hash: str) -> EvaluationRequest:
        values = dict(request_id="", run_fingerprint=self.run_fingerprint, phase=phase, generation=generation,
                      candidate_index=candidate_index, case=TrainingCase(**_json_copy(asdict(case))), method=self.config.method,
                      environment_seed=environment_seed, perturbation_seed=seed, perturbation_sigma=sigma,
                      actual_model_state_hash=actual_hash, initial_checkpoint_manifest_hash=self.policy.checkpoint_hash,
                      policy_state_id=self.policy.get_state_id())
        provisional = EvaluationRequest(**values)
        values["request_id"] = _fingerprint(provisional.slot_identity())
        return EvaluationRequest(**values)

    def _result_payload(self, result: EvaluationResult) -> dict[str, Any]:
        return _json_copy({**asdict(result), "result_paths": [str(Path(path).resolve()) for path in result.result_paths]})

    def _verify_result(self, request: EvaluationRequest, result: EvaluationResult) -> tuple[dict, list[dict]]:
        if _fingerprint(request.slot_identity()) != request.request_id:
            raise TrainingContractError("Callback mutated the frozen case/request identity.")
        if not isinstance(result, EvaluationResult) or result.request_id != request.request_id or result.complete is not True:
            raise TrainingContractError("Callback must return a complete EvaluationResult for this exact request.")
        if result.metric_name != request.metric_name or result.reward_source != request.reward_source:
            raise TrainingContractError("Only fixed official AUDC/reward is fitness; LLM self-scores and surrogate rewards are forbidden.")
        if isinstance(result.metric_value, bool) or not math.isfinite(result.metric_value):
            raise TrainingContractError("The official fitness must be a finite number.")
        if not isinstance(result.costs, Mapping) or not result.costs:
            raise TrainingContractError("Actual evaluation costs must be supplied, not inferred as zero.")
        for key, value in result.costs.items():
            if not isinstance(key, str) or type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise TrainingContractError("Costs must be named finite nonnegative measurements.")
        if not result.result_paths:
            raise TrainingContractError("Complete result and raw-evidence paths are required.")
        artifacts = []
        for path in result.result_paths:
            path = Path(path).resolve()
            if not path.is_file() or path == self._job_record_path(request.request_id):
                raise TrainingContractError("Evidence paths must be real result files, not the driver receipt itself.")
            artifacts.append({"path": str(path), "sha256": _file_hash(path), "size_bytes": path.stat().st_size})
        if len({a["path"] for a in artifacts}) != len(artifacts):
            raise TrainingContractError("Duplicate result evidence paths.")
        proof = _json_copy(dict(self.result_verifier(request, result)))
        if (proof.get("verified") is not True or not isinstance(proof.get("evidence"), str) or not proof["evidence"].strip()
                or proof.get("metric_name") != result.metric_name or proof.get("metric_value") != result.metric_value
                or proof.get("costs") != dict(result.costs)):
            raise TrainingContractError("Read-only result verifier must reconstruct and confirm official metric and actual costs.")
        for artifact in artifacts:
            if _file_hash(artifact["path"]) != artifact["sha256"]:
                raise TrainingContractError("Evidence changed while it was being verified.")
        return proof, artifacts

    def _job_record_path(self, request_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", request_id):
            raise TrainingContractError("Invalid request ID.")
        return self.output_dir / "jobs" / f"{request_id}.json"

    def _complete_result(self, request: EvaluationRequest, result: EvaluationResult, *, kind: str,
                         guard_current: EvaluationRequest | None = None) -> dict:
        proof, artifacts = self._verify_result(request, result)
        if guard_current is not None:
            self._callback_guard(guard_current)
        record = {"status": "complete", "request": request.to_dict(), "result": self._result_payload(result),
                  "artifacts": artifacts, "verification": proof, "completion_kind": kind, "completed_at": _now()}
        _write_json(self._job_record_path(request.request_id), record)
        return record

    def reconcile_result(self, request_id: str, result: EvaluationResult) -> None:
        """Register an independently recovered COMPLETE result; never retry a job."""
        with self._directory_lock():
            path = self._job_record_path(request_id)
            if not path.exists():
                raise TrainingContractError("Cannot reconcile an unknown request.")
            record = json.loads(path.read_text())
            request = EvaluationRequest.from_dict(record["request"])
            if request.run_fingerprint != self.run_fingerprint:
                raise TrainingContractError("Reconciliation belongs to a different training task.")
            if record["status"] == "complete":
                raise TrainingContractError("Completed evidence is immutable; do not overwrite it.")
            self._complete_result(request, result, kind="externally_reconciled")
            self._event("result_reconciled", request_id=request_id)

    def _evaluate_request(self, request: EvaluationRequest) -> EvaluationResult:
        path = self._job_record_path(request.request_id)
        if path.exists():
            record = json.loads(path.read_text())
            stored_request = EvaluationRequest.from_dict(record["request"])
            if stored_request.semantic_identity() != request.semantic_identity():
                raise TrainingContractError("Stored job belongs to different weights, seeds, cases, or method.")
            if record["status"] != "complete":
                recovered = self.recover_result(request, _json_copy(record)) if self.recover_result else None
                if recovered is None:
                    raise TrainingHalted(f"Unknown physical outcome for {request.request_id}; no automatic retry. Reconcile a complete result first.")
                record = self._complete_result(stored_request, recovered, kind="read_only_recovered", guard_current=request)
            result = EvaluationResult(**record["result"])
            proof, artifacts = self._verify_result(stored_request, result)
            if artifacts != record["artifacts"]:
                raise TrainingContractError("A previously completed result artifact changed; refusing reuse or re-execution.")
            self._session_stats["results_reused"] += 1
            self._event("result_reused", request_id=request.request_id,
                        original_policy_state_id=stored_request.policy_state_id,
                        current_policy_state_id=request.policy_state_id,
                        actual_model_state_hash=request.actual_model_state_hash,
                        verification=proof)
            self._callback_guard(request)
            return result
        durable_request = request.to_dict()
        _write_json(path, {"status": "started", "request": durable_request, "started_at": _now(),
                           "actual_costs": None, "automatic_retry_allowed": False})
        self._event("evaluation_started", request_id=request.request_id, phase=request.phase)
        self._session_stats["evaluations_executed"] += 1
        result = None
        try:
            job = self.job_factory(request)
            result = self.evaluate(self.policy, job, request)
            self._complete_result(request, result, kind="executed", guard_current=request)
            return result
        except BaseException as exc:
            reported_result = None
            if isinstance(result, EvaluationResult):
                try:
                    reported_result = self._result_payload(result)
                except (TypeError, ValueError):
                    pass
            _write_json(path, {"status": "outcome_unknown", "request": durable_request,
                               "halted_at": _now(), "exception_type": type(exc).__name__,
                               "error": str(exc), "reported_result_unverified": reported_result,
                               "actual_costs": None, "automatic_retry_allowed": False})
            raise TrainingHalted(f"Evaluation {request.request_id} did not return verified complete evidence; halted without retry.") from exc

    def _callback_guard(self, request: EvaluationRequest) -> None:
        if _fingerprint(request.slot_identity()) != request.request_id:
            raise TrainingContractError("Callback changed the frozen case/request payload.")
        if self.policy.get_state_id() != request.policy_state_id or self.optimizer.model_state_hash() != request.actual_model_state_hash:
            raise TrainingContractError("Callback changed policy state during the trajectory/evidence operation.")

    def _commit(self, generation: int) -> dict:
        directory = self._checkpoint_dir(generation)
        marker_path = directory / "complete.json"
        if marker_path.exists():
            raise TrainingContractError("A committed generation cannot be overwritten.")
        directory.mkdir(parents=True, exist_ok=True)
        hashes = self.optimizer.save_checkpoint(directory / "policy.pt")
        self.optimizer.save_history(directory / "es_history.json")
        state = {"run_fingerprint": self.run_fingerprint, "generation": generation,
                 "initial_actual_model_state_hash": self._initial_actual_hash,
                 "initial_checkpoint_manifest_hash": self.policy.checkpoint_hash,
                 "actual_model_state_hash": hashes["model_state_hash"],
                 "policy_state_id": self.policy.get_state_id(), "generations": self._completed,
                 "best_generation": self._best_generation, "best_metric": self._best_metric}
        _write_json(directory / "driver_state.json", state)
        marker = {"status": "complete", "run_fingerprint": self.run_fingerprint, "generation": generation,
                  "actual_model_state_hash": hashes["model_state_hash"],
                  "files": {"policy.pt": hashes["file_sha256"], "policy.pt.json": _file_hash(directory / "policy.pt.json"),
                            "es_history.json": _file_hash(directory / "es_history.json"),
                            "driver_state.json": _file_hash(directory / "driver_state.json")}}
        _write_json(marker_path, marker)  # commit LAST
        self._event("generation_committed", generation=generation,
                    actual_model_state_hash=hashes["model_state_hash"], policy_state_id=self.policy.get_state_id())
        return marker

    def _load_committed(self) -> int:
        markers = sorted((self.output_dir / "checkpoints").glob("generation_*/complete.json"))
        if not markers:
            raise TrainingContractError("Resume requires at least one COMPLETE checkpoint; incomplete files are never loaded.")
        for expected, path in enumerate(markers):
            marker = json.loads(path.read_text())
            if (marker.get("status") != "complete" or marker.get("generation") != expected
                    or marker.get("run_fingerprint") != self.run_fingerprint):
                raise TrainingContractError("Committed checkpoint sequence or training identity is invalid.")
            if set(marker.get("files", {})) != {"policy.pt", "policy.pt.json", "es_history.json", "driver_state.json"}:
                raise TrainingContractError("Checkpoint commit does not identify all required files.")
            for name, checksum in marker["files"].items():
                if name == "policy.pt" and not (path.parent / name).exists():
                    from .checkpoint_retention import verify_retired_checkpoint
                    verify_retired_checkpoint(self.output_dir, expected, marker, verify_retained_weights=False)
                    continue
                if _file_hash(path.parent / name) != checksum:
                    raise TrainingContractError("Committed checkpoint or history file is corrupt; no silent fallback.")
        latest = markers[-1].parent
        state = json.loads((latest / "driver_state.json").read_text())
        generation = state["generation"]
        if (generation > self.config.generations or generation != len(markers) - 1
                or len(state["generations"]) != generation or state["run_fingerprint"] != self.run_fingerprint):
            raise TrainingContractError("Driver checkpoint does not cover its declared complete generations.")
        self.optimizer.load_checkpoint(latest / "policy.pt")
        if (len(self.optimizer.history) != generation
                or self.optimizer.model_state_hash() != state["actual_model_state_hash"]):
            raise TrainingContractError("Restored ES history and driver generation disagree.")
        self.policy.mark_state("reload", generation=generation)
        self._completed = state["generations"]
        self._initial_actual_hash = state["initial_actual_model_state_hash"]
        self._best_generation, self._best_metric = state["best_generation"], state["best_metric"]
        expected_best, expected_score = None, None
        for index, record in enumerate(self._completed):
            plan, step = self.plan[index], self.optimizer.history[index]
            if (record["generation"] != index + 1 or record["sigma"] != plan["sigma"]
                    or record["actual_model_state_hash_before"] != step["state_hash_before"]
                    or record["actual_model_state_hash_after"] != step["state_hash_after"]
                    or record["population"] != step["metadata"]["population"]
                    or record["fitness"] != step["rewards"]
                    or step["seeds"] != plan["population_seeds"]):
                raise TrainingContractError("Committed driver generations disagree with the ES history or frozen schedule.")
            if len(record["dev"]) != (len(self.dev_cases) if plan["run_dev"] else 0):
                raise TrainingContractError("A committed development sweep omitted agreed cases.")
            if record["dev"]:
                mean = sum(row["metric"] for row in record["dev"]) / len(self.dev_cases)
                if mean != record["dev_mean"]:
                    raise TrainingContractError("Committed dev mean is inconsistent.")
                if expected_score is None or mean > expected_score:
                    expected_best, expected_score = record["generation"], mean
        if (self._best_generation, self._best_metric) != (expected_best, expected_score):
            raise TrainingContractError("Best generation was not selected solely from committed development results.")
        self._event("checkpoint_resumed", generation=generation,
                    actual_model_state_hash=state["actual_model_state_hash"], policy_state_id=self.policy.get_state_id())
        return generation

    @staticmethod
    def _layer_deltas(parameter_deltas: Mapping[str, float]) -> dict[str, Any]:
        grouped: dict[str, dict] = {}
        for name, norm in parameter_deltas.items():
            match = re.match(r"(.*\.layers\.\d+)\.", name)
            layer = "unused_visual" if "visual" in name.split(".") else match.group(1) if match else "other_text_or_shared"
            item = grouped.setdefault(layer, {"actual_delta_squared": 0.0, "changed_parameter_tensors": 0, "parameter_tensors": 0})
            item["actual_delta_squared"] += norm * norm
            item["changed_parameter_tensors"] += int(norm > 0)
            item["parameter_tensors"] += 1
        for item in grouped.values():
            item["actual_delta_l2"] = math.sqrt(item.pop("actual_delta_squared"))
        return grouped

    def _train_generation(self, plan: Mapping[str, Any]) -> None:
        generation, sigma = plan["generation"], plan["sigma"]
        baseline_hash = self.optimizer.model_state_hash()
        population, fitness = [], []
        for candidate, seed in enumerate(plan["population_seeds"]):
            candidate_record = {"candidate_index": candidate, "perturbation_seed": seed, "sigma": sigma, "results": []}
            try:
                with self.optimizer.perturbation(seed, sigma):
                    stamp = self.policy.mark_state("perturb", generation=generation - 1,
                                                  perturbation_seed=seed, perturbation_sigma=sigma)
                    actual_hash = self.optimizer.model_state_hash()
                    candidate_record.update(policy_state_id=stamp.state_id, actual_perturbed_model_state_hash=actual_hash)
                    for index, environment_seed in zip(plan["train_case_indices"], plan["environment_seeds"], strict=True):
                        request = self._request(phase="train", generation=generation, candidate_index=candidate,
                                                case=self.train_cases[index], environment_seed=environment_seed,
                                                seed=seed, sigma=sigma, actual_hash=actual_hash)
                        result = self._evaluate_request(request)
                        evaluation_state = json.loads(self._job_record_path(request.request_id).read_text())["request"]["policy_state_id"]
                        candidate_record["results"].append({"request_id": request.request_id, "case_id": request.case.case_id,
                                                            "environment_seed": environment_seed, "metric": result.metric_value,
                                                            "policy_state_id_at_evaluation": evaluation_state,
                                                            "current_policy_state_id": request.policy_state_id})
            finally:
                if self.optimizer.active_perturbation is None:
                    restored = self.policy.mark_state("restore", generation=generation - 1)
                    restored_hash = self.optimizer.model_state_hash()
                    self._event("perturbation_restored", generation=generation, candidate_index=candidate,
                                policy_state_id=restored.state_id, actual_model_state_hash=restored_hash,
                                perturbation_seed=seed, sigma=sigma)
                    if restored_hash != baseline_hash:
                        raise TrainingHalted("Exact candidate restoration failed; resume a committed checkpoint.")
                else:
                    raise TrainingHalted("A perturbation could not be restored; reload from a complete checkpoint.")
            mean = sum(item["metric"] for item in candidate_record["results"]) / self.config.cases_per_generation
            candidate_record["fitness"] = mean
            fitness.append(mean)
            population.append(candidate_record)
        try:
            update = self.optimizer.step(plan["population_seeds"], fitness, alpha=self.config.alpha, sigma=sigma,
                                         metadata={"run_fingerprint": self.run_fingerprint, "split": "train", "method": self.config.method,
                                                   "property_name": self.config.property_name, "metric_name": self.identity["metric_name"],
                                                   "reward_source": self.identity["reward_source"], "population": population})
        except BaseException:
            restored = self.policy.mark_state("restore", generation=generation - 1)
            rollback_hash = self.optimizer.model_state_hash()
            self._event("failed_update_restored", generation=generation, policy_state_id=restored.state_id,
                        actual_model_state_hash=rollback_hash)
            if rollback_hash != baseline_hash:
                raise TrainingHalted("Failed ES update did not restore the committed baseline; reload is required.")
            raise
        stamp = self.policy.mark_state("update", generation=generation)
        actual_hash = self.optimizer.model_state_hash()
        if actual_hash != update["state_hash_after"]:
            raise TrainingContractError("The actual updated weights disagree with the recorded ES step.")
        record = {"generation": generation, "sigma": sigma, "population": population, "fitness": fitness,
                  "actual_model_state_hash_before": baseline_hash, "actual_model_state_hash_after": actual_hash,
                  "initial_checkpoint_manifest_hash": self.policy.checkpoint_hash, "updated_policy_state_id": stamp.state_id,
                  "actual_parameter_delta_l2": update["parameter_delta_l2"],
                  "actual_layer_deltas": self._layer_deltas(update["parameter_delta_l2"]), "dev": []}
        if plan["run_dev"]:
            for case, environment_seed in zip(self.dev_cases, plan["dev_environment_seeds"], strict=True):
                request = self._request(phase="dev", generation=generation, candidate_index=None, case=case,
                                        environment_seed=environment_seed, seed=None, sigma=None, actual_hash=actual_hash)
                result = self._evaluate_request(request)
                evaluation_state = json.loads(self._job_record_path(request.request_id).read_text())["request"]["policy_state_id"]
                record["dev"].append({"case_id": case.case_id, "request_id": request.request_id,
                                      "environment_seed": environment_seed, "metric": result.metric_value,
                                      "policy_state_id_at_evaluation": evaluation_state,
                                      "current_policy_state_id": request.policy_state_id})
            score = sum(item["metric"] for item in record["dev"]) / len(self.dev_cases)
            record["dev_mean"] = score
            if self._best_metric is None or score > self._best_metric:
                self._best_generation, self._best_metric = generation, score
        self._completed.append(record)
        self._commit(generation)

    def _summary(self, reload_proof: Mapping[str, Any]) -> dict[str, Any]:
        expected = self.config.generations * self.config.population * self.config.cases_per_generation
        expected += sum(plan["run_dev"] for plan in self.plan) * len(self.dev_cases)
        records = [json.loads(path.read_text()) for path in (self.output_dir / "jobs").glob("*.json")]
        if len(records) != expected or any(record["status"] != "complete" for record in records):
            raise TrainingContractError("Complete training requires every planned train/dev evaluation and no unknown outcome.")
        expected_ids = {item["request_id"] for generation in self._completed
                        for candidate in generation["population"] for item in candidate["results"]}
        expected_ids.update(item["request_id"] for generation in self._completed for item in generation["dev"])
        if len(expected_ids) != expected or {record["request"]["request_id"] for record in records} != expected_ids:
            raise TrainingContractError("Result accounting does not match all committed generation jobs.")
        costs: dict[str, float] = {}
        for record in records:
            request = EvaluationRequest.from_dict(record["request"])
            if request.run_fingerprint != self.run_fingerprint:
                raise TrainingContractError("A result belongs to a different training task.")
            result = EvaluationResult(**record["result"])
            _, artifacts = self._verify_result(request, result)
            if artifacts != record["artifacts"]:
                raise TrainingContractError("Final accounting detected changed result evidence.")
            for key, value in result.costs.items():
                costs[key] = costs.get(key, 0) + value
        return {"status": "complete", "run_fingerprint": self.run_fingerprint,
                "completed_generations": len(self._completed), "required_generations": self.config.generations,
                "expected_evaluations": expected, "completed_evaluations": len(records),
                "actual_evaluator_costs": costs, "current_session": self._session_stats,
                "best_generation": self._best_generation, "best_dev_metric": self._best_metric,
                "best_checkpoint": str(self._checkpoint_dir(self._best_generation) / "policy.pt"),
                "best_actual_model_state_hash": self._completed[self._best_generation - 1]["actual_model_state_hash_after"],
                "final_checkpoint": str(self._checkpoint_dir(self.config.generations) / "policy.pt"),
                "final_actual_model_state_hash": self.optimizer.model_state_hash(),
                "initial_checkpoint_manifest_hash": self.policy.checkpoint_hash,
                "selection_uses": "all_declared_dev_cases_at_scheduled_trained_generations_including_final",
                "test_data_used": False, "clean_reload": dict(reload_proof),
                "clean_reload_checked_generation": self.config.generations,
                "result_ledger_directory": str(self.output_dir / "jobs"),
                "policy_left_at": "final_generation; selected best checkpoint is returned separately"}

    def run(self, *, resume: bool = False) -> dict[str, Any]:
        with _POLICY_LOCK_GUARD:
            lock = _POLICY_LOCKS.setdefault(self.policy.model, threading.Lock())
        if not lock.acquire(blocking=False):
            raise TrainingHalted("The same policy cannot run concurrent candidate trajectories or training tasks.")
        started = time.monotonic()
        try:
            with self._directory_lock():
                self._session_stats = {"evaluations_executed": 0, "results_reused": 0}
                if (_fingerprint(self.identity) != self.run_fingerprint
                        or [asdict(c) for c in self.train_cases] != self.identity["train_cases"]
                        or [asdict(c) for c in self.dev_cases] != self.identity["dev_cases"]):
                    raise TrainingContractError("Training config, cases, or schedule changed after registration.")
                manifest_path = self.output_dir / "run_manifest.json"
                if manifest_path.exists():
                    saved = json.loads(manifest_path.read_text())
                    if not resume or saved.get("run_fingerprint") != self.run_fingerprint or saved.get("identity") != self.identity:
                        raise TrainingContractError("Existing training task requires explicit resume with the identical method, plan, and callbacks.")
                elif resume:
                    raise TrainingContractError("No committed training run exists to resume.")
                elif any(p.name != ".training.lock" for p in self.output_dir.iterdir()):
                    raise TrainingContractError("Use an empty output directory for an independent ES task.")
                self._running = True
                for name, _ in self.policy.model.named_modules():
                    if {"risk", "risknet", "calibrator", "transcoder", "interpreter", "uncertainty", "probe", "sae"} & set(name.lower().replace("_", ".").split(".")):
                        raise TrainingContractError("Risk/interpreter/transcoder networks must remain outside the full-ES policy model.")
                self.optimizer = AgenticESOpt(self.policy.model, policy_model_id=f"{self.policy.model_id}@{self.policy.revision}",
                                              parameter_scope="full", noise_chunk_size=self.config.noise_chunk_size,
                                              normalization_epsilon=self.config.normalization_epsilon)
                if resume:
                    completed = self._load_committed()
                else:
                    stamp = self.policy.model_stamp
                    if stamp.generation != 0 or stamp.perturbation_seed is not None or stamp.perturbation_sigma is not None:
                        raise TrainingContractError("Independent training starts from the clean initial policy, not a prior trained task.")
                    self._initial_actual_hash = self.optimizer.model_state_hash()
                    report = self.policy.parameter_report()
                    _write_json(manifest_path, {"run_fingerprint": self.run_fingerprint, "identity": self.identity,
                                               "initial_actual_model_state_hash": self._initial_actual_hash,
                                               "full_parameter_manifest": self.optimizer.parameter_manifest(),
                                               "policy_parameter_report": report,
                                               "vision_scope_note": "Full ES includes retained visual parameters even when unused in text trajectories.",
                                               "created_at": _now()})
                    self._completed, self._best_generation, self._best_metric = [], None, None
                    self._commit(0)
                    completed = 0
                self._event("session_started", resume=resume, completed_generations=completed)
                for plan in self.plan[completed:]:
                    self._train_generation(plan)
                if len(self._completed) != self.config.generations or self._best_generation is None:
                    raise TrainingContractError("A pilot or partial generation schedule cannot satisfy complete training.")
                final_hash = self.optimizer.model_state_hash()
                final_state_id = self.policy.get_state_id()
                proof = _json_copy(dict(self.clean_reload_validator(self.policy,
                                                                   self._checkpoint_dir(self.config.generations) / "policy.pt", final_hash)))
                if (proof.get("verified") is not True or proof.get("fresh_instance") is not True
                        or proof.get("allclose") is not True or proof.get("loaded_state_hash") != final_hash
                        or proof.get("expected_state_hash") != final_hash
                        or proof.get("checkpoint_file_sha256") != _file_hash(self._checkpoint_dir(self.config.generations) / "policy.pt")
                        or not re.fullmatch(r"[0-9a-f]{64}", proof.get("probe_input_hash", ""))
                        or type(proof.get("max_abs_logit_difference")) not in (int, float)
                        or not math.isfinite(proof["max_abs_logit_difference"])):
                    raise TrainingContractError("Fresh-model forward-logit equality proof is required for clean-reload acceptance.")
                if self.optimizer.model_state_hash() != final_hash:
                    raise TrainingContractError("Clean-reload validator mutated the live final policy.")
                summary = self._summary(proof)
                if self.optimizer.model_state_hash() != final_hash or self.policy.get_state_id() != final_state_id:
                    raise TrainingContractError("Read-only final verification changed the final policy state.")
                summary["current_session_wall_seconds"] = time.monotonic() - started
                _write_json(self.output_dir / "training_summary.json", summary)
                self._event("training_complete", completed_generations=self.config.generations,
                            best_generation=self._best_generation, actual_model_state_hash=final_hash)
                return summary
        except BaseException as exc:
            if self.output_dir.exists() and self._running:
                self._event("training_halted", exception_type=type(exc).__name__, error=str(exc), automatic_physical_retry_allowed=False)
                _write_json(self.output_dir / "halted.json", {"status": "halted", "time": _now(),
                            "run_fingerprint": self.run_fingerprint, "exception_type": type(exc).__name__,
                            "error": str(exc), "automatic_physical_retry_allowed": False,
                            "current_session_wall_seconds": time.monotonic() - started})
            raise
        finally:
            self._running = False
            lock.release()
