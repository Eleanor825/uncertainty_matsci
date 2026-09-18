"""Serial execution of the immutable full final-test matrix with real evidence.

Filters select a scheduling batch, never a reduced study. RunLedger owns each
physical attempt. A timeout or interrupted action is quarantined, not replayed.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import gc
import json
from pathlib import Path
import time
import uuid

import torch

from .accounting import (
    METHODS, MODEL_KEYS, SEEDS, AccountingError, RunLedger, build_final_manifest,
    file_sha256, fingerprint, validate_manifest, verify_result, write_json_atomic,
)
from .esopt import AgenticESOpt, tensor_state_hash
from .policy import declared_runtime_options
from .training_jobs import (
    TrainingCondition, TrainingContractError, TrainingHalted, TrainingJobCallbacks,
    _read, _require, _rpc_pairs, load_frozen_controllers, reconstruct_scientific_evidence,
)

_ARTIFACT_HASH_CACHE = {}


def _verify_profile_artifacts(profile):
    # Shared selected weights may support hundreds of final episodes. Hash each
    # unchanged file once per process; ctime also prevents a restored mtime from
    # hiding same-size edits. Check identity again after hashing/cache lookup.
    from . import accounting
    from .immutable_hash_cache import metadata_allows_hash_reuse
    def identity(path):
        resolved = str(path.resolve())
        stat = path.stat()
        return (resolved, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    for name, expected in profile["artifacts"].items():
        path = Path(name)
        before = identity(path)
        reusable = metadata_allows_hash_reuse(before[4], before[5])
        if not reusable or _ARTIFACT_HASH_CACHE.get(before) != expected:
            _require(file_sha256(path) == expected, "Final checkpoint/controller provenance artifact checksum mismatch")
            _require(identity(path) == before, "Final checkpoint/controller artifact changed during hashing")
            if not reusable:
                # Recent writes can share a coarse kernel timestamp. An
                # independent content read catches a same-tick post-hash edit;
                # never publish such a file into this metadata-only cache.
                _require(accounting.file_sha256(path) == expected and identity(path) == before,
                         "Final checkpoint/controller artifact changed during hashing")
            else:
                _ARTIFACT_HASH_CACHE[before] = expected
        else:
            _require(identity(path) == before, "Final checkpoint/controller artifact changed during cache lookup")


def select_final_jobs(manifest, *, model_key, benchmark, method=None, seed=None, property_name=None):
    validate_manifest(manifest)
    _require(model_key in MODEL_KEYS and benchmark in {"made", "crystalgym"}, "Unknown final scheduling condition")
    _require(method is None or method in METHODS, "Unknown final method")
    _require(seed is None or seed in SEEDS, "Final scheduling seed must be 1 through 5")
    _require(property_name is None or benchmark == "crystalgym" and property_name in {"bm", "density", "band_gap"}, "Property filter applies only to CrystalGym")
    result = [job for job in manifest["jobs"] if job["model_key"] == model_key and job["benchmark"] == benchmark
              and (method is None or job["method"] == method) and (seed is None or job["seed"] == seed)
              and (property_name is None or job["task"]["property"]["id"] == property_name)]
    _require(bool(result), "Scheduling filter selected no declared final jobs")
    # Group shared policy checkpoints together; task order remains frozen.
    return sorted(result, key=lambda j: (j["method"], j["task"]["property"]["id"] if benchmark == "crystalgym" else "audc", j["seed"], j["task_id"]))


def selected_es_artifact(policy, condition: TrainingCondition, es_root: Path):
    """Verify all generations' history and the retained best/final tensor files."""
    directory = condition.output_directory(es_root)
    summary_path, manifest_path = directory / "training_summary.json", directory / "run_manifest.json"
    summary, manifest = _read(summary_path), _read(manifest_path)
    config = condition.config()
    identity = manifest["identity"]
    train, dev = condition.cases()
    expected_evaluations = config.generations * config.population * config.cases_per_generation
    expected_evaluations += sum(g % config.dev_every == 0 or g == config.generations for g in range(1, config.generations + 1)) * len(dev)
    _require(summary["status"] == "complete" and summary["completed_generations"] == summary["required_generations"] == 16
             and summary["expected_evaluations"] == summary["completed_evaluations"] == expected_evaluations
             and summary["test_data_used"] is False, "ES final evaluation requires the complete 16-generation training condition")
    _require(identity["config"] == asdict(config) and identity["train_cases"] == [asdict(c) for c in train]
             and identity["dev_cases"] == [asdict(c) for c in dev] and identity["parameter_scope"] == "full"
             and identity["model_id"] == policy.model_id and identity["model_revision"] == policy.revision
             and identity["initial_checkpoint_manifest_hash"] == policy.checkpoint_hash
             and identity["policy_configuration_fingerprint"] == policy.configuration_fingerprint,
             "ES checkpoint was trained with another policy, method, split, property or protocol")
    _require(summary["run_fingerprint"] == manifest["run_fingerprint"] == fingerprint(identity), "ES run identity hash mismatch")
    best = summary["best_generation"]
    _require(type(best) is int and best in range(4, 17, 4), "ES selection must use a scheduled trained dev generation")
    files = [summary_path, manifest_path]
    final_state = None
    # Old tensor snapshots need not all be resident: histories/markers remain
    # immutable, while best/final tensors are mandatory and independently hashed.
    for generation in range(17):
        folder = directory / "checkpoints" / f"generation_{generation:04d}"
        marker_path = folder / "complete.json"
        marker = _read(marker_path)
        _require(marker["status"] == "complete" and marker["generation"] == generation
                 and marker["run_fingerprint"] == summary["run_fingerprint"], "Missing complete ES generation marker")
        _require(set(marker["files"]) == {"policy.pt", "policy.pt.json", "driver_state.json", "es_history.json"}, "Incomplete ES checkpoint marker")
        for name in ("policy.pt.json", "driver_state.json", "es_history.json"):
            path = folder / name
            _require(file_sha256(path) == marker["files"][name], "ES checkpoint metadata/history checksum mismatch")
            files.append(path)
        if generation in {best, 16}:
            path = folder / "policy.pt"
            _require(file_sha256(path) == marker["files"]["policy.pt"], "Selected/final ES weights must be present and match their commit checksum")
            files.append(path)
        elif not (folder / "policy.pt").exists():
            from .checkpoint_retention import verify_retired_checkpoint
            verify_retired_checkpoint(directory, generation, marker, verify_retained_weights=False)
            files.append(directory / "checkpoint_retirement.json")
        files.append(marker_path)
        if generation == 16:
            final_state = _read(folder / "driver_state.json")
    records = final_state["generations"]
    _require(len(records) == 16 and [r["generation"] for r in records] == list(range(1, 17)), "Incomplete ES generation history")
    scores = [(r["dev_mean"], r["generation"]) for r in records if r["dev"]]
    for record in records:
        if record["dev"]:
            _require(len(record["dev"]) == len(dev) and record["dev_mean"] == sum(r["metric"] for r in record["dev"]) / len(dev), "ES development selection omitted cases or changed rewards")
    chosen = max(scores, key=lambda pair: (pair[0], -pair[1]))
    _require(chosen == (summary["best_dev_metric"], best)
             and records[best - 1]["actual_model_state_hash_after"] == summary["best_actual_model_state_hash"], "ES best checkpoint was not selected solely from scheduled dev results")
    proof = summary["clean_reload"]
    final_path = directory / "checkpoints/generation_0016/policy.pt"
    _require(proof["verified"] is True and proof["fresh_instance"] is True and proof["allclose"] is True
             and proof["loaded_state_hash"] == summary["final_actual_model_state_hash"]
             and proof["checkpoint_file_sha256"] == file_sha256(final_path), "Final ES checkpoint lacks matching clean-reload evidence")
    checkpoint = directory / "checkpoints" / f"generation_{best:04d}" / "policy.pt"
    optimizer = AgenticESOpt(policy.model, policy_model_id=f"{policy.model_id}@{policy.revision}", parameter_scope="full")
    optimizer.load_checkpoint(checkpoint)
    stamp = policy.mark_state("reload", generation=best)
    actual = optimizer.model_state_hash()
    _require(actual == summary["best_actual_model_state_hash"], "Loaded selected ES tensor hash differs from the verified training result")
    return {"training_directory": str(directory), "training_run_fingerprint": summary["run_fingerprint"],
            "selected_generation": best, "selected_checkpoint": str(checkpoint), "actual_model_state_hash": actual,
            "initial_checkpoint_manifest_hash": policy.checkpoint_hash, "parameter_scope": "full_including_retained_unused_visual",
            "training_costs": summary["actual_evaluator_costs"], "checkpoint_selection": "development_only"}, sorted(set(files))


def execution_job(job, policy, profile):
    return {**job, "split": "test", "group_id": f"{job['task_id']}:seed:{job['seed']}:partition:test",
            "source": "full_main_final_evaluation", "policy_configuration_fingerprint": policy.configuration_fingerprint,
            "initial_checkpoint_manifest_hash": policy.checkpoint_hash, "actual_model_state_hash": profile["actual_model_state_hash"],
            "policy_state_id": policy.get_state_id(), "selected_generation": policy.model_stamp.generation,
            "execution_profile_fingerprint": fingerprint(profile)}


def verify_final_envelope(path, job, manifest_fingerprint, *, tasks, expected_profile=None,
                          expected_mace_num_workers=None, expected_made_budget=50, core_protocol=None,
                          evaluation_extension=None, made_all30_extension=None, made_budget_sweep=None):
    verified = verify_result(path, job, manifest_fingerprint, expected_made_budget=expected_made_budget,
                             core_protocol=core_protocol, evaluation_extension=evaluation_extension,
                             made_all30_extension=made_all30_extension, made_budget_sweep=made_budget_sweep)
    if made_budget_sweep is not None:
        from .made_budget_sweep import admitted_parent
        parent = admitted_parent(made_budget_sweep)
        _require(core_protocol is None or core_protocol == parent, "Sweep parent core differs")
        core_protocol = parent
        frozen_profile = made_budget_sweep["execution_profiles"][job["method"]]
        _require(expected_profile is None or expected_profile == frozen_profile,
                 "Sweep cannot reselect checkpoints/controllers")
        expected_profile = frozen_profile
    elif made_all30_extension is not None:
        from .made_all30_extension import admitted_parent
        parent = admitted_parent(made_all30_extension)
        _require(core_protocol is None or core_protocol == parent, "All30 parent core differs")
        core_protocol = parent
        frozen_profile = made_all30_extension["execution_profiles"][job["method"]]
        _require(expected_profile is None or expected_profile == frozen_profile,
                 "All30 cannot reselect checkpoints/controllers")
        expected_profile = frozen_profile
    elif evaluation_extension is not None:
        # verify_result validated the sealed extension and exact job membership.
        from .core_protocol import read_core
        parent = read_core(evaluation_extension["parent_core"]["workspace"])
        _require(core_protocol is None or core_protocol == parent, "Extension parent core differs")
        core_protocol = parent
        frozen_profile = evaluation_extension["execution_profiles"][job["method"]]
        _require(expected_profile is None or expected_profile == frozen_profile,
                 "Extension cannot reselect parent checkpoints/controllers")
        expected_profile = frozen_profile
    envelope = _read(Path(path))
    _require(envelope.get("evidence_schema") == "final_official_rpc_v1", "Final result lacks full scientific-evidence verification")
    actual_job = envelope["execution_job"]
    _require(all(actual_job.get(key) == value for key, value in job.items()), "Final result changed the immutable matrix job")
    profile = envelope["execution_profile"]
    _require(envelope["execution_profile_fingerprint"] == fingerprint(profile)
             and actual_job["execution_profile_fingerprint"] == fingerprint(profile), "Final execution profile fingerprint mismatch")
    if expected_profile is not None:
        _require(profile == expected_profile, "Completed final result used another checkpoint/controller/source configuration")
    _verify_profile_artifacts(profile)
    output = Path(path).parent
    inventory = [item for item in TrainingJobCallbacks._inventory(output) if item["path"] != str(Path(path).resolve())]
    _require(inventory == envelope["artifacts"], "Final evidence inventory changed or omitted raw files")
    if job["benchmark"] == "made":
        expected_mace_num_workers = (profile.get("mace_num_workers", 1) if expected_mace_num_workers is None
                                     else expected_mace_num_workers)
        _require(profile.get("mace_num_workers", 1) == expected_mace_num_workers,
                 "Final profile differs from the fixed paired MACE executor")
    evidence = reconstruct_scientific_evidence(output, actual_job, tasks=tasks, partition="final_test",
        expected_mace_num_workers=expected_mace_num_workers if job["benchmark"] == "made" else None,
        expected_made_budget=expected_made_budget, core_protocol=core_protocol,
        made_all30_extension=made_all30_extension, made_budget_sweep=made_budget_sweep)
    _require(evidence == envelope["scientific_evidence"] and envelope["episodes"] == _read(output / "episodes.json"), "Final reported scientific values differ from raw complete trajectories")
    _require(actual_job["actual_model_state_hash"] == profile["actual_model_state_hash"], "Final result confused initial checkpoint identity with actual updated weights")
    if job["method"] in {"graph_risk", "esopt_graph_risk"}:
        from .native_attribution import CONTRACT
        decisions = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines() if line.strip()]
        for row in decisions:
            _require(row.get("graph_status") in {"succeeded", "unavailable"}, "Every graph decision requires a graph or an explicit unavailable record")
            if row["graph_status"] == "succeeded":
                metadata = row["graph_metadata"]
                _require(row["graph_contract"] == CONTRACT and metadata["policy"]["state_id"] == actual_job["policy_state_id"]
                         and metadata["policy"]["generation"] == actual_job["selected_generation"]
                         and metadata["full_prefix_hash"] == row["prefix_hash"], "Final graph was not attributed under the current selected policy and exact generated prefix")
    return verified


class FinalEvaluationRunner:
    """One process executes a scheduling batch; the ledger always contains 2160 jobs."""
    def __init__(self, project: Path, output: Path, *, risk_root: Path, transcoder_root: Path, es_root: Path,
                 device=None, cuda_memory_fraction=None, policy_factory=None, rollout_factory=None,
                 controller_loader=None):
        self.project, self.output = Path(project).resolve(), Path(output).resolve()
        self.risk_root, self.transcoder_root, self.es_root = map(lambda p: Path(p).resolve(), (risk_root, transcoder_root, es_root))
        self.manifest = build_final_manifest(self.project)
        self.tasks = _read(self.project / "configs/benchmark_tasks.json")
        self.protocol = _read(self.project / "configs/main_protocol.json")
        from .execution_contract import declared_mace_workers
        self.expected_mace_num_workers = declared_mace_workers(self.protocol)
        self.runtime = declared_runtime_options(self.protocol, device=device, cuda_memory_fraction=cuda_memory_fraction)
        self.device, self.cuda_memory_fraction = self.runtime["device"], self.runtime["cuda_memory_fraction"]
        self.policy_factory, self.rollout_factory, self.controller_loader = policy_factory, rollout_factory, controller_loader
        files = sorted((self.project / "src/matdiscovery").glob("*.py"))
        files += [self.project / path for path in ("scripts/run_final_evaluation.py", "configs/main_protocol.json", "configs/benchmark_tasks.json", "configs/model_manifest.json", "configs/made_splits.json")]
        # Both evaluator runtimes are fixed even when only one batch runs today.
        files += [self.project / path for path in ("configs/assets.runtime.json", "configs/qe.runtime.json", "data/raw/materials_project/index.json")]
        self.inputs = {str(path): file_sha256(path) for path in files}
        self.execution = {"inputs": self.inputs, "device": self.device, "cuda_memory_fraction": self.cuda_memory_fraction,
                          "risk_root": str(self.risk_root), "transcoder_root": str(self.transcoder_root), "es_root": str(self.es_root),
                          "artifact_layout": "model_key/benchmark; ES additionally property/method/seed",
                          "torch_cpu_threads": self.runtime["torch_cpu_threads"], "policy_runtime_declared": self.runtime}

    def _unchanged(self):
        _require(all(file_sha256(path) == digest for path, digest in self.inputs.items()), "Frozen final-evaluation sources, protocol or evaluator inputs changed")

    @contextmanager
    def _lock(self):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / ".runner.lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise TrainingHalted("Another final worker owns this study; one policy trajectory runs at a time") from exc
            yield

    def _load_policy(self, key):
        if self.policy_factory:
            return self.policy_factory(key)
        from .policy import DecodingConfig, QwenPolicyAdapter
        decode = self.protocol["decoding"]
        return QwenPolicyAdapter.from_verified_checkpoint(self.project / "configs/model_manifest.json", key,
            self.project / "data/models" / key,
            **{k: self.runtime[k] for k in ("device", "dtype", "attn_implementation", "sdpa_backend", "cpu_embedding_and_lm_head", "attention_checkpointing")},
            decoding=DecodingConfig(**{k: decode[k] for k in ("max_new_tokens", "temperature", "top_p", "top_k")}),
            enable_thinking=decode["enable_thinking"], max_input_tokens=decode["max_input_tokens"])

    def _profile(self, policy, job, base_state, base_hash):
        method, benchmark, key = job["method"], job["benchmark"], job["model_key"]
        policy.model.load_state_dict(base_state, strict=True)
        policy.mark_state("reload", generation=0)
        _require(tensor_state_hash(dict(policy.model.state_dict())) == base_hash, "Initial policy restore failed")
        selected, es_files = None, []
        if method.startswith("esopt"):
            condition = TrainingCondition(self.project, key, benchmark, method, job["seed"],
                                          job["task"]["property"]["id"] if benchmark == "crystalgym" else None)
            selected, es_files = selected_es_artifact(policy, condition, self.es_root)
        risk_method = "graph_risk" if method in {"graph_risk", "esopt_graph_risk"} else "hidden_risk" if method == "hidden_risk" else "entropy_risk"
        risk_path = self.risk_root / key / benchmark / (risk_method + ".pt")
        bank = self.transcoder_root / key / benchmark / "transcoder_manifest.json" if "graph_risk" in method else None
        loader = self.controller_loader or load_frozen_controllers
        risk, attributor, auxiliary = loader(policy, model_key=key, benchmark=benchmark, method=method,
            risk_checkpoint=risk_path, transcoder_manifest=bank, graph_config=self.protocol["graph"], device=self.device,
            failure_control=self.protocol.get("failure_control"))
        files = {str(path): file_sha256(path) for path in es_files + auxiliary}
        profile = {"model_key": key, "benchmark": benchmark, "method": method,
                   "seed": job["seed"] if method.startswith("esopt") else None,
                   "property": job["task"]["property"]["id"] if method.startswith("esopt") and benchmark == "crystalgym" else None,
                   "actual_model_state_hash": tensor_state_hash(dict(policy.model.state_dict())),
                   "initial_actual_model_state_hash": base_hash, "initial_checkpoint_manifest_hash": policy.checkpoint_hash,
                   "policy_configuration_fingerprint": policy.configuration_fingerprint,
                   "execution_inputs_fingerprint": fingerprint(self.execution), "artifacts": files,
                   "mace_num_workers": self.expected_mace_num_workers if benchmark == "made" else None,
                   "selected_es": selected, "auxiliary_modules_attached_to_policy": False}
        _require(profile["actual_model_state_hash"] == (selected["actual_model_state_hash"] if selected else base_hash), "Final policy has unexpected weights")
        return profile, risk, attributor

    def run(self, *, model_key, benchmark, method=None, seed=None, property_name=None, resume=False):
        jobs = select_final_jobs(self.manifest, model_key=model_key, benchmark=benchmark, method=method, seed=seed, property_name=property_name)
        from .rollouts import DiscoveryRollout, RolloutSettings
        rollout_factory = self.rollout_factory or DiscoveryRollout
        torch.set_num_threads(self.runtime["torch_cpu_threads"])
        if torch.device(self.device).type == "cuda":
            torch.cuda.set_per_process_memory_fraction(self.cuda_memory_fraction, device=self.device)
        with self._lock():
            self._unchanged()
            manifest_path, inputs_path = self.output / "manifest.json", self.output / "execution_inputs.json"
            if manifest_path.exists():
                _require(resume, "An existing final study requires explicit --resume, including later scheduling batches")
                _require(_read(manifest_path) == self.manifest and _read(inputs_path) == self.execution, "Final matrix or execution inputs changed")
            else:
                _require(not resume, "No final study exists to resume")
                write_json_atomic(manifest_path, self.manifest)
                write_json_atomic(inputs_path, self.execution)
            with RunLedger(self.output / "ledger.sqlite3", self.manifest) as ledger:
                if resume:
                    # Exclusive whole-study worker lock proves no peer is active;
                    # subprocess outcomes remain unknown and are only quarantined.
                    ledger.mark_orphaned(stale_before=time.time())
                policy = self._load_policy(model_key)
                base_state = {name: value.detach().cpu().clone() for name, value in policy.model.state_dict().items()}
                base_hash = tensor_state_hash(base_state)
                current_key, profile, risk, attributor = None, None, None, None
                artifact_stats = {}
                completed = []
                for job in jobs:
                    key = (job["method"], job["task"]["property"]["id"] if benchmark == "crystalgym" and job["method"].startswith("esopt") else None,
                           job["seed"] if job["method"].startswith("esopt") else None)
                    if key != current_key:
                        risk = attributor = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        profile, risk, attributor = self._profile(policy, job, base_state, base_hash)
                        artifact_stats = {path: (Path(path).stat().st_size, Path(path).stat().st_mtime_ns, Path(path).stat().st_ino)
                                          for path in profile["artifacts"]}
                        current_key = key
                    self._unchanged()
                    # Every artifact was SHA256-verified while loading this
                    # profile; guard file identity while rechecking live tensors
                    # per trajectory, without rereading 40 GB for every case.
                    _require(all((Path(path).stat().st_size, Path(path).stat().st_mtime_ns, Path(path).stat().st_ino) == stat
                                 for path, stat in artifact_stats.items()), "A selected checkpoint or controller changed during final evaluation")
                    state = ledger.get(job["job_id"])
                    if state["state"] == "succeeded":
                        proof = verify_final_envelope(state["result_path"], job, ledger.fingerprint, tasks=self.tasks, expected_profile=profile,
                            expected_mace_num_workers=self.expected_mace_num_workers)
                        _require(proof["result_sha256"] == state["result_sha256"], "Recorded final result bytes changed")
                        completed.append(job["job_id"])
                        continue
                    if state["state"] == "orphaned":
                        path = self.output / "jobs" / job["job_id"] / state["attempt_id"] / "result.json"
                        if path.is_file():
                            verify_final_envelope(path, job, ledger.fingerprint, tasks=self.tasks, expected_profile=profile,
                                expected_mace_num_workers=self.expected_mace_num_workers)
                            ledger.reconcile(job["job_id"], evidence="Existing complete envelope and closed official RPC independently verified; no action repeated", execution_stopped=True, result_path=path)
                            completed.append(job["job_id"])
                            continue
                        raise TrainingHalted("Final physical outcome is unknown; reconcile existing evidence without automatic retry: " + job["job_id"])
                    if state["state"] != "pending":
                        raise TrainingHalted("Final job requires explicit stopped-action reconciliation: " + job["job_id"])
                    claim = ledger.claim(job["job_id"], "serial-final-" + uuid.uuid4().hex)
                    _require(claim is not None, "Final job could not be exclusively claimed")
                    output = self.output / "jobs" / job["job_id"] / claim["attempt_id"]
                    actual_job = execution_job(job, policy, profile)
                    stamp = policy.get_state_id()
                    settings = RolloutSettings(max_generation_retries=self.protocol["risk_network"]["maximum_candidate_generations_per_decision"],
                        risk_threshold=self.protocol["risk_network"]["threshold"], history_results=self.protocol["memory"]["scientific_history_per_episode"],
                        recent_tools=self.protocol["memory"]["recent_tool_responses"],
                        failure_aware_control=benchmark in self.protocol.get("failure_control", {}).get("enabled_benchmarks", []))
                    try:
                        rollout_factory(self.project, policy, settings=settings, risk_model=risk, attributor=attributor).run(actual_job, output, collection=False)
                        _require(policy.get_state_id() == stamp and tensor_state_hash(dict(policy.model.state_dict())) == profile["actual_model_state_hash"], "Final rollout modified the selected policy")
                        evidence = reconstruct_scientific_evidence(output, actual_job, tasks=self.tasks, partition="final_test",
                            expected_mace_num_workers=self.expected_mace_num_workers if benchmark == "made" else None)
                        envelope = {"job_id": job["job_id"], "manifest_fingerprint": ledger.fingerprint, "model_revision": job["model_revision"],
                            "complete": True, "episodes": _read(output / "episodes.json"), "artifacts": TrainingJobCallbacks._inventory(output),
                            "evidence_schema": "final_official_rpc_v1", "scientific_evidence": evidence, "execution_job": actual_job,
                            "execution_profile": profile, "execution_profile_fingerprint": fingerprint(profile)}
                        path = output / "result.json"
                        write_json_atomic(path, envelope)
                        self._unchanged()
                        verify_final_envelope(path, job, ledger.fingerprint, tasks=self.tasks, expected_profile=profile,
                            expected_mace_num_workers=self.expected_mace_num_workers)
                        ledger.succeed(job["job_id"], claim["attempt_id"], path)
                        completed.append(job["job_id"])
                    except BaseException as exc:
                        # No claim of a stopped QE subprocess unless a closed RPC
                        # transcript proves it. Either state is non-runnable.
                        stopped = False
                        try:
                            _rpc_pairs(output / "rpc/rpc.jsonl")
                            stopped = True
                        except (OSError, ValueError, KeyError, TypeError):
                            pass
                        if stopped and not (output / "result.json").exists():
                            ledger.fail(job["job_id"], claim["attempt_id"], str(exc), execution_stopped=True,
                                        evidence_paths=[p for p in output.rglob("*") if p.is_file()])
                        else:
                            # A committed envelope can be reconciled read-only
                            # after a ledger-commit interruption. Invalid envelopes
                            # still fail verification; no branch dispatches a retry.
                            ledger.mark_orphaned(stale_before=time.time() + 1)
                        write_json_atomic(self.output / "last_halt.json", {"job_id": job["job_id"], "attempt_id": claim["attempt_id"],
                            "error_type": type(exc).__name__, "error": str(exc), "execution_stopped": stopped, "automatic_retry": False})
                        raise
                progress = ledger.completion()
                progress.update(batch_complete=len(completed) == len(jobs), batch_expected_jobs=len(jobs), batch_completed_jobs=len(completed),
                                batch_filters={"model_key": model_key, "benchmark": benchmark, "method": method, "seed": seed, "property": property_name})
                write_json_atomic(self.output / "progress.json", progress)
                return progress
