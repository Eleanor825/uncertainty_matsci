"""Execute only the 50 newly registered B10 trajectories in independent workers.

Workers own disjoint, sealed job IDs. A worker groups methods to load each fixed
profile once. Launch validates the parent evidence deeply; the explicit all30
admission subsequently checks the same process's authorization, immutable inputs,
job membership and profile without recursively re-auditing ten old trajectories.
Every new trajectory still undergoes the original raw RPC/scientific verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import socket
import sys
import time
import uuid

from .accounting import file_sha256, fingerprint, write_json_atomic
from .core_final import CoreFinalLedger, CoreFinalRunner, STATES, publish_once, read
from .evaluation_extension_runner import exclusive_lock, _preflight, _result_path, _pairs, _costs
from .made_all30_extension import read_all30, validate_all30, validate_launch, all30_jobs, audit_imported_results

SCOPE = "post_results_MADE_all30_B10_fixed_G2_evaluation"
METHODS = ("baseline", "esopt_graph_risk")
NEW_COUNTS = {"jobs": 50, "episodes": 50, "candidate_oracle_attempts": 500, "dft_episode_attempts": 0}
CUMULATIVE_COUNTS = {"jobs": 60, "episodes": 60, "candidate_oracle_attempts": 600, "dft_episode_attempts": 0}
MANIFEST_SCHEMA = "sealed_made_all30_worker_v1"
LEDGER_SCHEMA = "made_all30_worker_ledger_v1"
STAGE_SCHEMA = "made_all30_worker_stage_receipt_v1"
CHANGED_MODULES = ("accounting.py", "final_evaluation.py", "training_jobs.py")
ADDED_MODULES = ("made_all30_extension.py", "made_all30_runner.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def worker_spec(registration, worker_id):
    matches = [item for item in registration["execution_shards"] if item["worker_id"] == worker_id]
    require(len(matches) == 1, "Worker is not in the sealed ownership matrix")
    return matches[0]


def verify_execution_sources(registration):
    """All numerical/control/runtime source stays byte-identical to the old five."""
    package = Path(__file__).resolve().parent
    original = Path(registration["prior_extension"]["workspace"]) / "src/matdiscovery"
    sources = {Path(item["path"]).resolve(): item["sha256"] for item in registration["source_files"]}
    for name in (*CHANGED_MODULES, *ADDED_MODULES):
        path = package / name
        require(path in sources and file_sha256(path) == sources[path], "Unsealed actual all30 implementation: " + name)
    actual = {str(p.relative_to(package)) for p in package.rglob("*.py")}
    expected = {str(p.relative_to(original)) for p in original.rglob("*.py")}
    require(actual == expected | set(ADDED_MODULES) and not (expected & set(ADDED_MODULES)),
            "All30 changed the frozen Python inventory")
    for name in sorted(expected - set(CHANGED_MODULES)):
        require(file_sha256(package / name) == file_sha256(original / name),
                "All30 changed a frozen runtime/control module: " + name)


def build_worker_manifest(registration, worker_id):
    spec = worker_spec(registration, worker_id)
    jobs = sorted(all30_jobs(registration, worker_id=worker_id),
                  key=lambda job: (METHODS.index(job["method"]), job["task_id"], job["job_id"]))
    value = {"schema": MANIFEST_SCHEMA, "scope": SCOPE, "all30_fingerprint": registration["fingerprint"],
        "parent_core_fingerprint": registration["parent_core"]["core_fingerprint"],
        "worker": deepcopy(spec), "jobs": jobs,
        "expected_counts": {**NEW_COUNTS, "jobs": len(jobs), "episodes": len(jobs),
                            "candidate_oracle_attempts": 10 * len(jobs)},
        "all_new_expected_counts": dict(NEW_COUNTS), "global_study_complete": False,
        "accepts_only_worker_completion": True}
    value["fingerprint"] = fingerprint(value)
    return value


def validate_worker_manifest(manifest, registration):
    require(manifest == build_worker_manifest(registration, manifest.get("worker", {}).get("worker_id")),
            "Worker job matrix, ownership, or denominator changed")


class MadeAll30Ledger(CoreFinalLedger):
    """Original one-attempt transitions, independent exact worker admission."""
    def __init__(self, path, manifest, *, registration, readonly=False):
        validate_worker_manifest(manifest, registration)
        self.path, self.manifest, self.readonly = Path(path), manifest, readonly
        self.registration, self.fingerprint = registration, manifest["fingerprint"]
        if self.path.exists():
            self.data = read(self.path)
            self.validate()
        else:
            require(not readonly, "All30 worker ledger does not exist")
            self.data = {"schema": LEDGER_SCHEMA, "scope": SCOPE, "manifest_fingerprint": self.fingerprint,
                "all30_fingerprint": registration["fingerprint"], "worker_id": manifest["worker"]["worker_id"],
                "global_study_complete": False, "jobs": {job["job_id"]: {"state": "pending",
                    "attempt_number": 0, "attempt_id": None, "result_path": None, "result_sha256": None}
                    for job in manifest["jobs"]}, "events": []}
            self.save()

    def validate(self):
        require(self.data.get("schema") == LEDGER_SCHEMA and self.data.get("scope") == SCOPE
            and self.data.get("manifest_fingerprint") == self.fingerprint
            and self.data.get("all30_fingerprint") == self.registration["fingerprint"]
            and self.data.get("worker_id") == self.manifest["worker"]["worker_id"]
            and self.data.get("global_study_complete") is False, "Ledger belongs to another all30 worker")
        require(set(self.data.get("jobs", {})) == {job["job_id"] for job in self.manifest["jobs"]},
                "All30 worker ledger added or omitted a physical job")
        claims, attempts = Counter(), {}
        for index, event in enumerate(self.data.get("events", [])):
            require(event.get("sequence") == index and event.get("job_id") in self.data["jobs"],
                    "All30 ledger event identity changed")
            if event["event"] == "claimed":
                claims[event["job_id"]] += 1
                attempts[event["job_id"]] = event["details"]["attempt_id"]
        for job_id, state in self.data["jobs"].items():
            require(state["state"] in STATES and type(state["attempt_number"]) is int
                    and state["attempt_number"] == claims[job_id] and claims[job_id] <= 1,
                    "Repeated or unregistered all30 physical attempt")
            require((state["state"] == "pending") == (claims[job_id] == 0)
                    and state["attempt_id"] == attempts.get(job_id), "All30 claim/attempt identity changed")
            if state["state"] == "succeeded":
                require(state["result_path"] and state["result_sha256"], "Completed all30 job lacks evidence")

    def completion(self):
        states = {name: sum(row["state"] == name for row in self.data["jobs"].values()) for name in sorted(STATES)}
        return {"schema": "made_all30_worker_completion_v1", "scope": SCOPE,
            "worker_id": self.manifest["worker"]["worker_id"], "all30_fingerprint": self.registration["fingerprint"],
            "manifest_fingerprint": self.fingerprint, "expected_jobs": len(self.manifest["jobs"]),
            "completed_jobs": states["succeeded"], "worker_complete": states["succeeded"] == len(self.manifest["jobs"]),
            "all30_complete": False, "global_study_complete": False, "states": states,
            "expected_counts": deepcopy(self.manifest["expected_counts"]), "all_new_expected_counts": dict(NEW_COUNTS),
            "ledger_fingerprint": fingerprint(self.data)}


def verify_all30_envelope(path, job, manifest, registration, *, tasks, core):
    from .final_evaluation import verify_final_envelope
    validate_worker_manifest(manifest, registration)
    require(job in manifest["jobs"], "Result belongs to another all30 worker")
    envelope = read(path)
    require(envelope.get("scope") == SCOPE and envelope.get("all30_fingerprint") == registration["fingerprint"]
        and envelope.get("parent_core_fingerprint") == registration["parent_core"]["core_fingerprint"]
        and envelope.get("worker_id") == manifest["worker"]["worker_id"]
        and envelope.get("execution_site") == manifest["worker"]["execution_site"]
        and envelope.get("global_study_complete") is False, "All30 envelope identity changed")
    profile = registration["execution_profiles"][job["method"]]
    require(profile.get("mace_num_workers") == 4 and profile.get("orb_num_workers") == 1, "All30 evaluator settings changed")
    require(envelope["execution_job"].get("source") == job.get("source", "registered_made_all30_final"),
            "All30 execution source changed")
    generation = profile["selected_es"]["selected_generation"] if job["method"] == "esopt_graph_risk" else 0
    require(envelope["execution_job"].get("selected_generation") == generation, "All30 selected generation changed")
    proof = verify_final_envelope(path, job, manifest["fingerprint"], tasks=tasks, expected_profile=profile,
        expected_mace_num_workers=4, expected_made_budget=10, core_protocol=core, made_all30_extension=registration)
    for episode in envelope["episodes"]:
        require(episode["metrics"].get("mSUN") == episode["discovery_curve"][-1][1] / 10, "All30 mSUN differs from raw discoveries")
        require("failure_rate" not in episode["metrics"] or episode["metrics"]["failure_rate"] == 1 - episode["metrics"]["mSUN"],
                "All30 non-discovery failure rate differs from mSUN")
    return proof


def _verify_finished(output, ledger, registration, *, tasks, core):
    for job in ledger.manifest["jobs"]:
        state = ledger.data["jobs"][job["job_id"]]
        if state["state"] not in {"orphaned", "succeeded"}:
            continue
        path = _result_path(output, job, state)
        proof = verify_all30_envelope(path, job, ledger.manifest, registration, tasks=tasks, core=core)
        if state["state"] == "succeeded":
            require(state["result_path"] == str(path.resolve()) and state["result_sha256"] == proof["result_sha256"],
                    "Committed all30 result changed")
        else:
            ledger.finish(job["job_id"], path, proof["result_sha256"], reconciled=True)


class MadeAll30Runner:
    def __init__(self, workspace, *, worker_id, output=None, policy_factory=None, rollout_factory=None,
                 controller_loader=None, selected_loader=None):
        self.registration = read_all30(workspace)
        self.workspace = Path(self.registration["workspace"])
        self.worker = worker_spec(self.registration, worker_id)
        self.output = Path(self.worker["output_workspace"]).resolve()
        require(output is None or Path(output).resolve() == self.output, "Output override violates sealed worker ownership")
        require(not Path(self.worker["output_workspace"]).is_symlink(), "All30 worker output cannot be aliased")
        verify_execution_sources(self.registration)
        self.parent = CoreFinalRunner(self.registration["parent_core"]["workspace"], policy_factory=policy_factory,
            rollout_factory=rollout_factory, controller_loader=controller_loader, selected_loader=selected_loader)
        self.core, self.tasks, self.protocol, self.runtime = self.parent.core, self.parent.tasks, self.parent.protocol, self.parent.runtime
        require(self.core["fingerprint"] == self.registration["parent_core"]["core_fingerprint"], "All30 loaded another parent core")
        self.manifest = build_worker_manifest(self.registration, worker_id)
        self.execution = {"scope": SCOPE, "all30_fingerprint": self.registration["fingerprint"],
            "worker": deepcopy(self.worker), "parent_execution": self.parent.execution,
            "source_files": self.registration["source_files"]}
        self.rollout_factory = rollout_factory

    def unchanged(self):
        self.parent.unchanged()
        require(validate_all30(self.registration) == self.registration, "All30 registration changed during execution")
        verify_execution_sources(self.registration)

    def _publish(self, ledger):
        result = ledger.completion()
        require(result["worker_complete"], "Incomplete worker cannot publish completion")
        self.unchanged()
        result.update(complete=True, result_files=[{"path": ledger.data["jobs"][job["job_id"]]["result_path"],
            "sha256": ledger.data["jobs"][job["job_id"]]["result_sha256"]} for job in self.manifest["jobs"]])
        publish_once(self.output / "completion.json", result)
        stage = {"schema": STAGE_SCHEMA, "complete": True, "scope": SCOPE,
            "all30_fingerprint": self.registration["fingerprint"], "worker_id": self.worker["worker_id"],
            "execution_site": self.worker["execution_site"], "worker_complete": True,
            "all30_complete": False, "global_study_complete": False,
            "artifacts": [artifact(self.output / name) for name in ("manifest.json", "execution_inputs.json", "ledger.json", "completion.json")]}
        stage["fingerprint"] = fingerprint(stage)
        publish_once(self.output / "stage_receipt.json", stage)
        return result

    def run(self):
        with exclusive_lock(self.output / ".runner.lock"):
            self.unchanged()
            for name, value in (("manifest.json", self.manifest), ("execution_inputs.json", self.execution)):
                publish_once(self.output / name, value)
            jobs_root = self.output / "jobs"
            require((self.output / "ledger.json").exists() or not jobs_root.exists() or not any(jobs_root.iterdir()),
                    "Untracked all30 physical output exists without a ledger")
            ledger = MadeAll30Ledger(self.output / "ledger.json", self.manifest, registration=self.registration)
            if any(row["state"] == "running" for row in ledger.data["jobs"].values()):
                ledger.quarantine_running()
            _preflight(self.output, ledger)
            proof = validate_launch(self.registration)
            require(proof.get("complete") is True and proof.get("all30_fingerprint") == self.registration["fingerprint"],
                    "All30 launch deep validation did not complete")
            proof_path = self.output / "launch_checks" / (fingerprint(proof) + ".json")
            publish_once(proof_path, proof)
            _verify_finished(self.output, ledger, self.registration, tasks=self.tasks, core=self.core)
            if ledger.completion()["worker_complete"]:
                return self._publish(ledger)
            # Lock, registration, deep proof, attempt recovery and a durable
            # invocation claim all precede loading the policy.
            invocation = self.output / "launch_checks" / ("invocation-" + uuid.uuid4().hex + ".json")
            publish_once(invocation, {"schema": "made_all30_worker_invocation_v1", "all30_fingerprint": self.registration["fingerprint"],
                "worker_id": self.worker["worker_id"], "execution_site": self.worker["execution_site"],
                "hostname": socket.gethostname(), "pid": os.getpid(), "started_at": time.time(),
                "python_executable": sys.executable, "python_version": sys.version.split()[0],
                "registered_runtime": deepcopy(self.runtime),
                "deep_launch_proof": artifact(proof_path), "automatic_physical_replay": False})
            import torch
            from .esopt import tensor_state_hash
            from .final_evaluation import execution_job
            from .rollouts import DiscoveryRollout, RolloutSettings
            from .training_jobs import TrainingJobCallbacks, reconstruct_scientific_evidence, _rpc_pairs
            torch.set_num_threads(self.runtime["torch_cpu_threads"])
            if torch.device(self.runtime["device"]).type == "cuda":
                torch.cuda.set_per_process_memory_fraction(self.runtime["cuda_memory_fraction"], device=self.runtime["device"])
            policy = self.parent.load_policy()
            baseline = {name: value.detach().cpu().clone() for name, value in policy.model.state_dict().items()}
            base_hash, current_method = tensor_state_hash(baseline), None
            profile = risk = graph = None
            factory = self.rollout_factory or DiscoveryRollout
            for job in self.manifest["jobs"]:
                if ledger.data["jobs"][job["job_id"]]["state"] == "succeeded":
                    continue
                if job["method"] != current_method:
                    risk = graph = None
                    gc.collect()
                    if torch.device(self.runtime["device"]).type == "cuda":
                        torch.cuda.empty_cache()
                    profile, risk, graph = self.parent.profile(policy, job, baseline, base_hash)
                    require(profile == self.registration["execution_profiles"][job["method"]], "Loaded all30 profile differs from frozen G2/controller")
                    current_method = job["method"]
                self.unchanged()
                attempt = ledger.claim(job["job_id"])
                output = self.output / "jobs" / job["job_id"] / attempt
                require(not output.exists(), "All30 physical attempt directory already exists")
                actual = execution_job(job, policy, profile)
                actual["source"] = job.get("source", "registered_made_all30_final")
                actual["group_id"] = job.get("group_id", actual["group_id"])
                stamp = policy.get_state_id()
                settings = RolloutSettings(max_generation_retries=self.protocol["risk_network"]["maximum_candidate_generations_per_decision"],
                    risk_threshold=self.protocol["risk_network"]["threshold"], history_results=self.protocol["memory"]["scientific_history_per_episode"],
                    recent_tools=self.protocol["memory"]["recent_tool_responses"], failure_aware_control=True)
                try:
                    factory(self.parent.project, policy, settings=settings, risk_model=risk, attributor=graph).run(actual, output, collection=False)
                    require(policy.get_state_id() == stamp and tensor_state_hash(dict(policy.model.state_dict())) == profile["actual_model_state_hash"],
                            "All30 trajectory changed fixed policy weights")
                    evidence = reconstruct_scientific_evidence(output, actual, tasks=self.tasks, partition="final_test",
                        expected_mace_num_workers=4, expected_made_budget=10, core_protocol=self.core, made_all30_extension=self.registration)
                    value = {"scope": SCOPE, "all30_fingerprint": self.registration["fingerprint"],
                        "parent_core_fingerprint": self.core["fingerprint"], "worker_id": self.worker["worker_id"],
                        "execution_site": self.worker["execution_site"], "global_study_complete": False,
                        "job_id": job["job_id"], "manifest_fingerprint": self.manifest["fingerprint"],
                        "model_revision": job["model_revision"], "complete": True, "episodes": read(output / "episodes.json"),
                        "artifacts": TrainingJobCallbacks._inventory(output), "evidence_schema": "final_official_rpc_v1",
                        "scientific_evidence": evidence, "execution_job": actual, "execution_profile": profile,
                        "execution_profile_fingerprint": fingerprint(profile)}
                    path = output / "result.json"
                    write_json_atomic(path, value)
                    self.unchanged()
                    verification = verify_all30_envelope(path, job, self.manifest, self.registration, tasks=self.tasks, core=self.core)
                    ledger.finish(job["job_id"], path, verification["result_sha256"])
                except BaseException as error:
                    closed = False
                    try:
                        _rpc_pairs(output / "rpc/rpc.jsonl")
                        closed = not (output / "result.json").exists()
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                    ledger.halt(job["job_id"], error, closed=closed,
                        evidence_files=[p for p in output.rglob("*") if p.is_file()] if output.exists() else ())
                    raise
                write_json_atomic(self.output / "progress.json", ledger.completion())
            _verify_finished(self.output, ledger, self.registration, tasks=self.tasks, core=self.core)
            return self._publish(ledger)


def aggregate_all30(workspace):
    """CPU-only acceptance of 50 new + 10 original, never a new physical ledger."""
    registration = read_all30(workspace)
    verify_execution_sources(registration)
    root = Path(registration["workspace"])
    reports = root / "experiments/made_all30_reports"
    with exclusive_lock(reports / ".aggregate.lock"):
        missing = [s["worker_id"] for s in registration["execution_shards"] if not (Path(s["output_workspace"]) / "stage_receipt.json").is_file()]
        if missing:
            return {"complete": False, "all30_complete": False, "global_study_complete": False,
                "all30_fingerprint": registration["fingerprint"], "missing_workers": missing, "expected_new_counts": dict(NEW_COUNTS)}
        validate_launch(registration)
        core = read(registration["parent_core"]["path"])
        tasks = read(Path(core["workspace"]) / "configs/benchmark_tasks.json")
        records, evidence, new_ids = {}, [], set()
        for worker in registration["execution_shards"]:
            output = Path(worker["output_workspace"])
            with exclusive_lock(output / ".runner.lock"):
                manifest = read(output / "manifest.json")
                validate_worker_manifest(manifest, registration)
                ledger = MadeAll30Ledger(output / "ledger.json", manifest, registration=registration, readonly=True)
                _preflight(output, ledger)
                require(ledger.completion()["worker_complete"], "All30 worker has incomplete ledger jobs")
                stage = read(output / "stage_receipt.json")
                require(stage == {"schema": STAGE_SCHEMA, "complete": True, "scope": SCOPE,
                    "all30_fingerprint": registration["fingerprint"], "worker_id": worker["worker_id"],
                    "execution_site": worker["execution_site"], "worker_complete": True, "all30_complete": False,
                    "global_study_complete": False, "artifacts": [artifact(output / name) for name in
                        ("manifest.json", "execution_inputs.json", "ledger.json", "completion.json")], "fingerprint": stage.get("fingerprint")}
                    and stage.get("fingerprint") == fingerprint({k: v for k, v in stage.items() if k != "fingerprint"}),
                    "Invalid or changed all30 worker stage receipt")
                execution = read(output / "execution_inputs.json")
                require(execution == {"scope": SCOPE, "all30_fingerprint": registration["fingerprint"],
                    "worker": worker, "parent_execution": execution["parent_execution"], "source_files": registration["source_files"]}
                    and all(profile["execution_inputs_fingerprint"] == fingerprint(execution["parent_execution"])
                            for profile in registration["execution_profiles"].values()), "All30 execution input binding changed")
                expected = ledger.completion()
                expected.update(complete=True, result_files=[{"path": ledger.data["jobs"][job["job_id"]]["result_path"],
                    "sha256": ledger.data["jobs"][job["job_id"]]["result_sha256"]} for job in manifest["jobs"]])
                require(read(output / "completion.json") == expected, "All30 worker completion changed")
                for job in manifest["jobs"]:
                    state = ledger.data["jobs"][job["job_id"]]
                    path = _result_path(output, job, state)
                    proof = verify_all30_envelope(path, job, manifest, registration, tasks=tasks, core=core)
                    require(state["result_path"] == str(path.resolve()) and state["result_sha256"] == proof["result_sha256"],
                            "Aggregated all30 result differs from committed ledger")
                    require(job["job_id"] not in new_ids, "All30 physical job appears in multiple workers")
                    new_ids.add(job["job_id"])
                    records[(job["task_id"], job["method"])] = {"episode": read(path)["episodes"][0], "result": artifact(path)}
                    evidence.append(artifact(path))
                evidence.append(artifact(output / "stage_receipt.json"))
        require(new_ids == {job["job_id"] for job in registration["new_jobs"]} and len(new_ids) == 50,
                "All30 acceptance requires all 50 unique new jobs")
        imported = audit_imported_results(registration, require_complete=True)
        require(imported["complete"] is True and len(imported["results"]) == 10, "Original five-system results incomplete")
        old_tasks = set()
        for item in imported["results"]:
            job, reference = item["job"], item["result"]
            require(artifact(reference["path"]) == reference, "Imported all30 result changed")
            key = (job["task_id"], job["method"])
            require(key not in records, "Original result duplicated into new all30 work")
            records[key] = {"episode": item["episode"], "result": reference}
            old_tasks.add(job["task_id"])
            evidence.append(reference)
        new_tasks = {job["task_id"] for job in registration["new_jobs"]}
        require(len(records) == 60 and len(old_tasks) == 5 and len(new_tasks) == 25 and not old_tasks & new_tasks,
                "All30 requires exactly 30 paired systems")
        report = {"schema": "made_all30_B10_report_v1", "scope": SCOPE, "complete": True,
            "all30_complete": True, "global_study_complete": False, "original_full_study_complete": False,
            "original_b50_core_complete": False, "all30_fingerprint": registration["fingerprint"],
            "expected_new_counts": dict(NEW_COUNTS), "expected_cumulative_counts": dict(CUMULATIVE_COUNTS),
            "original_five_registered_before_expansion": _pairs(records, sorted(old_tasks)),
            "new_twenty_five_after_expansion": _pairs(records, sorted(new_tasks)),
            "cumulative_thirty_descriptive": _pairs(records, sorted(old_tasks | new_tasks)),
            "prospective_thirty_system_registration_claimed": False,
            "costs": {"scope": "final_evaluation_only_not_total_research_cost", "original_ten_by_method": _costs(records, old_tasks),
                "new_fifty_by_method": _costs(records, new_tasks), "cumulative_sixty_by_method": _costs(records, old_tasks | new_tasks),
                "additional_training_or_ES_cost": 0, "prior_training_not_recharged": True},
            "source_results_and_workers": evidence}
        report["fingerprint"] = fingerprint(report)
        require(validate_all30(registration) == registration, "All30 registration changed before publication")
        publish_once(reports / "report.json", report)
        stage = {"schema": "made_all30_global_stage_receipt_v1", "complete": True, "scope": SCOPE,
            "all30_fingerprint": registration["fingerprint"], "all30_complete": True, "global_study_complete": False,
            "expected_new_counts": dict(NEW_COUNTS), "expected_cumulative_counts": dict(CUMULATIVE_COUNTS),
            "report": artifact(reports / "report.json"), "source_results_and_workers": evidence}
        stage["fingerprint"] = fingerprint(stage)
        publish_once(root / "experiments/made_all30_stage_receipts/final.json", stage)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker")
    mode.add_argument("--aggregate", action="store_true")
    args = parser.parse_args(argv)
    from .immutable_hash_cache import immutable_hash_cache
    with immutable_hash_cache() as counters:
        result = aggregate_all30(args.workspace) if args.aggregate else MadeAll30Runner(args.workspace, worker_id=args.worker).run()
    print(json.dumps({"result": result, "hash_cache_io": counters}, indent=2))
    return 0 if result.get("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
