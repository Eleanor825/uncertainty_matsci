"""Persistent actors for the separately registered B30/B50 evaluation.

The directory reservation and immutable claim permit at most one physical
attempt per job. Claims are never leased, recycled, or reassigned. CPU
aggregation may reconcile a complete orphaned envelope under its execution
lock, but cannot construct a replacement trajectory. Optional actors do not
enter completion denominators: only the 120 registered jobs do.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import csv
import fcntl
import gc
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

from .accounting import file_sha256, fingerprint, write_json_atomic
from .core_final import CoreFinalRunner, publish_once, read
from .evaluation_extension_runner import _pairs, _costs
from .made_budget_sweep import read_sweep, validate_sweep, validate_launch, sweep_jobs, audit_B10_results

SCOPE = "post_results_MADE_B30_B50_fixed_G2_budget_sweep"
METHODS = ("baseline", "esopt_graph_risk")
BUDGETS = (30, 50)
CHANGED_MODULES = ("accounting.py", "final_evaluation.py", "training_jobs.py")
ADDED_MODULES = ("made_budget_sweep.py", "made_budget_runner.py")
NEW_COUNTS = {"jobs": 120, "episodes": 120, "candidate_oracle_attempts": 4800, "dft_episode_attempts": 0}
CUMULATIVE_COUNTS = {"jobs": 180, "episodes": 180, "candidate_oracle_attempts": 5400, "dft_episode_attempts": 0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def verify_artifact(value):
    require(set(value) == {"path", "sha256"} and artifact(value["path"]) == value, "Budget evaluation artifact changed")


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_once(path, value):
    """A failed/truncated exclusive write stays claimed; no automatic repair."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    sync_directory(Path(path).parent)


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BlockingIOError("Another process holds the budget actor/GPU/job lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def actor_spec(registration, actor_id):
    matches = [a for a in registration["actors"] if a["actor_id"] == actor_id]
    require(len(matches) == 1 and matches[0]["role"] == "gpu_actor"
            and matches[0]["initial_method"] in METHODS, "Unregistered budget actor or role")
    return matches[0]


def verify_execution_sources(registration):
    package = Path(__file__).resolve().parent
    original = Path(registration["prior_all30"]["workspace"]) / "src/matdiscovery"
    sources = {Path(a["path"]).resolve(): a["sha256"] for a in registration["source_files"]}
    for name in (*CHANGED_MODULES, *ADDED_MODULES):
        path = package / name
        require(path in sources and file_sha256(path) == sources[path], "Unsealed executing budget source: " + name)
    actual = {str(p.relative_to(package)) for p in package.rglob("*.py")}
    expected = {str(p.relative_to(original)) for p in original.rglob("*.py")}
    require(actual == expected | set(ADDED_MODULES) and not expected & set(ADDED_MODULES),
            "Budget sweep changed the frozen Python inventory")
    for name in sorted(expected - set(CHANGED_MODULES)):
        require(file_sha256(package / name) == file_sha256(original / name), "Budget sweep changed frozen runtime/control: " + name)


def gpu_identity():
    response = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
                              text=True, capture_output=True, check=True, timeout=20)
    rows = list(csv.reader(io.StringIO(response.stdout)))
    require(len(rows) == 1 and len(rows[0]) == 3, "Each actor requires one allocated GPU")
    index, identifier, name = [v.strip() for v in rows[0]]
    require(index == "0" and identifier.startswith("GPU-") and name == "NVIDIA H20", "Unexpected allocated GPU")
    return {"index": 0, "uuid": identifier, "name": name}


def job_manifest(registration, job):
    require(job in registration["new_jobs"], "Job is outside the new budget matrix")
    value = {"schema": "made_budget_single_job_manifest_v1", "scope": SCOPE,
        "sweep_fingerprint": registration["fingerprint"], "jobs": [deepcopy(job)],
        "expected_counts": {"jobs": 1, "episodes": 1, "candidate_oracle_attempts": job["budget"], "dft_episode_attempts": 0}}
    value["fingerprint"] = fingerprint(value)
    return value


def queue_state(registration):
    """Control-plane counts, never a scientific acceptance of result contents."""
    root = Path(registration["workspace"]) / "jobs"
    expected = {j["job_id"] for j in registration["new_jobs"]}
    if root.exists():
        require(not root.is_symlink(), "Aliased budget jobs directory")
        require(all(p.name in expected and p.is_dir() and not p.is_symlink() for p in root.iterdir()),
                "Unknown or aliased budget job directory")
    counts = {str(b): Counter() for b in BUDGETS}
    for job in registration["new_jobs"]:
        folder = root / job["job_id"]
        if not folder.exists(): state = "unclaimed"
        elif (folder / "receipt.json").is_file(): state = "receipt_present_not_reverified"
        elif (folder / "failure.json").is_file(): state = "failed_or_requires_reconciliation"
        elif (folder / "claim.json").is_file(): state = "claimed_not_complete"
        else: state = "unknown_reserved"
        counts[str(job["budget"])][state] += 1
    return {key: dict(value) for key, value in counts.items()}


def claim_next_job(registration, actor, invocation, gpu, current_method):
    """Atomic per-job reservation: B30 first, then current method within budget.

    Existing directories, including empty ones left by a crash, remain reserved.
    B50 becomes eligible only after every B30 directory has been reserved; it
    does not wait for already running B30 trajectories to finish.
    """
    require(actor == actor_spec(registration, actor["actor_id"]), "Actor assignment changed")
    verify_artifact(invocation)
    root = Path(registration["workspace"]) / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    queue_state(registration)
    jobs = sorted(registration["new_jobs"], key=lambda j: (j["budget"], j["method"] != current_method,
                   len(j["task"]["elements"]), j["task_id"], j["job_id"]))
    for job in jobs:
        directory = root / job["job_id"]
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        sync_directory(root)
        claim = {"schema": "made_budget_job_claim_v1", "scope": SCOPE, "sweep_fingerprint": registration["fingerprint"],
            "job_id": job["job_id"], "job_fingerprint": fingerprint(job), "budget": job["budget"],
            "actor_id": actor["actor_id"], "invocation": invocation, "gpu_uuid": gpu["uuid"],
            "hostname": socket.gethostname(), "pid": os.getpid(), "claimed_at": time.time(),
            "attempt_id": uuid.uuid4().hex, "attempt_number": 1, "automatic_physical_replay": False}
        create_once(directory / "claim.json", claim)
        publish_once(directory / "manifest.json", job_manifest(registration, job))
        return job, claim
    return None, None


def validate_claim(registration, job, claim):
    require(claim.get("schema") == "made_budget_job_claim_v1" and claim.get("scope") == SCOPE
        and claim.get("sweep_fingerprint") == registration["fingerprint"] and claim.get("job_id") == job["job_id"]
        and claim.get("job_fingerprint") == fingerprint(job) and claim.get("budget") == job["budget"]
        and claim.get("attempt_number") == 1 and claim.get("automatic_physical_replay") is False,
        "Budget claim identity or attempt changed")
    actor_spec(registration, claim["actor_id"])
    attempt = claim.get("attempt_id")
    require(isinstance(attempt, str) and len(attempt) == 32 and all(c in "0123456789abcdef" for c in attempt), "Invalid budget attempt ID")
    expected_invocation = Path(registration["workspace"]) / "actors" / claim["actor_id"] / "invocation.json"
    require(claim["invocation"]["path"] == str(expected_invocation), "Claim invocation is outside its actor")
    verify_artifact(claim["invocation"])
    invocation = read(expected_invocation)
    require(invocation.get("schema") == "made_budget_actor_invocation_v1" and invocation.get("scope") == SCOPE
        and invocation.get("automatic_physical_replay") is False
        and invocation["sweep_fingerprint"] == registration["fingerprint"] and invocation["actor"] == actor_spec(registration, claim["actor_id"])
        and invocation["gpu"]["uuid"] == claim["gpu_uuid"] and invocation["hostname"] == claim["hostname"]
        and invocation["pid"] == claim["pid"], "Claim owner differs from its immutable invocation")
    require(all(profile["execution_inputs_fingerprint"] == fingerprint(invocation["parent_execution"])
                for profile in registration["execution_profiles"].values())
            and invocation["registered_runtime"] == invocation["parent_execution"]["runtime"], "Actor changed its fixed execution inputs")
    verify_artifact(invocation["launch_validation"])
    proof = read(invocation["launch_validation"]["path"])
    require(proof.get("complete") is True and proof.get("sweep_fingerprint") == registration["fingerprint"]
            and proof.get("fingerprint") == fingerprint({k: v for k, v in proof.items() if k != "fingerprint"}),
            "Actor launch proof changed")


def verify_budget_envelope(path, job, registration, *, tasks, core):
    from .final_evaluation import verify_final_envelope
    directory = Path(registration["workspace"]) / "jobs" / job["job_id"]
    require(not directory.is_symlink(), "Aliased budget job")
    claim = read(directory / "claim.json"); validate_claim(registration, job, claim)
    expected_path = directory / claim["attempt_id"] / "result.json"
    require(Path(path) == expected_path and not expected_path.parent.is_symlink(), "Result is outside the sole claimed attempt")
    allowed = {"claim.json", "manifest.json", ".execution.lock", "receipt.json", "failure.json", claim["attempt_id"]}
    require({p.name for p in directory.iterdir()} <= allowed, "Unregistered additional budget attempt/evidence")
    manifest = job_manifest(registration, job)
    require(read(directory / "manifest.json") == manifest, "Budget job manifest changed")
    value = read(path)
    require(value.get("scope") == SCOPE and value.get("sweep_fingerprint") == registration["fingerprint"]
        and value.get("actor_id") == claim["actor_id"] and value.get("claim") == artifact(directory / "claim.json")
        and value.get("parent_core_fingerprint") == registration["parent_core"]["core_fingerprint"]
        and value.get("global_study_complete") is False, "Budget result provenance changed")
    profile = registration["execution_profiles"][job["method"]]
    require(profile.get("mace_num_workers") == 4 and profile.get("orb_num_workers") == 1, "Budget evaluator settings changed")
    actual = value["execution_job"]
    generation = profile["selected_es"]["selected_generation"] if job["method"] == METHODS[1] else 0
    require(actual.get("source") == job["source"] and actual.get("selected_generation") == generation,
            "Budget execution source or frozen generation changed")
    proof = verify_final_envelope(path, job, manifest["fingerprint"], tasks=tasks, expected_profile=profile,
        expected_mace_num_workers=4, expected_made_budget=job["budget"], core_protocol=core, made_budget_sweep=registration)
    for episode in value["episodes"]:
        require(episode["metrics"].get("mSUN") == episode["discovery_curve"][-1][1] / job["budget"], "Budget mSUN does not match raw discoveries")
        require("failure_rate" not in episode["metrics"] or episode["metrics"]["failure_rate"] == 1 - episode["metrics"]["mSUN"],
                "Budget non-discovery failure rate changed")
    return proof


def finish_job(registration, job, *, tasks, core):
    """Caller holds the job execution lock; no timestamp-dependent receipt."""
    directory = Path(registration["workspace"]) / "jobs" / job["job_id"]
    claim = read(directory / "claim.json"); validate_claim(registration, job, claim)
    path = directory / claim["attempt_id"] / "result.json"
    proof = verify_budget_envelope(path, job, registration, tasks=tasks, core=core)
    value = {"schema": "made_budget_job_receipt_v1", "scope": SCOPE, "complete": True,
        "sweep_fingerprint": registration["fingerprint"], "job_id": job["job_id"], "budget": job["budget"],
        "actor_id": claim["actor_id"], "claim": artifact(directory / "claim.json"),
        "manifest": artifact(directory / "manifest.json"), "result": {"path": str(path), "sha256": proof["result_sha256"]},
        "expected_counts": {"jobs": 1, "episodes": 1, "candidate_oracle_attempts": job["budget"], "dft_episode_attempts": 0},
        "global_study_complete": False}
    value["fingerprint"] = fingerprint(value)
    publish_once(directory / "receipt.json", value)
    return value


def global_receipt(registration):
    path = Path(registration["workspace"]) / "experiments/made_budget_stage_receipts/final.json"
    if not path.is_file(): return None
    value = read(path)
    require(value.get("schema") == "made_budget_global_stage_receipt_v1" and value.get("complete") is True
        and value.get("sweep_complete") is True and value.get("sweep_fingerprint") == registration["fingerprint"]
        and value.get("global_study_complete") is False and value.get("expected_new_counts") == NEW_COUNTS
        and value.get("expected_cumulative_counts") == CUMULATIVE_COUNTS
        and value.get("fingerprint") == fingerprint({k: v for k, v in value.items() if k != "fingerprint"}),
        "Invalid global budget receipt")
    verify_artifact(value["report"])
    root = Path(registration["workspace"])
    require(value["report"]["path"] == str(root / "experiments/made_budget_reports/report.json"), "Global budget report path changed")
    report = read(value["report"]["path"])
    require(report.get("schema") == "made_B10_B30_B50_report_v1" and report.get("complete") is True
        and report.get("sweep_complete") is True and report.get("sweep_fingerprint") == registration["fingerprint"]
        and report.get("expected_new_counts") == NEW_COUNTS and report.get("expected_cumulative_counts") == CUMULATIVE_COUNTS
        and report.get("global_study_complete") is False
        and report.get("fingerprint") == fingerprint({k: v for k, v in report.items() if k != "fingerprint"})
        and report.get("new_job_receipts") == value.get("new_job_receipts")
        and report.get("budget_reports") == value.get("budget_reports"), "Global budget report does not match its acceptance")
    references = value["new_job_receipts"]
    require(len(references) == 120 and {r["path"] for r in references}
        == {str(root / "jobs" / job["job_id"] / "receipt.json") for job in registration["new_jobs"]},
        "Global budget receipt omitted/duplicated a new job")
    for reference in references:
        verify_artifact(reference)
    require(set(value["budget_reports"]) == {"10", "30", "50"}, "Global report omitted a budget")
    for key, reference in value["budget_reports"].items():
        verify_artifact(reference)
        budget_report = read(reference["path"])
        require(budget_report.get("schema") == "made_single_budget_report_v1" and budget_report.get("complete") is True
            and budget_report.get("sweep_fingerprint") == registration["fingerprint"] and budget_report.get("budget") == int(key)
            and budget_report.get("expected_counts") == {"jobs": 60, "episodes": 60, "candidate_oracle_attempts": 60 * int(key), "dft_episode_attempts": 0},
            "Global report changed a budget denominator")
    return artifact(path)


class MadeBudgetActor:
    def __init__(self, workspace, *, actor_id, policy_factory=None, rollout_factory=None,
                 controller_loader=None, selected_loader=None, gpu_provider=None):
        self.registration = read_sweep(workspace)
        self.workspace = Path(self.registration["workspace"])
        self.actor = actor_spec(self.registration, actor_id)
        self.directory = self.workspace / "actors" / actor_id
        verify_execution_sources(self.registration)
        self.parent = CoreFinalRunner(self.registration["parent_core"]["workspace"], policy_factory=policy_factory,
            rollout_factory=rollout_factory, controller_loader=controller_loader, selected_loader=selected_loader)
        self.core, self.tasks, self.protocol, self.runtime = self.parent.core, self.parent.tasks, self.parent.protocol, self.parent.runtime
        self.rollout_factory, self.gpu_provider = rollout_factory, gpu_provider or gpu_identity

    def unchanged(self):
        self.parent.unchanged()
        require(validate_sweep(self.registration) == self.registration, "Budget registration changed")
        verify_execution_sources(self.registration)

    def run(self, *, wait_for_global=True, wait_seconds=36 * 3600, poll_seconds=30):
        """One invocation only. Test callers may inspect a drained actor without waiting."""
        require(0 < wait_seconds <= 36 * 3600 and 0 < poll_seconds <= 60, "Invalid administrative wait bound")
        with locked(self.directory / ".actor.lock"):
            self.unchanged()
            accepted = global_receipt(self.registration)
            if accepted is not None:
                return {"complete": True, "status": "global_already_accepted", "global_receipt": accepted, "new_physical_calls": 0}
            require(not (self.directory / "invocation.json").exists(), "Actor invocation already exists; reconcile without automatic restart")
            gpu = self.gpu_provider()
            with locked(self.workspace / "locks" / ("gpu-" + fingerprint(gpu["uuid"]) + ".lock")):
                proof = validate_launch(self.registration)
                require(proof.get("complete") is True and proof.get("sweep_fingerprint") == self.registration["fingerprint"], "Budget launch validation incomplete")
                publish_once(self.directory / "launch_validation.json", proof)
                invocation_path = self.directory / "invocation.json"
                create_once(invocation_path, {"schema": "made_budget_actor_invocation_v1", "scope": SCOPE,
                    "sweep_fingerprint": self.registration["fingerprint"], "actor": self.actor, "gpu": gpu,
                    "hostname": socket.gethostname(), "pid": os.getpid(), "started_at": time.time(),
                    "python_executable": sys.executable, "python_version": sys.version.split()[0],
                    "registered_runtime": self.runtime, "parent_execution": self.parent.execution,
                    "launch_validation": artifact(self.directory / "launch_validation.json"),
                    "automatic_physical_replay": False})
                invocation = artifact(invocation_path)
                state = {"schema": "made_budget_actor_status_v1", "actor_id": self.actor["actor_id"],
                    "sweep_fingerprint": self.registration["fingerprint"], "invocation": invocation,
                    "status": "ready", "completed_job_ids": [], "current_job_id": None}
                policy = baseline = profile = risk = graph = None
                current_method = self.actor["initial_method"]
                loaded_method = None
                job = claim = None
                try:
                    while True:
                        self.unchanged()
                        job, claim = claim_next_job(self.registration, self.actor, invocation, gpu, current_method)
                        if job is None: break
                        directory = self.workspace / "jobs" / job["job_id"]
                        with locked(directory / ".execution.lock"):
                            state.update(status="running_job", current_job_id=job["job_id"], budget=job["budget"], heartbeat_at=time.time())
                            write_json_atomic(self.directory / "status.json", state)
                            import torch
                            from .esopt import tensor_state_hash
                            if policy is None:
                                torch.set_num_threads(self.runtime["torch_cpu_threads"])
                                if torch.device(self.runtime["device"]).type == "cuda":
                                    torch.cuda.set_per_process_memory_fraction(self.runtime["cuda_memory_fraction"], device=self.runtime["device"])
                                policy = self.parent.load_policy()
                                baseline = {name: value.detach().cpu().clone() for name, value in policy.model.state_dict().items()}
                                base_hash = tensor_state_hash(baseline)
                            if loaded_method != job["method"]:
                                risk = graph = None; gc.collect()
                                if torch.device(self.runtime["device"]).type == "cuda": torch.cuda.empty_cache()
                                profile, risk, graph = self.parent.profile(policy, job, baseline, base_hash)
                                require(profile == self.registration["execution_profiles"][job["method"]], "Loaded budget profile differs from frozen G2/controller")
                                current_method = loaded_method = job["method"]
                            self._execute(policy, job, claim, profile, risk, graph)
                            state["completed_job_ids"].append(job["job_id"])
                            state.update(status="job_complete", current_job_id=None, heartbeat_at=time.time())
                            write_json_atomic(self.directory / "status.json", state)
                    # No claim is recycled. Release model memory while waiting
                    # for other actors and the independent CPU aggregator.
                    policy = baseline = profile = risk = graph = None; gc.collect()
                    if "torch" in locals() and torch.device(self.runtime["device"]).type == "cuda": torch.cuda.empty_cache()
                    deadline = time.monotonic() + wait_seconds
                    while True:
                        state.update(status="no_unclaimed_jobs_waiting_for_global_acceptance", current_job_id=None,
                                     queue_state=queue_state(self.registration), heartbeat_at=time.time())
                        write_json_atomic(self.directory / "status.json", state)
                        accepted = global_receipt(self.registration)
                        if accepted is not None:
                            state.update(status="completed", complete=True, global_receipt=accepted)
                            write_json_atomic(self.directory / "status.json", state)
                            publish_once(self.directory / "receipt.json", state)
                            return state
                        if not wait_for_global:
                            return {**state, "complete": False, "all_jobs_claimed_is_not_completion": True}
                        require(time.monotonic() < deadline, "Actor global-acceptance wait expired; no claims reclaimed")
                        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
                except BaseException as error:
                    state.update(status="failed_requires_reconciliation", complete=False,
                        error_type=type(error).__name__, error=str(error), heartbeat_at=time.time(), automatic_physical_replay=False)
                    write_json_atomic(self.directory / "status.json", state)
                    if job is not None:
                        directory = self.workspace / "jobs" / job["job_id"]
                        publish_once(directory / "failure.json", {"schema": "made_budget_job_failure_v1", "sweep_fingerprint": self.registration["fingerprint"],
                            "job_id": job["job_id"], "claim": artifact(directory / "claim.json"), "error_type": type(error).__name__,
                            "error": str(error), "requires_reconciliation": True, "automatic_physical_replay": False})
                    raise

    def _execute(self, policy, job, claim, profile, risk, graph):
        from .esopt import tensor_state_hash
        from .final_evaluation import execution_job
        from .rollouts import DiscoveryRollout, RolloutSettings
        from .training_jobs import TrainingJobCallbacks, reconstruct_scientific_evidence
        directory = self.workspace / "jobs" / job["job_id"]
        output = directory / claim["attempt_id"]
        require(not output.exists(), "Budget attempt output already exists; no replay")
        actual = execution_job(job, policy, profile)
        actual["source"], actual["group_id"] = job["source"], job["group_id"]
        stamp = policy.get_state_id()
        settings = RolloutSettings(max_generation_retries=self.protocol["risk_network"]["maximum_candidate_generations_per_decision"],
            risk_threshold=self.protocol["risk_network"]["threshold"], history_results=self.protocol["memory"]["scientific_history_per_episode"],
            recent_tools=self.protocol["memory"]["recent_tool_responses"], failure_aware_control=True)
        self.unchanged()
        (self.rollout_factory or DiscoveryRollout)(self.parent.project, policy, settings=settings, risk_model=risk, attributor=graph).run(actual, output, collection=False)
        require(policy.get_state_id() == stamp and tensor_state_hash(dict(policy.model.state_dict())) == profile["actual_model_state_hash"],
                "Budget trajectory changed the fixed policy parameters")
        evidence = reconstruct_scientific_evidence(output, actual, tasks=self.tasks, partition="final_test", expected_mace_num_workers=4,
            expected_made_budget=job["budget"], core_protocol=self.core, made_budget_sweep=self.registration)
        value = {"scope": SCOPE, "sweep_fingerprint": self.registration["fingerprint"], "parent_core_fingerprint": self.core["fingerprint"],
            "actor_id": claim["actor_id"], "claim": artifact(directory / "claim.json"), "global_study_complete": False,
            "job_id": job["job_id"], "manifest_fingerprint": job_manifest(self.registration, job)["fingerprint"],
            "model_revision": job["model_revision"], "complete": True, "episodes": read(output / "episodes.json"),
            "artifacts": TrainingJobCallbacks._inventory(output), "evidence_schema": "final_official_rpc_v1", "scientific_evidence": evidence,
            "execution_job": actual, "execution_profile": profile, "execution_profile_fingerprint": fingerprint(profile)}
        write_json_atomic(output / "result.json", value)
        self.unchanged()
        finish_job(self.registration, job, tasks=self.tasks, core=self.core)


def _budget_report(registration, records, budget, *, prior=False):
    tasks = sorted({task for task, method in records})
    require(len(tasks) == 30 and len(records) == 60 and set(records) == {(t, m) for t in tasks for m in METHODS},
            "Each budget requires 30 complete paired systems")
    for item in records.values():
        require(item["episode"]["costs"]["candidate_oracle_attempts"] == budget, "Budget report mixes evaluation horizons")
    value = {"schema": "made_single_budget_report_v1", "scope": SCOPE, "complete": True,
        "sweep_fingerprint": registration["fingerprint"], "budget": budget,
        "expected_counts": {"jobs": 60, "episodes": 60, "candidate_oracle_attempts": 60 * budget, "dft_episode_attempts": 0},
        "physical_origin": "read_only_original_B10" if prior else "new_registered_budget_evaluation",
        "paired_results": _pairs(records, tasks), "costs_by_method": _costs(records, tasks),
        "source_results": [item["result"] for _, item in sorted(records.items())],
        "global_study_complete": False, "B50_retraining_claimed": False,
        "ES_selection_budget": 10, "representation_collection_budget": 50}
    value["fingerprint"] = fingerprint(value)
    return value


def aggregate_sweep(workspace):
    """Publish each complete budget independently; global acceptance needs 180.

    Only complete orphaned envelopes can be reconciled, and only while their
    execution lock is free. Absent, active, truncated and failed partial work
    stays missing. No trajectory/generation function is called here.
    """
    registration = read_sweep(workspace)
    verify_execution_sources(registration)
    root = Path(registration["workspace"])
    reports = root / "experiments/made_budget_reports"
    with locked(reports / ".aggregate.lock"):
        validate_launch(registration)
        core = read(registration["parent_core"]["path"])
        tasks = read(Path(core["workspace"]) / "configs/benchmark_tasks.json")
        queue_state(registration)
        by_budget = {budget: {} for budget in BUDGETS}
        missing, receipts = [], []
        for job in registration["new_jobs"]:
            directory = root / "jobs" / job["job_id"]
            if not (directory / "claim.json").is_file():
                missing.append(job["job_id"]); continue
            claim = read(directory / "claim.json"); validate_claim(registration, job, claim)
            path = directory / claim["attempt_id"] / "result.json"
            if not path.is_file():
                missing.append(job["job_id"]); continue
            try:
                with locked(directory / ".execution.lock"):
                    receipt = finish_job(registration, job, tasks=tasks, core=core)
            except BlockingIOError:
                missing.append(job["job_id"]); continue
            by_budget[job["budget"]][(job["task_id"], job["method"])] = {"episode": read(path)["episodes"][0], "result": receipt["result"]}
            receipts.append(artifact(directory / "receipt.json"))
        budget_reports = {}
        for budget in BUDGETS:
            if len(by_budget[budget]) == 60:
                publish_once(reports / f"budget_{budget}.json", _budget_report(registration, by_budget[budget], budget))
                budget_reports[str(budget)] = artifact(reports / f"budget_{budget}.json")
        if missing:
            return {"complete": False, "sweep_complete": False, "sweep_fingerprint": registration["fingerprint"],
                "global_study_complete": False, "missing_new_jobs": missing, "completed_budget_reports": budget_reports,
                "expected_new_counts": dict(NEW_COUNTS)}
        imported = audit_B10_results(registration, require_complete=False)
        if not imported["complete"]:
            return {"complete": False, "sweep_complete": False, "sweep_fingerprint": registration["fingerprint"],
                "global_study_complete": False, "missing_new_jobs": [], "missing_B10_jobs": imported["missing_jobs"],
                "missing_B10_acceptance": imported.get("missing_acceptance", []),
                "completed_budget_reports": budget_reports, "expected_new_counts": dict(NEW_COUNTS)}
        require(len(imported["results"]) == 60, "Original B10 reference must contain 60 jobs")
        old = {}
        for item in imported["results"]:
            job = item["job"]
            key = (job["task_id"], job["method"])
            require(key not in old and job["budget"] == 10, "Duplicate or mislabeled B10 reference")
            verify_artifact(item["result"])
            old[key] = {"episode": item["episode"], "result": item["result"]}
        publish_once(reports / "budget_10.json", _budget_report(registration, old, 10, prior=True))
        budget_reports["10"] = artifact(reports / "budget_10.json")
        report = {"schema": "made_B10_B30_B50_report_v1", "scope": SCOPE, "complete": True, "sweep_complete": True,
            "sweep_fingerprint": registration["fingerprint"], "expected_new_counts": dict(NEW_COUNTS),
            "expected_cumulative_counts": dict(CUMULATIVE_COUNTS), "budget_reports": budget_reports,
            "new_job_receipts": receipts, "B10_source_evidence": imported["evidence_files"],
            "budgets_reported_separately": True, "cross_budget_pooled_metric_claimed": False,
            "new_training_or_ES_cost": 0, "historical_training_not_recharged": True,
            "B50_retraining_claimed": False, "original_full_study_complete": False, "global_study_complete": False}
        report["fingerprint"] = fingerprint(report)
        require(validate_sweep(registration) == registration, "Budget registration changed before publication")
        publish_once(reports / "report.json", report)
        stage = {"schema": "made_budget_global_stage_receipt_v1", "scope": SCOPE, "complete": True,
            "sweep_complete": True, "sweep_fingerprint": registration["fingerprint"], "global_study_complete": False,
            "expected_new_counts": dict(NEW_COUNTS), "expected_cumulative_counts": dict(CUMULATIVE_COUNTS),
            "report": artifact(reports / "report.json"), "budget_reports": budget_reports, "new_job_receipts": receipts}
        stage["fingerprint"] = fingerprint(stage)
        publish_once(root / "experiments/made_budget_stage_receipts/final.json", stage)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--actor-id")
    mode.add_argument("--aggregate", action="store_true")
    args = parser.parse_args(argv)
    from .immutable_hash_cache import immutable_hash_cache
    with immutable_hash_cache() as counters:
        result = aggregate_sweep(args.workspace) if args.aggregate else MadeBudgetActor(args.workspace, actor_id=args.actor_id).run()
    print(json.dumps({"result": result, "hash_cache_io": counters}, indent=2))
    return 0 if result.get("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
