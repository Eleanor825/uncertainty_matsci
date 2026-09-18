"""Execute sealed, disjoint evaluation shards without replaying the parent study.

The original four physical jobs remain read-only references. Each shard owns a
separate ledger/lock and exactly its sealed job IDs. Only aggregate_extension
can accept all six new jobs and report the cumulative five-system comparison.
No checkpoint selection, training, or runtime monkeypatching occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import gc
import json
import math
from pathlib import Path

from .accounting import file_sha256, fingerprint, write_json_atomic
from .core_final import CoreFinalLedger, CoreFinalRunner, STATES, publish_once, read
from .evaluation_extension import (COUNTS, CUMULATIVE_COUNTS, METHODS, NEW_TASKS,
    OLD_TASKS, EvaluationExtensionError, extension_jobs, read_extension, require)

SCOPE = "post_results_MADE_B10_evaluation_extension"
MANIFEST_SCHEMA = "sealed_evaluation_extension_shard_v1"
LEDGER_SCHEMA = "evaluation_extension_shard_ledger_v1"
EXTENSION_MODULES = ("evaluation_extension_runner.py", "evaluation_extension.py",
                     "accounting.py", "final_evaluation.py", "immutable_hash_cache.py")
FROZEN_RUNTIME_MODULES = tuple(sorted(str(p.relative_to(Path(__file__).resolve().parent))
    for p in Path(__file__).resolve().parent.rglob("*.py")
    if str(p.relative_to(Path(__file__).resolve().parent)) not in EXTENSION_MODULES))



def artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def shard_spec(extension, shard_id):
    matches = [s for s in extension["execution_shards"] if s["shard_id"] == shard_id]
    require(len(matches) == 1, "Shard is not in the sealed ownership matrix")
    return matches[0]


def verify_execution_sources(extension):
    """New admission code is sealed; every numerical/runtime module stays frozen."""
    package = Path(__file__).resolve().parent
    sources = {Path(a["path"]).resolve(): a["sha256"] for a in extension["source_files"]}
    for name in EXTENSION_MODULES:
        path = package / name
        require(path in sources and file_sha256(path) == sources[path],
                "Actual extension implementation is absent from its seal: " + name)
    original = Path(extension["parent_core"]["workspace"]) / "src/matdiscovery"
    actual = {str(p.relative_to(package)) for p in package.rglob("*.py")
              if str(p.relative_to(package)) not in EXTENSION_MODULES}
    expected = {str(p.relative_to(original)) for p in original.rglob("*.py")
                if str(p.relative_to(original)) not in EXTENSION_MODULES}
    require(actual == expected, "Extension changed the frozen parent Python module inventory")
    for name in sorted(expected):
        require(file_sha256(package / name) == file_sha256(original / name),
                "Extension changed the frozen parent runtime module: " + name)


def build_shard_manifest(extension, shard_id):
    shard = shard_spec(extension, shard_id)
    jobs = extension_jobs(extension, shard_id=shard_id)
    value = {"schema": MANIFEST_SCHEMA, "scope": SCOPE,
        "extension_fingerprint": extension["fingerprint"],
        "parent_core_fingerprint": extension["parent_core"]["core_fingerprint"],
        "shard": deepcopy(shard), "jobs": jobs,
        "expected_counts": {**COUNTS, "jobs": len(jobs), "episodes": len(jobs),
                            "candidate_oracle_attempts": 10 * len(jobs)},
        "all_new_expected_counts": dict(COUNTS),
        "global_study_complete": False, "accepts_only_shard_completion": True}
    value["fingerprint"] = fingerprint(value)
    return value


def validate_shard_manifest(manifest, extension):
    expected = build_shard_manifest(extension, manifest.get("shard", {}).get("shard_id"))
    require(manifest == expected, "Shard matrix, ownership, or denominator changed")


class EvaluationExtensionLedger(CoreFinalLedger):
    """Reuse claim/finish/halt transitions, with a separate exact shard schema.

    A failed attempt is never made pending. Orphaned complete envelopes can be
    verified and committed without another rollout. A partial directory remains
    quarantined. Imported parent results can never enter this physical ledger.
    """
    def __init__(self, path, manifest, *, extension, readonly=False):
        validate_shard_manifest(manifest, extension)
        self.path, self.manifest, self.readonly = Path(path), manifest, readonly
        self.extension = extension
        self.fingerprint = manifest["fingerprint"]
        if self.path.exists():
            self.data = read(self.path)
            self.validate()
        else:
            require(not readonly, "Extension shard ledger does not exist")
            self.data = {"schema": LEDGER_SCHEMA, "scope": SCOPE,
                "manifest_fingerprint": self.fingerprint,
                "extension_fingerprint": extension["fingerprint"],
                "shard_id": manifest["shard"]["shard_id"], "global_study_complete": False,
                "jobs": {j["job_id"]: {"state": "pending", "attempt_number": 0,
                    "attempt_id": None, "result_path": None, "result_sha256": None}
                         for j in manifest["jobs"]}, "events": []}
            self.save()

    def validate(self):
        require(self.data.get("schema") == LEDGER_SCHEMA and self.data.get("scope") == SCOPE
            and self.data.get("manifest_fingerprint") == self.fingerprint
            and self.data.get("extension_fingerprint") == self.extension["fingerprint"]
            and self.data.get("shard_id") == self.manifest["shard"]["shard_id"]
            and self.data.get("global_study_complete") is False,
            "Ledger belongs to another extension or shard")
        require(set(self.data.get("jobs", {})) == {j["job_id"] for j in self.manifest["jobs"]},
                "Shard ledger added or omitted a physical job")
        claims = Counter()
        attempts = {}
        for index, event in enumerate(self.data.get("events", [])):
            require(event.get("sequence") == index and event.get("job_id") in self.data["jobs"],
                    "Extension ledger event sequence changed")
            if event["event"] == "claimed":
                job = event["job_id"]
                claims[job] += 1
                attempts[job] = event["details"]["attempt_id"]
        for job, state in self.data["jobs"].items():
            require(state["state"] in STATES and type(state["attempt_number"]) is int
                    and state["attempt_number"] == claims[job] and claims[job] <= 1,
                    "Unregistered or repeated extension physical attempt")
            require((state["state"] == "pending") == (claims[job] == 0)
                    and state["attempt_id"] == attempts.get(job), "Shard claim/attempt identity changed")
            if state["state"] == "succeeded":
                require(state["result_path"] and state["result_sha256"], "Completed shard job lacks evidence")

    def completion(self):
        states = {name: sum(s["state"] == name for s in self.data["jobs"].values()) for name in sorted(STATES)}
        count = len(self.manifest["jobs"])
        return {"schema": "evaluation_extension_shard_completion_v1", "scope": SCOPE,
            "shard_id": self.manifest["shard"]["shard_id"],
            "extension_fingerprint": self.extension["fingerprint"],
            "manifest_fingerprint": self.fingerprint, "expected_jobs": count,
            "completed_jobs": states["succeeded"], "shard_complete": states["succeeded"] == count,
            "extension_complete": False, "global_study_complete": False,
            "states": states, "expected_counts": deepcopy(self.manifest["expected_counts"]),
            "all_new_expected_counts": dict(COUNTS), "ledger_fingerprint": fingerprint(self.data)}


def verify_extension_envelope(path, job, manifest, extension, *, tasks, core):
    from .final_evaluation import verify_final_envelope
    validate_shard_manifest(manifest, extension)
    require(job in manifest["jobs"], "A result belongs to another shard")
    output = read(path)
    require(output.get("scope") == SCOPE
        and output.get("extension_fingerprint") == extension["fingerprint"]
        and output.get("parent_core_fingerprint") == extension["parent_core"]["core_fingerprint"]
        and output.get("shard_id") == manifest["shard"]["shard_id"]
        and output.get("execution_site") == manifest["shard"]["execution_site"]
        and output.get("global_study_complete") is False, "Result extension/site identity changed")
    profile = extension["execution_profiles"][job["method"]]
    require(profile.get("mace_num_workers") == 4 and profile.get("orb_num_workers") == 1,
            "Extension changed the parent evaluator worker settings")
    actual_job = output["execution_job"]
    require(actual_job.get("source") == job.get("source", "registered_evaluation_extension_final"),
            "Extension result mislabeled its execution source")
    generation = profile["selected_es"]["selected_generation"] if job["method"] == "esopt_graph_risk" else 0
    require(actual_job.get("selected_generation") == generation,
            "Extension execution generation differs from the frozen selected profile")
    proof = verify_final_envelope(path, job, manifest["fingerprint"], tasks=tasks,
        expected_profile=profile, expected_mace_num_workers=4, expected_made_budget=10,
        core_protocol=core, evaluation_extension=extension)
    for episode in output["episodes"]:
        require(episode["metrics"].get("mSUN") == episode["discovery_curve"][-1][1] / 10,
                "Extension mSUN differs from the reconstructed official curve")
        if "failure_rate" in episode["metrics"]:
            require(episode["metrics"]["failure_rate"] == 1 - episode["metrics"]["mSUN"],
                    "Extension non-discovery failure_rate differs from mSUN")
    return proof


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EvaluationExtensionError("Another worker owns this extension shard/report") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _result_path(output, job, state):
    attempt = state["attempt_id"]
    require(isinstance(attempt, str) and len(attempt) == 32
            and all(c in "0123456789abcdef" for c in attempt), "Invalid attempt directory identity")
    return Path(output) / "jobs" / job["job_id"] / attempt / "result.json"


def _preflight(output, ledger):
    jobs_root = Path(output) / "jobs"
    if jobs_root.exists():
        require(jobs_root.is_dir() and not jobs_root.is_symlink(), "Aliased shard jobs root")
        for folder in jobs_root.iterdir():
            require(folder.is_dir() and not folder.is_symlink() and folder.name in ledger.data["jobs"],
                    "Unknown or aliased extension job directory")
            state = ledger.data["jobs"][folder.name]
            require(state["state"] != "pending"
                and {p.name for p in folder.iterdir()} == {state["attempt_id"]}
                and not (folder / state["attempt_id"]).is_symlink(),
                "Unregistered or duplicated extension physical attempt")
    for job in ledger.manifest["jobs"]:
        state = ledger.data["jobs"][job["job_id"]]
        require(state["state"] != "failed", "Failed extension job requires reconciliation; never replay")
        if state["state"] in {"orphaned", "succeeded"}:
            require(_result_path(output, job, state).is_file(),
                    "Unknown/partial extension outcome requires reconciliation; never replay")


def _verify_finished(output, ledger, extension, *, tasks, core):
    for job in ledger.manifest["jobs"]:
        state = ledger.data["jobs"][job["job_id"]]
        if state["state"] not in {"orphaned", "succeeded"}:
            continue
        path = _result_path(output, job, state)
        proof = verify_extension_envelope(path, job, ledger.manifest, extension, tasks=tasks, core=core)
        if state["state"] == "succeeded":
            require(state["result_path"] == str(path.resolve()) and state["result_sha256"] == proof["result_sha256"],
                    "Committed extension result changed")
        else:
            ledger.finish(job["job_id"], path, proof["result_sha256"], reconciled=True)


class EvaluationExtensionRunner:
    def __init__(self, workspace, *, shard="primary", output=None, policy_factory=None,
                 rollout_factory=None, controller_loader=None, selected_loader=None):
        self.extension = read_extension(workspace)
        self.workspace = Path(self.extension["workspace"])
        self.shard = shard_spec(self.extension, shard)
        self.output = Path(self.shard["output_workspace"]).resolve()
        require(output is None or Path(output).resolve() == self.output,
                "Output override differs from sealed shard ownership")
        require(not Path(self.shard["output_workspace"]).is_symlink(), "Shard output cannot be aliased")
        verify_execution_sources(self.extension)
        self.parent = CoreFinalRunner(self.extension["parent_core"]["workspace"],
            policy_factory=policy_factory, rollout_factory=rollout_factory,
            controller_loader=controller_loader, selected_loader=selected_loader)
        self.core, self.tasks, self.protocol, self.runtime = (
            self.parent.core, self.parent.tasks, self.parent.protocol, self.parent.runtime)
        require(self.core["fingerprint"] == self.extension["parent_core"]["core_fingerprint"],
                "Parent runner loaded another policy protocol")
        self.manifest = build_shard_manifest(self.extension, shard)
        self.registration = artifact(self.workspace / "configs/evaluation_extension.json")
        self.execution = {"scope": SCOPE, "extension_fingerprint": self.extension["fingerprint"],
            "registration": self.registration, "shard": deepcopy(self.shard),
            "parent_execution": self.parent.execution, "source_files": self.extension["source_files"]}
        self.rollout_factory = rollout_factory

    def unchanged(self):
        self.parent.unchanged()
        require(artifact(self.registration["path"]) == self.registration,
                "Extension registration changed during execution")
        for item in self.extension["source_files"] + self.extension["scientific_parent_inputs"]:
            require(artifact(item["path"]) == item, "Extension scientific/source input changed")
        verify_execution_sources(self.extension)

    def _publish(self, ledger):
        result = ledger.completion()
        require(result["shard_complete"], "Incomplete shard cannot publish completion")
        self.unchanged()
        result.update(complete=True, result_files=[{"path": ledger.data["jobs"][j["job_id"]]["result_path"],
            "sha256": ledger.data["jobs"][j["job_id"]]["result_sha256"]} for j in self.manifest["jobs"]])
        publish_once(self.output / "completion.json", result)
        stage = {"schema": "evaluation_extension_shard_stage_receipt_v1", "complete": True,
            "scope": SCOPE, "extension_fingerprint": self.extension["fingerprint"],
            "shard_id": self.shard["shard_id"], "execution_site": self.shard["execution_site"],
            "shard_complete": True, "extension_complete": False, "global_study_complete": False,
            "artifacts": [artifact(self.output / name) for name in
                          ("manifest.json", "execution_inputs.json", "ledger.json", "completion.json")]}
        stage["fingerprint"] = fingerprint(stage)
        publish_once(self.output / "stage_receipt.json", stage)
        return result

    def run(self):
        # Recovery checks and completed reuse happen before importing/loading a policy.
        with exclusive_lock(self.output / ".runner.lock"):
            self.unchanged()
            for name, value in (("manifest.json", self.manifest), ("execution_inputs.json", self.execution)):
                publish_once(self.output / name, value)
            jobs_root = self.output / "jobs"
            require((self.output / "ledger.json").exists() or not jobs_root.exists() or not any(jobs_root.iterdir()),
                    "Untracked physical output exists without an extension ledger")
            ledger = EvaluationExtensionLedger(self.output / "ledger.json", self.manifest, extension=self.extension)
            if any(s["state"] == "running" for s in ledger.data["jobs"].values()):
                ledger.quarantine_running()
            _preflight(self.output, ledger)
            _verify_finished(self.output, ledger, self.extension, tasks=self.tasks, core=self.core)
            if ledger.completion()["shard_complete"]:
                return self._publish(ledger)
            import torch
            from .esopt import tensor_state_hash
            from .final_evaluation import execution_job
            from .rollouts import DiscoveryRollout, RolloutSettings
            from .training_jobs import TrainingJobCallbacks, reconstruct_scientific_evidence, _rpc_pairs
            torch.set_num_threads(self.runtime["torch_cpu_threads"])
            if torch.device(self.runtime["device"]).type == "cuda":
                torch.cuda.set_per_process_memory_fraction(self.runtime["cuda_memory_fraction"], device=self.runtime["device"])
            policy = self.parent.load_policy()
            baseline = {n: v.detach().cpu().clone() for n, v in policy.model.state_dict().items()}
            base_hash = tensor_state_hash(baseline)
            current_method = None
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
                    require(profile == self.extension["execution_profiles"][job["method"]],
                            "Loaded extension profile differs from the original frozen arm")
                    current_method = job["method"]
                self.unchanged()
                attempt = ledger.claim(job["job_id"])
                output = self.output / "jobs" / job["job_id"] / attempt
                require(not output.exists(), "Extension physical attempt output already exists")
                actual = execution_job(job, policy, profile)
                actual["source"] = job.get("source", "registered_evaluation_extension_final")
                actual["group_id"] = job.get("group_id", actual["group_id"])
                stamp = policy.get_state_id()
                settings = RolloutSettings(
                    max_generation_retries=self.protocol["risk_network"]["maximum_candidate_generations_per_decision"],
                    risk_threshold=self.protocol["risk_network"]["threshold"],
                    history_results=self.protocol["memory"]["scientific_history_per_episode"],
                    recent_tools=self.protocol["memory"]["recent_tool_responses"], failure_aware_control=True)
                try:
                    factory(self.parent.project, policy, settings=settings, risk_model=risk, attributor=graph).run(actual, output, collection=False)
                    require(policy.get_state_id() == stamp
                        and tensor_state_hash(dict(policy.model.state_dict())) == profile["actual_model_state_hash"],
                        "Extension trajectory changed the fixed policy weights")
                    evidence = reconstruct_scientific_evidence(output, actual, tasks=self.tasks,
                        partition="final_test", expected_mace_num_workers=4,
                        expected_made_budget=10, core_protocol=self.core)
                    value = {"scope": SCOPE, "extension_fingerprint": self.extension["fingerprint"],
                        "parent_core_fingerprint": self.core["fingerprint"], "shard_id": self.shard["shard_id"],
                        "execution_site": self.shard["execution_site"], "global_study_complete": False,
                        "job_id": job["job_id"], "manifest_fingerprint": self.manifest["fingerprint"],
                        "model_revision": job["model_revision"], "complete": True,
                        "episodes": read(output / "episodes.json"), "artifacts": TrainingJobCallbacks._inventory(output),
                        "evidence_schema": "final_official_rpc_v1", "scientific_evidence": evidence,
                        "execution_job": actual, "execution_profile": profile,
                        "execution_profile_fingerprint": fingerprint(profile)}
                    path = output / "result.json"
                    write_json_atomic(path, value)
                    self.unchanged()
                    proof = verify_extension_envelope(path, job, self.manifest, self.extension, tasks=self.tasks, core=self.core)
                    ledger.finish(job["job_id"], path, proof["result_sha256"])
                except BaseException as exc:
                    closed = False
                    try:
                        _rpc_pairs(output / "rpc/rpc.jsonl")
                        closed = not (output / "result.json").exists()
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                    ledger.halt(job["job_id"], exc, closed=closed,
                        evidence_files=[p for p in output.rglob("*") if p.is_file()] if output.exists() else ())
                    raise
                write_json_atomic(self.output / "progress.json", ledger.completion())
            _verify_finished(self.output, ledger, self.extension, tasks=self.tasks, core=self.core)
            return self._publish(ledger)


def _pairs(records, task_ids):
    rows = []
    for task in task_ids:
        values = {method: records[(task, method)]["episode"]["metrics"] for method in METHODS}
        row = {"task_id": task, "methods": {m: {k: values[m].get(k) for k in ("AUDC", "mSUN")} for m in METHODS}}
        row["full_minus_baseline"] = {k: values[METHODS[1]][k] - values[METHODS[0]][k] for k in ("AUDC", "mSUN")}
        rows.append(row)
    return {"paired_systems": len(rows), "pairs": rows,
        "mean_full_minus_baseline": {k: sum(r["full_minus_baseline"][k] for r in rows) / len(rows) for k in ("AUDC", "mSUN")},
        "population_significance_claim": False, "inferential_p_value": None,
        "importance_or_causal_component_claim": False}


def _costs(records, tasks):
    costs = {method: Counter() for method in METHODS}
    for (task, method), record in records.items():
        if task in tasks:
            costs[method].update(record["episode"]["costs"])
    return {method: dict(value) for method, value in costs.items()}


def aggregate_extension(workspace):
    """CPU-only all-six acceptance; four parent episodes are referenced, not rerun.

    This never loads policy weights or creates a new physical ledger. An absent
    shard returns pending and publishes no acceptance. A present invalid/partial
    completion is rejected. Shared training costs are referenced but not added
    a second time to the final-evaluation cost totals.
    """
    from .core_protocol import read_core
    extension = read_extension(workspace)
    verify_execution_sources(extension)
    root = Path(extension["workspace"])
    reports = root / "experiments/evaluation_extension_reports"
    with exclusive_lock(reports / ".aggregate.lock"):
        missing = [s["shard_id"] for s in extension["execution_shards"]
                   if not (Path(s["output_workspace"]) / "stage_receipt.json").is_file()]
        if missing:
            return {"complete": False, "extension_complete": False, "global_study_complete": False,
                "extension_fingerprint": extension["fingerprint"], "missing_shards": missing,
                "expected_new_counts": dict(COUNTS)}
        core = read_core(extension["parent_core"]["workspace"])
        tasks = read(Path(core["workspace"]) / "configs/benchmark_tasks.json")
        records, evidence, new_ids = {}, [], set()
        for shard in extension["execution_shards"]:
            output = Path(shard["output_workspace"])
            with exclusive_lock(output / ".runner.lock"):
                manifest = read(output / "manifest.json")
                validate_shard_manifest(manifest, extension)
                ledger = EvaluationExtensionLedger(output / "ledger.json", manifest, extension=extension, readonly=True)
                _preflight(output, ledger)
                require(ledger.completion()["shard_complete"], "Shard completion has incomplete ledger jobs")
                stage = read(output / "stage_receipt.json")
                require(stage.get("schema") == "evaluation_extension_shard_stage_receipt_v1"
                    and stage.get("scope") == SCOPE
                    and stage.get("fingerprint") == fingerprint({k: v for k, v in stage.items() if k != "fingerprint"})
                    and stage.get("complete") is True and stage.get("shard_complete") is True
                    and stage.get("extension_complete") is False and stage.get("global_study_complete") is False
                    and stage.get("extension_fingerprint") == extension["fingerprint"]
                    and stage.get("shard_id") == shard["shard_id"] and stage.get("execution_site") == shard["execution_site"]
                    and stage.get("artifacts") == [artifact(output / name) for name in
                        ("manifest.json", "execution_inputs.json", "ledger.json", "completion.json")],
                    "Invalid or changed shard stage receipt")
                expected_inputs = {"scope": SCOPE, "extension_fingerprint": extension["fingerprint"],
                    "registration": artifact(root / "configs/evaluation_extension.json"),
                    "shard": shard,
                    "parent_execution": read(extension["parent_final"]["execution_inputs"]["path"]),
                    "source_files": extension["source_files"]}
                require(read(output / "execution_inputs.json") == expected_inputs,
                        "Shard execution source binding differs from the parent/extension seal")
                expected_completion = ledger.completion()
                expected_completion.update(complete=True, result_files=[{"path": ledger.data["jobs"][j["job_id"]]["result_path"],
                    "sha256": ledger.data["jobs"][j["job_id"]]["result_sha256"]} for j in manifest["jobs"]])
                require(read(output / "completion.json") == expected_completion, "Shard completion inventory changed")
                for job in manifest["jobs"]:
                    state = ledger.data["jobs"][job["job_id"]]
                    path = _result_path(output, job, state)
                    proof = verify_extension_envelope(path, job, manifest, extension, tasks=tasks, core=core)
                    require(state["result_path"] == str(path.resolve()) and state["result_sha256"] == proof["result_sha256"],
                            "Aggregated result differs from the committed shard ledger")
                    require(job["job_id"] not in new_ids, "A physical job appears in multiple shards")
                    new_ids.add(job["job_id"])
                    records[(job["task_id"], job["method"])] = {"episode": read(path)["episodes"][0], "result": artifact(path)}
                    evidence.append(artifact(path))
                evidence.append(artifact(output / "stage_receipt.json"))
        require(new_ids == {j["job_id"] for j in extension["new_jobs"]} and len(new_ids) == 6,
                "Cumulative acceptance requires all six unique new jobs")
        # read_extension already reverified the immutable four parent envelopes.
        for item in extension["imported_results"]:
            job, reference = item["job"], item["result"]
            require(artifact(reference["path"]) == reference, "Imported result changed during aggregation")
            key = (job["task_id"], job["method"])
            require(key not in records, "Imported result was duplicated into a physical extension job")
            records[key] = {"episode": read(reference["path"])["episodes"][0], "result": reference}
            evidence.append(reference)
        require(len(records) == 10, "Five paired systems require ten distinct episodes")
        report = {"schema": "cumulative_five_MADE_extension_report_v1", "scope": SCOPE,
            "complete": True, "extension_complete": True, "global_study_complete": False,
            "original_full_study_complete": False, "original_b50_core_complete": False,
            "extension_fingerprint": extension["fingerprint"], "registration": artifact(root / "configs/evaluation_extension.json"),
            "expected_new_counts": dict(COUNTS), "expected_cumulative_counts": dict(CUMULATIVE_COUNTS),
            "initial_two_known_before_expansion": _pairs(records, OLD_TASKS),
            "new_three_out_of_sample_after_expansion": _pairs(records, NEW_TASKS),
            "cumulative_five_descriptive": _pairs(records, (*OLD_TASKS, *NEW_TASKS)),
            "prospective_five_system_registration_claimed": False,
            "costs": {"scope": "final_evaluation_only_not_total_research_cost",
                "original_four_by_method": _costs(records, OLD_TASKS),
                "new_six_by_method": _costs(records, NEW_TASKS),
                "cumulative_ten_by_method": _costs(records, (*OLD_TASKS, *NEW_TASKS)),
                "additional_training_or_ES_cost": 0,
                "prior_training_not_recharged": True},
            "source_results_and_shards": evidence}
        report["fingerprint"] = fingerprint(report)
        require(read_extension(workspace) == extension, "Extension or parent evidence changed during aggregation")
        publish_once(reports / "report.json", report)
        receipt = {"schema": "evaluation_extension_global_stage_receipt_v1", "complete": True,
            "scope": SCOPE, "extension_fingerprint": extension["fingerprint"],
            "extension_complete": True, "global_study_complete": False,
            "expected_new_counts": dict(COUNTS), "expected_cumulative_counts": dict(CUMULATIVE_COUNTS),
            "report": artifact(reports / "report.json"), "source_results_and_shards": evidence}
        receipt["fingerprint"] = fingerprint(receipt)
        publish_once(root / "experiments/evaluation_extension_stage_receipts/final.json", receipt)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--shard", default=None)
    mode.add_argument("--aggregate", action="store_true")
    args = parser.parse_args(argv)
    from .immutable_hash_cache import immutable_hash_cache
    with immutable_hash_cache() as cache:
        result = (aggregate_extension(args.workspace) if args.aggregate else
                  EvaluationExtensionRunner(args.workspace, shard=args.shard or "primary").run())
    print(json.dumps({"result": result, "hash_cache_io": cache}, indent=2))
    return 0 if result.get("complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
