"""Explicit all-30 MADE B10 coverage, extending the unchanged five-system study.

The fifty new jobs never replay the old ten. Old results may still be pending at
registration; their immutable jobs/manifests are declared, not fabricated result
hashes. Deep ancestor verification is a launch prerequisite once per process.
Every subsequent scope check still verifies sealed inputs and exact job identity.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import re
from typing import Mapping

from .accounting import file_sha256, fingerprint, write_json_atomic
from .collection_provenance import read_json

SCHEMA = "made_all30_B10_extension_v1"
PARENT_CORE_FINGERPRINT = "725a5febe58990defc7a681dce24b503bbf605016f410d089ee36a128210b154"
PRIOR_EXTENSION_FINGERPRINT = "2e1c40d56bbde23fe1c4a42b1afda84dd7fb5ba4d1879723a90a24636219f274"
METHODS = ("baseline", "esopt_graph_risk")
OLD_TASKS = ("Al-Li-V", "Al-V-Zn", "Au-K-Tb", "Co-Dy-W", "Co-Mg-Na")
SOURCE = "registered_MADE_all30_B10_final_evaluation"
COUNTS = {"jobs": 50, "episodes": 50, "candidate_oracle_attempts": 500, "dft_episode_attempts": 0}
IMPORTED_COUNTS = {"jobs": 10, "episodes": 10, "candidate_oracle_attempts": 100, "dft_episode_attempts": 0}
CUMULATIVE_COUNTS = {"jobs": 60, "episodes": 60, "candidate_oracle_attempts": 600, "dft_episode_attempts": 0}
SHARED_ADMISSION_MODULES = ("accounting.py", "final_evaluation.py", "training_jobs.py")
NEW_MODULES = ("made_all30_extension.py", "made_all30_runner.py")
REGISTRATION = {
    "schema": "registered_all30_MADE_B10_remaining25_v1", "study_id": "made_all30_B10_extension_v1",
    "scope": "all30_MADE_B10_seed1_two_methods_not_original_B50_study",
    "model_key": "qwen35_4b", "methods": list(METHODS), "seed": 1, "execution_budget": 10,
    "selected_full_generation": 2,
    "official_test_systems": 30, "new_systems": 25, "imported_systems": 5,
    "selection_rule": "all_official_test_minus_exact_prior5_ordered_by_element_count_then_canonical_id",
    "registered_after_partial_prior5_results_known": True,
    "prior5_completion_required_before_new_launch": False, "prior10_verified_results_required_for_final_acceptance": True,
    "new_training_or_es": False, "checkpoint_reselection": False, "test_outcome_tuning": False,
    "original_B50_study_complete": False, "original_full_study_complete": False,
}
_LIVE_ADMISSIONS = {}  # Process-local capability; never reconstructed from JSON.


class MadeAll30Error(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise MadeAll30Error(message)


def artifact(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), "Admission evidence is not a file")
    return {"path": str(path), "sha256": file_sha256(path)}


def verify_artifact(value):
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"}, "Malformed all30 source artifact")
    require(artifact(value["path"]) == value, "All30 source artifact changed: " + str(value.get("path")))
    return dict(value)


def _reference(value, *, kind):
    name = "deadline_core_protocol.json" if kind == "parent" else "evaluation_extension.json"
    key = "core_fingerprint" if kind == "parent" else "fingerprint"
    if isinstance(value, Mapping):
        root = Path(value["workspace"]).resolve()
    else:
        root = Path(value).resolve()
        if root.is_file(): root = root.parent.parent
    path = root / "configs" / name
    data = read_json(path)
    if isinstance(value, Mapping): require(data == value, "Caller source differs from its sealed document")
    return {**artifact(path), "workspace": str(root), key: data["fingerprint"]}


def _sources(parent_reference, prior_reference):
    """Hash-bound JSON admission only; deliberately no recursive read_core here."""
    from .core_protocol import FAST_CAUSAL_GRAPH_REGISTRATION
    from .evaluation_extension import REGISTRATION as PRIOR_REGISTRATION
    values = []
    for ref, filename, fpkey, expected in (
        (parent_reference, "deadline_core_protocol.json", "core_fingerprint", PARENT_CORE_FINGERPRINT),
        (prior_reference, "evaluation_extension.json", "fingerprint", PRIOR_EXTENSION_FINGERPRINT)):
        require(set(ref) == {"path", "sha256", "workspace", fpkey}, "Malformed pinned source reference")
        root = Path(ref["workspace"]).resolve()
        require(ref["workspace"] == str(root) and ref["path"] == str(root / "configs" / filename), "Pinned source path differs")
        verify_artifact({k: ref[k] for k in ("path", "sha256")})
        data = read_json(ref["path"])
        excluded = {"fingerprint", "core_fingerprint"} if fpkey == "core_fingerprint" else {"fingerprint", "extension_fingerprint"}
        require(data["fingerprint"] == ref[fpkey] == expected
                and data["fingerprint"] == fingerprint({k: v for k, v in data.items() if k not in excluded}),
                "All30 requires the exact original FAST parent and five-system extension")
        alias = "core_fingerprint" if fpkey == "core_fingerprint" else "extension_fingerprint"
        require(data.get(alias, data["fingerprint"]) == data["fingerprint"], "Source fingerprint aliases differ")
        require(data["workspace"] == str(root), "Source workspace changed")
        values.append(data)
    parent, prior = values
    require(parent["registration"] == FAST_CAUSAL_GRAPH_REGISTRATION and parent["budget"] == 50
            and parent["execution_budget"] == 10 and parent["methods"] == list(METHODS), "Original FAST contract changed")
    require(prior["registration"] == PRIOR_REGISTRATION and prior["parent_core"] == parent_reference
            and prior["execution_budget"] == 10 and prior["expected_cumulative_counts"] == IMPORTED_COUNTS,
            "Original five-system source is not bound to the pinned parent")
    require(prior["execution_profiles"]["esopt_graph_risk"]["selected_es"]["selected_generation"] == 2,
            "All30 must retain the originally selected G2 checkpoint")
    for item in prior["scientific_parent_inputs"]:
        verify_artifact(item)
    return parent, prior


def _tasks(parent):
    root = Path(parent["workspace"])
    tasks = read_json(root / "configs/benchmark_tasks.json")["made"]["systems"]
    split = read_json(root / "configs/made_splits.json")["splits"]
    def task(elements):
        require(isinstance(elements, list) and elements and len(set(elements)) == len(elements), "Invalid system elements")
        values = sorted(elements)
        return {"id": "-".join(values), "elements": values}
    official = sorted((task(t["elements"]) for t in tasks), key=lambda t: (len(t["elements"]), t["id"]))
    ids = {t["id"] for t in official}
    require(len(official) == len(ids) == 30 and Counter(len(t["elements"]) for t in official) == {3: 10, 4: 10, 5: 10},
            "All30 requires the complete official system strata")
    test = [task(elements)["id"] for elements in split["test"]]
    require(len(test) == len(set(test)) == 30 and set(test) == ids, "Official30 differs from the frozen test split")
    require(not ids & {task(v)["id"] for s in ("train", "dev") for v in split[s]}, "All30 includes train/development chemistry")
    require(tuple(t["id"] for t in official[:5]) == OLD_TASKS, "Prior-five fixed system set changed")
    return official


def _jobs(parent, prior, tasks):
    require(prior["cumulative_tasks"] == tasks[:5] and len(prior["cumulative_jobs"]) == 10, "Prior five matrix changed")
    templates = {job["method"]: job for job in prior["new_jobs"]}
    result = []
    for method in METHODS:
        for task in tasks[5:]:
            job = deepcopy(templates[method])
            for key in ("job_id", "evaluation_extension_registration"):
                job.pop(key, None)
            job.update(study_id=REGISTRATION["study_id"], classification="registered_all30_B10_extension",
                source=SOURCE, task_id=task["id"], group_id=task["id"], task=deepcopy(task),
                prior_extension_fingerprint=prior["fingerprint"], made_all30_registration=REGISTRATION["schema"])
            job["job_id"] = "all30-final-made-" + fingerprint(job)[:24]
            result.append(job)
    return result


def planned_jobs(parent_core, prior_extension):
    parent, prior = _sources(_reference(parent_core, kind="parent"), _reference(prior_extension, kind="prior"))
    return _jobs(parent, prior, _tasks(parent))


def _dependencies(prior):
    result = []
    for item in prior["imported_results"]:
        result.append({"origin": "parent_core", "job": item["job"], "result": item["result"],
            "manifest": prior["parent_final"]["manifest"], "ledger_path": prior["parent_final"]["ledger"]["path"]})
    for job in prior["new_jobs"]:
        shard = next(s for s in prior["execution_shards"] if job["job_id"] in s["job_ids"])
        root = Path(shard["output_workspace"])
        manifest = read_json(root / "manifest.json")
        require(manifest["extension_fingerprint"] == prior["fingerprint"] and manifest["shard"] == shard
                and manifest["fingerprint"] == fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
                and job in manifest["jobs"], "Prior shard manifest is absent or inconsistent")
        result.append({"origin": "prior_five_extension", "job": job, "result": None,
            "manifest": artifact(root / "manifest.json"), "ledger_path": str(root / "ledger.json")})
    require(len(result) == len({x["job"]["job_id"] for x in result}) == 10, "Prior ten dependencies omitted/duplicated")
    return result


def _shards(value, workspace, jobs):
    from .evaluation_extension import _shards as existing_shards
    if value is None:
        value = [{"worker_id": "primary", "execution_site": "primary", "output_workspace": str(workspace / "shards/primary"),
                  "job_ids": [j["job_id"] for j in jobs]}]
    require(isinstance(value, list) and value, "No static all30 workers")
    converted = []
    for item in value:
        require(set(item) == {"worker_id", "execution_site", "output_workspace", "job_ids"}, "Invalid worker assignment")
        converted.append({"shard_id": item["worker_id"], **{k: v for k, v in item.items() if k != "worker_id"}})
    checked = existing_shards(converted, workspace, jobs)
    return [{"worker_id": item["shard_id"], **{k: v for k, v in item.items() if k != "shard_id"}} for item in checked]


def _payload(parent_ref, prior_ref, workspace, sources, validation, workers, registered_at):
    parent, prior = _sources(parent_ref, prior_ref)
    workspace = Path(workspace).resolve()
    for original in (Path(parent["workspace"]), Path(prior["workspace"])):
        require(workspace != original and original not in workspace.parents and workspace not in original.parents,
                "All30 must use a separate workspace without rewriting ancestors")
    from .evaluation_extension import _source_files
    sources = _source_files(sources)
    if validation is not None:
        verify_artifact(validation)
        require(read_json(validation["path"]).get("passed") is True, "All30 implementation validation has not passed")
    tasks = _tasks(parent); jobs = _jobs(parent, prior, tasks)
    return {"schema": SCHEMA, "registration": deepcopy(REGISTRATION), "workspace": str(workspace),
        "registered_at": registered_at, "parent_core": parent_ref, "prior_extension": prior_ref,
        "source_files": sources, "validation": validation, "execution_profiles": prior["execution_profiles"],
        "official_tasks": tasks, "imported_tasks": tasks[:5], "new_tasks": tasks[5:],
        "new_jobs": jobs, "imported_dependencies": _dependencies(prior),
        "execution_shards": _shards(workers, workspace, jobs), "expected_new_counts": dict(COUNTS),
        "expected_imported_counts": dict(IMPORTED_COUNTS), "expected_cumulative_counts": dict(CUMULATIVE_COUNTS),
        "execution_budget": 10, "training_jobs": [], "es_jobs": [], "global_study_complete": False,
        "original_b50_core_complete": False, "scientific_improvement_assumed": False,
        "launch_deep_validation_required_per_process": True}


def build_all30(parent_core, prior_extension, workspace, *, source_files=None, validation=None, execution_shards=None):
    """Publish a fixed fifty-job extension; pending old results remain dependencies."""
    from .evaluation_extension import _source_files
    parent_ref, prior_ref = _reference(parent_core, kind="parent"), _reference(prior_extension, kind="prior")
    sources = _source_files(source_files if source_files is not None else [Path(__file__).with_name(n)
        for n in (*SHARED_ADMISSION_MODULES, *NEW_MODULES)])
    validation = (verify_artifact(validation) if isinstance(validation, Mapping) else artifact(validation)) if validation is not None else None
    target = Path(workspace).resolve() / "configs/made_all30_extension.json"
    if target.exists():
        current = read_all30(target)
        expected = _payload(parent_ref, prior_ref, workspace, sources, validation, execution_shards, current["registered_at"])
        require({k: v for k, v in current.items() if k not in {"fingerprint", "all30_fingerprint"}} == expected,
                "Existing all30 registration cannot be replaced")
        return current
    value = _payload(parent_ref, prior_ref, workspace, sources, validation, execution_shards, datetime.now(timezone.utc).isoformat())
    value["fingerprint"] = value["all30_fingerprint"] = fingerprint(value)
    write_json_atomic(target, value)
    return value


def validate_all30(value):
    require(isinstance(value, Mapping), "All30 registration must be a sealed object")
    payload = {k: v for k, v in value.items() if k not in {"fingerprint", "all30_fingerprint"}}
    require(value.get("fingerprint") == value.get("all30_fingerprint") == fingerprint(payload), "All30 fingerprint changed")
    path = Path(value["workspace"]) / "configs/made_all30_extension.json"
    require(read_json(path) == value, "Caller all30 object differs from its sealed file")
    expected = _payload(value["parent_core"], value["prior_extension"], value["workspace"], value["source_files"],
                        value["validation"], value["execution_shards"], value["registered_at"])
    require(payload == expected, "All30 scope, task matrix, profiles, or source dependencies changed")
    return deepcopy(dict(value))


def read_all30(workspace_or_path):
    path = Path(workspace_or_path).resolve()
    if path.is_dir(): path = path / "configs/made_all30_extension.json"
    value = read_json(path)
    require(path == Path(value["workspace"]) / "configs/made_all30_extension.json", "All30 registration is in another workspace")
    return validate_all30(value)


def all30_jobs(value, *, worker_id=None):
    value = validate_all30(value)
    if worker_id is None: return value["new_jobs"]
    matches = [w for w in value["execution_shards"] if w["worker_id"] == worker_id]
    require(len(matches) == 1, "Unregistered all30 worker")
    return [j for j in value["new_jobs"] if j["job_id"] in matches[0]["job_ids"]]


def verify_runtime_sources(value):
    package = Path(__file__).resolve().parent
    original = Path(value["prior_extension"]["workspace"]) / "src/matdiscovery"
    exceptional = set(SHARED_ADMISSION_MODULES + NEW_MODULES)
    sources = {Path(a["path"]): a["sha256"] for a in value["source_files"]}
    for name in exceptional:
        path = package / name
        require(path in sources and file_sha256(path) == sources[path], "Unsealed executing all30 source: " + name)
    actual = {str(p.relative_to(package)) for p in package.rglob("*.py") if str(p.relative_to(package)) not in exceptional}
    expected = {str(p.relative_to(original)) for p in original.rglob("*.py") if str(p.relative_to(original)) not in exceptional}
    require(actual == expected, "All30 changed the frozen Python inventory")
    for name in expected:
        require(file_sha256(package / name) == file_sha256(original / name), "All30 changed frozen scientific/runtime code: " + name)


def validate_launch(value):
    """Deeply verify actual ancestors once, then grant a PID-local scope capability."""
    from .core_protocol import read_core
    from .evaluation_extension import read_extension
    value = validate_all30(value)
    key = (os.getpid(), value["fingerprint"])
    verify_runtime_sources(value)
    if key in _LIVE_ADMISSIONS:
        return deepcopy(_LIVE_ADMISSIONS[key]["proof"])
    parent, prior = _sources(value["parent_core"], value["prior_extension"])
    require(read_core(parent["workspace"]) == parent and read_extension(prior["workspace"]) == prior,
            "All30 ancestor deep admission differs from the pinned sources")
    proof = {"schema": "process_verified_all30_launch_v1", "complete": True,
        "all30_fingerprint": value["fingerprint"], "parent_core_fingerprint": parent["fingerprint"],
        "prior_extension_fingerprint": prior["fingerprint"], "prior_pending_results_not_treated_complete": True,
        "evidence_files": [{k: value[name][k] for k in ("path", "sha256")} for name in ("parent_core", "prior_extension")]
            + value["source_files"] + [d["manifest"] for d in value["imported_dependencies"]],
        "new_scientific_calls": 0}
    proof["fingerprint"] = fingerprint(proof)
    _LIVE_ADMISSIONS[key] = {"parent": deepcopy(parent), "prior": deepcopy(prior), "proof": deepcopy(proof)}
    return proof


def admitted_parent(value):
    value = validate_all30(value)
    entry = _LIVE_ADMISSIONS.get((os.getpid(), value["fingerprint"]))
    require(entry is not None, "All30 requires successful validate_launch in this process; JSON flags cannot authorize it")
    verify_runtime_sources(value)
    parent, prior = _sources(value["parent_core"], value["prior_extension"])
    require(parent == entry["parent"] and prior == entry["prior"], "Admitted ancestor changed")
    return parent


def validate_job_scope(value, job, *, expected_made_budget=10, core_protocol=None, execution=False):
    parent = admitted_parent(value)
    require(type(expected_made_budget) is int and expected_made_budget == 10, "All30 requires the registered B10 budget")
    require(core_protocol is None or core_protocol == parent, "All30 caller parent differs")
    matches = [j for j in value["new_jobs"] if j["job_id"] == job.get("job_id")]
    require(len(matches) == 1, "Job is outside the fifty new all30 jobs")
    expected = matches[0]
    require(all(job.get(k) == v for k, v in expected.items()) if execution else job == expected,
            "All30 job identity, chemistry, seed, or budget changed")
    if execution:
        profile = value["execution_profiles"][job["method"]]
        generation = profile["selected_es"]["selected_generation"] if job["method"] == METHODS[1] else 0
        require(job.get("actual_model_state_hash") == profile["actual_model_state_hash"]
                and job.get("selected_generation") == generation and job.get("execution_profile_fingerprint") == fingerprint(profile),
                "All30 executed job changed its original checkpoint/profile")
    return parent


def audit_imported_results(value, *, require_complete=True):
    """Only final aggregation resolves old pending results; no old job is replayed."""
    from .core_final import CoreFinalLedger, verify_core_envelope
    from .evaluation_extension_runner import EvaluationExtensionLedger, verify_extension_envelope
    parent = admitted_parent(value)
    prior = _sources(value["parent_core"], value["prior_extension"])[1]
    tasks = read_json(Path(parent["workspace"]) / "configs/benchmark_tasks.json")
    results, missing, evidence = [], [], []
    for dependency in value["imported_dependencies"]:
        manifest_path = Path(dependency["manifest"]["path"]); verify_artifact(dependency["manifest"])
        manifest = read_json(manifest_path); ledger_path = Path(dependency["ledger_path"])
        if not ledger_path.exists():
            missing.append(dependency["job"]["job_id"]); continue
        ledger = (CoreFinalLedger(ledger_path, manifest, readonly=True) if dependency["origin"] == "parent_core" else
                  EvaluationExtensionLedger(ledger_path, manifest, extension=prior, readonly=True))
        job = dependency["job"]; state = ledger.data["jobs"][job["job_id"]]
        if state["state"] != "succeeded":
            missing.append(job["job_id"]); continue
        path = Path(state["result_path"]).resolve()
        require(path == ledger_path.parent / "jobs" / job["job_id"] / state["attempt_id"] / "result.json",
                "Imported result is outside its original owner")
        result = artifact(path)
        require(result["sha256"] == state["result_sha256"] and (dependency["result"] is None or dependency["result"] == result),
                "Imported result hash differs from original committed identity")
        checked = (verify_core_envelope(path, job, manifest, tasks=tasks) if dependency["origin"] == "parent_core" else
                   verify_extension_envelope(path, job, manifest, prior, tasks=tasks, core=parent))
        require(checked["result_sha256"] == result["sha256"], "Imported result changed during verification")
        envelope = read_json(path)
        results.append({"job": job, "result": result, "episode": envelope["episodes"][0]})
        evidence.extend([dependency["manifest"], artifact(ledger_path), result])
    require(not require_complete or not missing, "All30 final acceptance still lacks one or more original ten results")
    return {"complete": not missing and len(results) == 10, "results": results, "missing_jobs": missing,
            "evidence_files": list({a["path"]: a for a in evidence}.values())}
