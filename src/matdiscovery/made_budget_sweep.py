"""Independent B30/B50 evaluation with immutable B10, training and ES ancestry.

The 120 new episodes have fresh identities. Prior B10 results are readonly final
dependencies, never prefixes or training data. Launch grants a PID-local scope
capability only after deep ancestor validation; a JSON flag cannot grant it.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import importlib.util
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Mapping

from .accounting import file_sha256, fingerprint, write_json_atomic
from .collection_provenance import read_json

SCHEMA = "made_B30_B50_budget_sweep_v1"
SCOPE = "post_results_MADE_B30_B50_fixed_G2_budget_sweep"
SOURCE = "registered_MADE_B30_B50_budget_sweep_final"
PARENT_CORE_FINGERPRINT = "725a5febe58990defc7a681dce24b503bbf605016f410d089ee36a128210b154"
PRIOR_ALL30_FINGERPRINT = "a33c801bf2cea72eb048a23ac231144ffcdb0cbbb32a7bbc141aad22764f4140"
METHODS = ("baseline", "esopt_graph_risk")
BUDGETS = (30, 50)
SHARED_ADMISSION_MODULES = ("accounting.py", "final_evaluation.py", "training_jobs.py")
NEW_MODULES = ("made_budget_sweep.py", "made_budget_runner.py")
NEW_COUNTS = {"jobs": 120, "episodes": 120, "candidate_oracle_attempts": 4800, "dft_episode_attempts": 0}
IMPORTED_COUNTS = {"jobs": 60, "episodes": 60, "candidate_oracle_attempts": 600, "dft_episode_attempts": 0}
CUMULATIVE_COUNTS = {"jobs": 180, "episodes": 180, "candidate_oracle_attempts": 5400, "dft_episode_attempts": 0}
REGISTRATION = {
    "schema": "registered_MADE_independent_B30_B50_v1", "study_id": SCHEMA, "scope": SCOPE,
    "model_key": "qwen35_4b", "seed": 1, "methods": list(METHODS), "evaluation_budgets": list(BUDGETS),
    "historical_collection_budget": 50, "selected_ES_execution_budget": 10, "selected_full_generation": 2,
    "official_test_systems": 30, "selection_rule": "all_official_test_ordered_by_element_count_then_canonical_id",
    "fresh_independent_episodes": True, "B10_prefix_extension": False,
    "registered_after_B10_results_known": True, "budget_response_is_post_results_analysis": True,
    "new_training_or_ES": False, "checkpoint_reselection": False, "test_outcome_tuning": False,
    "B10_completion_required_before_new_launch": False, "verified_B10_60_required_for_final_acceptance": True,
    "claim_priority": "unclaimed_B30_before_B50_without_preempting_running_B30",
    "actors_are_optional": True, "completion_denominator": "all_120_registered_jobs_not_actor_receipts",
    "original_multi_model_multi_seed_full_study_complete": False,
}
_LIVE_ADMISSIONS = {}


class MadeBudgetSweepError(ValueError):
    pass


def require(condition, message):
    if not condition: raise MadeBudgetSweepError(message)


def artifact(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), "Sweep evidence is not a regular file")
    return {"path": str(path), "sha256": file_sha256(path)}


def verify_artifact(value):
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"}, "Malformed sweep artifact")
    require(artifact(value["path"]) == value, "Sweep source artifact changed: " + str(value.get("path")))
    return dict(value)


def _reference(value, *, parent=False):
    root = Path(value["workspace"] if isinstance(value, Mapping) else value).resolve()
    if root.is_file(): root = root.parent.parent
    path = root / "configs" / ("deadline_core_protocol.json" if parent else "made_all30_extension.json")
    data = read_json(path)
    if isinstance(value, Mapping): require(data == value, "Caller ancestor differs from its sealed document")
    return {**artifact(path), "workspace": str(root), "core_fingerprint" if parent else "fingerprint": data["fingerprint"]}


def _sources(parent_ref, prior_ref):
    from .core_protocol import FAST_CAUSAL_GRAPH_REGISTRATION
    from .made_all30_extension import read_all30
    values = []
    for ref, filename, key, expected in (
        (parent_ref, "deadline_core_protocol.json", "core_fingerprint", PARENT_CORE_FINGERPRINT),
        (prior_ref, "made_all30_extension.json", "fingerprint", PRIOR_ALL30_FINGERPRINT)):
        require(set(ref) == {"path", "sha256", "workspace", key}, "Malformed pinned sweep ancestor")
        root = Path(ref["workspace"]).resolve()
        require(str(root) == ref["workspace"] and ref["path"] == str(root / "configs" / filename), "Ancestor path differs")
        verify_artifact({k: ref[k] for k in ("path", "sha256")})
        value = read_json(ref["path"])
        aliases = {"fingerprint", "core_fingerprint"} if key == "core_fingerprint" else {"fingerprint", "all30_fingerprint"}
        require(value["fingerprint"] == ref[key] == expected == fingerprint({k:v for k,v in value.items() if k not in aliases}),
                "Sweep requires the exact original FAST core and all30 B10 registration")
        values.append(value)
    parent, prior = values
    require(parent["registration"] == FAST_CAUSAL_GRAPH_REGISTRATION and parent["budget"] == 50
            and parent["execution_budget"] == 10 and parent["methods"] == list(METHODS), "Historical corpus/ES budget changed")
    require(prior["parent_core"] == parent_ref and prior["expected_cumulative_counts"] == IMPORTED_COUNTS
            and prior["execution_budget"] == 10, "Prior B10 ancestry or scope differs")
    require(read_all30(prior["workspace"]) == prior, "Prior all30 sealed matrix differs")
    require(prior["execution_profiles"][METHODS[1]]["selected_es"]["selected_generation"] == 2, "Sweep must keep selected G2")
    return parent, prior


def _jobs(parent, prior):
    from .made_all30_extension import _tasks
    tasks = _tasks(parent)
    require(tasks == prior["official_tasks"], "Official thirty systems changed")
    templates = {j["method"]: j for j in prior["new_jobs"]}
    jobs = []
    for budget in BUDGETS:
        for method in METHODS:
            for task in tasks:
                job = deepcopy(templates[method])
                for key in ("job_id", "made_all30_registration", "prior_extension_fingerprint"):
                    job.pop(key, None)
                job.update(study_id=SCHEMA, classification="registered_independent_budget_sweep", source=SOURCE,
                    task_id=task["id"], task=deepcopy(task), group_id=task["id"], budget=budget,
                    expected_counts={"episodes": 1, "candidate_oracle_attempts": budget, "dft_episode_attempts": 0},
                    made_budget_sweep_registration=REGISTRATION["schema"], prior_all30_fingerprint=prior["fingerprint"])
                require(job["seed"] == 1 and job["environment_seeds"] == [1] and job["episode_ids"] == ["0"]
                        and job["split"] == "test" and job["stage"] == "final_eval" and job["model_key"] == "qwen35_4b",
                        "Sweep cannot change the original model/seed/final-test contract")
                job["job_id"] = "budget-final-made-" + fingerprint(job)[:24]
                jobs.append(job)
    return tasks, jobs


def _actors(values):
    if values is None:
        values = [{"actor_id": f"actor_{i:02d}", "role": "gpu_actor", "initial_method": METHODS[0 if i < 2 else 1]} for i in range(10)]
    require(isinstance(values, list) and 1 <= len(values) <= 10, "Register one to ten optional GPU actors")
    require(len({v["actor_id"] for v in values}) == len(values), "Duplicate actor identity")
    for value in values:
        require(set(value) == {"actor_id", "role", "initial_method"} and value["role"] == "gpu_actor"
                and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value["actor_id"])
                and value["initial_method"] in METHODS, "Malformed registered actor")
    require(Counter(v["initial_method"] for v in values)["baseline"] == min(2, len(values)),
            "Initial scheduling must retain two baseline actors, remainder full")
    return deepcopy(values)


def _dependencies(prior):
    from .made_all30_runner import build_worker_manifest
    result = deepcopy(prior["imported_dependencies"])
    for spec in prior["execution_shards"]:
        manifest = build_worker_manifest(prior, spec["worker_id"])
        for job in manifest["jobs"]:
            result.append({"origin": "prior_all30_B10", "job": job, "result": None,
                "worker": deepcopy(spec), "manifest_path": str(Path(spec["output_workspace"]) / "manifest.json"),
                "manifest_fingerprint": manifest["fingerprint"],
                "ledger_path": str(Path(spec["output_workspace"]) / "ledger.json")})
    require(len(result) == len({d["job"]["job_id"] for d in result}) == 60, "Prior B10 dependencies omitted or duplicated")
    return result


def _payload(parent_ref, prior_ref, workspace, sources, validation, actors, registered_at):
    from .evaluation_extension import _source_files
    parent, prior = _sources(parent_ref, prior_ref)
    workspace = Path(workspace).resolve()
    for old in (Path(parent["workspace"]), Path(prior["workspace"])):
        require(workspace != old and workspace not in old.parents and old not in workspace.parents, "Sweep requires a separate fresh workspace")
    if validation is not None:
        verify_artifact(validation); require(read_json(validation["path"]).get("passed") is True, "Sweep implementation validation has not passed")
    tasks, jobs = _jobs(parent, prior)
    return {"schema": SCHEMA, "scope": SCOPE, "registration": deepcopy(REGISTRATION), "registered_at": registered_at,
        "workspace": str(workspace), "parent_core": parent_ref, "prior_all30": prior_ref,
        "source_files": _source_files(sources), "validation": validation,
        "execution_profiles": prior["execution_profiles"], "official_tasks": tasks, "new_jobs": jobs,
        "actors": _actors(actors), "actors_are_optional": True, "imported_dependencies": _dependencies(prior),
        "expected_new_counts": dict(NEW_COUNTS), "expected_imported_counts": dict(IMPORTED_COUNTS),
        "expected_cumulative_counts": dict(CUMULATIVE_COUNTS),
        "expected_counts_by_budget": {str(b): {"jobs": 60, "episodes": 60, "candidate_oracle_attempts": 60*b, "dft_episode_attempts": 0} for b in BUDGETS},
        "historical_collection_budget": 50, "selected_ES_execution_budget": 10, "evaluation_budgets": list(BUDGETS),
        "training_jobs": [], "es_jobs": [], "global_study_complete": False,
        "B10_results_modified": False, "scientific_improvement_assumed": False,
        "launch_deep_validation_required_per_process": True}


def build_sweep(parent_core, prior_all30, workspace, *, source_files=None, validation=None, actors=None):
    from .evaluation_extension import _source_files
    parent, prior = _reference(parent_core, parent=True), _reference(prior_all30)
    sources = _source_files(source_files if source_files is not None else [Path(__file__).with_name(n) for n in (*SHARED_ADMISSION_MODULES, *NEW_MODULES)])
    validation = (verify_artifact(validation) if isinstance(validation, Mapping) else artifact(validation)) if validation is not None else None
    path = Path(workspace).resolve() / "configs/made_budget_sweep.json"
    if path.exists():
        current = read_sweep(path)
        require(_payload(parent, prior, workspace, sources, validation, actors, current["registered_at"])
                == {k:v for k,v in current.items() if k not in {"fingerprint", "sweep_fingerprint"}}, "Existing sweep cannot be replaced")
        return current
    value = _payload(parent, prior, workspace, sources, validation, actors, datetime.now(timezone.utc).isoformat())
    value["fingerprint"] = value["sweep_fingerprint"] = fingerprint(value)
    write_json_atomic(path, value)
    return value


def validate_sweep(value):
    require(isinstance(value, Mapping), "Sweep must be a sealed object")
    payload = {k:v for k,v in value.items() if k not in {"fingerprint", "sweep_fingerprint"}}
    require(value.get("fingerprint") == value.get("sweep_fingerprint") == fingerprint(payload), "Sweep fingerprint differs")
    require(read_json(Path(value["workspace"]) / "configs/made_budget_sweep.json") == value, "Sweep differs from sealed registration")
    require(payload == _payload(value["parent_core"], value["prior_all30"], value["workspace"], value["source_files"],
        value["validation"], value["actors"], value["registered_at"]), "Sweep budgets/jobs/profiles/source/actor contract changed")
    return deepcopy(dict(value))


def read_sweep(workspace_or_path):
    path = Path(workspace_or_path).resolve()
    if path.is_dir(): path = path / "configs/made_budget_sweep.json"
    value = read_json(path)
    require(path == Path(value["workspace"]) / "configs/made_budget_sweep.json", "Sweep path differs")
    return validate_sweep(value)


def sweep_jobs(value, *, budget=None):
    value = validate_sweep(value)
    require(budget is None or type(budget) is int and budget in BUDGETS, "Unregistered sweep budget")
    return [j for j in value["new_jobs"] if budget is None or j["budget"] == budget]


def verify_runtime_sources(value):
    package = Path(__file__).resolve().parent
    original = Path(value["prior_all30"]["workspace"]) / "src/matdiscovery"
    exceptional = set(SHARED_ADMISSION_MODULES + NEW_MODULES)
    sources = {Path(a["path"]): a["sha256"] for a in value["source_files"]}
    for name in exceptional:
        path = package / name
        require(path in sources and file_sha256(path) == sources[path], "Unsealed executing sweep source: " + name)
    actual = {str(p.relative_to(package)) for p in package.rglob("*.py")}
    expected = {str(p.relative_to(original)) for p in original.rglob("*.py")}
    require(not expected & set(NEW_MODULES) and actual == expected | set(NEW_MODULES), "Sweep changed frozen Python inventory")
    for name in expected - set(SHARED_ADMISSION_MODULES):
        require(file_sha256(package / name) == file_sha256(original / name), "Sweep changed frozen scientific/runtime source: " + name)


def validate_launch(value):
    from .core_protocol import read_core
    from .evaluation_extension import read_extension
    value = validate_sweep(value); verify_runtime_sources(value)
    key = (os.getpid(), value["fingerprint"])
    if key in _LIVE_ADMISSIONS: return deepcopy(_LIVE_ADMISSIONS[key]["proof"])
    parent, prior = _sources(value["parent_core"], value["prior_all30"])
    require(read_core(parent["workspace"]) == parent, "Sweep parent deep admission differs")
    earlier = read_extension(prior["prior_extension"]["workspace"])
    require(earlier["fingerprint"] == prior["prior_extension"]["fingerprint"], "Sweep prior-five deep admission differs")
    proof = {"schema": "process_verified_MADE_budget_sweep_launch_v1", "complete": True,
        "sweep_fingerprint": value["fingerprint"], "parent_core_fingerprint": parent["fingerprint"],
        "prior_all30_fingerprint": prior["fingerprint"], "pending_B10_not_treated_complete": True,
        "evidence_files": [{k:value[n][k] for k in ("path", "sha256")} for n in ("parent_core", "prior_all30")]
            + value["source_files"] + prior["source_files"], "new_scientific_calls": 0}
    proof["fingerprint"] = fingerprint(proof)
    _LIVE_ADMISSIONS[key] = {"parent": deepcopy(parent), "prior": deepcopy(prior), "proof": deepcopy(proof)}
    return proof


def admitted_parent(value):
    value = validate_sweep(value)
    entry = _LIVE_ADMISSIONS.get((os.getpid(), value["fingerprint"]))
    require(entry is not None, "Sweep requires successful validate_launch in this process")
    verify_runtime_sources(value)
    parent, prior = _sources(value["parent_core"], value["prior_all30"])
    require(parent == entry["parent"] and prior == entry["prior"], "Admitted sweep ancestor changed")
    return parent


def validate_job_scope(value, job, *, expected_made_budget, core_protocol=None, execution=False):
    parent = admitted_parent(value)
    require(type(expected_made_budget) is int and expected_made_budget in BUDGETS, "Unregistered sweep execution budget")
    require(core_protocol is None or core_protocol == parent, "Sweep parent differs; never rewrite its budget")
    matches = [j for j in value["new_jobs"] if j["job_id"] == job.get("job_id")]
    require(len(matches) == 1 and matches[0]["budget"] == expected_made_budget, "Job is outside exact sweep budget/matrix")
    expected = matches[0]
    require(all(job.get(k) == v for k,v in expected.items()) if execution else job == expected, "Sweep task/model/seed/budget identity changed")
    if execution:
        profile = value["execution_profiles"][job["method"]]
        generation = profile["selected_es"]["selected_generation"] if job["method"] == METHODS[1] else 0
        require(job.get("actual_model_state_hash") == profile["actual_model_state_hash"]
                and job.get("selected_generation") == generation
                and job.get("execution_profile_fingerprint") == fingerprint(profile), "Sweep changed selected G2/checkpoint/controller")
    return parent


def _b10_modules(prior):
    """Load original B10 verifier code in its own source-bound namespace."""
    root = Path(prior["workspace"]); package = root / "src/matdiscovery"
    sources = {Path(a["path"]): a["sha256"] for a in prior["source_files"]}
    files = sorted(package.rglob("*.py"))
    require(files and package / "__init__.py" in files, "Original B10 source package absent")
    for path in files:
        require(path in sources and file_sha256(path) == sources[path], "Original B10 validator source is not sealed")
    namespace = "_budget_sweep_B10_" + fingerprint({"workspace": str(root), "sources": [(str(p), sources[p]) for p in files]})[:24]
    before = sys.dont_write_bytecode; sys.dont_write_bytecode = True
    try:
        if namespace not in sys.modules:
            spec = importlib.util.spec_from_file_location(namespace, package / "__init__.py", submodule_search_locations=[str(package)])
            module = importlib.util.module_from_spec(spec); sys.modules[namespace] = module; spec.loader.exec_module(module)
        return SimpleNamespace(protocol=importlib.import_module(namespace + ".made_all30_extension"),
            runner=importlib.import_module(namespace + ".made_all30_runner"))
    finally: sys.dont_write_bytecode = before


def audit_B10_results(value, *, require_complete=True):
    """Readonly original 60-result admission; never invoke its aggregate writer."""
    parent = admitted_parent(value); prior = _sources(value["parent_core"], value["prior_all30"])[1]
    modules = _b10_modules(prior)
    # The frozen package has its own ContextVar; the caller's namespace cache
    # cannot activate it. Keep all original checks inside its public context.
    cache = importlib.import_module(modules.protocol.__package__ + ".immutable_hash_cache")
    with cache.immutable_hash_cache():
        return _audit_B10_results(value, parent, prior, modules, require_complete=require_complete)


def _audit_B10_results(value, parent, prior, modules, *, require_complete):
    modules.protocol.validate_launch(prior)
    imported = modules.protocol.audit_imported_results(prior, require_complete=False)
    results, missing, evidence = list(imported["results"]), list(imported["missing_jobs"]), list(imported["evidence_files"])
    tasks = read_json(Path(parent["workspace"]) / "configs/benchmark_tasks.json")
    for spec in prior["execution_shards"]:
        output = Path(spec["output_workspace"])
        if not (output / "ledger.json").is_file(): missing.extend(spec["job_ids"]); continue
        manifest = read_json(output / "manifest.json")
        ledger = modules.runner.MadeAll30Ledger(output / "ledger.json", manifest, registration=prior, readonly=True)
        evidence.extend([artifact(output / "manifest.json"), artifact(output / "ledger.json")])
        for job in manifest["jobs"]:
            state = ledger.data["jobs"][job["job_id"]]
            if state["state"] != "succeeded": missing.append(job["job_id"]); continue
            path = output / "jobs" / job["job_id"] / state["attempt_id"] / "result.json"
            reference = artifact(path)
            require(str(path) == state["result_path"] and reference["sha256"] == state["result_sha256"], "Original B10 result changed")
            checked = modules.runner.verify_all30_envelope(path, job, manifest, prior, tasks=tasks, core=parent)
            require(checked["result_sha256"] == reference["sha256"], "Original B10 result changed during audit")
            results.append({"job": job, "result": reference, "episode": read_json(path)["episodes"][0]}); evidence.append(reference)
    expected = {d["job"]["job_id"] for d in value["imported_dependencies"]}
    actual = [r["job"]["job_id"] for r in results]
    require(len(actual) == len(set(actual)) and set(actual).isdisjoint(missing) and set(actual) | set(missing) == expected,
            "B10 audit omitted or duplicated registered dependencies")
    complete = len(actual) == 60 and not missing
    missing_acceptance = []
    if complete:
        path = Path(prior["workspace"]) / "experiments/made_all30_stage_receipts/final.json"
        if not path.is_file():
            complete = False; missing_acceptance.append(str(path))
        else:
            receipt = read_json(path); report = verify_artifact(receipt["report"])
            require(receipt.get("schema") == "made_all30_global_stage_receipt_v1"
                    and receipt.get("scope") == "post_results_MADE_all30_B10_fixed_G2_evaluation"
                    and receipt.get("complete") is True and receipt.get("all30_complete") is True
                    and receipt.get("all30_fingerprint") == prior["fingerprint"]
                    and receipt.get("fingerprint") == fingerprint({k:v for k,v in receipt.items() if k != "fingerprint"})
                    and receipt.get("expected_cumulative_counts") == IMPORTED_COUNTS, "Original B10 global acceptance differs")
            evidence.extend([artifact(path), report])
    require(not require_complete or complete, "Sweep final acceptance still lacks original B10 sixty results")
    return {"complete": complete, "results": results, "missing_jobs": missing, "missing_acceptance": missing_acceptance,
        "evidence_files": list({a["path"]: a for a in evidence}.values())}
