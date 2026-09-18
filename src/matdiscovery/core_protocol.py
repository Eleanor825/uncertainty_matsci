"""Independent registered MADE core scope; original full-study gates stay strict.

Workspace preparation copies code/configuration only. It verifies imported
complete training receipts but never loads a model, dispatches an oracle, or
changes the original source/queue. All runtime consumers read the sealed copy.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import shutil

from .accounting import file_sha256, fingerprint, write_json_atomic
from .collection_provenance import read_json

SCHEMA = "deadline_core_protocol_v1"
CORPUS_SCHEMA = "deadline_core_corpus_contract_v1"
REGISTRATION = {
    "schema": "deadline_core_registration_v1", "study_id": "made_core48h_v1",
    "scope": "independent_MADE_core_not_original_full_study", "model_key": "qwen35_4b",
    "training_seed": 1, "budget": 50, "methods": ["baseline", "esopt_graph_risk"],
    "training_task_id": "Al-Au-Hf", "required_import_seeds": [1, 2], "optional_complete_import_seeds": [3],
    "dev_selection": "fewest_elements_then_canonical_task_id_original_dev", "test_selection": "fewest_elements_then_canonical_task_id_original_official_test",
    "esopt": {"generations": 2, "population": 2, "cases_per_generation": 1, "validation_every_generations": 1},
    "graph": {"max_feature_nodes": 32, "max_backward_targets": 42}, "offline_graph_decisions": "all",
    "start_utc": "2026-09-15T17:25:00Z", "deadline_utc": "2026-09-17T17:25:00Z",
    "original_full_study_complete": False, "original_made_phase_complete": False,
}


TC64_REGISTRATION = {**REGISTRATION,
    "schema": "deadline_core_registration_tc64_v2", "study_id": "made_core48h_tc64_v2",
    "transcoder_epochs": 64, "amendment": "pretest_uniform_tc_epochs_16_to_64"}

NORMALIZED_RECIPE = "train_centered_scalar_rms_fp32_export_v1"
NORMALIZED_REGISTRATION = {**REGISTRATION,
    "schema": "deadline_core_registration_normalized_v3", "study_id": "made_core48h_normalized_v3",
    "transcoder_epochs": 64, "transcoder_recipe": NORMALIZED_RECIPE,
    "amendment": "pretest_uniform_train_centered_scalar_rms_fp32_export"}

CAUSAL_GRAPH_RULE = "qwen_final_mlp_causal_activation_v1"
CAUSAL_GRAPH_REGISTRATION = {**NORMALIZED_REGISTRATION,
    "schema": "deadline_core_registration_causal_graph_v4", "study_id": "made_core48h_causal_graph_v4",
    "graph": {**REGISTRATION["graph"], "source_selection_rule": CAUSAL_GRAPH_RULE},
    "transcoder_training": "reuse_source_verified_complete_normalized_v3_bank",
    "amendment": "pretest_qwen_final_mlp_causal_feature_eligibility"}

FAST_CAUSAL_GRAPH_REGISTRATION = {**CAUSAL_GRAPH_REGISTRATION,
    "schema": "fast_subset_core_registration_causal_graph_v5", "study_id": "made_fast_subset_causal_v5",
    "scope": "registered_MADE_B10_subset_not_B50_core_or_original_full_study",
    "execution_budget": 10, "historical_collection_budget": 50,
    "amendment": "pretest_user_prioritized_B10_new_rollouts_with_unchanged_B50_historical_corpus",
    "zero_update_outcome": "complete_with_no_evolution_signal_not_evidence_of_effectiveness"}
CAUSAL_GRAPH_REGISTRATIONS = (CAUSAL_GRAPH_REGISTRATION, FAST_CAUSAL_GRAPH_REGISTRATION)


def core_execution_budget(core):
    """New-rollout budget only; historical collection identity remains B50."""
    if core.get("registration") == FAST_CAUSAL_GRAPH_REGISTRATION:
        require(type(core.get("execution_budget")) is int and core["execution_budget"] == 10,
                "FAST subset execution budget differs from its exact registration")
        return 10
    require("execution_budget" not in core, "A prior registration cannot silently change its execution budget")
    return 50


def _registered_epochs(core):
    core_execution_budget(core)
    if core.get("registration") not in CAUSAL_GRAPH_REGISTRATIONS:
        require("passed_bank_reuse_contract" not in core and "causal_graph_amendment" not in core,
                "A prior registration cannot silently adopt the causal graph/reused-bank amendment")
    if core.get("registration") == REGISTRATION:
        require("imported_development" not in core and "tc64_amendment" not in core and "reused_collections" not in core
                and "normalization_amendment" not in core and "fit_recipe" not in core,
                "V1 cannot silently import development or change its training contract")
        return 16
    if core.get("registration") == TC64_REGISTRATION:
        require("normalization_amendment" not in core and "fit_recipe" not in core,
                "Raw TC64 cannot silently adopt normalized training")
        return 64
    require(core.get("registration") in (NORMALIZED_REGISTRATION, *CAUSAL_GRAPH_REGISTRATIONS)
            and core.get("fit_recipe") == NORMALIZED_RECIPE, "Unregistered core/transcoder revision")
    if core["registration"] in CAUSAL_GRAPH_REGISTRATIONS:
        require("passed_bank_reuse_contract" in core and "causal_graph_amendment" in core,
                "Causal graph registration requires explicit original-bank and failed-graph evidence")
    return 64


def registered_transcoder_epochs(core):
    """Return only the epoch count of an exact read_core-verified registration."""
    verified = read_core(core["workspace"])
    require(verified == core, "Caller core differs from the sealed registered protocol")
    return _registered_epochs(verified)


def registered_transcoder_recipe(core):
    """A recipe is admitted by a complete sealed protocol, never a loose flag."""
    verified = read_core(core["workspace"])
    require(verified == core, "Caller core differs from the sealed registered protocol")
    return NORMALIZED_RECIPE if verified["registration"] in (NORMALIZED_REGISTRATION, *CAUSAL_GRAPH_REGISTRATIONS) else "raw"


class CoreProtocolError(ValueError):
    pass


def require(value, message):
    if not value:
        raise CoreProtocolError(message)


def artifact(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def verify_artifact(item):
    require(artifact(item["path"]) == {"path": str(Path(item["path"]).resolve()), "sha256": item["sha256"]},
            "Core source/artifact changed: " + item["path"])


def _task(elements):
    values = sorted(elements)
    return {"id": "-".join(values), "elements": values}


def _tasks(root):
    splits = read_json(root / "configs/made_splits.json")["splits"]
    by_split = {split: sorted((_task(elements) for elements in splits[split]), key=lambda task: (len(task["elements"]), task["id"]))
                for split in ("train", "dev", "test")}
    sets = {split: {task["id"] for task in tasks} for split, tasks in by_split.items()}
    require(not (sets["train"] & sets["dev"] or sets["train"] & sets["test"] or sets["dev"] & sets["test"]),
            "Core train/development/test chemical systems overlap")
    official = {_task(row["elements"])["id"] for row in read_json(root / "configs/benchmark_tasks.json")["made"]["systems"]}
    require(official == sets["test"] and len(official) == 30, "Original official test split was changed")
    require(REGISTRATION["training_task_id"] in sets["train"] and by_split["dev"], "Missing registered training/development chemistry")
    return {"train_tasks": [next(task for task in by_split["train"] if task["id"] == REGISTRATION["training_task_id"])],
            "dev_tasks": by_split["dev"][:1], "test_tasks": by_split["test"][:2]}


def _job(core, task, method, split, stage):
    model = core["model"]
    budget = 50 if stage == "collection" else core_execution_budget(core)
    job = {"study_id": core["study_id"], "classification": "independent_deadline_core_experiment",
        "stage": stage, "benchmark": "made", "model_key": core["model_key"], "model_id": model["model_id"],
        "model_revision": model["revision"], "tokenizer_revision": model.get("tokenizer_revision", model["revision"]),
        "method": method, "split": split, "task_id": task["id"], "group_id": task["id"], "task": copy.deepcopy(task),
        "seed": 1, "environment_seeds": [1], "episode_ids": ["0"], "budget": budget,
        "budget_unit": "candidate_oracle_attempts_per_episode",
        "expected_counts": {"episodes": 1, "candidate_oracle_attempts": budget, "dft_episode_attempts": 0}}
    prefix = "collection-made-" if stage == "collection" else "core-final-made-"
    job["job_id"] = prefix + fingerprint(job)[:24]
    return job


def collection_jobs(core, *, include_imported=True):
    jobs = [copy.deepcopy(item["job"]) for item in core["imported_train"]] if include_imported else []
    development = copy.deepcopy(core["imported_development"]["job"]) if core.get("registration") in (TC64_REGISTRATION, NORMALIZED_REGISTRATION, *CAUSAL_GRAPH_REGISTRATIONS) else _job(core, core["dev_tasks"][0], "baseline", "dev", "collection")
    return jobs + [development]


def final_jobs(core):
    return [_job(core, task, method, "test", "final_eval")
            for method in core["methods"] for task in core["test_tasks"]]


def collection_manifest_paths(core):
    paths = [Path(core["workspace"]) / "experiments/collection/qwen35_4b/made" / item["job"]["job_id"] / "collection_manifest.json" for item in core["imported_train"]]
    dev = collection_jobs(core, include_imported=False)[0]
    return tuple(paths + [Path(core["workspace"]) / "experiments/collection/qwen35_4b/made" / dev["job_id"] / "collection_manifest.json"])


def _derived_main(parent, core):
    result = copy.deepcopy(parent)
    result.update(study=core["study_id"], scope=core["scope"], models=[core["model_key"]], methods=core["methods"], training_seeds=[1])
    if core.get("registration") in (TC64_REGISTRATION, NORMALIZED_REGISTRATION, *CAUSAL_GRAPH_REGISTRATIONS):
        result["transcoder"]["epochs"] = 64
    result["graph"].update(REGISTRATION["graph"])
    if core.get("registration") in CAUSAL_GRAPH_REGISTRATIONS:
        result["graph"].update(CAUSAL_GRAPH_REGISTRATION["graph"])
    result["esopt"].update(REGISTRATION["esopt"], independent_training_arms=["esopt_graph_risk"], total_conditions=1,
        full_training_and_development_candidate_oracle_attempts=300, full_training_and_development_dft_attempts=0,
        checkpoint_retention="after_complete_registered_core_generations_and_verified_reload_keep_best_and_final")
    result["collection"]["made"] = {"train_systems": 1, "development_systems": 1,
        "episodes_per_system": "explicit_core_jobs", "oracle_attempts_per_episode": 50}
    result["final_evaluation"] = {"made": "two_registered_heldout_chemistries_times_two_methods_times_one_seed_times_B50",
        "jobs": 4, "episodes": 4, "MADE_candidate_oracle_attempts": 200, "CrystalGym_DFT_attempts": 0}
    result["budget_interpretation"] = "Independent registered deadline core; original full study is not completed or reduced."
    if core.get("registration") == FAST_CAUSAL_GRAPH_REGISTRATION:
        result["execution_budget"] = core_execution_budget(core)
        result["esopt"]["full_training_and_development_candidate_oracle_attempts"] = 60
        result["final_evaluation"].update(made="two_registered_heldout_chemistries_times_two_methods_times_one_seed_times_B10",
            MADE_candidate_oracle_attempts=40)
        result["budget_interpretation"] = "User-prioritized B10 subset for new ES/final rollouts; all historical B50 collections retained. Not completion of B50 core or full study."
    return result


def _core_hash(core):
    return fingerprint({key: value for key, value in core.items() if key not in {"fingerprint", "core_fingerprint"}})


def read_core(project_or_workspace):
    workspace = Path(project_or_workspace).resolve()
    path = workspace / "configs/deadline_core_protocol.json"
    core = read_json(path)
    require(core.get("schema") == SCHEMA and core.get("fingerprint") == core.get("core_fingerprint") == _core_hash(core), "Unsealed/modified core protocol")
    epochs = _registered_epochs(core)
    registration = core["registration"]
    require(core.get("workspace") == str(workspace), "Core registration/workspace mismatch")
    require(all(core.get(key) == registration[key] for key in ("study_id", "scope", "model_key", "training_seed", "budget", "methods", "deadline_utc")), "Core scope changed")
    require(all(core.get(key) == value for key, value in _tasks(workspace).items()), "Registered chemical split/selection changed")
    model = next(item for item in read_json(workspace / "configs/model_manifest.json")["models"] if item["key"] == core["model_key"])
    require(core["model"] == model, "Core model/checkpoint differs from fixed manifest")
    seeds = [item["job"]["seed"] for item in core["imported_train"]]
    require(seeds in ([1, 2], [1, 2, 3]), "Core imports must contain every registered complete training seed")
    for item in core["imported_train"]:
        job = item["job"]
        require(job["task_id"] == "Al-Au-Hf" and job["split"] == "train" and job["model_key"] == "qwen35_4b"
                and job["budget"] == 50 and job["method"] == "baseline", "Imported training identity changed")
        verify_artifact(item["receipt"]); verify_artifact(item["manifest"])
    for item in core["snapshot_files"] + core["source_evidence"]:
        verify_artifact(item)
    main = read_json(workspace / "configs/main_protocol.json")
    parent = read_json(workspace / "configs/full_study_main_protocol.json")
    require(main == _derived_main(parent, core), "Core derived protocol changes unregistered scientific settings")
    require(main["transcoder"]["epochs"] == epochs and main["transcoder"]["max_development_output_fvu"] == .5,
            "Core transcoder fidelity/training contract changed")
    require(main["graph"]["validation_epsilon"] == .001 and main["graph"]["validation_rtol"] == .05
            and main["graph"]["validation_atol"] == .0001, "Core native numerical gates changed")
    require(main["made_execution"]["mace_num_workers"] == 4 and main["made_execution"]["orb_num_workers"] == 1
            and main["policy_runtime"]["dtype"] == "float32", "Core fixed evaluator/policy runtime changed")
    for key in ("esopt", "graph", "transcoder", "risk_network", "failure_control", "policy_runtime", "decoding", "memory", "made_execution"):
        require(core[key] == main[key], "Mirrored core configuration changed: " + key)
    require(core["original_full_study_complete"] is False and core["original_made_phase_complete"] is False, "Core scope cannot complete original study")
    source = read_json(workspace / "source_manifest.json")
    files = {str(Path(item["path"]).relative_to(workspace)): item["sha256"] for item in core["snapshot_files"] if Path(item["path"]) != workspace / "source_manifest.json"}
    require(source["files"] == files and source["source_fingerprint"] == fingerprint(files) and source["original_project"] == str(workspace), "Core source manifest differs from copied source")
    present = {str(p.relative_to(workspace)) for name in ("src", "scripts", "configs") for p in (workspace / name).rglob("*")
               if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name not in {"stage_queue.json", "deadline_core_protocol.json"}}
    require(present == set(files), "Core source tree added or lost an unregistered file")
    require(core["derived_main_protocol"] == artifact(workspace / "configs/main_protocol.json"), "Core main protocol hash differs")
    require(core["collection_jobs"] == collection_jobs(core) and core["final_jobs"] == final_jobs(core), "Core exact job matrix changed")
    if registration == TC64_REGISTRATION:
        from .tc64_recovery import validate_tc64_registration
        validate_tc64_registration(core)
    elif registration == NORMALIZED_REGISTRATION:
        from .normalized_recovery import validate_normalized_registration
        validate_normalized_registration(core)
    elif registration in CAUSAL_GRAPH_REGISTRATIONS:
        from .graph_recovery import validate_causal_graph_registration
        validate_causal_graph_registration(core)
    return core


def manifest_input_files(core):
    return tuple(sorted({Path(item["path"]).resolve() for item in core["snapshot_files"] + core["source_evidence"]}
                        | {Path(core["workspace"]) / "configs/deadline_core_protocol.json"}))


def corpus_contract(core, manifest_paths):
    paths = tuple(Path(path).resolve() for path in manifest_paths)
    expected = collection_manifest_paths(core)
    require(len(paths) == len(expected) and len(set(paths)) == len(paths) and set(paths) == set(expected), "Core corpus omitted, duplicated, or added a manifest")
    jobs = collection_jobs(core)
    entries = []
    for path, job in zip(expected, jobs, strict=True):
        manifest = read_json(path)
        require(manifest.get("complete") is True and manifest.get("job_id") == job["job_id"], "Core corpus has an incomplete/wrong manifest")
        entries.append({**artifact(path), "job_id": job["job_id"], "split": job["split"]})
    result = {"schema": CORPUS_SCHEMA, "core_fingerprint": core["fingerprint"],
        "protocol_path": str(Path(core["workspace"]) / "configs/deadline_core_protocol.json"),
        "model_key": core["model_key"], "benchmark": "made", "expected_collection_jobs": len(entries), "manifests": entries,
        **{key: core[key] for key in ("train_tasks", "dev_tasks", "test_tasks")}, "test_used_for_fit": False}
    if core["registration"] in CAUSAL_GRAPH_REGISTRATIONS:
        result["passed_bank_reuse_contract"] = core["passed_bank_reuse_contract"]
    result["fingerprint"] = fingerprint(result)
    return result


def validate_corpus_contract(contract, *, manifest_paths=None, model_key=None):
    require(isinstance(contract, dict) and contract.get("schema") == CORPUS_SCHEMA, "Unknown core corpus contract")
    core = read_core(Path(contract["protocol_path"]).parent.parent)
    expected = corpus_contract(core, [item["path"] for item in contract["manifests"]])
    require(contract == expected and (model_key is None or model_key == core["model_key"]), "Core corpus contract/model was modified")
    if manifest_paths is not None:
        require(corpus_contract(core, manifest_paths) == contract, "Typed fit manifest set differs from core corpus")
    return contract


def prepare_core_workspace(source_project, destination, *, reconciliation_path):
    """Snapshot registered core scope after the independently audited safe pause.

    Existing complete workspaces are verified, never refreshed from live source.
    A partial workspace requires reconciliation; this operation dispatches no
    scientific work. The fixed deadline is not restarted by preparation/resume.
    """
    from .core_collection import inspect_train_imports
    source, workspace = Path(source_project).resolve(), Path(destination).resolve()
    if workspace.exists():
        require((workspace / "configs/deadline_core_protocol.json").is_file(), "Partial core workspace requires reconciliation")
        existing = read_core(workspace)
        require(existing["source_project"] == str(source) and existing["reconciliation"] == artifact(reconciliation_path), "Existing core workspace uses another registration")
        return existing
    require(workspace != source and workspace not in source.parents, "Invalid core workspace")
    registration = read_json(source / "configs/deadline_core_protocol.json")
    require(registration == REGISTRATION, "Deadline registration differs from the fixed authorized core")
    pause = read_json(reconciliation_path)
    require(pause.get("schema") == "interruption_reconciliation_v1" and pause.get("all_requests_resolved") is True
            and pause.get("unknown_physical_outcomes", 0) == 0 and pause.get("previous_processes_stopped")
            and pause.get("used_for_training") is False and pause.get("used_for_final_evaluation") is False,
            "Core workspace requires a stopped, closed, separately accounted original run")
    require(pause.get("additional_incurred_costs_not_subtracted_from_main_budgets") is True
            and isinstance(pause.get("observed_physical_costs"), dict) and pause["observed_physical_costs"]
            and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in pause["observed_physical_costs"].values()),
            "Core pause reconciliation must preserve explicit known physical costs")
    archived = pause.get("artifacts")
    require(isinstance(archived, list) and archived and len({item["path_after"] for item in archived}) == len(archived),
            "Core pause reconciliation lacks unique archived evidence")
    archived_evidence = [{"path": item["path_after"], "sha256": item["sha256"]} for item in archived]
    for item in archived_evidence:
        verify_artifact(item)
    parent = (source / "experiments").resolve().parent
    imported = inspect_train_imports(parent)
    originals = {str(p.relative_to(source)): file_sha256(p) for name in ("src", "scripts", "configs") for p in (source / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name not in {"stage_queue.json", "deadline_core_protocol.json"}}
    workspace.mkdir(parents=True)
    for name in ("src", "scripts", "configs"):
        shutil.copytree(source / name, workspace / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "stage_queue.json", "deadline_core_protocol.json"))
    require(all(file_sha256(source / name) == digest == file_sha256(workspace / name) for name, digest in originals.items()), "Source changed while copying core snapshot; partial workspace requires reconciliation")
    for name in ("data", "vendor", "environments"):
        require((source / name).exists(), "Missing shared core input: " + name)
        (workspace / name).symlink_to((source / name).resolve(), target_is_directory=True)
    for name in ("experiments", "logs"):
        (workspace / name).mkdir()
    parent_main = read_json(source / "configs/main_protocol.json")
    write_json_atomic(workspace / "configs/full_study_main_protocol.json", parent_main)
    write_json_atomic(workspace / "configs/deadline_core_registration.json", registration)
    model = next(item for item in read_json(source / "configs/model_manifest.json")["models"] if item["key"] == "qwen35_4b")
    core = {"schema": SCHEMA, "registration": registration, **{key: registration[key] for key in
            ("study_id", "scope", "model_key", "training_seed", "budget", "methods", "start_utc", "deadline_utc")},
        "workspace": str(workspace), "source_project": str(source), "parent_project": str(parent), "model": model,
        **_tasks(workspace), "imported_train": imported, "reconciliation": artifact(reconciliation_path),
        "original_full_study_complete": False, "original_made_phase_complete": False}
    main = _derived_main(parent_main, core)
    for key in ("esopt", "graph", "transcoder", "risk_network", "failure_control", "policy_runtime", "decoding", "memory", "made_execution"):
        core[key] = copy.deepcopy(main[key])
    write_json_atomic(workspace / "configs/main_protocol.json", main)
    core["derived_main_protocol"] = artifact(workspace / "configs/main_protocol.json")
    from .core_collection import validate_import_compatibility
    for item in imported:
        validate_import_compatibility(core, item)
    core["collection_jobs"], core["final_jobs"] = collection_jobs(core), final_jobs(core)
    frozen = [path for name in ("src", "scripts", "configs") for path in (workspace / name).rglob("*")
              if path.is_file() and "__pycache__" not in path.parts and path.name != "stage_queue.json"]
    files = {str(path.relative_to(workspace)): file_sha256(path) for path in sorted(frozen)}
    write_json_atomic(workspace / "source_manifest.json", {"schema": "deadline_core_source_snapshot_v1",
        "original_project": str(workspace), "parent_project": str(parent), "source_project": str(source),
        "source_fingerprint": fingerprint(files), "files": files, "copied_source_inputs": originals})
    core["snapshot_files"] = [artifact(path) for path in sorted(frozen)] + [artifact(workspace / "source_manifest.json")]
    evidence = [core["reconciliation"], *archived_evidence]
    evidence += [item for imported_job in imported for item in imported_job["source_evidence"]]
    core["source_evidence"] = list({item["path"]: item for item in evidence}.values())
    core["fingerprint"] = core["core_fingerprint"] = _core_hash(core)
    write_json_atomic(workspace / "configs/deadline_core_protocol.json", core)
    return read_core(workspace)
