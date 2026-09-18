"""Explicit post-results extension from the completed FAST two-system test to five.

This registration references immutable parent results and scientific profiles. It
does not copy/rewrite them, train anything, select another checkpoint, or execute
an evaluator. The six new trajectories have their own namespace and ledger.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Mapping

from .accounting import file_sha256, fingerprint, write_json_atomic
from .collection_provenance import read_json

SCHEMA = "made_cumulative_five_evaluation_extension_v1"
PARENT_CORE_FINGERPRINT = "725a5febe58990defc7a681dce24b503bbf605016f410d089ee36a128210b154"
METHODS = ("baseline", "esopt_graph_risk")
OLD_TASKS = ("Al-Li-V", "Al-V-Zn")
NEW_TASKS = ("Au-K-Tb", "Co-Dy-W", "Co-Mg-Na")
REGISTRATION = {
    "schema": "post_results_cumulative_five_MADE_registration_v1",
    "study_id": "made_fast_cumulative_five_evaluation_v1",
    "scope": "cumulative_five_MADE_B10_systems_not_original_full_study",
    "parent_initial_test_systems": 2, "cumulative_test_systems": 5,
    "new_test_systems": 3, "methods": list(METHODS), "model_key": "qwen35_4b",
    "seed": 1, "execution_budget": 10,
    "selection_rule": "fewest_elements_then_canonical_task_id_original_official_test",
    "expansion_registered_after_initial_two_system_results_known": True,
    "prospective_five_system_registration_claimed": False,
    "checkpoint_reselection": False, "new_training_or_es": False,
    "test_outcome_tuning": False, "original_full_study_complete": False,
    "existing_four_episodes": "reference_original_verified_bytes_without_physical_replay",
}
COUNTS = {"jobs": 6, "episodes": 6, "candidate_oracle_attempts": 60, "dft_episode_attempts": 0}
CUMULATIVE_COUNTS = {**COUNTS, "jobs": 10, "episodes": 10, "candidate_oracle_attempts": 100}


class EvaluationExtensionError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvaluationExtensionError(message)


def _artifact(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), "Extension evidence must be a file")
    return {"path": str(path), "sha256": file_sha256(path)}


def _verify_artifact(value):
    require(isinstance(value, Mapping) and set(value) == {"path", "sha256"}, "Malformed extension artifact")
    require(_artifact(value["path"]) == value, "Extension source artifact changed: " + str(value.get("path")))
    return dict(value)


def _core(reference):
    from .core_protocol import FAST_CAUSAL_GRAPH_REGISTRATION, core_execution_budget, read_core
    require(isinstance(reference, Mapping) and set(reference) == {"path", "sha256", "core_fingerprint", "workspace"},
            "Missing exact parent core binding")
    workspace = Path(reference["workspace"]).resolve()
    require(reference["workspace"] == str(workspace)
            and reference["path"] == str(workspace / "configs/deadline_core_protocol.json"), "Parent protocol path differs")
    _verify_artifact({key: reference[key] for key in ("path", "sha256")})
    parent = read_core(workspace)
    require(parent["fingerprint"] == reference["core_fingerprint"] == PARENT_CORE_FINGERPRINT,
            "Extension requires the exact completed FAST v5 parent")
    require(parent.get("registration") == FAST_CAUSAL_GRAPH_REGISTRATION
            and core_execution_budget(parent) == 10 and parent.get("budget") == 50,
            "Extension cannot adopt another budget or core registration")
    return parent


def _tasks(parent):
    root = Path(parent["workspace"])
    official = read_json(root / "configs/benchmark_tasks.json")["made"]["systems"]
    splits = read_json(root / "configs/made_splits.json")["splits"]
    def task(elements):
        values = sorted(elements)
        require(len(values) == len(set(values)), "Repeated element in chemical system")
        return {"id": "-".join(values), "elements": values}
    tests = sorted((task(v) for v in splits["test"]), key=lambda t: (len(t["elements"]), t["id"]))
    require(len(tests) == len({t["id"] for t in tests}) == len(official) == 30,
            "Extension requires all 30 original official test systems")
    require({t["id"] for t in tests} == {task(t["elements"])["id"] for t in official}, "Official test split differs")
    seen = {task(v)["id"] for split in ("train", "dev") for v in splits[split]}
    require(not seen & {t["id"] for t in tests}, "Test systems overlap train/development")
    require(tuple(t["id"] for t in tests[:2]) == OLD_TASKS
            and tuple(t["id"] for t in tests[2:5]) == NEW_TASKS
            and parent["test_tasks"] == tests[:2], "Five-system official selection changed")
    return tests[:5]


def _parent_final(parent):
    """Recheck completed parent physical envelopes; never publish into the parent."""
    from .core_final import CoreFinalLedger, validate_core_final_manifest, verify_core_envelope
    from .core_protocol import final_jobs
    root = Path(parent["workspace"])
    base = root / "experiments/core_final"
    paths = {name: base / (name + ".json") for name in ("manifest", "ledger", "completion", "execution_inputs")}
    paths["stage_receipt"] = root / "experiments/core_stage_receipts/final.json"
    artifacts = {name: _artifact(path) for name, path in paths.items()}
    manifest, completion, stage = (read_json(paths[key]) for key in ("manifest", "completion", "stage_receipt"))
    validate_core_final_manifest(manifest)
    jobs = final_jobs(parent)
    require(manifest["jobs"] == jobs and manifest["core_protocol_fingerprint"] == parent["fingerprint"],
            "Parent final matrix differs from the fixed four jobs")
    ledger = CoreFinalLedger(paths["ledger"], manifest, readonly=True)
    expected = {"jobs": 4, "episodes": 4, "candidate_oracle_attempts": 40, "dft_episode_attempts": 0}
    require(completion.get("complete") is True and completion.get("core_final_complete") is True
            and completion.get("completed_jobs") == completion.get("expected_jobs") == 4
            and completion.get("expected_counts") == expected
            and completion.get("core_protocol_fingerprint") == parent["fingerprint"]
            and completion.get("manifest_fingerprint") == manifest["fingerprint"]
            and completion.get("execution_budget") == 10 and completion.get("global_study_complete") is False,
            "Parent final completion is missing, partial, or changes scope")
    require(stage.get("schema") == "deadline_core_stage_receipt_v1" and stage.get("stage") == "final"
            and stage.get("complete") is True and stage.get("core_final_complete") is True
            and stage.get("core_fingerprint") == stage.get("core_protocol_fingerprint") == parent["fingerprint"]
            and stage.get("global_study_complete") is False
            and stage.get("completion") == artifacts["completion"]
            and stage.get("fingerprint") == fingerprint({k: v for k, v in stage.items() if k != "fingerprint"}),
            "Parent final stage receipt is invalid")
    require(stage.get("artifacts") == [artifacts[key] for key in ("completion", "manifest", "ledger", "execution_inputs")],
            "Parent receipt inventory differs")
    tasks = read_json(root / "configs/benchmark_tasks.json")
    imported, profiles, seen = [], {}, set()
    for job in jobs:
        state = ledger.data["jobs"][job["job_id"]]
        require(state["state"] == "succeeded", "Parent has an unfinished/failed/orphaned final job")
        path = Path(state["result_path"]).resolve()
        require(path == base / "jobs" / job["job_id"] / state["attempt_id"] / "result.json"
                and path not in seen, "Duplicate or misplaced parent final result")
        seen.add(path)
        result = _artifact(path)
        require(result["sha256"] == state["result_sha256"], "Parent result differs from ledger")
        proof = verify_core_envelope(path, job, manifest, tasks=tasks)
        require(proof["result_sha256"] == result["sha256"], "Parent envelope changed during verification")
        profile = read_json(path)["execution_profile"]
        require(profile.get("mace_num_workers") == 4 and profile.get("orb_num_workers") == 1,
                "Parent evaluator worker settings changed")
        require(job["method"] not in profiles or profiles[job["method"]] == profile,
                "Parent method used different checkpoints/controllers across test systems")
        profiles[job["method"]] = profile
        imported.append({"job": job, "result": result})
    require(completion.get("result_files") == [item["result"] for item in imported], "Parent completion omits/duplicates/reorders results")
    require(set(profiles) == set(METHODS), "Parent profiles omit an arm")
    # These exact profiles are also required for every new result: no reselection.
    return artifacts, imported, profiles


def _new_jobs(parent, tasks):
    from .core_protocol import final_jobs
    templates = {job["method"]: job for job in final_jobs(parent)}
    result = []
    for method in METHODS:
        for task in tasks[2:]:
            job = deepcopy(templates[method])
            job.pop("job_id")
            job.update(study_id=REGISTRATION["study_id"], classification="post_results_evaluation_extension",
                task_id=task["id"], group_id=task["id"], task=deepcopy(task),
                parent_core_fingerprint=parent["fingerprint"], evaluation_extension_registration=REGISTRATION["schema"])
            job["job_id"] = "extension-final-made-" + fingerprint(job)[:24]
            result.append(job)
    return result


def _source_files(values):
    result = [_verify_artifact(v) if isinstance(v, Mapping) else _artifact(v) for v in values]
    require(result and len({r["path"] for r in result}) == len(result), "Empty/duplicate extension source inventory")
    return sorted(result, key=lambda r: r["path"])


def _shards(values, workspace, jobs):
    ids = [job["job_id"] for job in jobs]
    if values is None:
        values = [{"shard_id": "primary", "execution_site": "primary",
                   "output_workspace": str(workspace / "shards/primary"), "job_ids": ids}]
    require(isinstance(values, list) and 1 <= len(values) <= len(ids), "Missing/empty execution shard plan")
    assigned, names, outputs = [], set(), []
    result = []
    for value in values:
        require(isinstance(value, Mapping) and set(value) == {"shard_id", "execution_site", "output_workspace", "job_ids"},
                "Malformed static execution shard")
        name, site = value["shard_id"], value["execution_site"]
        require(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name)
                and name not in names and isinstance(site, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,31}", site),
                "Duplicate/invalid shard name or non-public site label")
        names.add(name)
        output = Path(value["output_workspace"]).resolve()
        require(str(output) == value["output_workspace"] and workspace in output.parents
                and all(output != old and output not in old.parents and old not in output.parents for old in outputs),
                "Shard outputs must be separate non-overlapping extension workspaces")
        outputs.append(output)
        members = value["job_ids"]
        require(isinstance(members, list) and members and all(isinstance(v, str) and v in ids for v in members)
                and len(set(members)) == len(members), "Shard has duplicate/unregistered jobs")
        assigned.extend(members)
        result.append({"shard_id": name, "execution_site": site, "output_workspace": str(output),
                       "job_ids": [jid for jid in ids if jid in members]})
    require(len(assigned) == len(set(assigned)) == len(ids) and set(assigned) == set(ids),
            "Static shards must cover every new job exactly once")
    return sorted(result, key=lambda v: v["shard_id"])


def _payload(parent_reference, workspace, sources, validation, registered_at, execution_shards=None):
    parent = _core(parent_reference)
    workspace = Path(workspace).resolve()
    original = Path(parent["workspace"]).resolve()
    require(workspace != original and original not in workspace.parents,
            "Extension workspace must be separate from the immutable parent")
    try:
        stamp = datetime.fromisoformat(registered_at.replace("Z", "+00:00"))
        require(stamp.utcoffset() is not None, "Extension registration timestamp needs timezone")
    except (TypeError, ValueError, AttributeError) as exc:
        raise EvaluationExtensionError("Invalid extension registration timestamp") from exc
    tasks = _tasks(parent)
    parent_final, imported, profiles = _parent_final(parent)
    if validation is not None:
        _verify_artifact(validation)
        require(read_json(validation["path"]).get("passed") is True, "Extension implementation validation did not pass")
    jobs = _new_jobs(parent, tasks)
    return {"schema": SCHEMA, "registration": deepcopy(REGISTRATION), "workspace": str(workspace),
        "registered_at": registered_at, "parent_core": dict(parent_reference), "parent_final": parent_final,
        "source_files": _source_files(sources), "validation": validation,
        "implementation_validation_supplied": validation is not None,
        "imported_results": imported, "execution_profiles": profiles,
        "existing_tasks": tasks[:2], "new_tasks": tasks[2:], "cumulative_tasks": tasks,
        "new_jobs": jobs, "cumulative_jobs": [i["job"] for i in imported] + jobs,
        "execution_shards": _shards(execution_shards, workspace, jobs),
        "shard_completion_rule": "all_six_new_jobs_verified_once_before_cumulative_ten_complete",
        "expected_new_counts": dict(COUNTS), "expected_imported_counts": {
            "jobs": 4, "episodes": 4, "candidate_oracle_attempts": 40, "dft_episode_attempts": 0},
        "expected_cumulative_counts": dict(CUMULATIVE_COUNTS),
        "execution_budget": 10, "training_jobs": [], "es_jobs": [],
        "new_training_candidate_oracle_attempts": 0,
        "scientific_improvement_assumed": False, "global_study_complete": False,
        "scientific_parent_inputs": [_artifact(original / name) for name in (
            "configs/main_protocol.json", "configs/benchmark_tasks.json", "configs/made_splits.json",
            "configs/model_manifest.json", "source_manifest.json")]}


def build_extension(parent_core, workspace, *, source_files=None, validation=None, execution_shards=None):
    """Verify original four outcomes and publish one immutable, explicit extension.

    ``source_files`` accepts paths or exact {path, sha256} artifacts. ``validation``
    optionally references an implementation report with passed=true; its absence
    never claims test validation. No training/evaluation is launched here.
    """
    from .core_protocol import read_core
    parent = read_core(parent_core["workspace"] if isinstance(parent_core, Mapping) else parent_core)
    require(not isinstance(parent_core, Mapping) or parent == parent_core, "Caller parent differs from sealed protocol")
    root = Path(parent["workspace"]).resolve()
    reference = {**_artifact(root / "configs/deadline_core_protocol.json"),
                 "core_fingerprint": parent["fingerprint"], "workspace": str(root)}
    sources = _source_files(source_files if source_files is not None else [Path(__file__),
        Path(__file__).with_name("accounting.py"), Path(__file__).with_name("final_evaluation.py"),
        Path(__file__).with_name("evaluation_extension_runner.py"), Path(__file__).with_name("immutable_hash_cache.py")])
    validation = (_verify_artifact(validation) if isinstance(validation, Mapping) else _artifact(validation)) if validation is not None else None
    path = Path(workspace).resolve() / "configs/evaluation_extension.json"
    if path.exists():
        existing = read_extension(path)
        require(existing["parent_core"] == reference and existing["source_files"] == sources
                and existing["validation"] == validation
                and existing["execution_shards"] == _shards(execution_shards, path.parent.parent, existing["new_jobs"]),
                "Existing extension registration cannot be replaced")
        return existing
    value = _payload(reference, workspace, sources, validation, datetime.now(timezone.utc).isoformat(), execution_shards)
    value["fingerprint"] = value["extension_fingerprint"] = fingerprint(value)
    write_json_atomic(path, value)
    return value


def validate_extension(extension):
    """Validate exact scope, parent source, original results and registered profiles."""
    require(isinstance(extension, Mapping), "Extension must be an explicit sealed object")
    payload = {k: v for k, v in extension.items() if k not in {"fingerprint", "extension_fingerprint"}}
    require(extension.get("fingerprint") == extension.get("extension_fingerprint") == fingerprint(payload),
            "Extension fingerprint mismatch")
    path = Path(extension["workspace"]) / "configs/evaluation_extension.json"
    require(read_json(path) == extension, "Caller extension differs from immutable registration")
    expected = _payload(extension["parent_core"], extension["workspace"], extension["source_files"],
                        extension["validation"], extension["registered_at"], extension["execution_shards"])
    require(payload == expected, "Extension scope, jobs, sources, or fixed parent profiles changed")
    return deepcopy(dict(extension))


def read_extension(workspace_or_path):
    path = Path(workspace_or_path).resolve()
    if path.is_dir():
        path = path / "configs/evaluation_extension.json"
    value = read_json(path)
    require(path == Path(value["workspace"]) / "configs/evaluation_extension.json", "Extension registration loaded from another location")
    return validate_extension(value)


def extension_jobs(extension, *, include_imported=False, shard_id=None):
    """Six new jobs by default; the cumulative ten include read-only parent jobs."""
    value = validate_extension(extension)
    if shard_id is not None:
        require(not include_imported, "Read-only imported jobs cannot belong to execution shards")
        shards = {item["shard_id"]: item for item in value["execution_shards"]}
        require(shard_id in shards, "Unregistered execution shard")
        return deepcopy([j for j in value["new_jobs"] if j["job_id"] in shards[shard_id]["job_ids"]])
    return deepcopy(value["cumulative_jobs" if include_imported else "new_jobs"])


def planned_extension_jobs(parent_core):
    """Read-only preview for constructing static shard assignments before sealing."""
    from .core_protocol import read_core
    parent = read_core(parent_core["workspace"] if isinstance(parent_core, Mapping) else parent_core)
    require(not isinstance(parent_core, Mapping) or parent == parent_core, "Caller parent differs from sealed protocol")
    root = Path(parent["workspace"]).resolve()
    reference = {**_artifact(root / "configs/deadline_core_protocol.json"),
                 "core_fingerprint": parent["fingerprint"], "workspace": str(root)}
    parent = _core(reference)
    return _new_jobs(parent, _tasks(parent))
