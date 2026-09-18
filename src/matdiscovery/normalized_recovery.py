"""Registered V3 normalized training, preserving failed runs and closed data.

No policy, oracle, fitting or process signal is invoked here. Every V3 layer is
trained afresh for 64 epochs; the four diagnostic layers cannot be imported as a
partial main bank. Reuse derives new small manifests from the exact V1 closed
corpus and keeps all original physical identities/receipts and activation bytes.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import shutil

from .accounting import file_sha256, fingerprint, write_json_atomic
from .collection_provenance import read_json
from .core_protocol import (SCHEMA, TC64_REGISTRATION, NORMALIZED_REGISTRATION,
    NORMALIZED_RECIPE, _core_hash, _derived_main, artifact, collection_jobs,
    collection_manifest_paths, final_jobs, read_core, require, verify_artifact)
from .tc64_recovery import (_pretest, _failed_bank, validate_tc64_registration,
    _verify_reused_collection, _import_reused_collections)

DIAGNOSTIC_LAYERS = [1, 12, 13, 16]
AMENDMENT_SPEC = {
    "schema": "pretest_train_centered_scalar_rms_fp32_export_amendment_v1",
    "from_recipe": "raw", "to_recipe": NORMALIZED_RECIPE, "layers": 32, "epochs": 64,
    "training": "restart_all_32_layers_from_original_seed_no_diagnostic_checkpoint_reuse",
    "statistics": "all_training_rows_coordinate_means_and_positive_global_centered_RMS_float64",
    "optimizer_coordinates": "normalized_input_and_output_MSE",
    "export": "deterministic_FP32_original_TopK_parameter_form",
    "selection": "minimum_actual_raw_FP32_MSE_over_all_complete_development_epochs",
    "FP32_pointwise_equivalence_claimed": False,
    "max_development_output_fvu": .5, "native_numerical_gates_unchanged": True,
    "model_chemical_splits_B50_ES_and_test_unchanged": True,
    "closed_training_jobs": 3, "closed_development_jobs": 1,
    "additional_collection_oracle_calls": 0, "workers": 4,
    "seed": "1729+layer", "development_informed_recipe_amendment": True,
    "test_or_es_outcomes_used": False,
}


def _jsonl(path):
    import json
    data = Path(path).read_bytes()
    require(data and data.endswith(b"\n"), "Partial/empty diagnostic history")
    return [json.loads(line) for line in data.splitlines()]


def _self_fingerprint(record):
    return record.get("fingerprint") == fingerprint({k: v for k, v in record.items() if k != "fingerprint"})


def _raw_history(meta, epochs, reference=None):
    history = meta.get("history", [])
    require(len(history) == epochs and meta.get("epochs") == epochs
        and [row.get("epoch") for row in history] == list(range(epochs)), "Incomplete prior raw training history")
    if reference is not None:
        require(meta["provenance"] == reference["provenance"] and history[:64] == reference["history"],
                "Raw128 diagnostic differs from its exact raw64 prefix/corpus")
    require(all(math.isfinite(row["dev"]["output_mse"]) and row["dev"]["output_mse"] >= 0
        and row["train_rows"] == meta["provenance"]["rows"]["train"]
        and row["dev"]["rows"] == meta["provenance"]["rows"]["dev"] for row in history), "Prior history omitted rows or has invalid metrics")
    selected = min(range(epochs), key=lambda e: history[e]["dev"]["output_mse"])
    require(meta["selected_epoch"] == selected and meta["dev"] == history[selected]["dev"]
        and meta["max_dev_fvu"] == .5, "Prior raw dev selection/fidelity setting changed")
    passed = not meta["dev"]["fvu_undefined"] and meta["dev"]["output_fvu"] <= .5
    require(meta["fidelity_gate_passed"] is passed, "Prior raw pass flag is false evidence")


def _diagnostic_task(task, index, previous, metadata, contract):
    expected_dev = metadata["execution_provenance"]["development_files"]
    require(task.get("core_fingerprint") == previous["fingerprint"] and task.get("layer_index") == index
        and task.get("policy_fingerprint") == contract["policy_fingerprint"] and task.get("layer_path") == contract["layer_path"]
        and task.get("seed") == 1729 + index and task.get("config") == contract["config"]
        and task.get("train_paths") == contract["train_paths"]
        and task.get("dev_paths") == [item["path"] for item in expected_dev],
        "Diagnostic changed the exact ordered train/dev files or fixed policy/layer")
    source_inventory = {item["path"]: item["sha256"] for item in task.get("sources", [])}
    require(all(source_inventory.get(item["path"]) == item["sha256"]
        for item in contract["training"]["files"] + expected_dev), "Diagnostic sources omitted train/dev file hashes")
    for item in task["sources"]: verify_artifact(item)


def _offline_history(amendment, previous, failed):
    """Validate primary histories; return one explicit entry per incurred run."""
    rows = []
    def entry(run_id, artifacts, layers, epochs):
        unique = list({x["path"]: x for x in artifacts}.values())
        for item in unique: verify_artifact(item)
        rows.append({"run_id": run_id, "role": run_id, "artifacts": unique,
            "expected_layers": layers, "epochs_if_known": epochs})

    old = previous["tc64_amendment"]
    entry("raw_tc16", old["prior_tc16_execution"] + [old["failed_bank"], old["failure_reconciliation"]], list(range(32)), 16)
    run = read_json(amendment["prior_precompute_result"]["path"])
    require(run.get("complete") is True and run.get("core_fingerprint") == previous["fingerprint"]
        and run.get("layers") == 32 and run.get("epochs_per_layer") == 64
        and len(run.get("layer_receipts", [])) == 32, "Raw64 run is incomplete or belongs to another core")
    contracts = []
    for index, item in enumerate(run["layer_receipts"]):
        verify_artifact(item); record = read_json(item["path"]); contract = record.get("contract", {})
        require(record.get("complete") is True and _self_fingerprint(record)
            and contract.get("epochs") == 64 and contract.get("seed") == 1729 + index
            and len(record.get("epochs", [])) == 64, "Raw64 per-layer precomputation is incomplete")
        history = failed["layers"][index]["metadata"]["history"]
        require(all(x["epoch"] == e and x["train_rows"] == history[e]["train_rows"]
            and x["train_output_mse"] == history[e]["train_output_mse"] for e, x in enumerate(record["epochs"])),
            "Raw64 selection and precomputation histories differ")
        require(failed["layers"][index]["metadata"]["execution_provenance"]["precomputation"] == item,
            "Raw64 failed selection belongs to another run")
        contracts.append(contract)
    closed = read_json(amendment["failure_reconciliation"]["path"])
    require(closed.get("schema") == "tc64_fidelity_failure_closed_v1" and closed.get("complete") is True
        and closed.get("closed") is True and closed.get("core_fingerprint") == previous["fingerprint"]
        and closed.get("failed_bank") == amendment["failed_bank"]
        and closed.get("training_result") == amendment["prior_precompute_result"]
        and closed.get("queue_was_not_launched") is True and closed.get("later_graph_risk_ES_test_not_started") is True
        and closed.get("evidence_preserved") is True and closed.get("new_policy_or_oracle_calls") == 0,
        "Raw64 failure/process closure is not registered")
    for key in ("failure", "selection"): verify_artifact(closed[key])
    entry("raw_tc64", [amendment[k] for k in ("prior_precompute_result", "failed_bank", "failure_reconciliation")]
        + [closed["failure"], closed["selection"]], list(range(32)), 64)

    raw128 = read_json(amendment["raw128_summary"]["path"])
    require(raw128.get("complete") is True and raw128.get("prefix64_exact") is True
        and raw128.get("scientific_main_result") is False and raw128.get("test_data_used") is False
        and raw128.get("all_four_fidelity_passed") is False and [x["index"] for x in raw128["layers"]] == DIAGNOSTIC_LAYERS,
        "Raw128 four-layer failed diagnostic evidence differs")
    raw128_artifacts = [amendment["raw128_summary"], raw128["protocol"]]
    for item in raw128["layers"]:
        verify_artifact(item["result"]); result = read_json(item["result"]["path"])
        require(result.get("complete") is True and _self_fingerprint(result) and result["index"] == item["index"], "Raw128 diagnostic result changed")
        _raw_history(result["metadata"], 128, failed["layers"][item["index"]]["metadata"])
        raw128_artifacts += [item["result"], result["checkpoint"], result["sidecar"]]
    entry("raw128_four_layer_diagnostic", raw128_artifacts, DIAGNOSTIC_LAYERS, 128)

    guard = read_json(amendment["normalized_guard_registration"]["path"])
    require(guard.get("schema") == "registered_normalized64_diagnostic_v1" and _self_fingerprint(guard)
        and guard.get("core_fingerprint") == previous["fingerprint"] and guard.get("layers") == DIAGNOSTIC_LAYERS
        and guard.get("epochs") == 64 and len(guard.get("tasks", [])) == 4, "Normalized guard failure registration differs")
    guard_artifacts = [amendment["normalized_guard_registration"], guard["source"], guard["affine_proof_source"]]
    for index, task in zip(DIAGNOSTIC_LAYERS, guard["tasks"], strict=True):
        _diagnostic_task(task, index, previous, failed["layers"][index]["metadata"], contracts[index])
        directory = Path(task["output"]); failure = read_json(directory / "failure.json")
        history = _jsonl(directory / "history.jsonl")
        require(task["layer_index"] == index and failure.get("complete") is False and failure.get("task") == task
            and "Raw affine folding differs numerically" in failure.get("error", "")
            and 0 < len(history) < 64 and [x["epoch"] for x in history] == list(range(len(history)))
            and failure.get("new_policy_or_oracle_calls") == 0, "Normalized prior failure is partial/unknown or not the recorded numeric guard")
        guard_artifacts += [artifact(directory / "failure.json"), artifact(directory / "history.jsonl")]
    entry("normalized64_guard_failed", guard_artifacts, DIAGNOSTIC_LAYERS, 64)

    boundary = read_json(amendment["fold_boundary_result"]["path"])
    require(boundary.get("fixed_epochs_complete") == 5 and boundary.get("new_policy_or_oracle_calls") == 0
        and boundary.get("float64_refold_support_equal") is True
        and boundary.get("float64_refold_output_max_error", math.inf) <= 1e-10
        and boundary.get("affected_rows", 0) > 0, "Missing actual rounding-boundary diagnosis")
    entry("fold_boundary_reproduction", [amendment["fold_boundary_result"],
        artifact(Path(amendment["fold_boundary_result"]["path"]).with_name("registration.json"))], [1], 5)

    summary = read_json(amendment["normalized_diagnostic_summary"]["path"])
    require(summary.get("complete") is True and summary.get("all_four_passed") is True
        and summary.get("new_policy_or_oracle_calls") == 0 and [x["layer"] for x in summary["cases"]] == DIAGNOSTIC_LAYERS,
        "Normalized diagnostic is not complete for all four fixed layers")
    verify_artifact(summary["registration"]); registration = read_json(summary["registration"]["path"])
    require(registration.get("schema") == "registered_normalized64_diagnostic_v2" and _self_fingerprint(registration)
        and registration.get("core_fingerprint") == previous["fingerprint"] and registration.get("epochs") == 64
        and registration.get("layers") == DIAGNOSTIC_LAYERS and registration.get("workers") == 4,
        "Normalized successful diagnostic registration differs")
    successful = [amendment["normalized_diagnostic_summary"], summary["registration"], amendment["normalized_export_audit"],
        registration["source"], registration["affine_proof_source"]]
    for index, task, case in zip(DIAGNOSTIC_LAYERS, registration["tasks"], summary["cases"], strict=True):
        _diagnostic_task(task, index, previous, failed["layers"][index]["metadata"], contracts[index])
        require(task["layer_index"] == index and task["core_fingerprint"] == previous["fingerprint"], "Normalized diagnostic task/core changed")
        directory = Path(task["output"]); result = read_json(directory / "result.json"); history = result.get("history", [])
        ref = failed["layers"][index]["metadata"]
        require(result.get("complete") is True and result.get("epochs") == 64 and result.get("task") == task
            and result.get("fidelity_gate_passed") is True and result.get("original_FVU_threshold") == .5
            and result.get("test_data_used") is False and result.get("new_policy_or_oracle_calls") == 0
            and len(history) == 64 and task["seed"] == 1729 + index and task["config"] == ref["config"],
            "Normalized diagnostic omitted epochs/data or changed the layer contract")
        for e, row in enumerate(history):
            require(row["epoch"] == e and row["train_rows"] == ref["provenance"]["rows"]["train"]
                and row["train_raw_end_epoch"]["rows"] == ref["provenance"]["rows"]["train"]
                and row["dev_raw"]["rows"] == ref["provenance"]["rows"]["dev"]
                and math.isfinite(row["dev_raw"]["output_mse"]) and row["dev_raw"]["output_mse"] >= 0,
                "Normalized diagnostic history is incomplete/nonfinite")
        best = min(range(64), key=lambda e: history[e]["dev_raw"]["output_mse"])
        require(result["selected_epoch"] == best and result["dev_raw"] == history[best]["dev_raw"]
            and not result["dev_raw"]["fvu_undefined"] and result["dev_raw"]["output_fvu"] <= .5
            and case["best_epoch"] == best and case["dev_FVU"] == result["dev_raw"]["output_fvu"] and case["passed"] is True,
            "Normalized diagnostic used another metric/selection or invented a pass")
        statistics = result["statistics"]
        require(statistics["rows"] == ref["provenance"]["rows"]["train"]
            and statistics["policy_fingerprint"] == ref["provenance"]["policy_fingerprint"]
            and statistics["layer_path"] == ref["provenance"]["layer_path"]
            and all(math.isfinite(statistics[k]) and statistics[k] > 0 for k in ("scale_x", "scale_y"))
            and statistics["sources"] == [artifact(p) for p in task["train_paths"]], "Normalization statistics did not bind all train-only files")
        for item in task["sources"]: verify_artifact(item)
        successful += [artifact(directory / "result.json"), result["weights"], statistics["tensor_file"],
            artifact(directory / "selected_normalized.pt")]
    audit = read_json(amendment["normalized_export_audit"]["path"])
    require(audit.get("complete") is True and audit.get("registration") == summary["registration"]
        and audit.get("new_optimizer_steps") == 0 and audit.get("new_policy_or_oracle_calls") == 0
        and audit.get("test_data_used") is False and [x["layer"] for x in audit["layers"]] == DIAGNOSTIC_LAYERS,
        "Selected export rounding audit is incomplete")
    for row in audit["layers"]:
        require(row["float64_fixed_train_roundtrip"]["passed"] is True
            and all(row["splits"][s]["rows"] == failed["layers"][row["layer"]]["metadata"]["provenance"]["rows"][s]
                for s in ("train", "dev")), "Export audit omitted train/dev rows or failed FP64 algebra")
        for item in row["evidence"]: verify_artifact(item)
    entry("normalized64_four_layer_success", successful, DIAGNOSTIC_LAYERS, 64)
    return rows


def validate_normalized_registration(core):
    require(core.get("registration") == NORMALIZED_REGISTRATION and core.get("fit_recipe") == NORMALIZED_RECIPE,
            "Unknown normalized V3 recipe registration")
    amendment = core.get("normalization_amendment", {})
    require(amendment.get("spec") == AMENDMENT_SPEC and read_json(Path(core["workspace"]) / "configs/normalization_amendment.json") == amendment,
            "Normalized scientific amendment changed")
    keys = ("predecessor_protocol", "failed_bank", "failure_reconciliation", "prior_precompute_result", "raw128_summary",
        "normalized_guard_registration", "fold_boundary_result", "normalized_diagnostic_summary", "normalized_export_audit")
    for key in keys: verify_artifact(amendment[key])
    previous = read_core(Path(amendment["predecessor_protocol"]["path"]).parent.parent)
    require(previous["registration"] == TC64_REGISTRATION and previous["fingerprint"] == amendment["predecessor_fingerprint"],
            "Normalized V3 must descend from the exact failed raw TC64 core")
    _pretest(previous); original = validate_tc64_registration(previous)
    for key in ("model", "model_key", "train_tasks", "dev_tasks", "test_tasks", "training_seed", "budget", "methods", "graph",
        "esopt", "transcoder", "risk_network", "failure_control", "policy_runtime", "decoding", "memory", "made_execution", "deadline_utc",
        "parent_project", "imported_train", "imported_development", "reused_collections", "tc64_amendment"):
        require(core[key] == previous[key], "Normalized amendment changed an unregistered setting: " + key)
    require(amendment["corpus_predecessor_protocol"] == artifact(Path(original["workspace"]) / "configs/deadline_core_protocol.json")
        and amendment["corpus_predecessor_fingerprint"] == original["fingerprint"], "Normalized reuse lost the original closed V1 corpus")
    failed = _failed_bank(amendment["failed_bank"]["path"], previous, deep=False, expected_epochs=64)
    require(amendment["offline_history"] == _offline_history(amendment, previous, failed), "Offline history is omitted, duplicated or altered")
    future = Path(amendment["normalized_precompute_output_root"]); bank = Path(amendment["normalized_bank_path"])
    require(future.is_absolute() and bank.is_absolute() and str(future / "result.json") == amendment["normalized_precompute_result_path"],
        "Normalized future output paths changed")
    for old in (Path(previous["workspace"]), Path(original["workspace"]), Path(amendment["prior_precompute_result"]["path"]).parent,
                Path(amendment["failed_bank"]["path"]).parent, Path(amendment["normalized_diagnostic_summary"]["path"]).parent):
        require(future != old and old not in future.parents and bank != old and old not in bank.parents,
                "Normalized outputs would overwrite a frozen/failed prior run")
    return original


def prepare_normalized_workspace(source_project, destination, *, predecessor_workspace,
        failed_bank_path, failure_reconciliation_path, prior_precompute_result, raw128_summary,
        normalized_guard_registration, fold_boundary_result, normalized_diagnostic_summary,
        normalized_export_audit, normalized_output_root, normalized_bank_path):
    """Seal V3 after strict prior closure; never collect, train or copy epochs.

    Repeated preparation only verifies an identical complete registration. Any
    partial target or changed evidence is rejected for explicit reconciliation.
    """
    from .core_collection import completed_core_collections, verify_stage_receipt
    previous = read_core(predecessor_workspace)
    require(previous["registration"] == TC64_REGISTRATION, "Normalized amendment requires the registered raw TC64 predecessor")
    _pretest(previous); original = validate_tc64_registration(previous)
    for stage in ("import", "collect"): verify_stage_receipt(previous, stage)
    require(len(completed_core_collections(previous)) == 4, "Normalized training requires all four closed corpus jobs")
    failed = _failed_bank(failed_bank_path, previous, expected_epochs=64)
    source, target = Path(source_project).resolve(), Path(destination).resolve()
    require(target != source and target not in source.parents and target not in (Path(previous["workspace"]), Path(original["workspace"])), "Invalid new normalized workspace")
    inputs = {"failed_bank": failed_bank_path, "failure_reconciliation": failure_reconciliation_path,
        "prior_precompute_result": prior_precompute_result, "raw128_summary": raw128_summary,
        "normalized_guard_registration": normalized_guard_registration, "fold_boundary_result": fold_boundary_result,
        "normalized_diagnostic_summary": normalized_diagnostic_summary, "normalized_export_audit": normalized_export_audit}
    amendment = {"spec": AMENDMENT_SPEC, **{key: artifact(path) for key, path in inputs.items()},
        "predecessor_protocol": artifact(Path(previous["workspace"]) / "configs/deadline_core_protocol.json"),
        "predecessor_fingerprint": previous["fingerprint"],
        "corpus_predecessor_protocol": artifact(Path(original["workspace"]) / "configs/deadline_core_protocol.json"),
        "corpus_predecessor_fingerprint": original["fingerprint"],
        "normalized_precompute_output_root": str(Path(normalized_output_root).resolve()),
        "normalized_precompute_result_path": str(Path(normalized_output_root).resolve() / "result.json"),
        "normalized_bank_path": str(Path(normalized_bank_path).resolve())}
    amendment["offline_history"] = _offline_history(amendment, previous, failed)
    if target.exists():
        core = read_core(target)
        require(core["source_project"] == str(source) and core["normalization_amendment"] == amendment,
                "Existing normalized workspace uses another source or amendment")
        return core
    for path in (Path(normalized_output_root), Path(normalized_bank_path)):
        require(not path.exists(), "New normalized numerical outputs already exist; no implicit reuse/overwrite")
    core = copy.deepcopy(previous)
    core.update(registration=copy.deepcopy(NORMALIZED_REGISTRATION), study_id=NORMALIZED_REGISTRATION["study_id"],
        workspace=str(target), source_project=str(source), schema=SCHEMA, fit_recipe=NORMALIZED_RECIPE,
        normalization_amendment=amendment)
    target.mkdir(parents=True); copied = {}
    for name in ("src", "scripts"):
        originals = {str(p.relative_to(source)): file_sha256(p) for p in (source / name).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        shutil.copytree(source / name, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        require(all(file_sha256(source / p) == digest == file_sha256(target / p) for p, digest in originals.items()), "Source changed while freezing normalized workspace")
        copied.update(originals)
    old_root = Path(previous["workspace"])
    shutil.copytree(old_root / "configs", target / "configs", ignore=shutil.ignore_patterns("stage_queue.json", "deadline_core_protocol.json"))
    for name in ("data", "vendor", "environments"):
        require((old_root / name).exists(), "Missing verified shared input: " + name)
        (target / name).symlink_to((old_root / name).resolve(), target_is_directory=True)
    for name in ("experiments", "logs"): (target / name).mkdir()
    write_json_atomic(target / "configs/deadline_core_registration.json", NORMALIZED_REGISTRATION)
    write_json_atomic(target / "configs/normalization_amendment.json", amendment)
    main = _derived_main(read_json(target / "configs/full_study_main_protocol.json"), core)
    write_json_atomic(target / "configs/main_protocol.json", main)
    core["derived_main_protocol"] = artifact(target / "configs/main_protocol.json")
    core["collection_jobs"], core["final_jobs"] = collection_jobs(core), final_jobs(core)
    frozen = sorted(p for name in ("src", "scripts", "configs") for p in (target / name).rglob("*") if p.is_file())
    files = {str(p.relative_to(target)): file_sha256(p) for p in frozen}
    write_json_atomic(target / "source_manifest.json", {"schema": "deadline_core_source_snapshot_v1", "original_project": str(target),
        "parent_project": core["parent_project"], "source_project": str(source), "predecessor_project": str(old_root),
        "files": files, "source_fingerprint": fingerprint(files), "copied_source_inputs": copied})
    core["snapshot_files"] = [artifact(p) for p in frozen] + [artifact(target / "source_manifest.json")]
    evidence = list(previous["source_evidence"]) + [amendment[k] for k in inputs] + [amendment["predecessor_protocol"]]
    evidence += [artifact(old_root / "experiments/core_stage_receipts" / (stage + ".json")) for stage in ("import", "collect")]
    core["source_evidence"] = list({item["path"]: item for item in evidence}.values())
    core["fingerprint"] = core["core_fingerprint"] = _core_hash(core)
    write_json_atomic(target / "configs/deadline_core_protocol.json", core)
    return read_core(target)


def verify_reused_collection(core, job_id):
    previous = validate_normalized_registration(core)
    return _verify_reused_collection(core, job_id, previous, reuse_key="normalized_corpus_reuse")


def import_reused_collections(core, *, split):
    previous = validate_normalized_registration(core)
    return _import_reused_collections(core, split=split, previous=previous, reuse_key="normalized_corpus_reuse")
