"""One explicit pre-test TC16->TC64 amendment and exact closed-corpus reuse.

This module never starts training, loads a policy, or dispatches an oracle. V1
failed evidence remains immutable. New manifests describe a path-only derivation
of three completed training jobs and the one completed V1 development job;
physical job IDs, old completion receipts and source plan identities are kept.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil

from .accounting import file_sha256, fingerprint, write_json_atomic
from .collection_provenance import read_json
from .core_protocol import (REGISTRATION, TC64_REGISTRATION, SCHEMA, _core_hash,
    _derived_main, artifact, collection_jobs, collection_manifest_paths, final_jobs,
    read_core, require, verify_artifact)

AMENDMENT_SPEC = {
    "schema": "pretest_uniform_tc_epochs_amendment_v1",
    "from_epochs": 16, "to_epochs": 64, "layers": 32,
    "training": "restart_every_layer_from_original_seed_no_optimizer_resume",
    "selection": "minimum_mse_over_all_complete_development_epochs",
    "max_development_output_fvu": .5, "native_numerical_gates_unchanged": True,
    "model_chemical_splits_B50_ES_and_test_unchanged": True,
    "closed_training_jobs": 3, "closed_development_jobs": 1,
    "additional_collection_oracle_calls": 0,
    "development_informed_hyperparameter_amendment": True,
    "test_or_es_outcomes_used": False,
}


def _pretest(previous):
    root = Path(previous["workspace"])
    for name in ("core_es", "core_final"):
        path = root / "experiments" / name
        require(not path.exists() or not any(path.rglob("*")), "V1 ES/test execution exists; not a pre-test amendment")
    for stage in ("esopt", "final", "report"):
        require(not (root / "experiments/core_stage_receipts" / (stage + ".json")).exists(), "V1 later stage already completed")
    status = root / "logs/pipeline_status.json"
    if status.exists():
        require(not any(name in read_json(status).get("stages", {}) for name in ("core-esopt", "core-final", "core-report")),
                "V1 ES/test stage was already started")


def _failed_bank(path, previous, *, deep=True, expected_epochs=16):
    """Validate failed fidelity evidence without weakening the success loader."""
    from .representation_training import QWEN_MLP_PATHS, _bank_fingerprint, read_collections
    from .core_representation import transcoder_config
    from dataclasses import asdict
    require((expected_epochs, previous["registration"]) in ((16, REGISTRATION), (64, TC64_REGISTRATION)),
            "Failed bank epoch/registration pair is not an exact prior raw recipe")
    bank = read_json(path)
    require(bank.get("bank_fingerprint") == _bank_fingerprint(bank) and bank.get("complete") is True
        and bank.get("status") == "failed" and bank.get("ready_for_graphs") is False
        and bank.get("completed_layers") == bank.get("expected_layers") == 32
        and bank.get("expected_layer_paths") == list(QWEN_MLP_PATHS)
        and [row["layer_path"] for row in bank.get("layers", [])] == list(QWEN_MLP_PATHS)
        and bank.get("configuration") == asdict(transcoder_config(previous))
        and not bank.get("source_integrity_error"), "V1 must have a complete, unchanged 16-epoch failed-fidelity bank")
    failure = []
    for index, row in enumerate(bank["layers"]):
        meta = row.get("metadata", {}); history = meta.get("history", [])
        require(all(meta.get(key) == bank["configuration"][key] for key in ("epochs", "learning_rate", "batch_size", "max_dev_fvu")), "V1 layer hyperparameters differ from the registered bank")
        require(row["layer_index"] == index and row["status"] in {"succeeded", "failed_fidelity"}
            and meta.get("epochs") == expected_epochs and meta.get("seed") == 1729 + index
            and meta.get("max_dev_fvu") == .5 and len(history) == expected_epochs,
            "Failure evidence omitted layers/epochs or includes a non-fidelity failure")
        for epoch, item in enumerate(history):
            require(item.get("epoch") == epoch and item.get("train_rows") == meta["provenance"]["rows"]["train"]
                and item.get("dev", {}).get("rows") == meta["provenance"]["rows"]["dev"], "V1 training history is incomplete")
            require(all(type(value) in (float, int) and math.isfinite(value) and value >= 0
                for value in (item["train_output_mse"], item["dev"]["output_mse"], item["dev"]["output_fvu"])), "Invalid V1 reconstruction metrics")
        selected = min(range(expected_epochs), key=lambda epoch: history[epoch]["dev"]["output_mse"])
        measured = history[selected]["dev"]
        passed = measured["fvu_undefined"] == 0 and measured["output_fvu"] <= .5
        require(meta["selected_epoch"] == selected and meta["dev"] == measured
            and meta["fidelity_gate_passed"] is passed
            and row["status"] == ("succeeded" if passed else "failed_fidelity"), "V1 failure flag contradicts actual dev selection")
        require(measured["fvu_undefined"] == 0, "Undefined development FVU is not the registered epoch-budget failure")
        if not passed: failure.append(index)
        if deep:
            checkpoint = Path(row["checkpoint"])
            require(file_sha256(checkpoint) == row["checkpoint_sha256"]
                and file_sha256(str(checkpoint) + ".json") == row["metadata_sha256"]
                and read_json(str(checkpoint) + ".json") == meta, "V1 failed checkpoint or sidecar changed")
            from .transcoders import load_transcoder
            model, loaded = load_transcoder(checkpoint, expected_policy_fingerprint=bank["checkpoint_hash"], expected_layer_path=row["layer_path"])
            require(loaded == meta and model.checkpoint_hash() == row["transcoder_hash"], "V1 failed checkpoint state does not match metadata")
            del model
    require(failure and bank["passed_layers"] == 32 - len(failure), "V1 bank did not actually fail the fixed FVU gate")
    if deep:
        collection = read_collections(collection_manifest_paths(previous), model_key=previous["model_key"])
        require(bank["collections"] == collection["collections"] and bank["checkpoint_hash"] == collection["checkpoint_hash"]
            and bank["collection_fingerprint"] == collection["collection_fingerprint"], "V1 failed bank used another corpus")
        original_entries = {(item["job_id"], item["layer_path"]): item for item in collection["activation_shards"]}
        bank_entries = {(item["job_id"], item["layer_path"]): item for item in bank["activation_shards"]}
        require(len(bank_entries) == len(bank["activation_shards"]) == len(original_entries) and set(bank_entries) == set(original_entries), "V1 failed fit omitted collection shards")
        for key, original in original_entries.items():
            require(all(bank_entries[key].get(k) == v for k, v in original.items()), "V1 activation source identity differs")
        from .transcoders import TranscoderConfig, validate_shard_splits
        for layer in bank["layers"]:
            rows = [item for item in bank["activation_shards"] if item["layer_path"] == layer["layer_path"]]
            provenance = validate_shard_splits([item["fit_shard_path"] for item in rows if item["split"] == "train"],
                [item["fit_shard_path"] for item in rows if item["split"] == "dev" and item["fit_shard_path"]],
                policy_fingerprint=bank["checkpoint_hash"], layer_path=layer["layer_path"], config=TranscoderConfig(**layer["metadata"]["config"]))
            require(layer["metadata"]["provenance"] == provenance, "V1 layer did not use its complete declared shard provenance")
        for item in bank["activation_shards"] + collection["source_files"] + collection["collections"]:
            require(file_sha256(item["path"]) == item["sha256"], "V1 source bytes changed")
    return bank


def validate_tc64_registration(core):
    require(core["registration"] == TC64_REGISTRATION, "Unknown TC64 amendment")
    amendment = core.get("tc64_amendment", {})
    require(amendment.get("spec") == AMENDMENT_SPEC
        and read_json(Path(core["workspace"]) / "configs/tc64_amendment.json") == amendment, "TC64 scientific amendment changed")
    for key in ("predecessor_protocol", "failed_bank", "failure_reconciliation", "prior_tc16_precompute_result"):
        verify_artifact(amendment[key])
    previous = read_core(Path(amendment["predecessor_protocol"]["path"]).parent.parent)
    require(previous["registration"] == REGISTRATION and previous["fingerprint"] == amendment["predecessor_fingerprint"], "TC64 predecessor is not the exact V1 core")
    _pretest(previous)
    require([item["job"]["seed"] for item in previous["imported_train"]] == [1, 2, 3]
        and core["imported_train"] == previous["imported_train"], "TC64 must retain all three completed V1 training jobs")
    old_jobs = collection_jobs(previous)
    old_manifests = collection_manifest_paths(previous)
    expected = []
    for job, manifest in zip(old_jobs, old_manifests, strict=True):
        receipt = manifest.parent / ("import_receipt.json" if job["split"] == "train" else "completion.json")
        expected.append({"job": job, "predecessor_manifest": artifact(manifest), "predecessor_receipt": artifact(receipt)})
    require(core["reused_collections"] == expected and core["imported_development"] == expected[-1], "TC64 corpus changed or omitted a closed job")
    for key in ("model", "model_key", "train_tasks", "dev_tasks", "test_tasks", "training_seed", "budget", "methods",
                "graph", "esopt", "risk_network", "failure_control", "policy_runtime", "decoding", "memory", "made_execution", "deadline_utc"):
        require(core[key] == previous[key], "TC64 changed an unregistered scientific setting: " + key)
    require(core["transcoder"] == {**previous["transcoder"], "epochs": 64}, "Only the uniform transcoder epoch count may change")
    require(core["parent_project"] == previous["parent_project"], "TC64 changed original corpus root")
    failed = _failed_bank(amendment["failed_bank"]["path"], previous, deep=False)
    run = read_json(amendment["prior_tc16_precompute_result"]["path"])
    require(run.get("complete") is True and run["core_fingerprint"] == previous["fingerprint"]
        and run["layers"] == 32 and run["epochs_per_layer"] == 16 and len(run["layer_receipts"]) == 32,
        "V1 precomputation result is incomplete/different")
    require(amendment["prior_tc16_execution"] == [amendment["prior_tc16_precompute_result"]], "Prior run cost evidence differs")
    for index, item in enumerate(run["layer_receipts"]):
        verify_artifact(item)
        record = read_json(item["path"])
        require(record.get("schema") == "all_epoch_transcoder_precomputation_v1" and record.get("complete") is True
            and record.get("fingerprint") == fingerprint({k: v for k, v in record.items() if k != "fingerprint"})
            and record["contract"]["epochs"] == 16 and record["contract"]["seed"] == 1729 + index
            and len(record["epochs"]) == 16, "V1 precomputation layer is incomplete/different")
        require(all(row["epoch"] == epoch and row["train_rows"] == failed["layers"][index]["metadata"]["history"][epoch]["train_rows"]
            and row["train_output_mse"] == failed["layers"][index]["metadata"]["history"][epoch]["train_output_mse"]
            for epoch, row in enumerate(record["epochs"])), "V1 failed selection and training-epoch record differ")
        execution = failed["layers"][index]["metadata"].get("execution_provenance", {})
        require(execution.get("precomputation") == item, "V1 failed selection belongs to another epoch precomputation")
    stopped = read_json(amendment["failure_reconciliation"]["path"])
    require(stopped.get("status") == "failed_requires_reconciliation" and stopped.get("publisher_exited") is True
        and stopped.get("cleanup_errors") == [] and stopped.get("supervisor_halt_requested") is True,
        "V1 publisher/supervisor failure is not closed")
    future = Path(amendment["tc64_precompute_output_root"])
    require(future.is_absolute() and str(future / "result.json") == amendment["tc64_precompute_result_path"]
        and Path(previous["workspace"]) not in future.parents
        and future != Path(amendment["prior_tc16_precompute_result"]["path"]).parent, "TC64 future evidence path overwrites V1")
    return previous


def prepare_tc64_workspace(source_project, destination, *, predecessor_workspace,
                           failed_bank_path, failure_reconciliation_path,
                           prior_precompute_result, tc64_output_root):
    """Create a new immutable source/config snapshot; no data collection/training.

    The subsequent normal ``import`` and ``collect`` stages copy/verify existing
    closed V1 data and issue new derivation receipts. They dispatch zero physical
    calls. All 64 epochs must later be trained from the original seed.
    """
    from .core_collection import completed_core_collections, verify_stage_receipt
    previous = read_core(predecessor_workspace)
    require(previous["registration"] == REGISTRATION, "TC64 amendment is only from V1")
    _pretest(previous)
    verify_stage_receipt(previous, "import"); verify_stage_receipt(previous, "collect")
    old_manifests = completed_core_collections(previous)
    require(len(old_manifests) == 4, "TC64 requires exactly 3 closed train and 1 closed development jobs")
    _failed_bank(failed_bank_path, previous, deep=True)
    source, target = Path(source_project).resolve(), Path(destination).resolve()
    require(target != source and target != Path(previous["workspace"]) and target not in source.parents, "Invalid new TC64 workspace")
    if target.exists():
        core = read_core(target)
        amendment = core["tc64_amendment"]
        require(core["source_project"] == str(source)
            and amendment["predecessor_protocol"] == artifact(Path(previous["workspace"]) / "configs/deadline_core_protocol.json")
            and amendment["failed_bank"] == artifact(failed_bank_path)
            and amendment["failure_reconciliation"] == artifact(failure_reconciliation_path)
            and amendment["prior_tc16_precompute_result"] == artifact(prior_precompute_result)
            and amendment["tc64_precompute_output_root"] == str(Path(tc64_output_root).resolve()),
            "Existing TC64 workspace has another source/amendment/evidence registration")
        return core
    core = copy.deepcopy(previous)
    core.update(registration=copy.deepcopy(TC64_REGISTRATION), study_id=TC64_REGISTRATION["study_id"],
        workspace=str(target), source_project=str(source), schema=SCHEMA)
    core["reused_collections"] = [{"job": job, "predecessor_manifest": artifact(manifest),
        "predecessor_receipt": artifact(manifest.parent / ("import_receipt.json" if job["split"] == "train" else "completion.json"))}
        for job, manifest in zip(collection_jobs(previous), old_manifests, strict=True)]
    core["imported_development"] = core["reused_collections"][-1]
    prior = artifact(prior_precompute_result)
    core["tc64_amendment"] = {"spec": AMENDMENT_SPEC,
        "predecessor_protocol": artifact(Path(previous["workspace"]) / "configs/deadline_core_protocol.json"),
        "predecessor_fingerprint": previous["fingerprint"], "failed_bank": artifact(failed_bank_path),
        "failure_reconciliation": artifact(failure_reconciliation_path),
        "prior_tc16_precompute_result": prior, "prior_tc16_execution": [prior],
        "tc64_precompute_output_root": str(Path(tc64_output_root).resolve()),
        "tc64_precompute_result_path": str(Path(tc64_output_root).resolve() / "result.json")}
    target.mkdir(parents=True)
    copied = {}
    for name in ("src", "scripts"):
        originals = {str(p.relative_to(source)): file_sha256(p) for p in (source / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        shutil.copytree(source / name, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        require(all(file_sha256(source / name) == value == file_sha256(target / name) for name, value in originals.items()), "New source changed while copying")
        copied.update(originals)
    old_root = Path(previous["workspace"])
    shutil.copytree(old_root / "configs", target / "configs", ignore=shutil.ignore_patterns("stage_queue.json", "deadline_core_protocol.json"))
    for name in ("data", "vendor", "environments"):
        (target / name).symlink_to((old_root / name).resolve(), target_is_directory=True)
    for name in ("experiments", "logs"): (target / name).mkdir()
    write_json_atomic(target / "configs/deadline_core_registration.json", TC64_REGISTRATION)
    write_json_atomic(target / "configs/tc64_amendment.json", core["tc64_amendment"])
    main = _derived_main(read_json(target / "configs/full_study_main_protocol.json"), core)
    core["transcoder"] = main["transcoder"]
    write_json_atomic(target / "configs/main_protocol.json", main)
    core["derived_main_protocol"] = artifact(target / "configs/main_protocol.json")
    core["collection_jobs"], core["final_jobs"] = collection_jobs(core), final_jobs(core)
    frozen = [p for name in ("src", "scripts", "configs") for p in (target / name).rglob("*") if p.is_file()]
    files = {str(p.relative_to(target)): file_sha256(p) for p in sorted(frozen)}
    write_json_atomic(target / "source_manifest.json", {"schema": "deadline_core_source_snapshot_v1", "original_project": str(target),
        "parent_project": core["parent_project"], "source_project": str(source), "predecessor_project": str(old_root),
        "files": files, "source_fingerprint": fingerprint(files), "copied_source_inputs": copied})
    core["snapshot_files"] = [artifact(p) for p in sorted(frozen)] + [artifact(target / "source_manifest.json")]
    evidence = list(previous["source_evidence"]) + [core["tc64_amendment"][key] for key in
        ("predecessor_protocol", "failed_bank", "failure_reconciliation", "prior_tc16_precompute_result")]
    evidence += [artifact(old_root / "experiments/core_stage_receipts" / (stage + ".json")) for stage in ("import", "collect")]
    core["source_evidence"] = list({item["path"]: item for item in evidence}.values())
    core["fingerprint"] = core["core_fingerprint"] = _core_hash(core)
    write_json_atomic(target / "configs/deadline_core_protocol.json", core)
    return read_core(target)


def _reuse_spec(core, item, *, reuse_key="tc64_corpus_reuse", predecessor_fingerprint=None):
    from .core_collection import _derived_bytes
    source = Path(item["predecessor_manifest"]["path"]).parent
    target = Path(core["workspace"]) / "experiments/collection/qwen35_4b/made" / item["job"]["job_id"]
    manifest = read_json(item["predecessor_manifest"]["path"])
    shards = {Path(entry["path"]).resolve() for entry in manifest["activation_shards"]}
    mapping = []
    import hashlib
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.resolve() in shards or path.name in {"import_receipt.json", "completion.json", "collection_manifest.json", "graph_features.jsonl"}: continue
        output = target / path.relative_to(source)
        mapping.append({"source": artifact(path), "derived": {"path": str(output),
            "sha256": hashlib.sha256(_derived_bytes(path, source, target)).hexdigest()}})
    by_source = {item["source"]["path"]: item["derived"] for item in mapping}
    result = copy.deepcopy(manifest)
    result["decision_files"] = [by_source[str(Path(entry["path"]).resolve())] for entry in manifest["decision_files"]]
    if "core_import" in result: result["predecessor_core_import"] = result.pop("core_import")
    result[reuse_key] = {"core_fingerprint": core["fingerprint"], "predecessor_fingerprint": predecessor_fingerprint or core["tc64_amendment"]["predecessor_fingerprint"],
        "predecessor_manifest": item["predecessor_manifest"], "predecessor_receipt": item["predecessor_receipt"],
        "physical_job_identity_unchanged": True, "new_physical_calls": 0, "split": item["job"]["split"],
        "mapping_fingerprint": fingerprint(mapping), "activation_files_remain_read_only_original_references": True}
    return target, result, mapping


def verify_reused_collection(core, job_id):
    previous = validate_tc64_registration(core)
    return _verify_reused_collection(core, job_id, previous)


def _verify_reused_collection(core, job_id, previous, *, reuse_key="tc64_corpus_reuse"):
    from .core_collection import verify_import, verify_development
    item = next(item for item in core["reused_collections"] if item["job"]["job_id"] == job_id)
    proof = (verify_import(previous, next(x for x in previous["imported_train"] if x["job"]["job_id"] == job_id))
             if item["job"]["split"] == "train" else verify_development(previous))
    target, manifest, mapping = _reuse_spec(core, item, reuse_key=reuse_key, predecessor_fingerprint=previous["fingerprint"])
    receipt = read_json(target / "reuse_receipt.json")
    require(receipt.get("complete") is True and receipt["core_fingerprint"] == core["fingerprint"]
        and receipt["fingerprint"] == fingerprint({k: v for k, v in receipt.items() if k != "fingerprint"})
        and receipt["source"] == item and receipt["mapping"] == mapping, "Incomplete or changed TC64 derivation receipt")
    require(read_json(target / "collection_manifest.json") == manifest and receipt["manifest"] == artifact(target / "collection_manifest.json"), "TC64 derived manifest changed")
    expected = {target / "reuse_receipt.json", target / "collection_manifest.json"}
    for row in mapping:
        verify_artifact(row["derived"]); expected.add(Path(row["derived"]["path"]))
    require({p for p in target.rglob("*") if p.is_file() and p != target / "graph_features.jsonl"} == expected, "Unknown/partial reused corpus files")
    proof["evidence_files"] += [artifact(p) for p in sorted(expected)]
    proof["derived_manifest"] = artifact(target / "collection_manifest.json")
    return proof


def import_reused_collections(core, *, split):
    previous = validate_tc64_registration(core)
    return _import_reused_collections(core, split=split, previous=previous)


def _import_reused_collections(core, *, split, previous, reuse_key="tc64_corpus_reuse"):
    from .core_collection import _derived_bytes, publish_stage_receipt, verify_stage_receipt
    require(split in {"train", "dev"}, "Only the registered closed train/dev corpus can be reused")
    if split == "dev": verify_stage_receipt(core, "import")
    files = []
    for item in core["reused_collections"]:
        if item["job"]["split"] != split: continue
        from .core_collection import verify_import, verify_development
        if split == "train": verify_import(previous, next(x for x in previous["imported_train"] if x["job"]["job_id"] == item["job"]["job_id"]))
        else: verify_development(previous)
        target, manifest, mapping = _reuse_spec(core, item, reuse_key=reuse_key, predecessor_fingerprint=previous["fingerprint"])
        if not target.exists():
            target.mkdir(parents=True)
            source = Path(item["predecessor_manifest"]["path"]).parent
            for row in mapping:
                path = Path(row["derived"]["path"]); path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(_derived_bytes(Path(row["source"]["path"]), source, target))
            write_json_atomic(target / "collection_manifest.json", manifest)
            receipt = {"schema": "closed_core_v1_corpus_reuse_v1", "complete": True, "core_fingerprint": core["fingerprint"],
                "source": item, "mapping": mapping, "manifest": artifact(target / "collection_manifest.json"), "new_physical_calls": 0}
            receipt["fingerprint"] = fingerprint(receipt)
            write_json_atomic(target / "reuse_receipt.json", receipt)
        require((target / "reuse_receipt.json").exists(), "Partial TC64 copy needs reconciliation; do not overwrite")
        _verify_reused_collection(core, item["job"]["job_id"], previous, reuse_key=reuse_key)
        files.append(target / "reuse_receipt.json")
    return publish_stage_receipt(core, "import" if split == "train" else "collect", files,
        collection_jobs=len(files), reused_closed_corpus=True, new_physical_calls=0,
        predecessor_fingerprint=previous["fingerprint"], candidate_oracle_attempts=50 * len(files),
        candidate_oracle_attempts_are_historical=True)
