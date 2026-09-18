"""Explicit deferred-epoch provider for the original full-bank coordinator.

Run in a dedicated publisher process. Only its in-memory trainer binding is
temporarily adapted; frozen files, core sources and original numeric functions
remain unchanged. The original coordinator performs all collection admission,
exact-prefix filtering, order, layer accounting and fidelity gates. Restoring
the original trainer followed by strict resume proves that the exported bank
is complete and read-verifiable without another optimizer step.
"""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import torch

from .accounting import file_sha256, fingerprint, write_json_atomic
from .collection_provenance import read_json
from .deferred_dev_transcoders import _verify_complete, _runtime, finalize_development
from .representation_training import (QWEN_MLP_PATHS, TranscoderStageConfig,
    _bank_fingerprint, train_transcoders_from_collections)


def require(value, message):
    if not value:
        raise ValueError(message)


def _inventory(directory):
    return {str(path.relative_to(directory)): {"sha256": file_sha256(path), "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def _files(paths):
    return [{"path": str(Path(path).resolve()), "sha256": file_sha256(path)} for path in paths]


def _selection_runtime(expected):
    torch.set_num_threads(expected["threads"])
    torch.set_float32_matmul_precision(expected["matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = expected["cuda_tf32"]
    torch.backends.cudnn.allow_tf32 = expected["cudnn_tf32"]
    torch.use_deterministic_algorithms(expected["deterministic"])
    require(_runtime() == expected, "Spawned development runtime differs")


def _selection_worker(task):
    """Only the original full-development selector performs numerical work."""
    started = time.monotonic()
    c = task["precompute_contract"]
    _selection_runtime(c["runtime"])
    if c["device"].startswith("cuda"):
        total = torch.cuda.get_device_properties(c["device"]).total_memory
        torch.cuda.set_per_process_memory_fraction(c["gpu_memory_bytes"] / total, c["device"])
        torch.cuda.reset_peak_memory_stats(c["device"])
    require(file_sha256(task["precomputation"]["path"]) == task["precomputation"]["sha256"], "Precomputation receipt changed")
    output = Path(task["output"])
    model, metadata = finalize_development(Path(task["precomputation"]["path"]).parent,
                                           [x["path"] for x in task["development_files"]], output_path=output)
    require(_files([x["path"] for x in task["development_files"]]) == task["development_files"], "Development bytes changed")
    require(metadata["transcoder_hash"] == model.checkpoint_hash(), "Selected tensor hash differs")
    result = {"schema": "parallel_development_selection_layer_v1", "complete": True,
        "task": task, "checkpoint": {"path": str(output), "sha256": file_sha256(output)},
        "sidecar": {"path": str(output) + ".json", "sha256": file_sha256(str(output) + ".json")},
        "transcoder_hash": metadata["transcoder_hash"], "pid": os.getpid(), "runtime": _runtime(),
        "elapsed_seconds": time.monotonic() - started, "optimizer_steps": 0,
        "epochs_scored": len(metadata["history"]), "fidelity_gate_passed": metadata["fidelity_gate_passed"],
        "gpu_memory_cap_bytes": c["gpu_memory_bytes"]}
    if c["device"].startswith("cuda"):
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(c["device"])
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(c["device"])
        require(result["peak_reserved_bytes"] <= c["gpu_memory_bytes"], "Selection exceeded its fixed allocator cap")
    result["fingerprint"] = fingerprint(result)
    write_json_atomic(output.with_suffix(".selection.json"), result)
    return {"path": str(output.with_suffix(".selection.json")), "sha256": file_sha256(output.with_suffix(".selection.json"))}


def _verify_selection(item, *, verify_precomputed=False):
    """Read real cached weights; explicit resume can also rehash every epoch."""
    from .transcoders import load_transcoder, validate_shard_splits, TranscoderConfig
    require(file_sha256(item["path"]) == item["sha256"], "Selection receipt changed")
    record = read_json(item["path"])
    require(record.get("complete") is True and record.get("schema") == "parallel_development_selection_layer_v1"
        and record["fingerprint"] == fingerprint({k: v for k, v in record.items() if k != "fingerprint"}), "Invalid selection receipt")
    task = record["task"]; c = task["precompute_contract"]
    pre = task["precomputation"]
    require(file_sha256(pre["path"]) == pre["sha256"], "Precomputation receipt changed after selection")
    require(read_json(pre["path"])["contract"] == c, "Selected precomputation contract differs")
    if verify_precomputed:
        _verify_complete(Path(pre["path"]).parent, c)
    require(c["source"] == {name: file_sha256(Path(__file__).with_name(name)) for name in
        ("deferred_dev_transcoders.py", "transcoders.py", "esopt.py")}, "Selected source code changed")
    require(_files(c["train_paths"]) == c["training"]["files"]
        and _files([x["path"] for x in task["development_files"]]) == task["development_files"], "Selection train/development bytes changed")
    for entry in (record["checkpoint"], record["sidecar"]):
        require(file_sha256(entry["path"]) == entry["sha256"], "Selection checkpoint/sidecar changed")
    model, metadata = load_transcoder(record["checkpoint"]["path"], expected_policy_fingerprint=c["policy_fingerprint"], expected_layer_path=c["layer_path"])
    require(metadata == read_json(record["sidecar"]["path"]) and metadata["transcoder_hash"] == record["transcoder_hash"], "Selected metadata/tensor differs")
    provenance = validate_shard_splits(c["train_paths"], [x["path"] for x in task["development_files"]],
        policy_fingerprint=c["policy_fingerprint"], layer_path=c["layer_path"], config=TranscoderConfig(**c["config"]))
    require(metadata["provenance"] == provenance and len(metadata["history"]) == record["epochs_scored"] == c["epochs"], "Selection omitted source rows/epochs")
    for key in ("config", "seed", "epochs", "batch_size", "learning_rate", "max_dev_fvu"):
        require(metadata[key] == c[key], "Selection training settings changed")
    complete = read_json(pre["path"])
    for i, row in enumerate(metadata["history"]):
        require(row["epoch"] == i and row["train_rows"] == provenance["rows"]["train"]
            and row["dev"]["rows"] == provenance["rows"]["dev"]
            and row["train_output_mse"] == complete["epochs"][i]["train_output_mse"], "Selection history changed")
    best = min(range(c["epochs"]), key=lambda i: metadata["history"][i]["dev"]["output_mse"])
    require(metadata["selected_epoch"] == best and metadata["dev"] == metadata["history"][best]["dev"]
        and metadata["transcoder_hash"] == complete["epochs"][best]["state_hash"], "Selection is not the original dev-minimum epoch")
    require(record["runtime"] == c["runtime"] and record["optimizer_steps"] == 0, "Selection execution changed")
    return model, metadata, record


def _selection_verify_worker(item):
    _selection_runtime(read_json(item["path"])["task"]["precompute_contract"]["runtime"])
    _verify_selection(item, verify_precomputed=True)
    return True


def _parallel_selection(bank_path, precomputed_root, workers, expected_registration):
    """Consume the original coordinator's complete, already-filtered manifest."""
    bank_path = Path(bank_path).resolve(); bank = read_json(bank_path)
    require(bank["bank_fingerprint"] == _bank_fingerprint(bank) and bank["expected_layer_paths"] == list(QWEN_MLP_PATHS), "Coordinator manifest changed")
    cache = bank_path.parent.with_name(bank_path.parent.name + ".selection_cache")
    require(not cache.exists(), "Selection cache already exists/partial; never silently reselect")
    tasks = []
    for index, layer in enumerate(QWEN_MLP_PATHS):
        folder = Path(precomputed_root).resolve() / f"layer_{index:02d}"
        receipt = folder / "complete.json"; complete = read_json(receipt); c = complete["contract"]
        require(complete["complete"] is True and complete["fingerprint"] == fingerprint({k: v for k, v in complete.items() if k != "fingerprint"}), "Invalid precomputation receipt")
        entries = [x for x in bank["activation_shards"] if x["layer_path"] == layer]
        train = [x["fit_shard_path"] for x in entries if x["split"] == "train"]
        dev = [x["fit_shard_path"] for x in entries if x["split"] == "dev" and x["fit_shard_path"] is not None]
        require(train == c["train_paths"] and dev and c["layer_path"] == layer and c["policy_fingerprint"] == bank["checkpoint_hash"], "Coordinator and precomputed source paths differ")
        require(not bank["development_exact_prefix_filter"]["empty_groups_by_layer"][layer], "A development group is empty after original filtering")
        cfg = bank["configuration"]
        require(all(c[k] == cfg[k] for k in ("epochs", "batch_size", "learning_rate", "device", "max_dev_fvu"))
            and c["seed"] == cfg["seed"] + index and c["config"]["top_k"] == cfg["top_k"]
            and c["config"]["feature_dim"] == (cfg["feature_dim"] or cfg["feature_multiplier"] * c["config"]["input_dim"]), "Selection configuration differs from original coordinator")
        require(c["runtime"] == _runtime(), "Parallel selection runtime differs")
        require(c["gpu_memory_bytes"] == 2 * 1024**3, "Selection requires the registered 2GiB allocator cap")
        if expected_registration is not None:
            require(c["registration"] == expected_registration[index], "Selection uses another core registration")
        tasks.append({"layer_index": index, "precomputation": {"path": str(receipt), "sha256": file_sha256(receipt)},
            "precompute_contract": c, "development_files": _files(dev), "output": str(cache / f"layer_{index:02d}.pt")})
    require(type(workers) is int and 1 < workers <= 4, "Parallel development requires two to four workers")
    devices = {x["precompute_contract"]["device"] for x in tasks}; require(len(devices) == 1, "Mixed selection devices")
    device = next(iter(devices))
    if device.startswith("cuda"):
        require(all(task["precompute_contract"]["runtime"]["threads"] == 2
            and task["precompute_contract"]["runtime"]["matmul_precision"] == "highest"
            and task["precompute_contract"]["runtime"]["cuda_tf32"] is False
            and task["precompute_contract"]["runtime"]["cudnn_tf32"] is False for task in tasks), "CUDA selection requires the fixed FP32/TF32-false/two-thread runtime")
        free, _ = torch.cuda.mem_get_info(device)
        require(free >= workers * (3 * 1024**3) + 4 * 1024**3, "Insufficient observed headroom for parallel development")
    cache.mkdir(parents=True)
    started = {"schema": "parallel_development_selection_v1", "workers": workers, "tasks": tasks,
               "provider_source_sha256": file_sha256(__file__), "started_at": time.time()}
    write_json_atomic(cache / "started.json", started)
    # A worker exception joins the pool; already dispatched layers are preserved.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        receipts = list(pool.map(_selection_worker, tasks))
    record = {**started, "complete": True, "finished_at": time.time(), "layer_receipts": receipts}
    record["fingerprint"] = fingerprint(record)
    write_json_atomic(cache / "complete.json", record)
    return {"path": str(cache / "complete.json"), "sha256": file_sha256(cache / "complete.json")}, receipts


@contextmanager
def _provider(precomputed_root, *, expected_registration=None, development_workers=1):
    """Process-local, explicitly audited adapter; never a fabricated fit result."""
    from . import transcoders
    original = transcoders.train_layer_transcoder
    invoked = []
    selection, selected = None, None
    selection_failed = False

    def deferred(config, train_paths, dev_paths, *, policy_fingerprint, layer_path,
                 output_path, seed=0, epochs=4, batch_size=256,
                 learning_rate=1e-3, device="cpu", max_dev_fvu=None):
        nonlocal selection, selected, selection_failed
        index = QWEN_MLP_PATHS.index(layer_path)
        folder = Path(precomputed_root).resolve() / f"layer_{index:02d}"
        if development_workers > 1:
            require(not selection_failed, "Parallel selection already failed; preserve all partial outputs")
            if selected is None:
                try:
                    selection, selected = _parallel_selection(Path(output_path).parent / "transcoder_manifest.json", precomputed_root,
                                                              development_workers, expected_registration)
                except BaseException:
                    selection_failed = True
                    raise
            model, metadata, cached = _verify_selection(selected[index])
            complete = read_json(folder / "complete.json")
        else:
            complete = _verify_complete(folder)
        c = complete["contract"]
        expected = {"config": asdict(config), "train_paths": [str(Path(p).resolve()) for p in train_paths],
            "policy_fingerprint": policy_fingerprint, "layer_path": layer_path, "seed": seed,
            "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
            "device": str(device), "max_dev_fvu": max_dev_fvu}
        require(all(c.get(key) == value for key, value in expected.items()), "Precomputation differs from original full-corpus trainer call")
        require(c["runtime"] == _runtime(), "Publisher runtime differs from precomputation")
        if expected_registration is not None:
            require(c["registration"] == expected_registration[index], "Precomputation uses another core/source registration")
        require(index not in invoked, "A layer provider was called twice")
        if development_workers > 1:
            require(cached["task"]["development_files"] == _files(dev_paths), "Cached development does not match this original trainer call")
        else:
            model, metadata = finalize_development(folder, dev_paths, output_path=output_path)
        evidence = {"schema": "deferred_dev_transcoder_execution_v1",
            "execution": "all_training_epochs_precomputed_then_complete_filtered_development_selection",
            "original_serial_execution_claimed": False,
            "precomputation": {"path": str(folder / "complete.json"), "sha256": file_sha256(folder / "complete.json")},
            "precomputation_fingerprint": complete["fingerprint"],
            "provider_source": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(__file__)},
            "precomputation_source": c["source"], "registration": c["registration"],
            "epochs_scored": epochs, "selected_state_hash": complete["epochs"][metadata["selected_epoch"]]["state_hash"],
            "development_files": [{"path": str(Path(path).resolve()), "sha256": file_sha256(path)} for path in dev_paths],
            "original_training_call": expected,
            "runtime": c["runtime"], "test_data_used": False}
        if development_workers > 1:
            evidence["parallel_selection"] = {"execution": "spawned_original_finalize_development",
                "workers": development_workers, "cache_manifest": selection, "layer_receipt": selected[index]}
        metadata["execution_provenance"] = evidence
        # Preserve original checkpoint schema/tensor values, adding truthful
        # provenance inside metadata. The original strict loader permits this
        # extra field and still checks all fixed mathematical/history fields.
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save({"state_dict": state, "metadata": metadata}, output_path)
        Path(str(output_path) + ".json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
        invoked.append(index)
        return model, metadata

    transcoders.train_layer_transcoder = deferred
    try:
        yield invoked
    finally:
        transcoders.train_layer_transcoder = original


def _verify_execution_provenance(bank, precomputed_root, expected_registration, *, verify_precomputed=True):
    parallel = [layer["metadata"].get("execution_provenance", {}).get("parallel_selection") for layer in bank["layers"]]
    if any(parallel):
        require(all(parallel) and len(parallel) == 32, "Mixed or incomplete parallel selection provenance")
        manifests = {json.dumps(item["cache_manifest"], sort_keys=True) for item in parallel}
        workers = {item["workers"] for item in parallel}
        require(len(manifests) == len(workers) == 1 and 1 < next(iter(workers)) <= 4, "Selection cache/workers changed")
        cache_item = parallel[0]["cache_manifest"]
        require(file_sha256(cache_item["path"]) == cache_item["sha256"], "Selection manifest changed")
        cache = read_json(cache_item["path"])
        require(cache["complete"] is True and cache["fingerprint"] == fingerprint({k: v for k, v in cache.items() if k != "fingerprint"})
            and cache["provider_source_sha256"] == file_sha256(__file__)
            and cache["layer_receipts"] == [item["layer_receipt"] for item in parallel], "Selection cache inventory differs")
        directory = Path(cache_item["path"]).parent
        expected_files = {directory / name for name in ("started.json", "complete.json")}
        for index in range(32):
            expected_files.update(directory / f"layer_{index:02d}{suffix}" for suffix in (".pt", ".pt.json", ".selection.json"))
        require({p for p in directory.rglob("*") if p.is_file()} == expected_files, "Selection cache has partial/missing/unknown files")
        require(read_json(directory / "started.json") == {k: v for k, v in cache.items()
            if k not in {"complete", "finished_at", "layer_receipts", "fingerprint"}}, "Selection start/complete contract differs")
        if verify_precomputed:
            with ProcessPoolExecutor(max_workers=next(iter(workers)), mp_context=multiprocessing.get_context("spawn")) as pool:
                require(all(pool.map(_selection_verify_worker, cache["layer_receipts"])), "Parallel source verification failed")
    for index, layer in enumerate(bank["layers"]):
        evidence = layer["metadata"].get("execution_provenance", {})
        folder = Path(precomputed_root).resolve() / f"layer_{index:02d}"
        receipt = folder / "complete.json"
        require(evidence.get("original_serial_execution_claimed") is False
            and evidence.get("provider_source", {}).get("sha256") == file_sha256(__file__)
            and evidence.get("precomputation") == {"path": str(receipt), "sha256": file_sha256(receipt)},
            "Deferred execution provenance was omitted/changed")
        if parallel[index]:
            _, cached_metadata, selected = _verify_selection(parallel[index]["layer_receipt"])
            require(selected["task"] == cache["tasks"][index]
                and cached_metadata == {k: v for k, v in layer["metadata"].items() if k != "execution_provenance"}, "Bank does not contain its selected cached model")
            complete = read_json(folder / "complete.json")
        else:
            complete = _verify_complete(folder)
        require(complete["fingerprint"] == evidence["precomputation_fingerprint"]
            and complete["contract"]["registration"] == evidence["registration"], "Precomputation source identity differs")
        if expected_registration is not None:
            require(evidence["registration"] == expected_registration[index], "Existing bank uses another core registration")
        for item in evidence["development_files"]:
            require(file_sha256(item["path"]) == item["sha256"], "Actual development selection source changed")


def build_external_bank(manifest_paths, precomputed_root, output_dir, *, model_key,
                        config: TranscoderStageConfig, cpu_threads=2, expected_registration=None,
                        development_workers=1):
    """Use the untouched coordinator, then original strict read-only resume."""
    torch.set_num_threads(cpu_threads)
    require(type(development_workers) is int and 1 <= development_workers <= 4, "Development workers must be one to four")
    output = Path(output_dir).resolve()
    lock_path = output.with_name(output.name + ".publisher.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = (output / "transcoder_manifest.json").exists()
        if existing:
            # Existing complete banks can only take the original recovery path;
            # a partial bank cannot trigger another deferred selection silently.
            require(read_json(output / "transcoder_manifest.json").get("complete") is True,
                    "Partial publisher bank requires reconciliation")
            invoked = []
            bank = train_transcoders_from_collections(manifest_paths, output, model_key=model_key, config=config, resume=True)
        else:
            require(not output.exists() or not any(output.iterdir()), "Unknown publisher output requires reconciliation")
            with _provider(precomputed_root, expected_registration=expected_registration, development_workers=development_workers) as invoked:
                bank = train_transcoders_from_collections(manifest_paths, output, model_key=model_key, config=config, resume=True)
            require(invoked == list(range(32)), "Provider did not consume all 32 independent layers")
        require(bank["status"] == "succeeded" and bank["ready_for_graphs"] is True,
                "Deferred bank failed an original fidelity/integrity gate")
        # Fresh selectors already verified every epoch inside finalize_development.
        # Later reuse rechecks their full sources concurrently, never per provider.
        _verify_execution_provenance(bank, precomputed_root, expected_registration, verify_precomputed=existing)
        before = _inventory(output)
        # The real original trainer is restored here. Complete-bank recovery
        # must return before entering its training loop and write nothing.
        checked = train_transcoders_from_collections(manifest_paths, output, model_key=model_key, config=config, resume=True)
        require(checked == bank and _inventory(output) == before, "Original resume changed the completed bank")
        proof = {"schema": "deferred_bank_original_resume_validation_v1", "complete": True,
            "bank_fingerprint": bank["bank_fingerprint"], "all_layers": 32,
            "original_resume_verified": True, "all_file_hashes_and_mtimes_unchanged": True,
            "provider_source_sha256": file_sha256(__file__), "files": before}
        proof["fingerprint"] = fingerprint(proof)
        proof_path = output.with_name(output.name + ".resume_validation.json")
        if proof_path.exists():
            require(read_json(proof_path) == proof, "External bank validation receipt changed")
        else:
            write_json_atomic(proof_path, proof)
        return bank


def install_verified_bank(manifest_paths, external_bank, destination, *, model_key, config):
    """Copy verified bank data, rebase only derived paths, then strict resume.

    Source bank/provenance remain outside CORE. This operation changes no source
    code or protocol; call only after pausing the supervising stage boundary.
    Unknown destinations or interrupted copies are not overwritten.
    """
    source, target = Path(external_bank).resolve(), Path(destination).resolve()
    proof = read_json(source.with_name(source.name + ".resume_validation.json"))
    require(proof["fingerprint"] == fingerprint({k: v for k, v in proof.items() if k != "fingerprint"})
        and proof["bank_fingerprint"] == read_json(source / "transcoder_manifest.json")["bank_fingerprint"]
        and proof["complete"] is True and proof["original_resume_verified"] is True
        and proof["files"] == _inventory(source), "External bank changed after original-resume validation")
    source_bank = read_json(source / "transcoder_manifest.json")
    precomputed_root = Path(source_bank["layers"][0]["metadata"]["execution_provenance"]["precomputation"]["path"]).parent.parent
    _verify_execution_provenance(source_bank, precomputed_root, None)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name(target.name + ".import.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not target.exists(), "Destination bank already exists; explicit reconciliation required")
        pending = target.with_name(target.name + ".deferred_import_pending")
        require(not pending.exists(), "Partial import exists; explicit reconciliation required")
        shutil.copytree(source, pending)
        bank = read_json(pending / "transcoder_manifest.json")
        def rebase(value):
            path = Path(value) if value is not None else None
            return str(target / path.relative_to(source)) if path is not None and source in path.parents else value
        for layer in bank["layers"]:
            layer["checkpoint"] = rebase(layer["checkpoint"])
        for shard in bank["activation_shards"]:
            shard["fit_shard_path"] = rebase(shard["fit_shard_path"])
        for audit in bank["development_exact_prefix_filter"]["dev_shard_audits"]:
            if "filtered_path" in audit:
                audit["filtered_path"] = rebase(audit["filtered_path"])
        bank["bank_fingerprint"] = _bank_fingerprint(bank)
        write_json_atomic(pending / "transcoder_manifest.json", bank)
        for relative, item in proof["files"].items():
            if relative != "transcoder_manifest.json":
                require(file_sha256(pending / relative) == item["sha256"], "Copied bank artifact changed")
        os.rename(pending, target)
        before = _inventory(target)
        checked = train_transcoders_from_collections(manifest_paths, target, model_key=model_key, config=config, resume=True)
        require(checked == bank and before == _inventory(target), "Imported original-resume verification changed bank bytes")
        receipt = {"schema": "deferred_bank_import_v1", "complete": True,
            "external_validation": {"path": str(source.with_name(source.name + ".resume_validation.json")),
                                    "sha256": file_sha256(source.with_name(source.name + ".resume_validation.json"))},
            "bank": {"path": str(target / "transcoder_manifest.json"), "sha256": file_sha256(target / "transcoder_manifest.json")},
            "bank_fingerprint": bank["bank_fingerprint"], "original_resume_verified": True,
            "all_file_hashes_and_mtimes_unchanged_on_resume": True, "files": before}
        receipt["fingerprint"] = fingerprint(receipt)
        write_json_atomic(target.with_name(target.name + ".deferred_import.json"), receipt)
        return receipt


def publish_core_precomputed_bank(core, precomputed_root, external_bank, *, install=False, development_workers=1):
    from .core_protocol import read_core
    from .core_collection import completed_core_collections
    from .core_representation import transcoder_config, paths_for
    from .deferred_dev_transcoders import prepare_core_training_tasks
    require(read_core(core["workspace"]) == core, "Core/source registration changed")
    external = Path(external_bank).resolve()
    require(Path(core["workspace"]) not in external.parents, "Build the bank outside the frozen core first")
    manifests = completed_core_collections(core)
    tasks = prepare_core_training_tasks(core, precomputed_root)
    config = transcoder_config(core)
    bank = build_external_bank(manifests, precomputed_root, external, model_key=core["model_key"], config=config,
        cpu_threads=core["policy_runtime"]["torch_cpu_threads"], expected_registration=[task["kwargs"]["registration"] for task in tasks],
        development_workers=development_workers)
    require(read_core(core["workspace"]) == core, "Core source changed during bank construction")
    if install:
        return install_verified_bank(manifests, external, paths_for(core)["bank"], model_key=core["model_key"], config=config)
    return bank
