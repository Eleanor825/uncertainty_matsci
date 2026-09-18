"""External train-only precomputation; exact development selection is deferred.

This module never modifies a frozen CORE directory or loads a policy/oracle.
The original TopK network contains only Linear/ReLU/topk/scatter: eval mode has
no stateful effect. Development metrics do not update optimizer/RNG/parameters.
Consequently all epochs can be computed first, retaining every epoch's exact
weights. Once complete development data exist, original reconstruction_metrics
and minimum-MSE selection recreate the original trainer's metadata/checkpoint.
No early stopping, different seed, sample cap, or alternate fidelity gate exists.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import fcntl
import json
import math
import multiprocessing
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .accounting import file_sha256, fingerprint, write_json_atomic
from .collection_provenance import read_json
from .esopt import tensor_state_hash
from .transcoders import (TranscoderConfig, TopKTranscoder, _load_shard,
    iter_activation_batches, reconstruction_metrics, validate_shard_splits)

SCHEMA = "all_epoch_transcoder_precomputation_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _files(paths):
    return [{"path": str(Path(p).resolve()), "sha256": file_sha256(p)} for p in paths]


def _runtime():
    return {"torch": str(torch.__version__), "threads": torch.get_num_threads(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic": torch.are_deterministic_algorithms_enabled()}


def _training_provenance(paths, config, policy, layer):
    require(paths, "Training shards are required")
    rows, identities = 0, []
    for path in paths:
        shard = _load_shard(path); meta = shard["metadata"]
        require(meta["split"] == "train" and meta["policy_fingerprint"] == policy and meta["layer_path"] == layer,
                "Precomputation accepts only the registered training layer")
        require((meta["input_dim"], meta["output_dim"]) == (config.input_dim, config.output_dim)
            and shard["inputs"].shape == (meta["rows"], config.input_dim)
            and shard["outputs"].shape == (meta["rows"], config.output_dim)
            and len(meta["group_ids"]) == len(meta["prefix_hashes"]) == meta["rows"], "Training shape/row provenance differs")
        rows += meta["rows"]; identities.append(meta)
    return {"rows": rows, "shards": identities, "files": _files(paths)}


def _verify_complete(directory, contract=None):
    directory = Path(directory)
    record = read_json(directory / "complete.json")
    require(record.get("schema") == SCHEMA and record.get("complete") is True
            and record["fingerprint"] == fingerprint({k: v for k, v in record.items() if k != "fingerprint"}), "Invalid precomputation completion")
    actual = record["contract"]
    require(contract is None or actual == contract, "Precomputation config/source/runtime changed")
    require(actual["training"]["files"] == _files(actual["train_paths"]), "Training bytes changed after precomputation")
    require(actual["source"] == {name: file_sha256(Path(__file__).with_name(name)) for name in
        ("deferred_dev_transcoders.py", "transcoders.py", "esopt.py")}, "Precomputation code changed")
    require(len(record["epochs"]) == actual["epochs"], "Missing precomputed epoch")
    expected = {directory / "complete.json", directory / "started.json", directory / ".lock"}
    for index, epoch in enumerate(record["epochs"]):
        path = directory / f"epoch_{index:03d}.pt"; expected.add(path)
        require(epoch["epoch"] == index and epoch["train_rows"] == actual["training"]["rows"]
                and epoch["path"] == str(path) and epoch["sha256"] == file_sha256(path), "Epoch bytes/order/rows changed")
        state = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        require(tensor_state_hash(state) == epoch["state_hash"], "Epoch tensor state changed")
    require({p for p in directory.iterdir() if p.is_file()} == expected, "Unknown or partial precomputation files")
    require(read_json(directory / "started.json") == actual, "Started/complete training contract differs")
    return record


def precompute_layer(config, train_paths, output_dir, *, policy_fingerprint, layer_path,
                     seed=0, epochs=16, batch_size=256, learning_rate=4e-4,
                     device="cuda:0", max_dev_fvu=.5, cpu_threads=2, gpu_memory_bytes=2 * 1024**3,
                     registration=None):
    """Save every epoch with original SGD order; no development data is read."""
    config.validate()
    require(epochs > 0 and batch_size > 0 and math.isfinite(learning_rate) and learning_rate > 0, "Invalid fixed training settings")
    torch.set_num_threads(cpu_threads)
    if str(device).startswith("cuda"):
        require(torch.get_float32_matmul_precision() == "highest" and torch.backends.cuda.matmul.allow_tf32 is False,
                "Precomputation requires unchanged highest/TF32-false FP32 runtime")
    paths = [str(Path(p).resolve()) for p in train_paths]
    training = _training_provenance(paths, config, policy_fingerprint, layer_path)
    contract = {"config": asdict(config), "train_paths": paths, "training": training,
        "policy_fingerprint": policy_fingerprint, "layer_path": layer_path, "seed": seed, "epochs": epochs,
        "batch_size": batch_size, "learning_rate": learning_rate, "device": str(device), "max_dev_fvu": max_dev_fvu,
        "runtime": _runtime(), "registration": registration, "gpu_memory_bytes": gpu_memory_bytes,
        "source": {name: file_sha256(Path(__file__).with_name(name)) for name in
                   ("deferred_dev_transcoders.py", "transcoders.py", "esopt.py")}}
    directory = Path(output_dir).resolve(); directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "complete.json").exists():
            return _verify_complete(directory, contract)
        require({p.name for p in directory.iterdir()} == {".lock"}, "In-flight/partial precomputation requires reconciliation; never retrain")
        write_json_atomic(directory / "started.json", contract)
        if str(device).startswith("cuda"):
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(gpu_memory_bytes / total, device)
        transcoder = TopKTranscoder(config, seed=seed).to(device)
        require(all(type(module) in {TopKTranscoder, nn.Linear} for module in transcoder.modules()),
                "Deferred validation requires the reviewed stateless TopK/Linear architecture")
        optimizer = torch.optim.AdamW(transcoder.parameters(), lr=learning_rate, weight_decay=0.0)
        saved = []
        for epoch in range(epochs):
            train_loss, rows = 0.0, 0
            transcoder.train()
            for x, y in iter_activation_batches(paths, batch_size, seed=seed + epoch, shuffle=True):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(transcoder(x), y)
                require(bool(torch.isfinite(loss)), "Nonfinite transcoder training loss")
                loss.backward(); optimizer.step(); transcoder.normalize_decoder_()
                train_loss += loss.item() * x.shape[0]; rows += x.shape[0]
            state = {key: value.detach().cpu().clone() for key, value in transcoder.state_dict().items()}
            path = directory / f"epoch_{epoch:03d}.pt"
            torch.save({"state_dict": state}, path)
            saved.append({"epoch": epoch, "train_rows": rows, "train_output_mse": train_loss / rows,
                "path": str(path), "sha256": file_sha256(path), "state_hash": tensor_state_hash(state)})
        require(training["files"] == _files(paths), "Training source changed during precomputation")
        result = {"schema": SCHEMA, "complete": True, "contract": contract, "epochs": saved,
                  "development_read": False, "test_read": False, "selected_checkpoint": None}
        result["fingerprint"] = fingerprint(result)
        write_json_atomic(directory / "complete.json", result)
        return _verify_complete(directory, contract)


def finalize_development(precomputed_dir, dev_paths, *, output_path):
    """Score all epochs on the complete filtered dev; produce original metadata.

    This does not itself publish a graph-ready 32-layer bank. Its caller must
    supply the unchanged exact-prefix-filtered dev and verify all 32 layers.
    Failed FVU is retained as failed evidence; no automatic refit is performed.
    """
    record = _verify_complete(Path(precomputed_dir).resolve()); c = record["contract"]
    torch.set_num_threads(c["runtime"]["threads"])
    require(_runtime() == c["runtime"], "Development runtime differs from precomputed training runtime")
    config = TranscoderConfig(**c["config"])
    dev_paths = [str(Path(p).resolve()) for p in dev_paths]
    provenance = validate_shard_splits(c["train_paths"], dev_paths,
        policy_fingerprint=c["policy_fingerprint"], layer_path=c["layer_path"], config=config)
    dev_files = _files(dev_paths)
    output = Path(output_path).resolve()
    require(not output.exists() and not Path(str(output) + ".json").exists(), "Finalized/partial output exists; never overwrite or reselect")
    if c["device"].startswith("cuda"):
        total = torch.cuda.get_device_properties(c["device"]).total_memory
        torch.cuda.set_per_process_memory_fraction(c["gpu_memory_bytes"] / total, c["device"])
    transcoder = TopKTranscoder(config, seed=c["seed"]).to(c["device"])
    history, best_loss, best_state, best_epoch = [], math.inf, None, None
    for epoch in record["epochs"]:
        state = torch.load(epoch["path"], map_location="cpu", weights_only=True)["state_dict"]
        transcoder.load_state_dict(state); transcoder.eval()
        dev = reconstruction_metrics(transcoder, iter_activation_batches(dev_paths, c["batch_size"], seed=c["seed"], shuffle=False))
        history.append({"epoch": epoch["epoch"], "train_rows": epoch["train_rows"],
                        "train_output_mse": epoch["train_output_mse"], "dev": dev})
        if dev["output_mse"] < best_loss:
            best_loss, best_state, best_epoch = dev["output_mse"], state, epoch["epoch"]
    require(dev_files == _files(dev_paths), "Development source changed during selection")
    transcoder.load_state_dict(best_state); transcoder.eval(); dev = history[best_epoch]["dev"]
    metadata = {"schema_version": 1, "method": "per_mlp_topk_input_to_output_v1", "config": c["config"],
        **{key: c[key] for key in ("seed", "epochs", "learning_rate", "batch_size")},
        "selected_epoch": best_epoch, "history": history, "dev": dev, "max_dev_fvu": c["max_dev_fvu"],
        "fidelity_gate_passed": c["max_dev_fvu"] is not None and not dev["fvu_undefined"] and dev["output_fvu"] <= c["max_dev_fvu"],
        "provenance": provenance, "transcoder_hash": transcoder.checkpoint_hash()}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "metadata": metadata}, output)
    Path(str(output) + ".json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return transcoder, metadata


def _task_worker(task):
    return precompute_layer(TranscoderConfig(**task["config"]), task["train_paths"], task["output_dir"], **task["kwargs"])


def precompute_parallel(tasks, *, max_workers=2, gpu_reserve_bytes=4 * 1024**3):
    """Two spawned training processes by default; no duplicate output writers."""
    tasks = list(tasks)
    require(tasks and 1 <= max_workers <= 4 and len({t["output_dir"] for t in tasks}) == len(tasks), "Invalid/duplicate parallel tasks")
    devices = {str(task["kwargs"].get("device", "cuda:0")) for task in tasks}
    require(len(devices) == 1, "Use one fixed precomputation device")
    device = next(iter(devices))
    if device.startswith("cuda"):
        cap = max(task["kwargs"].get("gpu_memory_bytes", 2 * 1024**3) for task in tasks)
        free, _ = torch.cuda.mem_get_info(device)
        require(free >= max_workers * (cap + 1024**3) + gpu_reserve_bytes, "Insufficient observed GPU headroom; no worker launched")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(_task_worker, tasks))


def prepare_core_training_tasks(core, output_root):
    """Read only the exact complete imported train corpus; output stays external.

    Returns all 32 tasks, seed 1729+layer, complete per-layer shard order, and
    unchanged core TC settings. It deliberately does not require/read the new
    development collection or any test data. The core/source and original
    import receipts are revalidated before any CUDA worker can be dispatched.
    """
    from .core_protocol import read_core, collection_manifest_paths, registered_transcoder_epochs
    from .core_collection import verify_import
    from .representation_training import read_collections, QWEN_MLP_PATHS
    require(read_core(core["workspace"]) == core, "Core/source registration changed")
    output = Path(output_root).resolve()
    require(output != Path(core["workspace"]) and Path(core["workspace"]) not in output.parents,
            "Precomputation must remain outside the immutable CORE workspace")
    for item in core["imported_train"]:
        verify_import(core, item)
    manifests = collection_manifest_paths(core)[:-1]
    collection = read_collections(manifests, model_key=core["model_key"])
    require(all(row["split"] == "train" for row in collection["records"]), "Only training prefixes may precompute")
    tc = core["transcoder"]
    require(tc["epochs"] == registered_transcoder_epochs(core)
            and tc["max_development_output_fvu"] == .5, "Core training/fidelity contract changed")
    tasks = []
    for index, layer in enumerate(QWEN_MLP_PATHS):
        entries = [item for item in collection["activation_shards"] if item["layer_path"] == layer]
        require(len(entries) == len(manifests) and all(item["split"] == "train" for item in entries), "Layer is missing a complete training shard")
        dimensions = set()
        for entry in entries:
            shard = _load_shard(entry["path"]); meta = shard["metadata"]
            require(meta["policy_fingerprint"] == collection["checkpoint_hash"]
                and all(meta[k] == entry[k] for k in ("split", "layer_path", "tensor_hash"))
                and set(meta["group_ids"]) <= set(entry["source_groups"])
                and set(meta["prefix_hashes"]) <= set(entry["source_prefixes"]), "Training shard provenance differs from exact captured prefixes")
            dimensions.add((meta["input_dim"], meta["output_dim"]))
        require(len(dimensions) == 1, "Mixed layer activation dimensions")
        input_dim, output_dim = next(iter(dimensions))
        config = TranscoderConfig(input_dim, output_dim, tc["feature_expansion"] * input_dim, tc["top_k"])
        tasks.append({"config": asdict(config), "train_paths": [item["path"] for item in entries],
            "output_dir": str(output / f"layer_{index:02d}"), "kwargs": {
                "policy_fingerprint": collection["checkpoint_hash"], "layer_path": layer,
                "seed": 1729 + index, "epochs": tc["epochs"], "batch_size": tc["batch_size"],
                "learning_rate": tc["learning_rate"], "device": "cuda:0",
                "max_dev_fvu": tc["max_development_output_fvu"], "cpu_threads": core["policy_runtime"]["torch_cpu_threads"],
                "registration": {"core_fingerprint": core["fingerprint"], "model_key": core["model_key"],
                    "collection_fingerprint": collection["collection_fingerprint"], "manifests": _files(manifests),
                    "layer_index": index, "complete_training_data": True, "development_or_test_read": False}}})
    return tasks
