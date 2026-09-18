"""Full32 normalized fits through the original corpus/manifest coordinator.

Four processes fit independent layers. The original coordinator still owns all
data admission, exact-prefix filtering, layer inventory and FVU acceptance.
Finished worker checkpoints are imported explicitly with truthful provenance;
the original trainer is then restored and complete-bank resume must write zero
bytes. No source file is patched and no policy or material oracle is invoked.
"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import torch

from .accounting import file_sha256, fingerprint, write_json_atomic
from .collection_provenance import read_json
from .normalized_transcoders import train_normalized_layer, verify_normalized_metadata
from .representation_training import (QWEN_MLP_PATHS, _bank_fingerprint,
    train_transcoders_from_collections)
from .transcoders import TranscoderConfig, load_transcoder, _load_shard

RECIPE = "train_centered_scalar_rms_fp32_export_v1"
CAP = 2 * 1024**3


def require(value, message):
    if not value: raise ValueError(message)


def artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def files(paths):
    return [artifact(path) for path in paths]


def inventory(directory):
    return {str(p.relative_to(directory)): {"sha256": file_sha256(p), "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def fixed_runtime():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def worker(task):
    fixed_runtime()
    directory = Path(task["output_dir"])
    require(not directory.exists(), "Unknown/partial normalized layer requires reconciliation")
    directory.mkdir(parents=True)
    started = time.time()
    write_json_atomic(directory / "started.json", {"request": task, "pid": os.getpid(), "started_at": started})
    try:
        require(all(file_sha256(item["path"]) == item["sha256"] for item in task["input_files"] + task["source_files"]),
                "Normalized worker input/source changed")
        checkpoint = directory / "selected.pt"
        model, metadata = train_normalized_layer(TranscoderConfig(**task["config"]), task["train_paths"], task["dev_paths"],
            output_path=checkpoint, **task["kwargs"])
        require(metadata["fit_recipe"] == RECIPE and len(metadata["history"]) == 64
                and metadata["transcoder_hash"] == model.checkpoint_hash(), "Incomplete/wrong normalized fit")
        require(all(file_sha256(item["path"]) == item["sha256"] for item in task["input_files"] + task["source_files"]),
                "Normalized worker input/source changed during fit")
        result = {"schema": "parallel_normalized_layer_v1", "complete": True,
            "layer_index": task["layer_index"], "epochs": 64, "trained_epochs": 64, "history_length": 64,
            "fit_recipe": RECIPE, "statistics_train_only": metadata["normalization"]["train_only"],
            "request": task, "checkpoint": artifact(checkpoint), "metadata": artifact(str(checkpoint) + ".json"),
            "transcoder_hash": model.checkpoint_hash(), "pid": os.getpid(), "started_at": started,
            "finished_at": time.time(), "elapsed_seconds": time.time() - started,
            "runtime": metadata["runtime"], "fidelity_gate_passed": metadata["fidelity_gate_passed"]}
        result["fingerprint"] = fingerprint(result)
        write_json_atomic(directory / "result.json", result)
        return artifact(directory / "result.json")
    except BaseException:
        write_json_atomic(directory / "failure.json", {"request": task, "traceback": traceback.format_exc(),
            "failed_at": time.time(), "new_policy_or_material_oracle_calls": 0})
        raise


def verify_worker(item, *, expected_registration=None):
    require(file_sha256(item["path"]) == item["sha256"], "Normalized worker receipt changed")
    receipt = read_json(item["path"])
    require(receipt["complete"] is True and receipt["schema"] == "parallel_normalized_layer_v1"
        and receipt["fingerprint"] == fingerprint({k: v for k, v in receipt.items() if k != "fingerprint"}), "Invalid worker receipt")
    task = receipt["request"]
    for source in task["input_files"] + task["source_files"] + [receipt["checkpoint"], receipt["metadata"]]:
        require(file_sha256(source["path"]) == source["sha256"], "Normalized source or selected model changed")
    model, metadata = load_transcoder(receipt["checkpoint"]["path"],
        expected_policy_fingerprint=task["kwargs"]["policy_fingerprint"], expected_layer_path=task["kwargs"]["layer_path"])
    require(metadata == read_json(receipt["metadata"]["path"]) and model.checkpoint_hash() == receipt["transcoder_hash"],
            "Normalized checkpoint/sidecar/tensor disagree")
    verify_normalized_metadata(metadata, config=TranscoderConfig(**task["config"]),
        policy_fingerprint=task["kwargs"]["policy_fingerprint"], layer_path=task["kwargs"]["layer_path"],
        train_paths=task["train_paths"], dev_paths=task["dev_paths"], expected_registration=expected_registration)
    require(receipt["epochs"] == receipt["trained_epochs"] == receipt["history_length"] == len(metadata["history"]) == 64
        and receipt["fit_recipe"] == RECIPE and receipt["statistics_train_only"] is True, "Worker omitted registered training")
    return model, metadata, receipt


def prepare_tasks(bank_path, cache, registration):
    bank = read_json(bank_path); config = bank["configuration"]
    require(bank["bank_fingerprint"] == _bank_fingerprint(bank)
        and bank["expected_layer_paths"] == list(QWEN_MLP_PATHS)
        and config["epochs"] == 64 and config["max_dev_fvu"] == .5,
        "Original coordinator settings/preparation differ")
    tasks = []
    code = files([Path(__file__), Path(__file__).with_name("normalized_transcoders.py"),
                  Path(__file__).with_name("transcoders.py"), Path(__file__).with_name("esopt.py")])
    for index, layer in enumerate(QWEN_MLP_PATHS):
        entries = [item for item in bank["activation_shards"] if item["layer_path"] == layer]
        train = [item["fit_shard_path"] for item in entries if item["split"] == "train"]
        dev = [item["fit_shard_path"] for item in entries if item["split"] == "dev" and item["fit_shard_path"] is not None]
        require(train and dev and not bank["development_exact_prefix_filter"]["empty_groups_by_layer"][layer],
                "Original filtering left incomplete training/development data")
        meta = _load_shard(train[0])["metadata"]
        shape = TranscoderConfig(meta["input_dim"], meta["output_dim"],
            config["feature_dim"] or config["feature_multiplier"] * meta["input_dim"], config["top_k"])
        tasks.append({"layer_index": index, "config": asdict(shape), "train_paths": train, "dev_paths": dev,
            "input_files": files(train + dev), "source_files": code,
            "output_dir": str(Path(cache).resolve() / f"layer_{index:02d}"),
            "kwargs": {"policy_fingerprint": bank["checkpoint_hash"], "layer_path": layer,
                "seed": config["seed"] + index, "epochs": 64, "batch_size": config["batch_size"],
                "learning_rate": config["learning_rate"], "device": config["device"], "max_dev_fvu": .5,
                "cpu_threads": 2, "gpu_memory_bytes": CAP,
                "registration": {**registration, "layer_index": index, "collection_fingerprint": bank["collection_fingerprint"]}}})
    return tasks


@contextmanager
def provider(run_root, registration, *, workers=4):
    from . import transcoders
    original = transcoders.train_layer_transcoder
    receipts, tasks, failed, invoked = None, None, False, []

    def cached(config, train_paths, dev_paths, *, policy_fingerprint, layer_path, output_path,
               seed=0, epochs=4, batch_size=256, learning_rate=1e-3, device="cpu", max_dev_fvu=None):
        nonlocal receipts, tasks, failed
        require(not failed, "Parallel fit already failed; preserve partial outputs")
        index = QWEN_MLP_PATHS.index(layer_path)
        if receipts is None:
            try:
                tasks = prepare_tasks(Path(output_path).parent / "transcoder_manifest.json", Path(run_root) / "layers", registration)
                write_json_atomic(Path(run_root) / "tasks.json", tasks)
                if str(device).startswith("cuda"):
                    require(torch.cuda.mem_get_info(device)[0] >= workers * (CAP + 1024**3) + 4 * 1024**3,
                            "Insufficient observed memory for four normalized workers")
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                    receipts = list(pool.map(worker, tasks))
                require(len(receipts) == 32, "Incomplete normalized layer matrix")
                write_json_atomic(Path(run_root) / "worker_completion.json", {"complete": True, "layer_receipts": receipts})
            except BaseException:
                failed = True
                raise
        task = tasks[index]
        expected = {"policy_fingerprint": policy_fingerprint, "layer_path": layer_path, "seed": seed,
                    "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                    "device": str(device), "max_dev_fvu": max_dev_fvu}
        require(asdict(config) == task["config"] and [str(Path(p).resolve()) for p in train_paths] == task["train_paths"]
            and [str(Path(p).resolve()) for p in dev_paths] == task["dev_paths"]
            and all(task["kwargs"][key] == value for key, value in expected.items()),
            "Cached fit does not match the original full-corpus call")
        model, metadata, receipt = verify_worker(receipts[index], expected_registration=task["kwargs"]["registration"])
        require(index not in invoked, "A layer was consumed twice")
        metadata["execution_provenance"] = {"schema": "parallel_normalized_bank_layer_v1", "fit_recipe": RECIPE,
            "original_serial_execution_claimed": False, "worker_receipt": receipts[index],
            "provider_source": artifact(__file__), "registration": task["kwargs"]["registration"],
            "original_coordinator_call": expected, "new_policy_or_material_oracle_calls": 0}
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save({"state_dict": state, "metadata": metadata}, output_path)
        write_json_atomic(str(output_path) + ".json", metadata)
        invoked.append(index)
        return model, metadata

    transcoders.train_layer_transcoder = cached
    try:
        yield invoked
    finally:
        transcoders.train_layer_transcoder = original


def verify_bank_recipe(bank, *, core_fingerprint=None):
    require(bank["complete"] is True and len(bank["layers"]) == 32, "Normalized bank must contain all32 layers")
    for index, layer in enumerate(bank["layers"]):
        metadata = layer["metadata"]
        evidence = metadata.get("execution_provenance", {})
        require(evidence.get("schema") == "parallel_normalized_bank_layer_v1" and evidence.get("fit_recipe") == RECIPE
            and evidence.get("original_serial_execution_claimed") is False
            and evidence["provider_source"]["sha256"] == file_sha256(__file__)
            and evidence["registration"]["layer_index"] == index, "Normalized execution provenance changed")
        if core_fingerprint is not None:
            require(evidence["registration"]["core_fingerprint"] == core_fingerprint, "Bank belongs to another normalized core")
        _, cached, receipt = verify_worker(evidence["worker_receipt"], expected_registration=evidence["registration"])
        require({k: v for k, v in metadata.items() if k != "execution_provenance"} == cached
            and layer["transcoder_hash"] == receipt["transcoder_hash"], "Bank differs from its real parallel fit")


def build_bank(manifests, output, run_root, *, model_key, config, registration, workers=4):
    fixed_runtime()
    output, run_root = Path(output).resolve(), Path(run_root).resolve()
    require(workers == 4 and not output.exists() and not run_root.exists(), "Fresh registered four-worker output is required")
    run_root.mkdir(parents=True)
    with (run_root / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        state = {"schema": "full_parallel_normalized_bank_v1", "complete": False,
            "core_fingerprint": registration["core_fingerprint"], "registration": registration,
            "pid": os.getpid(), "layers": 32, "epochs_per_layer": 64, "gpu_workers": 4,
            "fit_recipe": RECIPE, "started_at": started, "state": "preparing_original_complete_corpus",
            "new_policy_or_material_oracle_calls": 0}
        write_json_atomic(run_root / "run.json", state)
        try:
            with provider(run_root, registration, workers=workers) as invoked:
                bank = train_transcoders_from_collections(manifests, output, model_key=model_key, config=config, resume=True)
            require(invoked == list(range(32)), "Not all32 trained layers were consumed")
            require(bank["status"] == "succeeded" and bank["ready_for_graphs"] is True,
                    "Normalized bank failed the unchanged original fidelity gate")
            verify_bank_recipe(bank, core_fingerprint=registration["core_fingerprint"])
            before = inventory(output)
            checked = train_transcoders_from_collections(manifests, output, model_key=model_key, config=config, resume=True)
            require(checked == bank and inventory(output) == before, "Original resume changed completed normalized bank")
            state.update(complete=True, state="complete_all32_fidelity_and_original_resume_verified",
                finished_at=time.time(), elapsed_seconds=time.time() - started,
                layer_receipts=read_json(run_root / "worker_completion.json")["layer_receipts"],
                bank=artifact(output / "transcoder_manifest.json"), original_resume_verified=True,
                all_bank_hashes_and_mtimes_unchanged_on_resume=True)
            write_json_atomic(run_root / "result.json", state); write_json_atomic(run_root / "run.json", state)
            return bank
        except BaseException:
            state.update(state="failed_requires_reconciliation", failed_at=time.time(), traceback=traceback.format_exc(),
                complete_layers=[p.parent.name for p in sorted((run_root / "layers").glob("*/result.json"))],
                started_layers=[p.parent.name for p in sorted((run_root / "layers").glob("*/started.json"))])
            write_json_atomic(run_root / "failure.json", state); write_json_atomic(run_root / "run.json", state)
            raise
