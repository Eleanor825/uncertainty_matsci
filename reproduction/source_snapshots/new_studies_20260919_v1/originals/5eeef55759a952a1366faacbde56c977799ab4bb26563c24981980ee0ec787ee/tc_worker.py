"""One isolated normalized64 TC fit; no policy/oracle loading or GPU concurrency.

The parent waits for this child in the same process group. The registered
2GiB bound is the PyTorch allocator cap, not a total-NVML-memory promise.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

GIB = 1024**3
SCHEMA = "snar_normalized_tc_layer_request_v1"
SOURCE_NAMES = ("normalized_transcoders.py", "transcoders.py", "esopt.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def sources():
    path = Path(importlib.util.find_spec("matdiscovery.normalized_transcoders").origin).resolve()
    return {name: {"path": str(path.with_name(name)), "sha256": sha(path.with_name(name))} for name in SOURCE_NAMES}


def make_request(config, train_paths, dev_paths, *, policy_fingerprint, layer_path,
                 output_path, seed, registration, expected_gpu_uuid=None, device="cuda:0",
                 epochs=64, batch_size=256, learning_rate=4e-4, max_dev_fvu=.5,
                 cpu_threads=2, gpu_memory_bytes=2*GIB, minimum_free_bytes=12*GIB,
                 unit_test=False):
    configuration = asdict(config) if not isinstance(config, dict) else dict(config)
    files = {split: [{"path": str(Path(p).resolve()), "sha256": sha(p)} for p in paths]
             for split, paths in (("train", train_paths), ("dev", dev_paths))}
    request = {"schema": SCHEMA, "config": configuration, "source_files": files,
               "policy_fingerprint": policy_fingerprint, "layer_path": layer_path,
               "output_path": str(Path(output_path).resolve()), "seed": seed,
               "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
               "device": device, "max_dev_fvu": max_dev_fvu, "cpu_threads": cpu_threads,
               "gpu_memory_bytes": gpu_memory_bytes, "minimum_free_bytes": minimum_free_bytes,
               "expected_gpu_uuid": expected_gpu_uuid, "registration": registration,
               "unit_test": unit_test, "trainer_sources": sources(),
               "worker_source_sha256": sha(__file__), "parent_pid": os.getpid(),
               "parent_pgid": os.getpgrp(), "parent_waits_without_policy_forward": True,
               "scientific_oracle_calls": 0, "loads_policy_model": False}
    request["fingerprint"] = fingerprint(request)
    validate_request(request)
    return request


def validate_request(request):
    if request.get("schema") != SCHEMA or request.get("fingerprint") != fingerprint({k:v for k,v in request.items() if k != "fingerprint"}):
        raise ValueError("TC worker request fingerprint differs")
    if request.get("worker_source_sha256") != sha(__file__) or request["trainer_sources"] != sources():
        raise ValueError("TC worker/trainer source changed")
    if (request["epochs"] != 64 or request["cpu_threads"] != 2 or request["gpu_memory_bytes"] != 2*GIB
            or request["max_dev_fvu"] != .5 or request["learning_rate"] != 4e-4
            or type(request["batch_size"]) is not int or request["batch_size"] < 1):
        raise ValueError("Normalized64 mathematical/runtime contract changed")
    if type(request["seed"]) is not int or request["seed"] < 0 or not request["registration"]:
        raise ValueError("Explicit seed/registration required")
    if request["loads_policy_model"] is not False or request["scientific_oracle_calls"] != 0 or request["parent_waits_without_policy_forward"] is not True:
        raise ValueError("Only an isolated TC fit is permitted")
    if request["unit_test"] is True:
        if request["device"] != "cpu":
            raise ValueError("Unit fixture is CPU-only")
    elif request["unit_test"] is False:
        if (request["device"] != "cuda:0" or request["batch_size"] != 256
                or request["config"] != {"input_dim":2560,"output_dim":2560,"feature_dim":5120,"top_k":64}
                or not re.fullmatch(r"model\.language_model\.layers\.(?:[0-9]|[12][0-9]|3[01])\.mlp", request["layer_path"])
                or not isinstance(request["expected_gpu_uuid"], str) or not request["expected_gpu_uuid"].startswith("GPU-")
                or request["minimum_free_bytes"] < 12*GIB):
            raise ValueError("Registered Qwen4B CUDA TC request differs")
    else:
        raise ValueError("Explicit boolean unit_test flag required")
    for split in ("train", "dev"):
        if not request["source_files"][split]:
            raise ValueError("Complete train and dev files required")
        names = [entry["path"] for entry in request["source_files"][split]]
        if len(set(names)) != len(names):
            raise ValueError("Repeated activation shard")
        for entry in request["source_files"][split]:
            if not Path(entry["path"]).is_absolute() or sha(entry["path"]) != entry["sha256"]:
                raise ValueError("Activation source bytes changed")
    return request


def gpu_snapshot(uuid):
    result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.free,memory.total", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15)
    records = []
    for line in result.stdout.splitlines():
        fields = [x.strip() for x in line.split(",")]
        if len(fields) == 3 and fields[0] == uuid:
            records.append({"uuid": fields[0], "free_bytes": int(fields[1])*1024**2,
                            "total_bytes": int(fields[2])*1024**2, "time_epoch": time.time()})
    if len(records) != 1:
        raise RuntimeError("Expected physical GPU UUID not found exactly once")
    return records[0]


def require_capacity(request, snapshot):
    if snapshot["uuid"] != request["expected_gpu_uuid"] or snapshot["free_bytes"] < request["minimum_free_bytes"]:
        raise RuntimeError("Insufficient current CUDA headroom for independent TC child")


def execute_request(request, receipt_path):
    """Called only inside the isolated child. A completed bad FVU remains bad."""
    started = time.time()
    receipt_path = Path(receipt_path)
    output = Path(request["output_path"])
    if receipt_path.exists() or output.exists() or Path(str(output)+".json").exists():
        raise FileExistsError("Existing/partial TC output needs reconciliation, never automatic retraining")
    if os.getppid() != request["parent_pid"] or os.getpgrp() != request["parent_pgid"]:
        raise RuntimeError("TC child must remain under the original parent and process group")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "2":
            raise ValueError("TC subprocess requires two CPU threads")
    try:
        validate_request(request)
        from matdiscovery.normalized_transcoders import train_normalized_layer, verify_normalized_metadata
        from matdiscovery.transcoders import TranscoderConfig, load_transcoder, validate_shard_splits
        import torch
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("Native FP32 initialization required")
        config = TranscoderConfig(**request["config"]); config.validate()
        train = [x["path"] for x in request["source_files"]["train"]]
        dev = [x["path"] for x in request["source_files"]["dev"]]
        population = validate_shard_splits(train, dev, policy_fingerprint=request["policy_fingerprint"],
                                          layer_path=request["layer_path"], config=config)
        capacity = None
        if not request["unit_test"]:
            capacity = gpu_snapshot(request["expected_gpu_uuid"])
            require_capacity(request, capacity)  # After CPU source reads, just before GPU fitting.
        trained, metadata = train_normalized_layer(config, train, dev,
            policy_fingerprint=request["policy_fingerprint"], layer_path=request["layer_path"],
            output_path=output, seed=request["seed"], epochs=request["epochs"],
            batch_size=request["batch_size"], learning_rate=request["learning_rate"],
            device=request["device"], max_dev_fvu=request["max_dev_fvu"], cpu_threads=request["cpu_threads"],
            gpu_memory_bytes=request["gpu_memory_bytes"], registration=request["registration"])
        del trained
        actual, actual_metadata = load_transcoder(output, expected_policy_fingerprint=request["policy_fingerprint"],
                                                  expected_layer_path=request["layer_path"])
        if actual_metadata != metadata or metadata["source_files"] != request["source_files"]:
            raise ValueError("Saved checkpoint metadata differs from actual completed fitting")
        verify_normalized_metadata(metadata, config=config, policy_fingerprint=request["policy_fingerprint"],
                                   layer_path=request["layer_path"], train_paths=train, dev_paths=dev,
                                   expected_registration=request["registration"])
        if metadata["runtime"]["device"] != request["device"]:
            raise ValueError("Actual fitting device differs from request")
        validate_request(request)
        receipt = {"schema":"snar_normalized_tc_layer_receipt_v1", "complete":True,
            "status":"succeeded" if metadata["fidelity_gate_passed"] else "failed_fidelity",
            "fidelity_gate_passed":metadata["fidelity_gate_passed"], "request_fingerprint":request["fingerprint"],
            "checkpoint":{"path":str(output),"sha256":sha(output)},
            "sidecar":{"path":str(output)+".json","sha256":sha(str(output)+".json")},
            "transcoder_hash":actual.checkpoint_hash(), "epochs":len(metadata["history"]),
            "selected_epoch":metadata["selected_epoch"], "dev":metadata["dev"], "population_rows":population["rows"],
            "runtime":metadata["runtime"], "capacity_before_fitting":capacity,
            "pid":os.getpid(),"parent_pid":os.getppid(),"pgid":os.getpgrp(),
            "elapsed_seconds":time.time()-started,"scientific_oracle_calls":0,"policy_models_loaded":0,
            "unit_test":request["unit_test"],"allocation_cap_is_not_total_nvml_memory":True}
        receipt["fingerprint"] = fingerprint(receipt)
        write_once(receipt_path, receipt)
        return receipt
    except BaseException as exc:
        failure = {"schema":"snar_normalized_tc_layer_failure_v1", "complete":False,
            "request_fingerprint":request.get("fingerprint"), "error_type":type(exc).__name__,
            "error":str(exc),"traceback":traceback.format_exc(),"pid":os.getpid(),"pgid":os.getpgrp(),
            "elapsed_seconds":time.time()-started,"scientific_oracle_calls":0,"automatic_retry":False}
        write_once(receipt_path.with_name(receipt_path.name+".failure.json"), failure)
        raise


def child_command(python, request_path, receipt_path):
    return [str(python), "-B", str(Path(__file__).resolve()), "--request", str(request_path), "--receipt", str(receipt_path)]


def run_layer_subprocess(config, train_paths, dev_paths, *, worker_directory,
                         expected_gpu_uuid=None, python=None, **kwargs):
    """Drop-in parent helper returning the actual CPU-loaded raw TC and metadata.

    worker_directory must be outside the bank inventory. This function neither
    changes parent's CUDA allocator fraction nor starts a new session/group.
    """
    directory = Path(worker_directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("TC worker directory already used; reconcile instead of retry")
    output = Path(kwargs["output_path"]).resolve()
    if directory == output.parent or output.parent in directory.parents:
        raise ValueError("Worker administrative files must be outside bank inventory")
    request = make_request(config, train_paths, dev_paths, expected_gpu_uuid=expected_gpu_uuid, **kwargs)
    directory.mkdir(parents=True, exist_ok=True)
    request_path, receipt_path = directory/"request.json", directory/"receipt.json"
    write_once(request_path, request)
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    if request["unit_test"]:
        env["CUDA_VISIBLE_DEVICES"] = ""
    with (directory/"worker.log").open("x") as log:
        result = subprocess.run(child_command(python or sys.executable, request_path, receipt_path),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=False)
    if result.returncode != 0 or not receipt_path.is_file():
        raise RuntimeError("Isolated TC fit failed; preserve worker log/output, no automatic retry")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("complete") is not True or receipt.get("request_fingerprint") != request["fingerprint"]
            or receipt.get("fingerprint") != fingerprint({k:v for k,v in receipt.items() if k!="fingerprint"})
            or receipt["pgid"] != os.getpgrp() or receipt["parent_pid"] != os.getpid()
            or sha(output) != receipt["checkpoint"]["sha256"] or sha(str(output)+".json") != receipt["sidecar"]["sha256"]):
        raise ValueError("TC worker receipt/source/group/checkpoint differs")
    from matdiscovery.transcoders import load_transcoder
    from matdiscovery.normalized_transcoders import verify_normalized_metadata
    trained, metadata = load_transcoder(output, expected_policy_fingerprint=kwargs["policy_fingerprint"],
                                        expected_layer_path=kwargs["layer_path"])
    verify_normalized_metadata(metadata, config=config, policy_fingerprint=kwargs["policy_fingerprint"],
        layer_path=kwargs["layer_path"], train_paths=train_paths, dev_paths=dev_paths,
        expected_registration=kwargs["registration"])
    if (trained.checkpoint_hash() != receipt["transcoder_hash"] or metadata["runtime"]["device"] != request["device"]
            or metadata["fidelity_gate_passed"] != receipt["fidelity_gate_passed"]):
        raise ValueError("Actual raw TC differs from worker receipt")
    return trained, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True); parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    receipt = execute_request(json.loads(Path(args.request).read_text()), args.receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
