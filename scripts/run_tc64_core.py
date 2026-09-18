#!/usr/bin/env python3
"""Execute the registered TC64 amendment and release its existing core queue.

Every layer starts from its original seed. The first16 state hashes must match
the preserved V1 computation. No physical collection is rerun by this helper.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

# Spawn children execute this module before importing the worker function.
import torch
torch.set_num_threads(2)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.core_protocol import read_core, registered_transcoder_epochs
from matdiscovery.core_collection import verify_stage_receipt
from matdiscovery.deferred_dev_transcoders import prepare_core_training_tasks, precompute_parallel
from matdiscovery.deferred_transcoder_publisher import publish_core_precomputed_bank


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def bound_source(core, path):
    path = Path(path).resolve()
    wanted = {"path": str(path), "sha256": file_sha256(path)}
    require(any({key: item[key] for key in wanted} == wanted
                for item in core["source_evidence"] + core["snapshot_files"]),
            "Prior evidence is not bound to the new core registration")
    return wanted


def verify_original_prefix(core, prior_result, records):
    """Compare already-produced tensor hashes, not optimizer continuation."""
    prior_source = bound_source(core, prior_result)
    prior = read(prior_result)
    require(prior.get("complete") is True and prior["layers"] == 32
            and prior["epochs_per_layer"] == 16 and len(prior["layer_receipts"]) == 32,
            "Prior evidence is not the complete original16 computation")
    comparisons = []
    for index, (item, new) in enumerate(zip(prior["layer_receipts"], records, strict=True)):
        require(file_sha256(item["path"]) == item["sha256"], "Prior layer receipt changed")
        old = read(item["path"])
        require(old["fingerprint"] == fingerprint({k: v for k, v in old.items() if k != "fingerprint"}),
                "Prior layer fingerprint changed")
        a, b = old["contract"], new["contract"]
        keys = ("config", "policy_fingerprint", "layer_path", "seed", "batch_size",
                "learning_rate", "device", "max_dev_fvu", "runtime")
        require(all(a[key] == b[key] for key in keys) and a["seed"] == 1729 + index
                and a["epochs"] == 16 and b["epochs"] == 64,
                "Amendment changed settings beyond the declared epoch count")
        require([x["sha256"] for x in a["training"]["files"]]
                == [x["sha256"] for x in b["training"]["files"]],
                "Amendment changed actual training shard bytes/order")
        require(len(new["epochs"]) == 64 and all(
            old["epochs"][epoch]["state_hash"] == new["epochs"][epoch]["state_hash"]
            and old["epochs"][epoch]["train_rows"] == new["epochs"][epoch]["train_rows"]
            and old["epochs"][epoch]["train_output_mse"] == new["epochs"][epoch]["train_output_mse"]
            for epoch in range(16)), "The from-seed run differs from the original16 prefix")
        comparisons.append({"layer_index": index, "original_receipt": item,
                            "matching_initial_epochs": 16, "all_states_and_training_losses_exact": True})
    return {"complete": True, "prior_result": prior_source, "layers": comparisons,
            "optimizer_state_resumed": False, "all64_epochs_trained_from_seed": True}


def execute(args):
    core = read_core(args.project)
    require(registered_transcoder_epochs(core) == 64, "This launcher only executes the explicit TC64 amendment")
    amendment = core["tc64_amendment"]
    require(args.output.resolve() == Path(amendment["tc64_precompute_output_root"])
            and args.prior_result.resolve() == Path(amendment["prior_tc16_precompute_result"]["path"]),
            "Launcher evidence/output paths differ from the registered amendment")
    require(not (Path(core["workspace"]) / "logs/pipeline_status.json").exists(),
            "A core queue already started; do not race its representation stage")
    verify_stage_receipt(core, "import"); verify_stage_receipt(core, "collect")
    prior_source = bound_source(core, args.prior_result)
    output, bank = args.output.resolve(), args.bank.resolve()
    require(not output.exists() and not bank.exists(), "Existing/partial outputs require explicit reconciliation")
    require(Path(core["workspace"]) not in output.parents and Path(core["workspace"]) not in bank.parents,
            "External precomputation/selection outputs must remain outside the frozen core")
    output.mkdir(parents=True)
    module = Path(sys.modules["matdiscovery.deferred_dev_transcoders"].__file__)
    source = {"path": str(module.resolve()), "sha256": file_sha256(module)}
    state = {"schema": "registered_core_parallel_transcoder_execution_v1", "pid": os.getpid(),
        "started_at": time.time(), "core_fingerprint": core["fingerprint"], "core": core["workspace"],
        "state": "validating_complete_reused_corpus", "scope": "pretest_uniform_tc64_amendment",
        "gpu_workers": 4, "layers": 32, "epochs_per_layer": 64,
        "allocator_cap_bytes_per_worker": 2 * 1024**3, "training_data_only": True,
        "new_policy_or_material_oracle_calls": 0, "empirical_improvement_claimed": False,
        "source": source, "launcher_sha256": file_sha256(__file__), "prior_result": prior_source,
        "hardware": {"name": torch.cuda.get_device_name(0), "free_total_bytes": list(torch.cuda.mem_get_info(0))}}
    write_json_atomic(output / "run.json", state)
    try:
        tasks = prepare_core_training_tasks(core, output / "layers")
        require(len(tasks) == 32 and all(t["kwargs"]["epochs"] == 64 for t in tasks), "Incomplete TC64 matrix")
        write_json_atomic(output / "tasks.json", tasks)
        state.update(state="training_all_epochs", training_started_at=time.time(), tasks_sha256=file_sha256(output / "tasks.json"))
        write_json_atomic(output / "run.json", state)
        records = precompute_parallel(tasks, max_workers=4)
        proof = verify_original_prefix(core, args.prior_result, records)
        write_json_atomic(output / "original16_prefix_verification.json", proof)
        state.update(state="all_training_epochs_complete_pending_development_selection", complete=True,
            finished_at=time.time(), original16_prefix_verified=True,
            prefix_verification={"path": str(output / "original16_prefix_verification.json"),
                "sha256": file_sha256(output / "original16_prefix_verification.json")},
            layer_receipts=[{"path": str(Path(task["output_dir"]) / "complete.json"),
                "sha256": file_sha256(Path(task["output_dir"]) / "complete.json")} for task in tasks])
        write_json_atomic(output / "result.json", state); write_json_atomic(output / "run.json", state)
        selection = {"started_at": time.time(), "development_workers": 4, "status": "selecting_complete_development"}
        write_json_atomic(output / "selection.json", selection)
        try:
            installed = publish_core_precomputed_bank(core, output / "layers", bank, install=True, development_workers=4)
        except BaseException:
            selection.update(status="failed_requires_reconciliation", finished_at=time.time(), traceback=traceback.format_exc())
            write_json_atomic(output / "selection.json", selection)
            raise
        selection.update(status="installed_verified", finished_at=time.time(), installation=installed)
        write_json_atomic(output / "selection.json", selection)
        require(read_core(core["workspace"]) == core, "Core source changed during representation work")
        project = Path(core["workspace"])
        require(not (project / "logs/pipeline_status.json").exists(), "A competing core queue started during training")
        command = [str(project / "environments/policy-py312/bin/python"), "-B",
                   str(project / "scripts/run_core_study.py"), "--project", str(project)]
        environment = dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONHASHSEED="0",
                           PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
        with (output / "core_queue.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=project, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        write_json_atomic(output / "queue_launch.json", {"pid": process.pid, "command": command,
            "launched_at": time.time(), "core_fingerprint": core["fingerprint"],
            "representation_complete": True, "scientific_core_complete": False})
    except BaseException:
        write_json_atomic(output / "failure.json", {"failed_at": time.time(), "traceback": traceback.format_exc(),
            "scope": "offline_tc64_training_selection_or_queue_launch", "physical_collection_replayed": False,
            "started_layers": [p.parent.name for p in sorted((output / "layers").glob("layer_*/started.json"))],
            "complete_layers": [p.parent.name for p in sorted((output / "layers").glob("layer_*/complete.json"))]})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--prior-result", type=Path, required=True)
    execute(parser.parse_args())
