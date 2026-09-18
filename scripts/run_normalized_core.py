#!/usr/bin/env python3
"""Fit all32 registered normalized layers before releasing the MADE core queue."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback

import torch
torch.set_num_threads(2)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.collection_provenance import read_json
from matdiscovery.core_protocol import read_core, registered_transcoder_recipe
from matdiscovery.core_collection import verify_stage_receipt, completed_core_collections
from matdiscovery.core_representation import paths_for, transcoder_config, _bank
from matdiscovery.esopt import tensor_state_hash
from matdiscovery.parallel_normalized_bank import RECIPE, build_bank, artifact, require


def verify_diagnostic_equivalence(core, bank):
    """The production recipe must reproduce the four completed diagnostic fits."""
    reference = core["normalization_amendment"]["normalized_diagnostic_summary"]
    require(file_sha256(reference["path"]) == reference["sha256"], "Diagnostic summary changed")
    summary = read_json(reference["path"])
    source = summary["registration"]
    require(file_sha256(source["path"]) == source["sha256"], "Diagnostic registration changed")
    declaration = read_json(source["path"])
    require(declaration["layers"] == [1, 12, 13, 16] and summary["all_four_passed"] is True,
            "Incomplete conditioning diagnostic")
    proofs = []
    for task in declaration["tasks"]:
        index = task["layer_index"]
        path = Path(task["output"]) / "result.json"
        result = read_json(path)
        require(result["complete"] is True and result["task"] == task and result["epochs"] == 64,
                "Diagnostic layer result differs from its declared task")
        weights = result["weights"]
        require(file_sha256(weights["path"]) == weights["sha256"], "Diagnostic selected weights changed")
        actual = torch.load(weights["path"], map_location="cpu", weights_only=True)["state_dict"]
        layer = bank["layers"][index]; metadata = layer["metadata"]
        require(tensor_state_hash(actual) == layer["transcoder_hash"] == metadata["transcoder_hash"]
                and metadata["selected_epoch"] == result["selected_epoch"], "Production normalized weights do not reproduce diagnostic")
        require(len(metadata["history"]) == len(result["history"]) == 64, "Missing comparison epochs")
        for produced, observed in zip(metadata["history"], result["history"], strict=True):
            require(produced["epoch"] == observed["epoch"] and produced["train_rows"] == observed["train_rows"]
                and produced["train_normalized_batch_mse"] == observed["train_normalized_batch_mse"]
                and produced["train_raw_end_epoch"] == observed["train_raw_end_epoch"]
                and produced["dev"] == observed["dev_raw"], "Production normalized history differs from diagnostic")
        proofs.append({"layer_index": index, "diagnostic_result": artifact(path), "diagnostic_weights": weights,
                       "all64_losses_and_raw_metrics_exact": True, "selected_tensor_hash_exact": True})
    return {"complete": True, "diagnostic_summary": reference, "layers": proofs,
            "all32_production_layers_trained_fresh": True, "new_policy_or_material_oracle_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    core = read_core(args.project)
    require(registered_transcoder_recipe(core) == RECIPE, "Only the registered normalized V3 core is permitted")
    project = Path(core["workspace"]); amendment = core["normalization_amendment"]
    output = Path(amendment["normalized_precompute_output_root"])
    paths = paths_for(core)
    require(Path(amendment["normalized_bank_path"]) == paths["bank"], "Normalized bank must use the exact registered CORE path")
    require(not (project / "logs/pipeline_status.json").exists(), "Another queue already started; do not race its stages")
    verify_stage_receipt(core, "import"); verify_stage_receipt(core, "collect")
    manifests = completed_core_collections(core)
    try:
        bank = build_bank(manifests, paths["bank"], output, model_key=core["model_key"], config=transcoder_config(core),
            registration={"core_fingerprint": core["fingerprint"], "fit_recipe": RECIPE})
        proof = verify_diagnostic_equivalence(core, bank)
        write_json_atomic(output / "diagnostic_equivalence.json", proof)
        _bank(core, manifests, paths)
        require(read_core(project) == core and not (project / "logs/pipeline_status.json").exists(),
                "Core source changed or a competing queue started")
        command = [str(project / "environments/policy-py312/bin/python"), "-B",
                   str(project / "scripts/run_core_study.py"), "--project", str(project)]
        env = dict(os.environ, PYTHONPATH=str(project / "src"), PYTHONHASHSEED="0",
                   PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
        with (output / "core_queue.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=project, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        write_json_atomic(output / "queue_launch.json", {"pid": child.pid, "command": command,
            "launched_at": time.time(), "core_fingerprint": core["fingerprint"],
            "all32_normalized_fidelity_passed": True, "diagnostic_equivalence": artifact(output / "diagnostic_equivalence.json"),
            "scientific_core_complete": False})
    except BaseException:
        output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output / "driver_failure.json", {"failed_at": time.time(), "traceback": traceback.format_exc(),
            "new_policy_or_material_oracle_calls": 0, "core_fingerprint": core["fingerprint"]})
        raise


if __name__ == "__main__":
    main()
