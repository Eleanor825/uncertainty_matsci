"""Tiny CPU provenance fixtures; no transcoder fitting or scientific results."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

from matdiscovery.accounting import file_sha256, write_json_atomic
from matdiscovery.esopt import tensor_state_hash
from torch_runtime_fixture import restore_torch_runtime


@pytest.fixture
def launcher():
    # The production entry point configures torch at import time. Import inside
    # the runtime fixture, never during collection of unrelated test modules.
    spec = importlib.util.spec_from_file_location("normalized_launcher_script",
        Path(__file__).resolve().parents[1] / "scripts/run_normalized_core.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def artifact(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def fixture(root):
    tasks = []
    bank = {"layers": [{} for _ in range(32)]}
    for index in (1, 12, 13, 16):
        directory = root / f"layer_{index:02d}"; directory.mkdir()
        task = {"layer_index": index, "output": str(directory), "unit_only": True}
        tasks.append(task)
        state = {"tiny_weight": torch.tensor([[index, .25], [-.5, index + 1]], dtype=torch.float32)}
        weights = directory / "selected_folded.pt"
        torch.save({"state_dict": state}, weights)
        history = []
        for epoch in range(64):
            train = {"rows": 27., "output_mse": .04 / (epoch + 1), "output_fvu": .2 / (epoch + 1),
                     "fvu_undefined": 0., "mean_active_features": 3.}
            dev = {"rows": 13., "output_mse": .06 / (epoch + 1), "output_fvu": .3 / (epoch + 1),
                   "fvu_undefined": 0., "mean_active_features": 3.}
            history.append({"epoch": epoch, "train_rows": 27, "train_normalized_batch_mse": .7 / (epoch + 1),
                            "train_raw_end_epoch": train, "dev_raw": dev})
        result = {"complete": True, "task": task, "epochs": 64, "weights": artifact(weights),
                  "selected_epoch": 63, "history": history}
        write_json_atomic(directory / "result.json", result)
        produced = [{**row, "train_output_mse": row["train_raw_end_epoch"]["output_mse"],
                     "dev": row["dev_raw"]} for row in copy.deepcopy(history)]
        for row in produced: del row["dev_raw"]
        bank["layers"][index] = {"transcoder_hash": tensor_state_hash(state), "metadata": {
            "transcoder_hash": tensor_state_hash(state), "selected_epoch": 63, "history": produced}}
    registration = root / "registration.json"
    write_json_atomic(registration, {"layers": [1, 12, 13, 16], "tasks": tasks})
    summary = root / "summary.json"
    write_json_atomic(summary, {"registration": artifact(registration), "all_four_passed": True})
    return {"normalization_amendment": {"normalized_diagnostic_summary": artifact(summary)}}, bank


def test_real_tensor_hash_and_all64_history_match_completed_diagnostic(tmp_path, launcher):
    core, bank = fixture(tmp_path)
    proof = launcher.verify_diagnostic_equivalence(core, bank)
    assert proof["complete"] and [item["layer_index"] for item in proof["layers"]] == [1, 12, 13, 16]
    assert all(item["all64_losses_and_raw_metrics_exact"] and item["selected_tensor_hash_exact"] for item in proof["layers"])
    for item in proof["layers"]:
        assert artifact(Path(item["diagnostic_result"]["path"])) == item["diagnostic_result"]


def test_wrong_tensor_state_or_one_history_value_is_rejected(tmp_path, launcher):
    core, bank = fixture(tmp_path)
    wrong = copy.deepcopy(bank)
    other_hash = tensor_state_hash({"tiny_weight": torch.zeros(2, 2)})
    wrong["layers"][1]["transcoder_hash"] = other_hash
    wrong["layers"][1]["metadata"]["transcoder_hash"] = other_hash
    with pytest.raises(ValueError, match="weights do not reproduce"):
        launcher.verify_diagnostic_equivalence(core, wrong)
    wrong = copy.deepcopy(bank)
    wrong["layers"][12]["metadata"]["history"][17]["dev"]["output_mse"] += .001
    with pytest.raises(ValueError, match="history differs"):
        launcher.verify_diagnostic_equivalence(core, wrong)
