"""Synthetic provenance guards only; no model, GPU or scientific experiment."""
import copy
import importlib.util
from pathlib import Path

import pytest

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic


def launcher():
    path = Path(__file__).parents[1] / "scripts/run_tc64_core.py"
    spec = importlib.util.spec_from_file_location("tc64_launcher_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prefix_fixture(tmp_path):
    prior, records = [], []
    for index in range(32):
        contract = {"config": {"tiny_fixture_only": True}, "policy_fingerprint": "unit",
            "layer_path": str(index), "seed": 1729 + index, "batch_size": 8,
            "learning_rate": .01, "device": "cpu", "max_dev_fvu": .5,
            "runtime": {"unit_only": True}, "epochs": 16,
            "training": {"files": [{"sha256": "unchanged-raw-shard"}]}}
        epochs = [{"state_hash": f"unit-layer-{index}-epoch-{epoch}", "train_rows": 8,
                   "train_output_mse": 1 / (epoch + 1)} for epoch in range(64)]
        old = {"contract": contract, "epochs": epochs[:16]}
        old["fingerprint"] = fingerprint(old)
        path = tmp_path / f"old-{index}.json"
        write_json_atomic(path, old)
        prior.append({"path": str(path), "sha256": file_sha256(path)})
        records.append({"contract": {**copy.deepcopy(contract), "epochs": 64}, "epochs": epochs})
    result = tmp_path / "prior-result.json"
    write_json_atomic(result, {"complete": True, "layers": 32, "epochs_per_layer": 16, "layer_receipts": prior})
    core = {"source_evidence": [{"path": str(result), "sha256": file_sha256(result)}], "snapshot_files": []}
    return core, result, records


def test_all32_from_seed_prefix_must_match_the_frozen16(prefix_fixture):
    module = launcher(); core, result, records = prefix_fixture
    proof = module.verify_original_prefix(core, result, records)
    assert proof["complete"] and len(proof["layers"]) == 32
    assert proof["optimizer_state_resumed"] is False
    records[11]["epochs"][14]["state_hash"] = "changed"
    with pytest.raises(RuntimeError, match="differs from the original16"):
        module.verify_original_prefix(core, result, records)


def test_unbound_prior_evidence_and_changed_train_order_are_refused(prefix_fixture):
    module = launcher(); core, result, records = prefix_fixture
    with pytest.raises(RuntimeError, match="not bound"):
        module.verify_original_prefix({"source_evidence": [], "snapshot_files": []}, result, records)
    records[0]["contract"]["training"]["files"][0]["sha256"] = "other-input"
    with pytest.raises(RuntimeError, match="training shard bytes/order"):
        module.verify_original_prefix(core, result, records)
