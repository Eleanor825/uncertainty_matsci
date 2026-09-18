"""Synthetic provenance/cost fixtures only; no model training or scientific call."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from matdiscovery.accounting import fingerprint, write_json_atomic
from matdiscovery.core_final import CoreFinalError
from matdiscovery.core_report import offline_transcoder_report
from test_core_report_reused_development import artifact, document, offline_inputs
from torch_runtime_fixture import restore_torch_runtime


LAYERS = [1, 12, 13, 16]
RECIPE = "train_centered_scalar_rms_fp32_export_v1"


def binary(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("unit_test_only:" + value).encode())
    return artifact(path)


def revise(path, **changes):
    value = json.loads(path.read_text()); value.update(changes)
    if "bank_fingerprint" in value:
        value["bank_fingerprint"] = fingerprint({k: v for k, v in value.items() if k != "bank_fingerprint"})
    write_json_atomic(path, value)
    return value


def normalized_fixture(tmp_path):
    prior = offline_inputs(tmp_path / "raw")
    raw64 = Path(prior["workspace"]) / "experiments/transcoders/qwen35_4b/made/transcoder_manifest.json"
    revise(raw64, ready_for_graphs=False)
    core = {"workspace": str(tmp_path / "core_v3"), "fingerprint": "unit-v3-normalized", "transcoder": {"epochs": 64},
            "tc64_amendment": deepcopy(prior["tc64_amendment"])}
    groups = []

    def group(name, items, layers, epochs):
        groups.append({"run_id": name, "role": name, "artifacts": items, "expected_layers": layers, "epochs_if_known": epochs})

    old = prior["tc64_amendment"]
    group("raw_tc16", old["prior_tc16_execution"] + [old["failed_bank"], old["failure_reconciliation"]], list(range(32)), 16)
    group("raw_tc64", [artifact(raw64), artifact(tmp_path / "raw/training_v2/result.json"),
        document(tmp_path / "raw64/selection.json", {"started_at": 160., "finished_at": 220., "development_workers": 4})], list(range(32)), 64)

    raw128 = []; raw_cases = []
    for index in LAYERS:
        path = tmp_path / "raw128" / f"layer_{index:02d}"
        weights = binary(path / "weights.pt", str(index))
        metadata = {"epochs": 128, "history": [{"epoch": epoch} for epoch in range(128)]}
        sidecar = document(path / "weights.pt.json", metadata)
        result = document(path / "result.json", {"complete": True, "index": index, "metadata": metadata,
            "checkpoint": weights, "sidecar": sidecar, "fidelity_gate_passed": index == 16, "elapsed_seconds": 85.})
        raw128 += [result, weights, sidecar]
        raw_cases.append({"index": index, "result": result})
    raw128.append(document(tmp_path / "raw128/summary.json", {"complete": True, "full_32_layer_bank": False,
        "test_data_used": False, "policy_loaded": False, "counters": {k: 0 for k in
            ["candidate_oracle_attempts", "initialization_oracle_attempts", "surrogate_oracle_attempts", "dft_episode_attempts"]},
        "layers": raw_cases, "elapsed_seconds": 90.83}))
    group("raw128_four_layer_diagnostic", raw128, LAYERS, 128)

    guard = [document(tmp_path / "guard/registration.json", {"policy_or_oracle_calls": 0})]
    for index in LAYERS:
        path = tmp_path / "guard" / f"layer_{index:02d}"
        guard.append(document(path / "failure.json", {"complete": False, "new_policy_or_oracle_calls": 0,
            "task": {"layer_index": index}, "error": "unit-only auxiliary numerical guard failure"}))
        log = path / "history.jsonl"
        log.write_text("".join(json.dumps({"epoch": epoch}) + "\n" for epoch in range(index % 5)))
        guard.append(artifact(log))
    group("normalized64_guard_failed", guard, LAYERS, 64)
    group("fold_boundary_reproduction", [document(tmp_path / "boundary/result.json",
        {"fixed_epochs_complete": 5, "new_policy_or_oracle_calls": 0})], [1], 5)

    success = []; cases = []
    for index in LAYERS:
        path = tmp_path / "diagnostic_success" / f"layer_{index:02d}"
        weights = binary(path / "raw.pt", str(index)); tensor = binary(path / "stats.pt", str(index))
        result = document(path / "result.json", {"complete": True, "new_policy_or_oracle_calls": 0, "epochs": 64,
            "test_data_used": False, "fidelity_gate_passed": True, "history": [{"epoch": epoch} for epoch in range(64)],
            "weights": weights, "statistics": {"tensor_file": tensor}, "task": {"layer_index": index}, "elapsed_seconds": 75.})
        success += [result, weights, tensor]; cases.append({"layer": index})
    success.append(document(tmp_path / "diagnostic_success/summary.json", {"complete": True,
        "new_policy_or_oracle_calls": 0, "all_four_passed": True, "cases": cases, "elapsed_seconds": 79.624}))
    group("normalized64_four_layer_success", success, LAYERS, 64)

    runroot = tmp_path / "current_run"
    bankroot = Path(core["workspace"]) / "experiments/transcoders/qwen35_4b/made"
    layer_rows, receipts = [], []
    train_source = binary(tmp_path / "unit_train_shard.pt", "all-real-shapes-are-synthetic-fixtures")
    for index in range(32):
        path = runroot / "layers" / f"layer_{index:02d}"
        weights = binary(path / "weights.pt", str(index))
        history = [{"epoch": epoch, "train_rows": 3, "dev": {"output_mse": 1. / (epoch + 1), "output_fvu": .4, "fvu_undefined": 0}} for epoch in range(64)]
        metadata = {"fit_recipe": RECIPE, "epochs": 64, "seed": 1729 + index, "registration": {"core_fingerprint": core["fingerprint"]},
            "history": history, "dev": history[-1]["dev"], "selected_epoch": 63, "max_dev_fvu": .5, "fidelity_gate_passed": True,
            "test_data_used": False, "runtime": {"parameter_dtype": "torch.float32"}, "source_files": {"train": [train_source]},
            "normalization": {"train_only": True, "sources": [train_source], "rows": 3, "export": {"dtype": "torch.float32"}}}
        metadata_item = document(path / "weights.pt.json", metadata)
        metadata = json.loads(Path(metadata_item["path"]).read_text())
        receipt = document(path / "result.json", {"schema": "parallel_normalized_layer_v1", "complete": True,
            "layer_index": index, "epochs": 64, "trained_epochs": 64, "history_length": 64, "fit_recipe": RECIPE,
            "request": {"layer_index": index, "output_dir": str(path), "input_files": [train_source], "source_files": [train_source],
                "kwargs": {"epochs": 64, "seed": metadata["seed"], "registration": metadata["registration"]}},
            "statistics_train_only": True, "checkpoint": weights, "metadata": metadata_item, "transcoder_hash": f"unit-layer-{index}",
            "started_at": 300., "finished_at": 400., "elapsed_seconds": 100.})
        record = json.loads(Path(receipt["path"]).read_text()); record["fingerprint"] = fingerprint(record)
        write_json_atomic(receipt["path"], record); receipt = artifact(Path(receipt["path"]))
        receipts.append(receipt)
        layer_rows.append({"layer_index": index, "metadata": {**metadata, "execution_provenance": {"worker_receipt": receipt}},
            "checkpoint": weights["path"], "checkpoint_sha256": weights["sha256"], "transcoder_hash": f"unit-layer-{index}"})
    bank = {"complete": True, "ready_for_graphs": True, "configuration": {"epochs": 64, "max_dev_fvu": .5}, "layers": layer_rows}
    bank["bank_fingerprint"] = fingerprint(bank)
    bankpath = bankroot / "transcoder_manifest.json"; write_json_atomic(bankpath, bank)
    document(runroot / "result.json", {"complete": True, "core_fingerprint": core["fingerprint"], "layers": 32, "epochs_per_layer": 64,
        "new_policy_or_material_oracle_calls": 0, "layer_receipts": receipts, "bank": artifact(bankpath),
        "original_resume_verified": True, "all_bank_hashes_and_mtimes_unchanged_on_resume": True,
        "started_at": 290., "finished_at": 450., "elapsed_seconds": 160.})
    core["normalization_amendment"] = {"offline_history": groups,
        "predecessor_protocol": document(tmp_path / "v2_protocol.json", prior), "predecessor_fingerprint": prior["fingerprint"],
        "normalized_precompute_output_root": str(runroot), "normalized_precompute_result_path": str(runroot / "result.json"),
        "normalized_bank_path": str(bankroot)}
    return core


def test_all_failed_diagnostic_and_full_fit_costs_are_preserved_without_double_wall(tmp_path):
    result = offline_transcoder_report(normalized_fixture(tmp_path))
    assert result["complete"] and len(result["historical_attempts"]) == 6
    history = {row["run_id"]: row for row in result["historical_attempts"]}
    assert history["raw_tc16"]["development_fidelity_outcome"] == "failed"
    assert history["raw_tc64"]["development_fidelity_outcome"] == "failed"
    assert history["raw_tc64"]["development_selection_wall_seconds"] == 60.
    assert history["raw128_four_layer_diagnostic"]["observed_layer_epochs"] == 512
    assert history["normalized64_four_layer_success"]["wall_seconds"] == 79.624
    assert history["normalized64_guard_failed"]["observed_layer_epochs_lower_bound"] == 7
    assert history["normalized64_guard_failed"]["wall_seconds"] is None
    assert history["fold_boundary_reproduction"]["observed_layer_epochs"] == 5
    assert history["fold_boundary_reproduction"]["wall_seconds"] is None
    assert all(not row["used_as_current_full_bank"] for row in history.values())
    current = result["current_full_fit"]
    assert current["observed_layer_epochs"] == 2048 and current["wall_seconds"] == 160
    assert current["registered_bank_equals_core_bank"]
    assert result["observed_layer_epochs_lower_bound"] == 5388
    assert result["wall_time_total"] is None and result["gpu_hours"] is None
    assert result["scientific_oracle_calls"] == 0 and not result["layer_epoch_total_exact"]


@pytest.mark.parametrize("mutation", ["missing_history", "duplicate_history", "shrink_diag", "changed_source", "incomplete_log", "nonzero_oracle", "repro_cost_omitted"])
def test_history_cannot_omit_or_reclassify_observed_cost(tmp_path, mutation):
    core = normalized_fixture(tmp_path); amendment = core["normalization_amendment"]
    groups = {row["run_id"]: row for row in amendment["offline_history"]}
    if mutation == "missing_history": amendment["offline_history"].pop()
    elif mutation == "duplicate_history": amendment["offline_history"][-1] = amendment["offline_history"][0]
    elif mutation == "shrink_diag": groups["raw128_four_layer_diagnostic"]["expected_layers"] = [1]
    else:
        name = "normalized64_guard_failed" if mutation == "incomplete_log" else "fold_boundary_reproduction"
        item = next(item for item in groups[name]["artifacts"] if item["path"].endswith("history.jsonl" if mutation == "incomplete_log" else "result.json"))
        path = Path(item["path"])
        if mutation == "changed_source": path.write_text("tampered")
        elif mutation == "incomplete_log": path.write_bytes(b'{"epoch":0}'); item["sha256"] = artifact(path)["sha256"]
        else:
            revise(path, **({"new_policy_or_oracle_calls": 1} if mutation == "nonzero_oracle" else {"fixed_epochs_complete": 0}))
            item["sha256"] = artifact(path)["sha256"]
    with pytest.raises((CoreFinalError, ValueError)): offline_transcoder_report(core)


@pytest.mark.parametrize("mutation", ["only_four", "failed_bank", "wrong_core", "duplicate_worker", "missing_worker", "bad_worker_hash", "missing_time"])
def test_diagnostic_success_or_partial_fit_cannot_pass_full_main_report_gate(tmp_path, mutation):
    core = normalized_fixture(tmp_path); a = core["normalization_amendment"]
    resultpath = Path(a["normalized_precompute_result_path"]); result = json.loads(resultpath.read_text())
    if mutation == "only_four": result["layers"] = 4; result["layer_receipts"] = result["layer_receipts"][:4]
    elif mutation == "failed_bank":
        bankpath = Path(result["bank"]["path"]); revise(bankpath, ready_for_graphs=False); result["bank"] = artifact(bankpath)
    elif mutation == "wrong_core": result["core_fingerprint"] = "different"
    elif mutation == "duplicate_worker": result["layer_receipts"][1] = result["layer_receipts"][0]
    elif mutation == "missing_worker": result["layer_receipts"].pop()
    elif mutation == "bad_worker_hash": Path(result["layer_receipts"][0]["path"]).write_text("changed")
    elif mutation == "missing_time": result.pop("finished_at")
    write_json_atomic(resultpath, result)
    with pytest.raises(CoreFinalError): offline_transcoder_report(core)


def test_cost_gate_accepts_actual_tiny_cpu_32_layer_normalized_provider_outputs(tmp_path):
    """Execute real tiny numerical training; this fixture is never study data."""
    from matdiscovery.accounting import file_sha256
    from matdiscovery.parallel_normalized_bank import build_bank
    from matdiscovery.representation_training import TranscoderStageConfig
    from matdiscovery.core_report import _normalized_full_fit_cost
    from test_deferred_transcoder_publisher import collections
    items = collections(tmp_path / "unit_collections")
    workspace = tmp_path / "unit_core"
    bank = workspace / "experiments/transcoders/qwen35_4b/made"
    runs = tmp_path / "unit_actual_training"
    identity = "unit_test_only_no_physical_calls"
    build_bank([item["manifest"] for item in items], bank, runs, model_key="qwen35_4b",
        config=TranscoderStageConfig(feature_dim=8, top_k=4, epochs=64, batch_size=8,
            learning_rate=.03, max_dev_fvu=.5, device="cpu"),
        registration={"core_fingerprint": identity, "fit_recipe": RECIPE})
    core = {"fingerprint": identity, "workspace": str(workspace), "normalization_amendment": {
        "normalized_precompute_output_root": str(runs), "normalized_precompute_result_path": str(runs / "result.json"),
        "normalized_bank_path": str(bank)}}
    def source(item, *, parse=True):
        assert file_sha256(item["path"]) == item["sha256"]
        return json.loads(Path(item["path"]).read_text()) if parse else None
    result = _normalized_full_fit_cost(core, source)
    assert result["complete"] and result["observed_layer_epochs"] == 2048
    assert len(result["worker_cases"]) == 32 and result["registered_bank_equals_core_bank"]
    assert result["wall_seconds"] > 0 and result["gpu_hours"] is None
