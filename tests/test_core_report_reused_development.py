"""Synthetic cost/provenance fixtures only; no model or physical result."""
from copy import deepcopy
from pathlib import Path

import pytest

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.core_final import CoreFinalError
from matdiscovery.core_report import core_cost_report, offline_transcoder_report
from test_core_final_report import cost_inputs, records, unit_manifest
from test_core_final_integration import project, report_dependencies, runner_fixture


def artifact(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def document(path, value):
    write_json_atomic(path, {"classification": "unit_test_only", **value})
    return artifact(path)


def reused_inputs(tmp_path):
    corpus, es = cost_inputs(tmp_path)
    receipts = [document(tmp_path / f"physical_episode_{i}.json", {"episode": i}) for i in range(4)]
    corpus["imported_train"].update(jobs=3, episodes=3, receipts=receipts[:3],
        costs={"candidate_oracle_attempts": 150, "initialization_oracle_attempts": 84, "surrogate_oracle_attempts": 2191})
    corpus["reused_development"] = {"jobs": 1, "episodes": 1, "receipts": receipts[3:],
        "execution_origin": "historical_core_v1_completed_development",
        "costs": {"candidate_oracle_attempts": 50, "initialization_oracle_attempts": 26, "surrogate_oracle_attempts": 596}}
    zeros = {"candidate_oracle_attempts": 0, "initialization_oracle_attempts": 0, "surrogate_oracle_attempts": 0, "dft_episode_attempts": 0}
    corpus["new_development"] = {"jobs": 0, "episodes": 0, "receipts": [], "costs": zeros}
    corpus["incremental_physical_costs"] = dict(zeros)
    corpus["total"] = {"jobs": 4, "episodes": 4,
        "costs": {"candidate_oracle_attempts": 200, "initialization_oracle_attempts": 110, "surrogate_oracle_attempts": 2787}}
    corpus["evidence_files"] += receipts
    return corpus, es


def report(corpus, es):
    return core_cost_report(corpus, es, records(unit_manifest()), expected_imported_jobs=3,
                            expected_reused_development=True)


def test_v2_reused_dev_is_historical_and_every_physical_cost_counted_once(tmp_path):
    corpus, es = reused_inputs(tmp_path)
    result = report(corpus, es)
    assert result["categories"]["matched_final_evaluation"]["candidate_oracle_attempts"] == 200
    assert result["incremental_core_costs"]["candidate_oracle_attempts"] == 500
    assert result["historical_reused_costs"]["candidate_oracle_attempts"] == 200
    assert result["reused_plus_incremental_costs"]["candidate_oracle_attempts"] == 700
    assert result["incremental_core_costs"]["initialization_oracle_attempts"] == 8
    assert result["historical_reused_costs"]["initialization_oracle_attempts"] == 110
    assert result["reused_plus_incremental_costs"]["surrogate_oracle_attempts"] == 2787
    assert result["categories"]["new_development_collection"]["candidate_oracle_attempts"] == 0
    assert result["development_collection_origin"] == "reused_completed_v1"
    assert "reused_development_collection" not in result["incremental_categories"]


def test_v1_keeps_new_dev_incremental_and_cannot_silently_adopt_v2_category(tmp_path):
    corpus, es = cost_inputs(tmp_path)
    result = core_cost_report(corpus, es, records(unit_manifest()), expected_imported_jobs=3)
    assert result["incremental_core_costs"]["candidate_oracle_attempts"] == 550
    assert result["reused_plus_incremental_costs"]["candidate_oracle_attempts"] == 700
    assert result["development_collection_origin"] == "new_core_execution"
    corpus, es = reused_inputs(tmp_path)
    with pytest.raises(CoreFinalError, match="Unregistered"):
        core_cost_report(corpus, es, records(unit_manifest()), expected_imported_jobs=3)


def test_final_report_uses_registered_reuse_and_keeps_matched_200(project, tmp_path, monkeypatch):
    from matdiscovery import core_collection, core_protocol, core_report
    calls = []; runner = runner_fixture(project, tmp_path, calls)
    runner.output = project / "experiments/core_final"
    runner.run()
    report_dependencies(monkeypatch, runner, tmp_path)
    core = core_protocol.read_core(project)
    core["imported_development"] = {"classification": "unit_test_only"}
    corpus, _ = reused_inputs(tmp_path)
    monkeypatch.setattr(core_collection, "audit_core_corpus_costs", lambda value: corpus)
    result = core_report.report_core(project)
    assert result["core_complete"] and not result["global_study_complete"]
    assert result["costs"]["incremental_core_costs"]["candidate_oracle_attempts"] == 500
    assert result["costs"]["reused_plus_incremental_costs"]["candidate_oracle_attempts"] == 700
    assert result["costs"]["categories"]["matched_final_evaluation"]["candidate_oracle_attempts"] == 200
    assert len(calls) == 4


@pytest.mark.parametrize("mutation", ["new_cost", "incremental_cost", "wrong_origin", "missing_receipt", "duplicate_receipt", "double_total", "changed_source"])
def test_v2_rejects_ambiguous_or_double_counted_physical_sources(tmp_path, mutation):
    corpus, es = reused_inputs(tmp_path)
    if mutation == "new_cost": corpus["new_development"]["costs"]["candidate_oracle_attempts"] = 50
    elif mutation == "incremental_cost": corpus["incremental_physical_costs"]["surrogate_oracle_attempts"] = 1
    elif mutation == "wrong_origin": corpus["reused_development"]["execution_origin"] = "assumed_reuse"
    elif mutation == "missing_receipt": corpus["evidence_files"].pop()
    elif mutation == "duplicate_receipt": corpus["reused_development"]["receipts"] = corpus["imported_train"]["receipts"][:1]
    elif mutation == "double_total": corpus["total"]["costs"]["candidate_oracle_attempts"] = 250
    elif mutation == "changed_source": Path(corpus["reused_development"]["receipts"][0]["path"]).write_text("tampered")
    with pytest.raises(CoreFinalError): report(corpus, es)


def tc_run(tmp_path, name, core_fingerprint, epochs, passed, *, times=True):
    folder = tmp_path / name
    receipt_files, layers = [], []
    for index in range(32):
        data = {"classification": "unit_test_only", "complete": True, "development_read": False, "test_read": False,
            "contract": {"epochs": epochs, "registration": {"core_fingerprint": core_fingerprint, "layer_index": index}},
            "epochs": [{"epoch": epoch, "train_rows": 3} for epoch in range(epochs)]}
        data["fingerprint"] = fingerprint(data)
        path = folder / "layers" / f"layer_{index:02d}" / "complete.json"
        write_json_atomic(path, data)
        item = artifact(path); receipt_files.append(item)
        layers.append({"layer_index": index, "metadata": {"epochs": epochs, "execution_provenance": {
            "epochs_scored": epochs, "test_data_used": False, "precomputation": item}}})
    execution = {"complete": True, "core_fingerprint": core_fingerprint, "layers": 32,
        "epochs_per_layer": epochs, "training_data_only": True, "new_policy_or_material_oracle_calls": 0,
        "layer_receipts": receipt_files}
    if times: execution.update(started_at=100., training_started_at=110., finished_at=150.)
    run_item = document(folder / "result.json", execution)
    run_copy = document(folder / "run.json", execution)
    bank = {"classification": "unit_test_only", "complete": True, "ready_for_graphs": passed,
        "configuration": {"epochs": epochs, "max_dev_fvu": .5}, "layers": layers}
    bank["bank_fingerprint"] = fingerprint(bank)
    return bank, run_item, run_copy


def offline_inputs(tmp_path, *, times=True):
    core = {"workspace": str(tmp_path / "core_v2"), "fingerprint": "unit-core-v2", "transcoder": {"epochs": 64}}
    prior_bank, old, run = tc_run(tmp_path, "training_v1", "unit-core-v1", 16, False, times=times)
    current_bank, _, _ = tc_run(tmp_path, "training_v2", "unit-core-v2", 64, True, times=times)
    prior_path = tmp_path / "failed_v1_bank.json"; write_json_atomic(prior_path, prior_bank)
    current_path = Path(core["workspace"]) / "experiments/transcoders/qwen35_4b/made/transcoder_manifest.json"
    write_json_atomic(current_path, current_bank)
    core["tc64_amendment"] = {"predecessor_fingerprint": "unit-core-v1",
        "predecessor_protocol": document(tmp_path / "v1_protocol.json", {"fingerprint": "unit-core-v1"}),
        "failure_reconciliation": document(tmp_path / "failure.json", {"fidelity_failed": True}),
        "failed_bank": artifact(prior_path), "prior_tc16_execution": [old, run],
        "tc64_precompute_result_path": str(tmp_path / "training_v2/result.json")}
    return core


def test_tc16_failure_and_tc64_execution_each_retained_once_without_gpu_hour_invention(tmp_path):
    result = offline_transcoder_report(offline_inputs(tmp_path))
    assert result["historical_failed_tc16"]["development_fidelity_outcome"] == "failed"
    assert result["historical_failed_tc16"]["observed_layer_epochs"] == 512
    assert result["current_tc64"]["development_fidelity_outcome"] == "passed"
    assert result["current_tc64"]["observed_layer_epochs"] == 2048
    assert result["observed_layer_epochs_across_attempts"] == 2560
    for run in (result["historical_failed_tc16"], result["current_tc64"]):
        assert run["precompute_total_wall_seconds"] == 50
        assert run["precompute_training_phase_wall_seconds"] == 40
        assert run["development_selection_wall_seconds"] is None
        assert run["gpu_hours"] is None
        assert run["scientific_oracle_calls"] == 0
    assert result["not_added_to_episode_wall_or_physical_cost_totals"]


def test_missing_tc_timings_remain_unknown_and_v1_does_not_require_new_schema(tmp_path):
    result = offline_transcoder_report(offline_inputs(tmp_path, times=False))
    assert result["historical_failed_tc16"]["precompute_total_wall_seconds"] is None
    assert result["current_tc64"]["precompute_training_phase_wall_seconds"] is None
    assert offline_transcoder_report({"fingerprint": "unit-original-v1"}) is None


@pytest.mark.parametrize("mutation", ["source", "core", "missing_old", "result_replay", "duplicate_layer", "epochs", "timing", "unregistered_path"])
def test_offline_tc_provenance_drift_and_fake_execution_are_rejected(tmp_path, mutation):
    core = offline_inputs(tmp_path)
    current = Path(core["workspace"]) / "experiments/transcoders/qwen35_4b/made/transcoder_manifest.json"
    if mutation == "source": Path(core["tc64_amendment"]["failed_bank"]["path"]).write_text("changed")
    elif mutation == "core": core["tc64_amendment"]["predecessor_fingerprint"] = "another-core"
    elif mutation == "missing_old": core["tc64_amendment"]["prior_tc16_execution"] = []
    elif mutation == "unregistered_path": core["tc64_amendment"]["tc64_precompute_result_path"] = str(tmp_path / "other/result.json")
    elif mutation in ("duplicate_layer", "epochs"):
        from matdiscovery.core_report import read
        bank = read(current)
        if mutation == "duplicate_layer": bank["layers"][1] = deepcopy(bank["layers"][0])
        else: bank["configuration"]["epochs"] = 16
        bank["bank_fingerprint"] = fingerprint({k: v for k, v in bank.items() if k != "bank_fingerprint"})
        write_json_atomic(current, bank)
    else:
        from matdiscovery.core_report import read
        target = tmp_path / "training_v2/result.json"; record = read(target)
        if mutation == "result_replay": record["epochs_per_layer"] = 16
        else: record["finished_at"] = 5
        write_json_atomic(target, record)
    with pytest.raises(CoreFinalError): offline_transcoder_report(core)
