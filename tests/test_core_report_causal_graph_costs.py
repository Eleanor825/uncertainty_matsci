"""Synthetic offline accounting, not Qwen/materials empirical results.

The source-bank admission boundary has independent real tiny tensor tests and
a real source-only V3 audit. These cases exercise the actual cost readers and
complete failed-graph/diagnostic record shapes without invoking any model.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from matdiscovery.accounting import fingerprint, write_json_atomic
from matdiscovery.core_final import CoreFinalError
from matdiscovery.core_protocol import CAUSAL_GRAPH_REGISTRATION
from matdiscovery import core_report
from test_core_report_normalized_costs import normalized_fixture
from test_core_report_reused_development import artifact, document


def sealed_document(path, value):
    value = deepcopy(value)
    value["fingerprint"] = fingerprint({k: v for k, v in value.items() if k != "fingerprint"})
    return document(path, value)


@pytest.fixture
def causal_cost_case(tmp_path, monkeypatch):
    source_core = normalized_fixture(tmp_path / "original_tc")
    run_path = Path(source_core["normalization_amendment"]["normalized_precompute_result_path"])
    value = json.loads(run_path.read_text())
    value.update(elapsed_seconds=787.388, finished_at=value["started_at"] + 787.388)
    write_json_atomic(run_path, value)
    prior = core_report.offline_normalized_transcoder_report(source_core)
    bank_path = Path(source_core["workspace"]) / "experiments/transcoders/qwen35_4b/made/transcoder_manifest.json"
    bank = json.loads(bank_path.read_text())
    source_protocol = document(Path(source_core["workspace"]) / "configs/unit-source-protocol.json", source_core)
    manifests = []
    for index in range(4):
        path = Path(source_core["workspace"]) / f"experiments/collection/unit-{index}/collection_manifest.json"
        document(path, {"classification": "unit_only_not_materials", "job_id": f"unit-{index}"})
        manifests.append(str(path))
    graph_path = Path(manifests[0]).parent / "graph_features.jsonl"
    rows = []
    for index in range(16):
        row = {"decision_id": f"unit-source-{index}", "complete": True, "graph_status": "unavailable",
            "transcoder_bank_fingerprint": bank["bank_fingerprint"], "graph_pipeline_fingerprint": "unit-old-pipeline",
            "failure_kind": "action_target_mapping_failed" if index < 5 else "backend_validation_failed",
            "elapsed_seconds": 30. if index < 15 else 80.1104}
        row["record_fingerprint"] = fingerprint(row); rows.append(row)
    graph_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report_path = Path(source_core["workspace"]) / "experiments/graph_reports/qwen35_4b/made/report.json"
    document(report_path, {"complete": False, "status": "failed", "expected_decisions": 386,
        "completed_decisions": 16, "unavailable_graphs": 16, "successful_graphs": 0,
        "failures_by_kind": {"action_target_mapping_failed": 5, "backend_validation_failed": 11},
        "transcoder_bank_fingerprint": bank["bank_fingerprint"], "graph_pipeline_fingerprint": "unit-old-pipeline"})
    pipeline = {"state": "halted_requires_reconciliation", "study_complete": False,
        "stages": {"core-graphs": {"status": "failed", "started_at": 100., "finished_at": 838.493616}}}
    status_path = Path(source_core["workspace"]) / "logs/pipeline_status.json"
    document(status_path, pipeline)
    closure_path = tmp_path / "pause/closure.json"
    preserved = []
    for path in (graph_path, report_path, status_path):
        target = closure_path.parent / "preserved" / path.relative_to(source_core["workspace"])
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        preserved.append({**artifact(path), "preserved": str(target)})
    closure = document(closure_path, {"closed": True, "processes_exited": {"12345": True},
        "new_policy_or_material_oracle_calls": 0, "pipeline": pipeline, "artifacts": preserved})
    target = tmp_path / "target-v4"
    contract = {"target_workspace": str(target), "source_core_protocol": source_protocol,
        "source_core_fp": source_core["fingerprint"], "source_bank_manifest": artifact(bank_path),
        "source_run_result": artifact(run_path), "source_artifacts": prior["evidence_files"],
        "source_manifest_paths": manifests, "source_counts": {"decisions": 386}, "pause_closure": closure}
    contract_artifact = document(target / "configs/passed_bank_reuse_contract.json", contract)
    history, terminals = [], {}

    def attempt(name, *, schema, pid, start, wall, complete, gate=False, terminal=True):
        directory = tmp_path / "diagnostics" / name
        source_file = directory / "run.py"; source_file.parent.mkdir(parents=True)
        source_file.write_text("# synthetic source artifact; never executed\n" + name)
        value = {"classification": "unit_test_only", "schema": schema, "source": artifact(source_file),
            "pid": pid, "started_at": start, "complete": False}
        if gate:
            value.update(source_core_fingerprint=source_core["fingerprint"], source_bank_manifest=artifact(bank_path),
                new_policy_or_oracle_calls=0, new_policy_generation_calls=0, new_material_oracle_calls=0,
                validation_attempts=0, attribute_attempts=0, passed=False, validation=None)
        else:
            value.update(new_oracle_calls=0, main_graph_files_written=False, capture_calls=0, additional_score_VJP_calls=0)
        history.append(document(directory / "started.json", value))
        if not terminal: return
        value.update(finished_at=start + wall, elapsed_seconds=wall, complete=complete)
        if not complete: value.update(error="unit failure: insufficient memory" if gate else "unit action mapping guard", failed_phase="policy_load")
        elif gate: value.update(passed=True, validation={"passed": True}, validation_attempts=1, attribute_attempts=1)
        else:
            value.update(core_fingerprint=source_core["fingerprint"], bank_fingerprint=bank["bank_fingerprint"],
                native_FD_gate_run=False, native_FD_gate_passed=None, capture_calls=1, additional_score_VJP_calls=1,
                assemblies=[{"feature_cap": 8, "elapsed_seconds": 7.8}, {"feature_cap": 32, "elapsed_seconds": 25.5}])
        item = sealed_document(directory / ("result.json" if complete else "failure.json"), value)
        history.append(item); terminals[name] = item

    probe_schema = "single_prefix_native_feature_probe_v1"
    gate_schema = "causal_native_single_prefix_validation_v1"
    attempt("probe-v1", schema=probe_schema, pid=1, start=1000., wall=90.1806557, complete=False)
    attempt("probe-v2", schema=probe_schema, pid=2, start=1200., wall=128.907835, complete=True)
    attempt("gate-resource-failed", schema=gate_schema, pid=3, start=1500., wall=19.9539156, complete=False, gate=True)
    attempt("gate-success", schema=gate_schema, pid=4, start=1700., wall=45., complete=True, gate=True)
    core = {"workspace": str(target), "fingerprint": "unit-new-core", "registration": CAUSAL_GRAPH_REGISTRATION,
        "normalization_amendment": source_core["normalization_amendment"], "passed_bank_reuse_contract": contract_artifact,
        "causal_graph_amendment": {"source_core_protocol": source_protocol, "pause_closure": closure,
            "validated_probe": terminals["gate-success"], "diagnostic_history": history}}
    monkeypatch.setattr(core_report, "_source_v3_cost_context", lambda value: (contract, source_core, bank, prior))
    return {"core": core, "contract": contract, "source_core": source_core, "bank": bank, "prior": prior,
            "closure": closure, "graph_path": graph_path, "report_path": report_path,
            "history": history, "terminals": terminals, "attempt": attempt}


def test_reuse_charges_original_tc_once_and_all_graph_diagnostics_without_final(causal_cost_case):
    case = causal_cost_case
    result = core_report.offline_transcoder_report(case["core"])
    assert result["complete"] and result["schema"] == "core_causal_graph_offline_costs_v1"
    assert not (Path(case["core"]["workspace"]) / "experiments/core_final").exists()
    history = result["historical_transcoder_attempts"]
    assert len(history) == len({row["run_id"] for row in history}) == 7
    assert history[-1]["category"] == "historical_reused_passed_bank_fit"
    assert history[-1]["wall_seconds"] == 787.388 and history[-1]["observed_layer_epochs"] == 2048
    assert result["current_transcoder_training"]["training_calls"] == 0
    assert result["current_transcoder_training"]["fit_wall_seconds"] == 0
    assert result["current_transcoder_training"]["reuse_validation_cpu_seconds"] is None
    graph = result["historical_graph_attempts"][0]
    assert graph["phase_wall_seconds"] == pytest.approx(738.493616)
    assert graph["persisted_decision_wall_seconds"] == pytest.approx(530.1104)
    assert graph["unattributed_stage_wall_seconds"] == pytest.approx(208.383216)
    assert graph["failed_graphs"] == graph["persisted_terminal_decisions"] == 16
    assert graph["not_terminal_decisions"] == 370 and graph["successful_graphs"] == 0
    assert graph["unpersisted_or_in_flight_computation"].startswith("unknown")
    diagnostics = result["diagnostic_attempts"]
    assert len(diagnostics) == 4
    assert [row["phase_wall_seconds"] for row in diagnostics] == [90.1806557, 128.907835, 19.9539156, 45.]
    assert len(diagnostics[1]["contained_phases"]) == 2
    assert all(row["scientific_oracle_calls"] == 0 and row["gpu_hours"] is None for row in diagnostics)
    assert diagnostics[-1]["single_prefix_gate_passed"] is True
    assert result["wall_time_total"] is None and result["gpu_hours"] is None
    assert result["scientific_oracle_calls"] == 0 and not result["full_graphs_NN_ES_or_final_acceptance_claimed"]


@pytest.mark.parametrize("mutation", ["omitted_probe", "duplicate_artifact", "duplicate_terminal", "changed_source", "nonzero_calls", "unknown_calls", "contradictory_calls", "timing", "missing_start", "missing_validated", "failed_validated"])
def test_diagnostic_identity_cost_and_source_gates(causal_cost_case, mutation):
    case = causal_cost_case; history = case["history"]
    item = case["terminals"]["probe-v1"]
    if mutation == "omitted_probe":
        directory = Path(item["path"]).parent
        history[:] = [x for x in history if Path(x["path"]).parent != directory]
    elif mutation == "duplicate_artifact": history.append(dict(item))
    elif mutation == "duplicate_terminal":
        path = Path(item["path"]).parent / "result.json"
        shutil.copyfile(item["path"], path); history.append(artifact(path))
    elif mutation == "changed_source": Path(item["path"]).write_text("{}")
    elif mutation == "missing_start":
        history[:] = [x for x in history if Path(x["path"]) != Path(item["path"]).parent / "started.json"]
    elif mutation == "missing_validated": history.remove(case["terminals"]["gate-success"])
    else:
        if mutation == "failed_validated": item = case["terminals"]["gate-success"]
        value = json.loads(Path(item["path"]).read_text())
        if mutation == "nonzero_calls": value["new_oracle_calls"] = 1
        elif mutation == "unknown_calls": del value["new_oracle_calls"]
        elif mutation == "contradictory_calls": value["new_material_oracle_calls"] = 1
        elif mutation == "timing": value["elapsed_seconds"] += 500.
        else: value["passed"] = False
        updated = sealed_document(Path(item["path"]), value)
        history[history.index(item)] = updated
        if mutation == "failed_validated": case["core"]["causal_graph_amendment"]["validated_probe"] = updated
    with pytest.raises(CoreFinalError): core_report.offline_transcoder_report(case["core"])


def test_unknown_unclosed_extra_diagnostic_is_preserved_not_zero(causal_cost_case):
    case = causal_cost_case
    case["attempt"]("unclosed-extra", schema="causal_native_single_prefix_validation_v1", pid=5,
        start=1800., wall=None, complete=False, gate=True, terminal=False)
    result = core_report.offline_transcoder_report(case["core"])
    last = result["diagnostic_attempts"][-1]
    assert last["phase_wall_seconds"] is None and last["outcome"] == "not_terminal_unknown_work"
    assert last["terminal"] is None and last["unrecorded_work"] == "unknown"


def test_cpu_source_audit_is_separate_from_tc_training_and_graph_wall(causal_cost_case, tmp_path):
    case = causal_cost_case
    script = tmp_path / "cpu-source.py"; script.write_text("# unit read-only audit source\n")
    summary = document(tmp_path / "cpu-audit/summary.json", {"schema": "passed_bank_source_audit_v1", "complete": True,
        "source_core_fingerprint": case["source_core"]["fingerprint"], "target_corpus_verified": False,
        "scientific_oracle_calls": 0, "new_policy_or_material_oracle_calls": 0, "gpu_model_calls": 0,
        "source": artifact(script), "audit_source": artifact(script), "elapsed_seconds": 381.927,
        "started_at_utc": "2026-09-16T11:56:34+00:00", "finished_at_utc": "2026-09-16T12:02:55.927+00:00",
        "build_elapsed_seconds": 191.580, "verify_elapsed_seconds": 188.605})
    case["history"].append(summary)
    result = core_report.offline_transcoder_report(case["core"])
    assert len(result["diagnostic_attempts"]) == 4 and len(result["cpu_admission_audits"]) == 1
    audit = result["cpu_admission_audits"][0]
    assert audit["cpu_task_wall_seconds"] == 381.927 and audit["cpu_seconds"] is None
    assert audit["gpu_model_calls"] == 0 and audit["scientific_oracle_calls"] == 0
    assert result["current_transcoder_training"]["fit_wall_seconds"] == 0
    assert result["historical_transcoder_attempts"][-1]["wall_seconds"] == 787.388


@pytest.mark.parametrize("mutation", ["not_closed", "changed_journal", "missing_journal", "duplicate_rows", "claimed_success", "shrunk_denominator"])
def test_graph_pause_failed_and_partial_denominators_are_not_dropped(causal_cost_case, mutation):
    case = causal_cost_case
    closure = json.loads(Path(case["closure"]["path"]).read_text())
    if mutation == "not_closed": closure["closed"] = False
    elif mutation == "missing_journal":
        closure["artifacts"] = [x for x in closure["artifacts"] if x["path"] != str(case["graph_path"])]
    elif mutation == "changed_journal": case["graph_path"].write_text("{}\n")
    elif mutation == "duplicate_rows":
        lines = case["graph_path"].read_text().splitlines()
        case["graph_path"].write_text("\n".join(lines + lines[:1]) + "\n")
        for item in closure["artifacts"]:
            if item["path"] == str(case["graph_path"]):
                item.update(artifact(case["graph_path"])); shutil.copyfile(case["graph_path"], item["preserved"])
    else:
        report = json.loads(case["report_path"].read_text())
        report["successful_graphs" if mutation == "claimed_success" else "expected_decisions"] = 1 if mutation == "claimed_success" else 16
        write_json_atomic(case["report_path"], report)
        for item in closure["artifacts"]:
            if item["path"] == str(case["report_path"]):
                item.update(artifact(case["report_path"])); shutil.copyfile(case["report_path"], item["preserved"])
    updated = document(Path(case["closure"]["path"]), closure)
    case["contract"]["pause_closure"] = updated
    case["core"]["causal_graph_amendment"]["pause_closure"] = updated
    with pytest.raises(CoreFinalError): core_report.offline_transcoder_report(case["core"])
