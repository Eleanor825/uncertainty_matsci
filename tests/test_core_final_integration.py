"""Tiny CPU policy + official-shaped synthetic B50 RPC, never main data."""
import json
from pathlib import Path

import pytest
import torch

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.core_final import CoreFinalError, CoreFinalLedger, CoreFinalRunner, verify_core_envelope
from matdiscovery.core_report import reliability
from test_core_final_report import unit_manifest
from test_final_evaluation import TinyPolicy
from test_training_jobs import project, write_mock_scientific_trajectory


def runner_fixture(project, tmp_path, calls, *, interrupted=False):
    runner = object.__new__(CoreFinalRunner)
    runner.project, runner.output = project, tmp_path / "core_workspace/experiments/core_final"
    runner.manifest = unit_manifest()
    runner.core = {"fingerprint": runner.manifest["core_protocol_fingerprint"]}
    runner.protocol = json.loads((project / "configs/main_protocol.json").read_text())
    runner.tasks = json.loads((project / "configs/benchmark_tasks.json").read_text())
    runner.runtime = {"device": "cpu", "torch_cpu_threads": 1, "cuda_memory_fraction": .2}
    runner.inputs = {str(project / "configs/benchmark_tasks.json"): file_sha256(project / "configs/benchmark_tasks.json")}
    runner.execution = {"classification": "unit_test_only", "inputs": runner.inputs}
    runner.corpus = {"classification": "unit_test_only"}
    runner.risk_root, runner.transcoder_manifest = tmp_path / "unit-risk", tmp_path / "unit-tc.json"
    runner.policy_factory = lambda key: TinyPolicy({"model_id": "unit/tiny-model", "revision": "unit-revision"})
    runner.controller_loader = lambda policy, **kwargs: (None, None, [])

    def selected(policy, directory, **kwargs):
        from matdiscovery.esopt import tensor_state_hash
        base = tensor_state_hash(dict(policy.model.state_dict()))
        with torch.no_grad():
            policy.model.weight.add_(.01)
        policy.mark_state("reload", generation=1)
        return {"classification": "unit_test_only", "selected_generation": 1,
                "actual_model_state_hash": tensor_state_hash(dict(policy.model.state_dict())),
                "initial_actual_model_state_hash": base, "artifacts": {}}
    runner.selected_loader = selected

    class UnitRollout:
        def __init__(self, root, policy, **kwargs):
            self.project = root

        def run(self, job, output, *, collection):
            assert not collection
            calls.append(job["job_id"])
            summaries = write_mock_scientific_trajectory(self.project, job, output)
            summaries[0]["metrics"]["mSUN"] = summaries[0]["discovery_curve"][-1][1] / 50
            write_json_atomic(output / "episodes.json", summaries)
            rows = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
            for row in rows:
                row.update(disposition="executed", label_future_failure=0, predicted_failure_probability=.2)
                if job["method"] == "esopt_graph_risk":
                    row.update(graph_status="unavailable", graph_error="explicit unit fixture does not construct a graph")
            (output / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            if interrupted:
                raise RuntimeError("unit interruption after closed physics-shaped transcript")
            return summaries
    runner.rollout_factory = UnitRollout
    return runner


def test_all_four_core_jobs_use_real_envelope_verifier_and_resume_without_replay(project, tmp_path):
    calls = []; runner = runner_fixture(project, tmp_path, calls)
    result = runner.run()
    assert result["complete"] and result["core_final_complete"] and result["completed_jobs"] == 4
    assert result["global_study_complete"] is False
    assert len(calls) == len(set(calls)) == 4
    assert (project / "experiments/core_stage_receipts/final.json").exists()
    original_receipts = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (
        runner.output / "completion.json", project / "experiments/core_stage_receipts/final.json")}
    runner.run()
    assert len(calls) == 4
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == original for path, original in original_receipts.items())
    record = next((runner.output / "jobs").glob("*/*/result.json"))
    envelope = json.loads(record.read_text())
    metrics = reliability(envelope, record.parent)
    assert metrics["confidence_complete"] and metrics["metrics"]["n"] == 1


def test_failed_core_attempt_is_kept_and_never_rerun(project, tmp_path):
    calls = []; runner = runner_fixture(project, tmp_path, calls, interrupted=True)
    with pytest.raises(RuntimeError, match="unit interruption"):
        runner.run()
    assert len(calls) == 1
    with pytest.raises(CoreFinalError, match="reconciliation"):
        runner.run()
    assert len(calls) == 1
    assert not (runner.output / "completion.json").exists()


def test_complete_envelope_before_commit_is_reconciled_not_replayed(project, tmp_path, monkeypatch):
    calls = []; runner = runner_fixture(project, tmp_path, calls)
    original = CoreFinalLedger.finish
    def crash(*args, **kwargs):
        raise KeyboardInterrupt("unit commit interruption")
    monkeypatch.setattr(CoreFinalLedger, "finish", crash)
    with pytest.raises(KeyboardInterrupt):
        runner.run()
    monkeypatch.setattr(CoreFinalLedger, "finish", original)
    result = runner.run()
    assert result["core_final_complete"] and len(calls) == len(set(calls)) == 4


def test_core_source_or_raw_evidence_tamper_cannot_be_skipped(project, tmp_path):
    calls = []; runner = runner_fixture(project, tmp_path, calls)
    runner.run()
    raw = next((runner.output / "jobs").glob("*/*/episodes.json"))
    raw.write_text("[]")
    with pytest.raises(ValueError):
        runner.run()
    assert len(calls) == 4


def test_core_msun_must_match_same_official_curve_even_after_rehash(project, tmp_path):
    calls = []; runner = runner_fixture(project, tmp_path, calls)
    runner.run()
    path = next((runner.output / "jobs").glob("*/*/result.json"))
    envelope = json.loads(path.read_text())
    episode_path = path.parent / "episodes.json"
    episodes = json.loads(episode_path.read_text()); episodes[0]["metrics"]["mSUN"] = .99
    write_json_atomic(episode_path, episodes)
    envelope["episodes"] = episodes
    from matdiscovery.training_jobs import TrainingJobCallbacks
    envelope["artifacts"] = [item for item in TrainingJobCallbacks._inventory(path.parent) if Path(item["path"]) != path]
    write_json_atomic(path, envelope)
    job = next(j for j in runner.manifest["jobs"] if j["job_id"] == envelope["job_id"])
    with pytest.raises(CoreFinalError, match="mSUN"):
        verify_core_envelope(path, job, runner.manifest, tasks=runner.tasks)


def report_dependencies(monkeypatch, runner, tmp_path):
    from matdiscovery import core_protocol, core_collection, core_es
    from test_core_final_report import cost_inputs
    corpus, es = cost_inputs(tmp_path)
    archive = tmp_path / "unit-prior-interruption"
    archive.mkdir()
    raw = archive / "unit_rpc.txt"; raw.write_text("synthetic prior closed-prefix fixture")
    reconciliation = archive / "reconciliation.json"
    write_json_atomic(reconciliation, {"schema": "interruption_reconciliation_v1", "all_requests_resolved": True,
        "unknown_physical_outcomes": 0, "used_for_training": False, "used_for_final_evaluation": False,
        "additional_incurred_costs_not_subtracted_from_main_budgets": True,
        "observed_physical_costs": {"initialization_oracle_attempts": 28, "candidate_oracle_attempts": 0},
        "artifacts": [{"path_after": str(raw), "sha256": file_sha256(raw)}]})
    core = {"workspace": str(runner.project), "fingerprint": runner.core["fingerprint"], "imported_train": [{}, {}, {}],
            "reconciliation": {"path": str(reconciliation), "sha256": file_sha256(reconciliation)}}
    monkeypatch.setattr(core_protocol, "read_core", lambda project: core)
    monkeypatch.setattr(core_protocol, "final_jobs", lambda source: runner.manifest["jobs"])
    # These two independently tested admission helpers are replaced only to
    # isolate final/report orchestration. Raw final verification remains real.
    monkeypatch.setattr(core_collection, "audit_core_corpus_costs", lambda source: corpus)
    monkeypatch.setattr(core_es, "audit_core_es_costs", lambda directory, source: es)


def test_core_report_emits_both_acceptance_receipts_only_after_all_four(project, tmp_path, monkeypatch):
    from matdiscovery.core_report import report_core
    calls = []; runner = runner_fixture(project, tmp_path, calls)
    runner.output = project / "experiments/core_final"
    runner.run()
    report_dependencies(monkeypatch, runner, tmp_path)
    report = report_core(project)
    assert report["core_complete"] and report["global_study_complete"] is False
    assert report["costs"]["incremental_core_costs"]["candidate_oracle_attempts"] == 550
    assert report["costs"]["prior_interrupted"]["costs"]["initialization_oracle_attempts"] == 28
    assert report["paired_results"]["aggregate"]["AUDC"]["interpretation"] == "no_observed_difference"
    assert (project / "experiments/core_reports/core_completion.json").exists()
    stage = json.loads((project / "experiments/core_stage_receipts/report.json").read_text())
    assert stage["complete"] and stage["global_study_complete"] is False
    assert len(calls) == 4


def test_incomplete_core_report_keeps_missing_jobs_and_never_publishes_acceptance(project, tmp_path, monkeypatch):
    from matdiscovery.core_report import report_core
    calls = []; runner = runner_fixture(project, tmp_path, calls, interrupted=True)
    runner.output = project / "experiments/core_final"
    with pytest.raises(RuntimeError):
        runner.run()
    report_dependencies(monkeypatch, runner, tmp_path)
    report = report_core(project)
    assert report["core_complete"] is False and report["completed_jobs"] == 0
    assert len(report["missing_jobs"]) == 4
    assert not (project / "experiments/core_reports/core_completion.json").exists()
    assert not (project / "experiments/core_stage_receipts/report.json").exists()


def test_reliability_preserves_missing_predictions_and_proposal_denominator(project, tmp_path):
    calls = []; runner = runner_fixture(project, tmp_path, calls)
    runner.run()
    path = next((runner.output / "jobs").glob("*/*/result.json"))
    envelope = json.loads(path.read_text())
    data_path = path.parent / "decisions.jsonl"
    rows = [json.loads(line) for line in data_path.read_text().splitlines()]
    rows[0]["predicted_failure_probability"] = None
    rejected = dict(rows[0], decision_id="unit-unexecuted", disposition="rejected_not_executed", label_future_failure=None)
    rows.append(rejected)
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    for item in envelope["artifacts"]:
        if Path(item["path"]) == data_path:
            item["sha256"] = file_sha256(data_path)
    result = reliability(envelope, path.parent)
    assert result["metrics"] is None and not result["confidence_complete"]
    assert result["executed_denominator"] == 1
    assert result["counts"]["total_proposed"] == 2
    assert result["counts"]["not_executed_no_counterfactual_label"] == 1
