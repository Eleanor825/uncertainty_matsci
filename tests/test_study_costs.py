"""Synthetic cost accounting only: these tests invoke no physical evaluators."""
import hashlib
import json
from pathlib import Path

import pytest

from matdiscovery.accounting import file_sha256, fingerprint
from matdiscovery.experiment_plan import collection_jobs
from matdiscovery import study_costs as costs
from matdiscovery.collection_provenance import CollectionProvenanceError
from collection_cost_fixtures import write_cost_collection_fixtures


PROJECT = Path(__file__).resolve().parents[1]


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_shared_all_720_collection_jobs_counted_once(tmp_path, monkeypatch):
    jobs = collection_jobs(PROJECT)
    monkeypatch.setattr(costs, "collection_jobs", lambda root: jobs)
    directories = write_cost_collection_fixtures(tmp_path, jobs)
    report = costs.audit_collection_costs(tmp_path)
    assert report["completed_jobs"] == 720 and report["costs"]["completed_episodes"] == 2220
    assert report["costs"]["candidate_oracle_attempts"] == 21000
    assert report["costs"]["dft_episode_attempts"] == 1800
    assert report["costs"]["initialization_oracle_attempts"] == 840
    assert report["costs"]["surrogate_oracle_attempts"] == 420
    assert len(report["conditions"]) == 4 and report["counted_once_across_all_method_arms"]
    journals = [row["source_provenance"]["oracle_attempt_journal"] for row in report["jobs"] if row["benchmark"] == "made"]
    assert len(journals) == 420 and all(j["valid"] and j["closed"] for j in journals)
    assert sum(j["counts"]["candidate_oracle_attempts"] for j in journals) == 21000
    raw = directories[-1] / "decisions.jsonl"
    raw.write_text("changed")
    with pytest.raises(CollectionProvenanceError, match="changed"):
        costs.audit_collection_costs(tmp_path)


def stage_fixture(root):
    planned, states = [], {}
    counts = {"collect": 4, "transcoders": 4, "graphs": 4, "risk": 4, "es": 80, "final": 4}
    for kind, count in counts.items():
        for i in range(count):
            stage = {"id": f"{kind}-synthetic-{i}", "command": ["never-executed-fixture", kind, str(i)]}
            start = 1000 + len(planned) * 10
            states[stage["id"]] = {"status": "succeeded", "returncode": 0, "started_at": start, "finished_at": start + 5,
                "command": stage["command"], "command_fingerprint": hashlib.sha256(json.dumps(stage, sort_keys=True).encode()).hexdigest()}
            planned.append(stage)
    write(root / "configs/stage_queue.json", {"stages": planned})
    write(root / "logs/pipeline_status.json", {"stages": states})
    return planned, states


def test_representation_and_all_stage_wall_times_include_offline_training(tmp_path):
    _, states = stage_fixture(tmp_path)
    report = costs.audit_stage_times(tmp_path)
    assert report["complete"] and report["known_stage_wall_seconds"] == 500
    assert report["representation_expected_stages"] == 12 and report["representation_wall_seconds"] == 60
    assert report["by_kind_known_wall_seconds"]["es"] == 400
    states["risk-synthetic-0"]["finished_at"] = None
    write(tmp_path / "logs/pipeline_status.json", {"stages": states})
    report = costs.audit_stage_times(tmp_path)
    assert not report["complete"] and report["representation_wall_seconds"] is None
    assert report["representation_known_wall_seconds"] == 55 and len(report["unknown"]) == 1


def archive_fixture(root):
    directory = root / "experiments/interrupted/001_prompt_context_failure"
    raw = directory / "condition/raw.json"
    write(raw, {"synthetic_only": True})
    receipt = {"schema": "interruption_reconciliation_v1", "all_requests_resolved": True, "previous_processes_stopped": [123],
        "used_for_training": False, "used_for_final_evaluation": False, "additional_incurred_costs_not_subtracted_from_main_budgets": True,
        "observed_physical_costs": {"initialization_oracle_attempts": 28, "surrogate_oracle_attempts": 6, "candidate_oracle_attempts": 1},
        "artifacts": [{"path_after": str(raw), "sha256": file_sha256(raw)}]}
    write(directory / "reconciliation.json", receipt)
    write(directory / "pipeline_status_before.json", {"stages": {"collect-old": {"status": "failed", "started_at": 10., "finished_at": 228.2}}})
    return directory


def test_archive_and_technical_calls_separate_no_borrowed_counter_double_count(tmp_path):
    directory = archive_fixture(tmp_path)
    report = costs.audit_interrupted_costs(tmp_path)
    assert report["costs"] == {"initialization_oracle_attempts": 28, "surrogate_oracle_attempts": 6, "candidate_oracle_attempts": 1}
    assert report["known_archived_stage_wall_seconds"] == pytest.approx(218.2)
    write(tmp_path / "technical_not_main/prompt_summary_regression.json", {"scientific_calls": 0,
          "last_scientific_counts": {"initialization_oracle_attempts": 28, "surrogate_oracle_attempts": 6, "candidate_oracle_attempts": 1}})
    write(tmp_path / "technical_not_main/dft/summary.json", {"classification": "technical_not_main", "scientific_main_result": False,
          "counters": {"dft_episode_attempts": 1}, "elapsed_seconds": 200., "status": "diagnostic"})
    write(tmp_path / "technical_not_main/unrecognized/summary.json", {"classification": "technical_not_main", "scientific_main_result": False,
          "configured_budget": 50})
    technical = costs.audit_technical_costs(tmp_path)
    assert technical["known_scientific_call_counts"] == {"dft_episode_attempts": 1}
    assert technical["excluded_from_main_results"] and len(technical["unknown"]) == 1
    (directory / "condition/raw.json").write_text("corrupt")
    with pytest.raises(RuntimeError, match="Archived"):
        costs.audit_interrupted_costs(tmp_path)


def test_end_to_end_keeps_matched_budget_and_counts_shared_cost_once(tmp_path, monkeypatch):
    _, states = stage_fixture(tmp_path)
    archive_fixture(tmp_path)
    monkeypatch.setattr(costs, "audit_collection_costs", lambda root: {"complete": True, "costs": {"candidate_oracle_attempts": 100, "wall_seconds": 1000}, "evidence_files": []})
    monkeypatch.setattr(costs, "audit_technical_costs", lambda root: {"known_scientific_call_counts": {"dft_episode_attempts": 4}, "evidence_files": [], "unknown": [{"reason": "fixture unknown"}]})
    summaries = [{"model_key": "qwen35_4b", "benchmark": "made", "method": method, "task_id": "fixture", "costs": {"candidate_oracle_attempts": 5, "wall_seconds": 2}}
                 for method in ("baseline", "esopt")]
    es = {str(i): {"costs": {"candidate_oracle_attempts": 1, "wall_seconds": 1}} for i in range(80)}
    prior = {"known_prior_attempt_costs": {"candidate_oracle_attempts": 2}, "all_attempt_costs_accounted": True}
    report = costs.build_study_cost_report(tmp_path, summaries, es, prior)
    assert report["matched_final_budget"]["costs"]["candidate_oracle_attempts"] == 10
    assert report["experimental_end_to_end"]["known_physical_call_counts"]["candidate_oracle_attempts"] == 193
    assert report["experimental_end_to_end"]["stage_wall_seconds"] == pytest.approx(718.2)
    assert report["experimental_end_to_end"]["known_measurement_subtotals"]["wall_seconds"] == 1084
    assert report["known_physical_calls_including_technical"]["dft_episode_attempts"] == 4
    assert report["experimental_cost_accounting_complete"] is True
    states["graphs-synthetic-0"]["finished_at"] = None
    write(tmp_path / "logs/pipeline_status.json", {"stages": states})
    report = costs.build_study_cost_report(tmp_path, summaries, es, prior)
    assert not report["experimental_cost_accounting_complete"] and report["experimental_end_to_end"]["stage_wall_seconds"] is None


def test_supervisor_referenced_missing_archive_is_unknown_not_free(tmp_path):
    path = tmp_path / "experiments/interrupted/missing/reconciliation.json"
    write(tmp_path / "logs/pipeline_status.json", {"reconciliations": [str(path)]})
    report = costs.audit_interrupted_costs(tmp_path)
    assert not report["complete"] and report["costs"] == {} and report["unknown"]
