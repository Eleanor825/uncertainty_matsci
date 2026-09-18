"""Synthetic control-flow fixtures only; no physical results or real predictions."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
from matdiscovery.accounting import file_sha256


def load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def envelope(tmp_path, rows):
    path = tmp_path / "decisions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"artifacts": [{"path": str(path), "sha256": file_sha256(path)}],
            "episodes": [{"episode_id": "0", "metrics": {"synthetic_score": 1}}]}


def row(disposition="executed", label=1, confidence=.2):
    return {"unit_test_fixture": True, "episode_index": 0, "disposition": disposition,
            "label_future_failure": label, "predicted_failure_probability": confidence}


def test_rejected_candidates_are_not_counterfactual_risk_observations(tmp_path):
    report = load("report_full_study")
    value = envelope(tmp_path, [row(), row(disposition="rejected_not_executed", label=None, confidence=.9)])
    derived, counts, pairs = report.reliability_for_job(value, tmp_path)
    assert pairs == [(1, .2)]
    assert counts["not_executed_no_counterfactual_label"] == 1
    assert derived[0]["metrics"]["decision_brier"] == pytest.approx(.64)


def test_missing_confidence_is_not_silently_removed_from_paired_episode_metric(tmp_path):
    report = load("report_full_study")
    value = envelope(tmp_path, [row(), row(confidence=None)])
    derived, counts, pairs = report.reliability_for_job(value, tmp_path)
    assert len(pairs) == 1
    assert counts["executed_without_valid_confidence"] == 1
    assert derived[0]["metrics"]["decision_brier"] is None
    assert derived[0]["metrics"]["decision_overconfident_error_rate"] is None


def test_reporting_rejects_modified_confidence_evidence(tmp_path):
    report = load("report_full_study")
    value = envelope(tmp_path, [row()])
    (tmp_path / "decisions.jsonl").write_text(json.dumps(row(confidence=.99)) + "\n")
    with pytest.raises(RuntimeError, match="missing or changed"):
        report.reliability_for_job(value, tmp_path)


def test_training_stage_requires_complete_collection_matrix(tmp_path):
    stage = load("run_representation_condition")
    # Missing collection files must fail before any training or evaluator call.
    with pytest.raises((FileNotFoundError, RuntimeError)):
        stage.completed_collections(tmp_path, "qwen35_4b", "made")
