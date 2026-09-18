"""Statistical contract tests on unmistakably synthetic episodes."""

import copy
from pathlib import Path

import pytest

from matdiscovery.accounting import AccountingError, build_final_manifest
from matdiscovery.metrics import (
    IncompleteResultsError, audit_summaries, paired_cluster_bootstrap, summarize_results,
)


@pytest.fixture(scope="module")
def manifest():
    return build_final_manifest(Path(__file__).resolve().parents[1])


@pytest.fixture(scope="module")
def full_rows(manifest):
    rows = []
    for job in manifest["jobs"]:
        for episode, env_seed in zip(job["episode_ids"], job["environment_seeds"]):
            rows.append({
                **{key: job[key] for key in ("benchmark", "model_key", "method", "task_id", "seed")},
                "episode_id": episode, "environment_seed": env_seed,
                "complete": True, "status": "succeeded",
                "metrics": {"synthetic_score": job["seed"] + (2 if job["method"] == "graph_risk" else 0)},
                "costs": {"candidate_oracle_attempts": 50 if job["benchmark"] == "made" else 0,
                          "dft_episode_attempts": int(job["benchmark"] == "crystalgym")},
            })
    return rows


def test_final_report_requires_every_expected_episode_and_budget(manifest, full_rows):
    assert audit_summaries(full_rows, manifest, final=True)["complete"] is True
    with pytest.raises(IncompleteResultsError) as failure:
        summarize_results(full_rows[:-1], manifest)
    assert failure.value.report["missing_episodes"] == 1
    assert failure.value.report["failure_denominator_expected"] == 3600
    changed = copy.deepcopy(full_rows)
    name = "candidate_oracle_attempts" if changed[0]["benchmark"] == "made" else "dft_episode_attempts"
    changed[0]["costs"][name] -= 1
    with pytest.raises(IncompleteResultsError) as failure:
        summarize_results(changed, manifest)
    assert failure.value.report["budget_mismatch_episodes"] == 1


def test_failed_scientific_episodes_remain_in_denominators(manifest, full_rows):
    rows = copy.deepcopy(full_rows)
    row = rows[0]
    row["status"] = "failed"
    row["metrics"] = {"synthetic_score": -10, "property_abs_error": None}
    report = summarize_results(rows, manifest)
    assert report["accounting"]["failed_episode_count"] == 1
    assert report["accounting"]["failure_denominator_expected"] == 3600
    task = next(t for t in report["tasks"] if all(t[k] == row[k] for k in ("benchmark", "model_key", "method", "task_id")))
    assert task["failed_episodes"] == 1
    assert task["metrics"]["synthetic_score"]["missing_values"] == 0
    assert task["metrics"]["property_abs_error"]["mean"] is None
    assert task["metrics"]["property_abs_error"]["expected_denominator"] == task["expected_episodes"]


def test_paired_cluster_bootstrap_known_delta_and_order_invariance(manifest, full_rows):
    arguments = dict(benchmark="crystalgym", model_key="qwen35_4b", method="graph_risk", reference="baseline", metric="synthetic_score", manifest=manifest, n_bootstrap=300, seed=41)
    result = paired_cluster_bootstrap(full_rows, **arguments)
    assert result["delta"] == pytest.approx(2)
    assert result["ci"] == pytest.approx([2, 2])
    assert result["n_tasks"] == 6
    assert result["n_seed_pairs"] == 30
    assert result["n_episodes_per_method"] == 150
    assert paired_cluster_bootstrap(reversed(full_rows), **arguments) == result


def test_cluster_means_equal_weight_tasks_and_seeds_not_episode_rows():
    rows = []
    # One task has many within-seed episodes. It must still have half the weight.
    for task, episode_count, difference in (("A", 20, 0), ("B", 1, 4)):
        for seed in (1, 2):
            for episode in range(episode_count):
                for method in ("baseline", "graph_risk"):
                    rows.append({"benchmark": "synthetic", "model_key": "tiny",
                                 "method": method, "task_id": task, "seed": seed,
                                 "episode_id": str(episode), "complete": True,
                                 "status": "succeeded", "costs": {},
                                 "metrics": {"score": difference if method == "graph_risk" else 0}})
    result = paired_cluster_bootstrap(rows, benchmark="synthetic", model_key="tiny", method="graph_risk", reference="baseline", metric="score", n_bootstrap=100, seed=2)
    assert result["delta"] == 2
    assert result["n_seed_pairs"] == 4
    assert result["coverage_verified_against_manifest"] is False


def test_missing_paired_episode_and_undefined_failed_metric_are_rejected(manifest, full_rows):
    arguments = dict(benchmark="made", model_key="qwen35_4b", method="graph_risk", reference="baseline", metric="synthetic_score", manifest=manifest, n_bootstrap=10)
    rows = [r for r in full_rows if not (r["benchmark"] == "made" and r["model_key"] == "qwen35_4b" and r["method"] == "baseline" and r["seed"] == 1)]
    with pytest.raises(AccountingError, match="missing expected"):
        paired_cluster_bootstrap(rows, **arguments)
    rows = copy.deepcopy(full_rows)
    row = next(r for r in rows if r["benchmark"] == "made" and r["model_key"] == "qwen35_4b" and r["method"] == "graph_risk")
    row["status"] = "failed"
    row["metrics"]["synthetic_score"] = None
    with pytest.raises(AccountingError, match="dropping failed"):
        paired_cluster_bootstrap(rows, **arguments)


def test_repeated_step_rows_cannot_inflate_effective_sample_size(manifest, full_rows):
    with pytest.raises(AccountingError, match="Duplicate episode"):
        audit_summaries(full_rows + [full_rows[0]], manifest)


def test_unlike_property_units_are_not_silently_averaged(manifest, full_rows):
    rows = copy.deepcopy(full_rows)
    for row in rows:
        row["metrics"]["property_abs_error"] = 1.0
        row["costs"]["llm_tokens"] = 3
    assert audit_summaries(rows, manifest)["observed_costs"]["llm_tokens"] == 10800
    with pytest.raises(AccountingError, match="different units"):
        paired_cluster_bootstrap(rows, benchmark="crystalgym", model_key="qwen35_4b", method="graph_risk", reference="baseline", metric="property_abs_error", manifest=manifest, n_bootstrap=10)
