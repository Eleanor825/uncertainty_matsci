"""Core bookkeeping/report contracts using synthetic unit records, never physics."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.core_final import (
    CoreFinalError, CoreFinalLedger, build_core_final_manifest, validate_core_final_manifest,
)
from matdiscovery.core_report import core_cost_report, pair_report


def unit_jobs():
    jobs = []
    for method in ("baseline", "esopt_graph_risk"):
        for elements in (["Al", "Li", "V"], ["Al", "V", "Zn"]):
            task = "-".join(elements)
            job = {"classification": "unit_test_only", "stage": "final_eval", "benchmark": "made", "model_key": "qwen35_4b",
                "model_id": "unit/tiny-model", "model_revision": "unit-revision", "method": method,
                "task_id": task, "task": {"id": task, "elements": elements}, "group_id": task, "split": "test",
                "seed": 1, "environment_seeds": [1], "episode_ids": ["0"], "budget": 50,
                "budget_unit": "candidate_oracle_attempts_per_episode",
                "expected_counts": {"episodes": 1, "candidate_oracle_attempts": 50, "dft_episode_attempts": 0}}
            job["job_id"] = "core-final-made-" + fingerprint(job)[:24]
            jobs.append(job)
    return jobs


def unit_manifest():
    return build_core_final_manifest(fingerprint("unit-core-not-study"), unit_jobs())


def records(manifest, *, full_deltas=(.1, -.1)):
    result = {}
    tasks = sorted({job["task_id"] for job in manifest["jobs"]})
    for job in manifest["jobs"]:
        delta = full_deltas[tasks.index(job["task_id"])] if job["method"] == "esopt_graph_risk" else 0
        result[job["job_id"]] = {"episode": {**{k: job[k] for k in ("benchmark", "model_key", "method", "task_id", "seed")},
            "status": "succeeded", "metrics": {"AUDC": .2 + delta, "mSUN": .3 + delta},
            "costs": {"candidate_oracle_attempts": 50, "initialization_oracle_attempts": 2}},
            "reliability": {"metrics": {"brier": .2 - delta, "overconfident_error_rate_p_le_0_1": .2 - delta}},
            "result": {"path": "unit-result", "sha256": "unit-hash"}}
    return result


def test_exact_independent_core_denominator_and_no_full_ledger(tmp_path):
    manifest = unit_manifest()
    assert manifest["expected_counts"]["jobs"] == 4
    assert manifest["expected_counts"]["candidate_oracle_attempts"] == 200
    ledger = CoreFinalLedger(tmp_path / "ledger.json", manifest)
    assert ledger.completion()["expected_jobs"] == 4
    assert ledger.completion()["global_study_complete"] is False
    bad = deepcopy(manifest); bad["jobs"].pop(); bad["fingerprint"] = fingerprint({k: v for k, v in bad.items() if k != "fingerprint"})
    with pytest.raises(CoreFinalError):
        validate_core_final_manifest(bad)


@pytest.mark.parametrize("state", ["failed", "orphaned"])
def test_core_failed_or_unknown_work_is_never_automatically_reclaimed(tmp_path, state):
    manifest = unit_manifest(); path = tmp_path / "ledger.json"
    ledger = CoreFinalLedger(path, manifest)
    job = manifest["jobs"][0]["job_id"]
    attempt = ledger.claim(job)
    ledger.halt(job, RuntimeError("synthetic incomplete physical outcome"), closed=state == "failed")
    resumed = CoreFinalLedger(path, manifest)
    assert resumed.data["jobs"][job]["state"] == state
    assert resumed.data["jobs"][job]["attempt_id"] == attempt
    with pytest.raises(CoreFinalError, match="blindly replayed"):
        resumed.claim(job)


def test_running_claim_is_orphaned_and_complete_commit_reconciles_without_second_claim(tmp_path):
    manifest = unit_manifest(); ledger = CoreFinalLedger(tmp_path / "ledger.json", manifest)
    job = manifest["jobs"][0]["job_id"]
    ledger.claim(job)
    resumed = CoreFinalLedger(ledger.path, manifest)
    resumed.quarantine_running()
    assert resumed.data["jobs"][job]["state"] == "orphaned"
    # This tests the commit operation only; runner must first verify the envelope.
    result = tmp_path / "unit-only-result.json"; write_json_atomic(result, {"classification": "unit_test_only"})
    resumed.finish(job, result, file_sha256(result), reconciled=True)
    assert resumed.data["jobs"][job]["attempt_number"] == 1
    assert resumed.completion()["completed_jobs"] == 1
    assert not resumed.completion()["core_final_complete"]


def test_changed_core_identity_or_fabricated_pending_history_is_rejected(tmp_path):
    manifest = unit_manifest(); path = tmp_path / "ledger.json"
    ledger = CoreFinalLedger(path, manifest)
    changed = build_core_final_manifest(fingerprint("different-core"), unit_jobs())
    with pytest.raises(CoreFinalError):
        CoreFinalLedger(path, changed)
    data = json.loads(path.read_text())
    data["jobs"][manifest["jobs"][0]["job_id"]]["attempt_number"] = 1
    write_json_atomic(path, data)
    with pytest.raises(CoreFinalError):
        CoreFinalLedger(path, manifest)


def test_report_keeps_missing_pairs_unknown_and_does_not_claim_small_n_significance():
    manifest = unit_manifest(); actual = records(manifest)
    report = pair_report(manifest, actual)
    assert report["aggregate"]["AUDC"]["interpretation"] == "mixed_or_tied_system_outcomes"
    assert report["aggregate"]["AUDC"]["population_significance_claim"] is False
    assert report["aggregate"]["AUDC"]["inferential_p_value"] is None
    actual.pop(manifest["jobs"][-1]["job_id"])
    incomplete = pair_report(manifest, actual)
    assert incomplete["aggregate"]["AUDC"]["mean_full_minus_baseline"] is None
    assert incomplete["aggregate"]["AUDC"]["paired_systems_observed"] == 1


@pytest.mark.parametrize("delta,label", [(.1, "positive_on_both_observed_systems"), (-.1, "negative_on_both_observed_systems"), (0., "no_observed_difference")])
def test_pair_outcome_direction_is_observed_not_assumed(delta, label):
    manifest = unit_manifest()
    report = pair_report(manifest, records(manifest, full_deltas=(delta, delta)))
    assert report["aggregate"]["AUDC"]["interpretation"] == label
    assert report["aggregate"]["brier"]["interpretation"] == label


def cost_inputs(tmp_path, imported=3):
    path = tmp_path / "cost_source.json"; write_json_atomic(path, {"classification": "unit_test_only"})
    evidence = [{"path": str(path), "sha256": file_sha256(path)}]
    corpus = {"complete": True, "imported_train": {"costs": {"candidate_oracle_attempts": imported * 50}},
        "new_development": {"costs": {"candidate_oracle_attempts": 50}}, "evidence_files": evidence}
    es = {"complete": True, "training": {"costs": {"candidate_oracle_attempts": 200}},
        "development": {"costs": {"candidate_oracle_attempts": 100}}, "evidence_files": evidence,
        "summary_total_ignored": {"candidate_oracle_attempts": 300}}
    return corpus, es


@pytest.mark.parametrize("imported", [2, 3])
def test_costs_separate_historical_imports_and_do_not_double_add_es_total(tmp_path, imported):
    manifest = unit_manifest(); corpus, es = cost_inputs(tmp_path, imported)
    report = core_cost_report(corpus, es, records(manifest), expected_imported_jobs=imported)
    assert report["incremental_core_costs"]["candidate_oracle_attempts"] == 550
    assert report["reused_plus_incremental_costs"]["candidate_oracle_attempts"] == 550 + imported * 50
    assert len(report["evidence_files"]) == 1


def test_incomplete_cost_source_or_missing_final_never_becomes_complete(tmp_path):
    manifest = unit_manifest(); corpus, es = cost_inputs(tmp_path)
    actual = records(manifest); actual.pop(next(iter(actual)))
    with pytest.raises(CoreFinalError):
        core_cost_report(corpus, es, actual, expected_imported_jobs=3)
    corpus["complete"] = False
    with pytest.raises(CoreFinalError):
        core_cost_report(corpus, es, records(manifest), expected_imported_jobs=3)


def test_report_cost_evidence_tampering_is_rejected(tmp_path):
    corpus, es = cost_inputs(tmp_path)
    Path(corpus["evidence_files"][0]["path"]).write_text("changed")
    with pytest.raises(CoreFinalError):
        core_cost_report(corpus, es, records(unit_manifest()), expected_imported_jobs=3)
