#!/usr/bin/env python3
"""Verify the COMPLETE current two-model MADE phase; never mark the study done.

The immutable 2160-job full-study manifest is retained and validated unchanged.
Only its MADE projection is eligible for this separate phase receipt. Shared
full-report evidence/confidence/failed-attempt checks and statistical estimators
are reused, and the default full-study accounting gate is never relaxed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import sqlite3

from matdiscovery.accounting import (
    METHODS, MODEL_KEYS, SEEDS, build_final_manifest, file_sha256, fingerprint,
    validate_manifest, write_json_atomic,
)
from matdiscovery.metrics import calibration_metrics, paired_cluster_bootstrap, summarize_results
from matdiscovery.failure_reporting import FailureTypeReporter, failure_control_required
from matdiscovery.study_costs import build_study_cost_report


_SHARED_PATH = Path(__file__).with_name("report_full_study.py")
_spec = importlib.util.spec_from_file_location("_shared_full_study_report_for_made", _SHARED_PATH)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
# Exactly the existing scientific receipt verifier, including all raw hashes.
verify_final_envelope = _shared.verify_final_envelope
reliability_for_job = _shared.reliability_for_job
audit_attempt_history = _shared.audit_attempt_history

SCOPE = "full_current_two_model_MADE_phase"
SCHEMA = "made_phase_execution_audit_v1"
FINAL_COUNTS = {"jobs": 1800, "episodes": 1800, "candidate_oracle_attempts": 90000, "dft_episode_attempts": 0}
COLLECTION_COUNTS = {"jobs": 420, "episodes": 420, "candidate_oracle_attempts": 21000, "dft_episode_attempts": 0}
ES_CONDITIONS = 20
ES_ORB_ATTEMPTS = 176000
REPORT_ARTIFACTS = {"verified_made_final_evidence.json", "made_summary.json", "made_paired_estimates.json",
                    "made_calibration.json", "made_failure_type_calibration.json", "made_ledger_attempt_events.json", "made_attempt_accounting.json", "made_study_costs.json"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def made_jobs(manifest):
    validate_manifest(manifest)  # FULL immutable matrix, not a re-fingerprinted subset.
    jobs = [job for job in manifest["jobs"] if job["benchmark"] == "made"]
    counts = {"jobs": len(jobs), **{key: sum(job["expected_counts"][key] for job in jobs) for key in FINAL_COUNTS if key != "jobs"}}
    require(counts == FINAL_COUNTS and manifest["expected_by_benchmark"]["made"] == FINAL_COUNTS,
            "MADE phase requires all 1800 jobs/episodes and 90000 candidate ORB attempts")
    return jobs


def summarize_made_phase(summaries, manifest):
    """Reuse the full auditor, then project its accounting without imputing CG."""
    made_jobs(manifest)
    require(all(row["benchmark"] == "made" for row in summaries), "MADE phase summaries contain another benchmark")
    global_partial = summarize_results(summaries, manifest, final=False)
    audit = global_partial["accounting"]
    missing = [key for key in audit["missing_episode_keys"] if key[0] == "made"]
    incomplete = [key for key in audit["incomplete_episode_keys"] if key[0] == "made"]
    mismatched = [key for key in audit["budget_mismatch_episode_keys"] if key[0] == "made"]
    require(not missing and not incomplete and not mismatched and audit["observed_counts"] == FINAL_COUNTS,
            f"MADE phase incomplete: {len(missing)} missing, {len(incomplete)} incomplete, {len(mismatched)} budget mismatches")
    phase_audit = {key: audit[key] for key in ("observed_counts", "observed_costs", "completed_jobs", "completed_episodes", "status_counts", "failed_episode_count", "observed_failed_fraction")}
    phase_audit.update(expected_counts=dict(FINAL_COUNTS), complete=True, missing_episodes=0, incomplete_episodes=0,
                       budget_mismatch_episodes=0, failure_denominator_expected=1800, failure_denominator_observed=1800,
                       full_manifest_fingerprint=manifest["manifest_fingerprint"], scope=SCOPE)
    return {"report_kind": "complete_benchmark_phase", "benchmark": "made", "scope": SCOPE,
            "global_study_complete": False, "accounting": phase_audit,
            "tasks": [row for row in global_partial["tasks"] if row["benchmark"] == "made"],
            "other_benchmarks": {"crystalgym": "outside_this_phase_not_counted_as_MADE_failures"}}


def validate_es_conditions(conditions):
    expected = {(model, method, seed) for model in MODEL_KEYS for method in ("esopt", "esopt_graph_risk") for seed in SEEDS}
    observed = []
    totals = Counter()
    for entry in conditions.values():
        require(entry["benchmark"] == "made" and entry.get("property") is None, "MADE ES receipt belongs to another benchmark/property")
        observed.append((entry["model_key"], entry["method"], entry["seed"]))
        costs = entry["costs"]
        require(type(costs.get("candidate_oracle_attempts")) is int and costs["candidate_oracle_attempts"] == 8800
                and costs.get("dft_episode_attempts", 0) == 0,
                "Each MADE ES condition needs all 8800 training/development candidate ORB attempts")
        totals.update(costs)
    require(len(conditions) == ES_CONDITIONS and len(set(observed)) == ES_CONDITIONS and set(observed) == expected,
            "MADE phase needs the exact 2 models x 2 ES methods x 5 seeds = 20 independent conditions")
    require(totals["candidate_oracle_attempts"] == ES_ORB_ATTEMPTS, "MADE ES physical budget must total 176000")
    return dict(totals)


def validate_cost_gate(costs, attempts):
    require(attempts["all_attempt_costs_accounted"], "MADE prior failed-attempt physical costs remain unknown")
    require(costs["benchmark_scope"] == "made" and costs["experimental_cost_accounting_complete"],
            "MADE end-to-end cost accounting remains incomplete")
    collection = costs["shared_collection"]
    require(collection["complete"] and collection["expected_jobs"] == collection["completed_jobs"] == 420
            and collection["costs"]["completed_episodes"] == 420
            and collection["costs"]["candidate_oracle_attempts"] == 21000
            and collection["costs"].get("dft_episode_attempts", 0) == 0,
            "MADE collection needs all 420 jobs/episodes and 21000 candidate ORB attempts")
    stages = costs["offline_and_other_stage_times"]
    require(stages["complete"] and stages["expected_stages"] == 30 and stages["representation_expected_stages"] == 6,
            "MADE cost gate requires 30 complete experimental stages including six representation stages")
    categories = costs["experimental_end_to_end"]["cost_categories"]
    require(categories["independent_es_training"]["candidate_oracle_attempts"] == 176000
            and categories["matched_final_evaluation"]["candidate_oracle_attempts"] == 90000,
            "MADE ES/final physical counters do not match the complete phase")


def make_phase_acceptance(manifest, summary, conditions, attempts, costs, *, evidence_hash, artifacts, primary):
    made_jobs(manifest)
    require(summary["accounting"]["complete"] and summary["accounting"]["expected_counts"] == FINAL_COUNTS
            and summary["accounting"]["observed_counts"] == FINAL_COUNTS, "Incomplete MADE result denominator")
    validate_es_conditions(conditions)
    validate_cost_gate(costs, attempts)
    require(len(artifacts) == len(REPORT_ARTIFACTS) and {Path(item["path"]).name for item in artifacts} == REPORT_ARTIFACTS,
            "MADE acceptance requires every audited report artifact")
    for item in artifacts:
        require(file_sha256(item["path"]) == item["sha256"], "MADE report artifact changed before acceptance")
    evidence = next(item for item in artifacts if Path(item["path"]).name == "verified_made_final_evidence.json")
    require(evidence["sha256"] == evidence_hash, "MADE evidence inventory hash mismatch")
    receipt = {"schema": SCHEMA, "status": "complete", "scope": SCOPE,
        "made_phase_complete": True, "all_made_final_experiments_complete": True,
        "global_study_complete": False, "all_final_experiments_complete": False,
        "completed_models": list(MODEL_KEYS), "completed_model_count": 2,
        "latest_3_to_4_model_extension_complete": False,
        "method_alignment_gate_evaluated_by_this_report": False,
        "requested_full_method_completion_not_asserted": True,
        "remaining_scope": ["CrystalGym full 360 final jobs / 1800 DFT episodes", "User-requested latest 3-to-4-model extension remains unfulfilled"],
        "expected_counts": dict(FINAL_COUNTS), "expected_final_counts": dict(FINAL_COUNTS), "expected_collection_counts": dict(COLLECTION_COUNTS),
        "expected_es_conditions": ES_CONDITIONS, "accounted_es_conditions": len(conditions),
        "expected_es_candidate_oracle_attempts": ES_ORB_ATTEMPTS,
        "full_study_manifest_fingerprint": manifest["manifest_fingerprint"],
        "full_study_expected_counts_unchanged": manifest["expected_counts"],
        "verified_made_evidence_sha256": evidence_hash,
        "all_attempt_costs_accounted": True, "all_es_training_conditions_accounted": True,
        "all_end_to_end_experimental_costs_accounted": True,
        "ledger_events_fingerprint": attempts["ledger_events_fingerprint"],
        "primary_full_method_comparisons": primary,
        "improvement_is_assessed_per_endpoint_not_assumed": True,
        "negative_or_inconclusive_results_must_be_reported": True,
        "technical_diagnostics_are_not_main_results": True,
        "artifacts": artifacts}
    receipt["fingerprint"] = fingerprint(receipt)
    return receipt


def phase_estimates(summaries, manifest, protocol):
    require(protocol["models"] == list(MODEL_KEYS) and protocol["methods"] == list(METHODS)
            and protocol["training_seeds"] == list(SEEDS), "MADE reporting protocol lost models, methods or seeds")
    contrasts = [(method, "baseline") for method in protocol["methods"] if method != "baseline"]
    contrasts += [("graph_risk", "entropy_risk"), ("graph_risk", "hidden_risk"), ("esopt_graph_risk", "esopt")]
    estimates = []
    for model in MODEL_KEYS:
        for method, reference in contrasts:
            selected = [row for row in summaries if row["model_key"] == model and row["method"] in {method, reference}]
            for metric in ("AUDC", "mSUN", "decision_brier", "decision_overconfident_error_rate"):
                if any(row["metrics"].get(metric) is None for row in selected):
                    estimates.append({"benchmark": "made", "model_key": model, "stratum": "all_systems",
                        "method": method, "reference": reference, "metric": metric,
                        "status": "undefined_incomplete_confidence_coverage" if metric.startswith("decision_") else "undefined_scientific_metric_no_failed_episode_dropping"})
                    continue
                result = paired_cluster_bootstrap(summaries, benchmark="made", model_key=model, method=method,
                    reference=reference, metric=metric, manifest=manifest,
                    n_bootstrap=protocol["statistics"]["bootstrap_replicates"], confidence=protocol["statistics"]["confidence_level"], seed=1729)
                lower_better = metric.startswith("decision_")
                result.update(stratum="all_systems", status="estimated", higher_is_better=not lower_better,
                    interval_excludes_no_improvement=result["ci"][1] < 0 if lower_better else result["ci"][0] > 0,
                    interval_multiplicity="unadjusted_pointwise_95_percent_not_a_familywise_claim")
                estimates.append(result)
    return estimates


def report_made_phase(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = build_final_manifest(root)
    jobs = made_jobs(manifest)
    final = root / "experiments/final_evaluation"
    require(json.loads((final / "manifest.json").read_text()) == manifest,
            "Stored final manifest differs from the immutable full study")
    database = sqlite3.connect(f"file:{final / 'ledger.sqlite3'}?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    try:
        all_states = {row["job_id"]: dict(row) for row in database.execute("SELECT * FROM jobs")}
        all_events = [{**dict(row), "details": json.loads(row["details"])} for row in database.execute("SELECT * FROM events ORDER BY event_id")]
    finally:
        database.close()
    require(set(all_states) == {job["job_id"] for job in manifest["jobs"]},
            "Ledger must preserve all 2160 declared jobs even while CrystalGym is deferred")
    made_ids = {job["job_id"] for job in jobs}
    states = {job_id: state for job_id, state in all_states.items() if job_id in made_ids}
    require(len(states) == 1800 and all(state["state"] == "succeeded" for state in states.values()),
            "MADE phase requires all 1800 verified completed final jobs")
    events = [event for event in all_events if event["job_id"] in made_ids]
    attempts = audit_attempt_history(events, states)
    tasks = json.loads((root / "configs/benchmark_tasks.json").read_text())
    protocol = json.loads((root / "configs/main_protocol.json").read_text())
    summaries, evidence, counts_by_method, scores_by_method = [], [], defaultdict(Counter), defaultdict(list)
    conditions = {}
    failure_types = FailureTypeReporter()
    for index, job in enumerate(jobs):
        state = states[job["job_id"]]
        path = Path(state["result_path"])
        require(file_sha256(path) == state["result_sha256"], "MADE ledger result changed after completion")
        from matdiscovery.execution_contract import declared_mace_workers
        verify_final_envelope(path, job, manifest["manifest_fingerprint"], tasks=tasks,
            expected_mace_num_workers=declared_mace_workers(protocol))
        envelope = json.loads(path.read_text())
        training = envelope["execution_profile"].get("selected_es")
        require(bool(training) == (job["method"] in {"esopt", "esopt_graph_risk"}), "MADE method and selected ES provenance disagree")
        if training:
            identity = training["training_run_fingerprint"]
            entry = {"model_key": job["model_key"], "benchmark": "made", "method": job["method"], "seed": job["seed"],
                     "property": None, "training_directory": training["training_directory"], "costs": training["training_costs"]}
            require(identity not in conditions or conditions[identity] == entry,
                    "ES training was reused across independent MADE model/method/seed conditions")
            conditions[identity] = entry
        derived, counts, pairs = reliability_for_job(envelope, path.parent)
        failure_types.add_job(envelope, path.parent, job, required=failure_control_required(protocol, "made"))
        summaries.extend(derived)
        key = (job["model_key"], job["method"])
        counts_by_method[key].update(counts)
        scores_by_method[key].extend(pairs)
        evidence.append({"job_id": job["job_id"], "path": str(path), "sha256": state["result_sha256"]})
        if (index + 1) % 25 == 0:
            print(json.dumps({"verified_made_final_jobs": index + 1, "expected_jobs": 1800}), flush=True)
    summary = summarize_made_phase(summaries, manifest)
    es_costs = validate_es_conditions(conditions)
    attempts.update(expected_es_training_conditions=20, accounted_es_training_conditions=20,
                    es_training_conditions=conditions, actual_es_training_costs=es_costs,
                    verified_final_costs=summary["accounting"]["observed_costs"], scope=SCOPE,
                    non_MADE_ledger_events_excluded_from_phase_not_assumed_success=True)
    costs = build_study_cost_report(root, summaries, conditions, attempts, benchmark="made")
    estimates = phase_estimates(summaries, manifest, protocol)
    calibration = [{"benchmark": "made", "model_key": model, "method": method, "property": "stability",
                    "coverage_counts": dict(counts),
                    "pooled_descriptive_step_metrics": calibration_metrics(*zip(*scores_by_method[(model, method)])) if scores_by_method[(model, method)] else None,
                    "unit_warning": "Dependent within-episode decisions; pooled calibration is descriptive, paired intervals cluster by task/seed."}
                   for (model, method), counts in sorted(counts_by_method.items())]
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {"verified_made_final_evidence.json": evidence, "made_summary.json": summary,
                 "made_paired_estimates.json": estimates, "made_calibration.json": calibration,
                 "made_failure_type_calibration.json": failure_types.report(),
                 "made_ledger_attempt_events.json": events, "made_attempt_accounting.json": attempts,
                 "made_study_costs.json": costs}
    for name, payload in artifacts.items():
        write_json_atomic(output / name, payload)
    primary = [row for row in estimates if row["method"] == "esopt_graph_risk" and row["reference"] == "baseline"]
    acceptance = make_phase_acceptance(manifest, summary, conditions, attempts, costs,
        evidence_hash=file_sha256(output / "verified_made_final_evidence.json"),
        artifacts=[{"path": str(output / name), "sha256": file_sha256(output / name)} for name in artifacts], primary=primary)
    write_json_atomic(output / "made_phase_acceptance.json", acceptance)
    return acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = report_made_phase(args.project, args.output)
    except Exception as error:
        # A failed rerun cannot leave an old successful gate at the public path.
        failure = {"schema": SCHEMA, "scope": SCOPE, "status": "audit_failed", "made_phase_complete": False,
                   "all_made_final_experiments_complete": False, "global_study_complete": False,
                   "all_final_experiments_complete": False, "completed_models": [],
                   "latest_3_to_4_model_extension_complete": False, "error": str(error)}
        failure["fingerprint"] = fingerprint(failure)
        write_json_atomic(args.output / "made_phase_acceptance.json", failure)
        raise
    print(json.dumps({"status": "complete_current_two_model_MADE_phase", "global_study_complete": False,
                      "acceptance": str(args.output / "made_phase_acceptance.json"), "fingerprint": receipt["fingerprint"]}), flush=True)


if __name__ == "__main__":
    main()
