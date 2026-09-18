#!/usr/bin/env python3
"""Audit all final evidence before reporting paired material and risk outcomes."""
import argparse
from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path
import sqlite3

from matdiscovery.accounting import build_final_manifest, file_sha256, fingerprint, write_json_atomic
from matdiscovery.final_evaluation import verify_final_envelope
from matdiscovery.failure_reporting import FailureTypeReporter, failure_control_required
from matdiscovery.metrics import calibration_metrics, paired_cluster_bootstrap, summarize_results
from matdiscovery.study_costs import build_study_cost_report


def reliability_for_job(envelope, directory):
    path = directory / "decisions.jsonl"
    candidates = [a for a in envelope["artifacts"] if Path(a["path"]).resolve() == path.resolve()]
    if len(candidates) != 1 or file_sha256(path) != candidates[0]["sha256"]:
        raise RuntimeError("Decision confidence evidence is missing or changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    grouped = defaultdict(list)
    counts = Counter(total_proposed=len(rows))
    for row in rows:
        if row.get("graph_status") in {"succeeded", "unavailable"}:
            counts["graph_proposed"] += 1
            counts["graph_" + row["graph_status"]] += 1
        if row["disposition"] != "executed":
            counts["not_executed_no_counterfactual_label"] += 1
            continue
        counts["executed"] += 1
        label, probability = row.get("label_future_failure"), row.get("predicted_failure_probability")
        if label not in (0, 1):
            counts["executed_without_observed_label"] += 1
            continue
        if type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1:
            counts["executed_without_valid_confidence"] += 1
            continue
        grouped[str(row["episode_index"])].append((label, probability))
        counts["scored_executed"] += 1
    values = []
    for episode in envelope["episodes"]:
        scored = grouped[episode["episode_id"]]
        total = sum(r["disposition"] == "executed" and str(r["episode_index"]) == episode["episode_id"] for r in rows)
        metrics = calibration_metrics(*zip(*scored)) if scored else None
        # A paired episode metric is unavailable if even one executed decision
        # lacks a valid probability/label; no favourable complete-case filtering.
        complete = bool(scored) and len(scored) == total
        derived = copy.deepcopy(episode)
        for source, target in (("brier", "decision_brier"), ("overconfident_error_rate_p_le_0_1", "decision_overconfident_error_rate")):
            derived["metrics"][target] = metrics[source] if complete else None
        values.append(derived)
    return values, dict(counts), [pair for pairs in grouped.values() for pair in pairs]


def audit_attempt_history(events, states):
    """Preserve failed/retried attempt costs; unknown cost is never zero."""
    active, failures, claims = {}, {}, Counter()
    for event in events:
        job_id, details = event["job_id"], event["details"]
        if event["event"] == "claimed":
            active[job_id] = details["attempt_id"]
            claims[job_id] += 1
        if event["event"] == "failed" or event["event"] == "reconciled" and details.get("state") == "failed":
            attempt = details.get("attempt_id", active.get(job_id))
            if attempt is None:
                raise RuntimeError("Failed physical attempt has no durable claim")
            for artifact in details.get("artifacts", []):
                if file_sha256(artifact["path"]) != artifact["sha256"]:
                    raise RuntimeError("An earlier failed attempt's raw evidence changed")
            costs = details.get("costs")
            if costs is not None and any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in costs.values()):
                raise RuntimeError("Invalid prior-attempt cost measurement")
            physical_key = "candidate_oracle_attempts" if "made" in job_id else "dft_episode_attempts"
            failures[(job_id, attempt)] = {"job_id": job_id, "attempt_id": attempt, "costs": costs,
                "physical_cost_known": isinstance(costs, dict) and physical_key in costs,
                "evidence_paths": [a["path"] for a in details.get("artifacts", [])]}
    if any(claims[job_id] != row["attempt_number"] for job_id, row in states.items()):
        raise RuntimeError("Ledger claim history and actual attempt counts disagree")
    known = Counter()
    for failure in failures.values():
        if failure["costs"]:
            known.update(failure["costs"])
    return {"claimed_attempts": sum(claims.values()), "jobs_with_multiple_attempts": sum(n > 1 for n in claims.values()),
        "prior_failed_attempts": list(failures.values()), "known_prior_attempt_costs": dict(known),
        "prior_attempts_with_unknown_physical_cost": sum(not f["physical_cost_known"] for f in failures.values()),
        "all_attempt_costs_accounted": all(f["physical_cost_known"] for f in failures.values()),
        "ledger_events_fingerprint": fingerprint(events), "unknown_costs_are_not_zero": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.project.resolve(), args.output.resolve()
    final = root / "experiments/final_evaluation"
    manifest = build_final_manifest(root)
    stored = json.loads((final / "manifest.json").read_text())
    if stored != manifest:
        raise RuntimeError("Final execution manifest differs from the complete study")
    database = sqlite3.connect(f"file:{final / 'ledger.sqlite3'}?mode=ro", uri=True)
    database.row_factory = sqlite3.Row
    states = {r["job_id"]: dict(r) for r in database.execute("SELECT * FROM jobs")}
    events = [{**dict(row), "details": json.loads(row["details"])} for row in database.execute("SELECT * FROM events ORDER BY event_id")]
    database.close()
    if set(states) != {j["job_id"] for j in manifest["jobs"]} or any(s["state"] != "succeeded" for s in states.values()):
        raise RuntimeError("A final report requires all 2160 jobs with complete verified physical budgets")
    attempt_accounting = audit_attempt_history(events, states)
    tasks = json.loads((root / "configs/benchmark_tasks.json").read_text())
    protocol = json.loads((root / "configs/main_protocol.json").read_text())
    summaries, evidence, confidence_counts, scored_groups = [], [], defaultdict(Counter), defaultdict(list)
    es_conditions = {}
    failure_types = FailureTypeReporter()
    for index, job in enumerate(manifest["jobs"]):
        state = states[job["job_id"]]
        path = Path(state["result_path"])
        if file_sha256(path) != state["result_sha256"]:
            raise RuntimeError("A ledger result changed after completion")
        from matdiscovery.execution_contract import declared_mace_workers
        verify_final_envelope(path, job, manifest["manifest_fingerprint"], tasks=tasks,
            expected_mace_num_workers=declared_mace_workers(protocol))
        envelope = json.loads(path.read_text())
        training = envelope["execution_profile"].get("selected_es")
        if training:
            identity = training["training_run_fingerprint"]
            entry = {"model_key": job["model_key"], "benchmark": job["benchmark"], "method": job["method"], "seed": job["seed"],
                "property": job["task"]["property"]["id"] if job["benchmark"] == "crystalgym" else None,
                "training_directory": training["training_directory"], "costs": training["training_costs"]}
            if identity in es_conditions and es_conditions[identity] != entry:
                raise RuntimeError("An ES training run was incorrectly reused across independent conditions")
            es_conditions[identity] = entry
        derived, counts, scored = reliability_for_job(envelope, path.parent)
        failure_types.add_job(envelope, path.parent, job, required=failure_control_required(protocol, job["benchmark"]))
        summaries.extend(derived)
        property_key = job["task"]["property"]["id"] if job["benchmark"] == "crystalgym" else "stability"
        key = (job["benchmark"], job["model_key"], job["method"], property_key)
        confidence_counts[key].update(counts)
        scored_groups[key].extend(scored)
        evidence.append({"job_id": job["job_id"], "path": str(path), "sha256": state["result_sha256"]})
        if (index + 1) % 25 == 0:
            print(json.dumps({"verified_final_jobs": index + 1, "expected_jobs": len(states)}), flush=True)
    summary = summarize_results(summaries, manifest, final=True)
    if len(es_conditions) != 80:
        raise RuntimeError("Final results must account for all 80 independently completed ES training conditions")
    es_costs = Counter()
    for condition in es_conditions.values():
        es_costs.update(condition["costs"])
    attempt_accounting.update(expected_es_training_conditions=80, accounted_es_training_conditions=len(es_conditions),
        es_training_conditions=es_conditions, actual_es_training_costs=dict(es_costs),
        verified_final_costs=summary["accounting"]["observed_costs"],
        cost_scope="See study_costs.json for shared collection once, offline representation, ES, final, prior failures, archived interruptions and separate diagnostics")
    print(json.dumps({"phase": "auditing_end_to_end_costs", "expected_collection_jobs": 720, "expected_representation_stages": 12}), flush=True)
    study_costs = build_study_cost_report(root, summaries, es_conditions, attempt_accounting)
    contrasts = [(method, "baseline") for method in protocol["methods"] if method != "baseline"]
    contrasts += [("graph_risk", "entropy_risk"), ("graph_risk", "hidden_risk"), ("esopt_graph_risk", "esopt")]
    estimates = []
    for model in protocol["models"]:
        for benchmark in ("made", "crystalgym"):
            strata = [("all_systems", None, ["AUDC", "mSUN"])] if benchmark == "made" else [
                (prop, sorted({j["task_id"] for j in manifest["jobs"] if j["benchmark"] == benchmark and j["task"]["property"]["id"] == prop}), ["reward"])
                for prop in ("bm", "density", "band_gap")]
            for stratum, task_ids, metrics in strata:
                for method, reference in contrasts:
                    for metric in metrics + ["decision_brier", "decision_overconfident_error_rate"]:
                        selected = [s for s in summaries if s["benchmark"] == benchmark and s["model_key"] == model
                            and s["method"] in {method, reference} and (task_ids is None or s["task_id"] in task_ids)]
                        if any(s["metrics"].get(metric) is None for s in selected):
                            estimates.append({"benchmark": benchmark, "model_key": model, "stratum": stratum,
                                "method": method, "reference": reference, "metric": metric, "status": "undefined_incomplete_confidence_coverage"})
                            continue
                        result = paired_cluster_bootstrap(summaries, benchmark=benchmark, model_key=model, method=method,
                            reference=reference, metric=metric, manifest=manifest, task_ids=task_ids,
                            n_bootstrap=protocol["statistics"]["bootstrap_replicates"], confidence=protocol["statistics"]["confidence_level"], seed=1729)
                        lower_better = metric.startswith("decision_")
                        result.update(stratum=stratum, status="estimated", higher_is_better=not lower_better,
                            interval_excludes_no_improvement=(result["ci"][1] < 0 if lower_better else result["ci"][0] > 0),
                            interval_multiplicity="unadjusted_pointwise_95_percent_not_a_familywise_claim")
                        estimates.append(result)
    calibration = []
    for key, counts in sorted(confidence_counts.items()):
        pairs = scored_groups[key]
        calibration.append({**dict(zip(("benchmark", "model_key", "method", "property"), key)),
            "coverage_counts": dict(counts), "pooled_descriptive_step_metrics": calibration_metrics(*zip(*pairs)) if pairs else None,
            "unit_warning": "Within-episode decisions are dependent; pooled metrics are descriptive. Paired CIs use episode/task/seed clusters."})
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "verified_final_evidence.json", evidence)
    write_json_atomic(output / "summary.json", summary)
    write_json_atomic(output / "paired_estimates.json", estimates)
    write_json_atomic(output / "calibration.json", calibration)
    write_json_atomic(output / "failure_type_calibration.json", failure_types.report())
    write_json_atomic(output / "ledger_attempt_events.json", events)
    write_json_atomic(output / "attempt_accounting.json", attempt_accounting)
    write_json_atomic(output / "study_costs.json", study_costs)
    primary = [e for e in estimates if e["method"] == "esopt_graph_risk" and e["reference"] == "baseline"
               and e["metric"] in ("AUDC", "reward", "decision_brier", "decision_overconfident_error_rate")]
    acceptance = {"schema": "full_study_execution_audit_v1", "all_final_experiments_complete": True,
        "expected_counts": manifest["expected_counts"], "manifest_fingerprint": manifest["manifest_fingerprint"],
        "verified_evidence_sha256": file_sha256(output / "verified_final_evidence.json"),
        "all_attempt_costs_accounted": attempt_accounting["all_attempt_costs_accounted"],
        "all_es_training_conditions_accounted": len(es_conditions) == 80,
        "all_end_to_end_experimental_costs_accounted": study_costs["experimental_cost_accounting_complete"],
        "technical_diagnostic_unknowns_reported_separately": True,
        "ledger_events_fingerprint": attempt_accounting["ledger_events_fingerprint"],
        "primary_full_method_comparisons": primary, "improvement_is_assessed_per_endpoint_not_assumed": True,
        "negative_or_inconclusive_results_must_be_reported": True,
        "artifacts": [{"path": str(output / name), "sha256": file_sha256(output / name)} for name in
            ("summary.json", "paired_estimates.json", "calibration.json", "failure_type_calibration.json", "attempt_accounting.json", "ledger_attempt_events.json", "study_costs.json")]}
    acceptance["fingerprint"] = fingerprint(acceptance)
    write_json_atomic(output / "acceptance.json", acceptance)
    print(json.dumps({"status": "full_final_evidence_verified", "acceptance": str(output / "acceptance.json"),
        "improvement_not_assumed": True}), flush=True)


if __name__ == "__main__":
    main()
