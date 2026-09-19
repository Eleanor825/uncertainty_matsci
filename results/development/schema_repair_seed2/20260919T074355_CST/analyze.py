"""Recompute the completed DEVELOPMENT seed-2 pair, without remote/science calls."""
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text())


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(name, rows):
    with (HERE / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


e = read("evidence.json")
j = read("additional_evidence.json")
identity = read("identity_evidence.json")
assert all(v["complete"] and v["common_profile_exact_equal"] for v in identity["results"].values())
arms = {a["arm"]: a for a in e["arms"]}
assert set(arms) == {"original_controller", "schema_repaired_controller"}
summary = {"scope": e["scope"], "environment_seed": 2, "task": "Al-Pd-Sm",
           "budget_per_arm": 50, "split": "dev", "independent_seed_pairs": 1,
           "run_to_run_variance": None,
           "snapshot_time_CST": datetime.fromtimestamp(e["read_at"], ZoneInfo("Asia/Shanghai")).isoformat(),
           "registration": e["registration"], "controller_sources": e["controller_sources"],
           "execution_profile": e["execution_profile"], "arms": {}}
step_rows, decision_rows, comparison = [], [], []
for name, a in arms.items():
    job = a["job"]
    assert (job["task_id"], job["seed"], job["budget"], job["split"]) == ("Al-Pd-Sm", 2, 50, "dev")
    assert a["episode"]["complete"] and a["close"]["ok"]
    rows, steps = a["rows"], a["steps"]
    assert len(steps) == 50 and all(s["ok"] for s in steps)
    assert [s["metrics"]["queries_used"] for s in steps] == list(range(1, 51))
    curves = [[0, 0]] + [[s["step"], s["metrics"]["num_newly_discovered_stable"]] for s in steps]
    assert curves == a["episode"]["discovery_curve"]
    # Frozen MADE definition normalizes trapezoid area by the ideal B^2/2.
    audc = 2 * sum((curves[i][1] + curves[i-1][1]) / 2 for i in range(1, 51)) / 2500
    sun = curves[-1][1]
    assert math.isclose(audc, a["episode"]["metrics"]["AUDC"], abs_tol=1e-12)
    assert math.isclose(sun / 50, a["episode"]["metrics"]["mSUN"], abs_tol=1e-12)
    journal = j["arms"][name]
    assert journal["source"] == a["data_sources"]["oracle_attempts.jsonl"]
    assert journal["unclosed_attempts"] == 0
    assert journal["attempts_by_role_phase"]["orb|candidate"] == 50
    assert journal["attempts_by_role_phase"]["orb|initialization"] == 26
    physical = [x for x in journal["candidate_hashes"] if x["role"] == "orb"]
    assert [x["candidate_hash"] for x in physical] == [s["candidate_hash"] for s in steps]
    assert len(rows) == a["costs"]["llm_calls"]
    groups = {}
    for r in rows:
        assert r["seed"] == 2 * 10000019 + r["sequence"] * 97 + r["candidate_index"]
        assert r["controller"]["threshold"] == .6
        groups.setdefault(r["sequence"], []).append(r)
        decision_rows.append({"arm": name, "decision_id": r["local_decision_id"],
            "source_line": r["line"], "generation_success": r["generation_success"],
            "tool": (r["action"] or {}).get("tool"), "disposition": r["disposition"],
            "graph_status": r["graph_status"], "scalar_risk": r["predicted_failure_probability"],
            "typed_generation_invalid": r["failure_type_probabilities"]["generation_invalid"],
            "observed_generation_invalid": r["observed_failure_types"]["generation_invalid"],
            "trigger": r["controller"]["trigger"], "trigger_source": r["controller"]["trigger_source"],
            "request_retry": r["controller"]["request_retry"],
            "schema_issues": "|".join(r["controller"]["local_schema_issues"] or []),
            "original_schema_replay": "|".join(r["schema_replay"]["original_controller"]),
            "repaired_schema_replay": "|".join(r["schema_replay"]["schema_repaired_controller"]),
            "selected_index": r["selection"]["selected_index"]})
    valid = [r for r in rows if r["generation_success"]]
    supported = [r for r in valid if r["graph_status"] == "succeeded"]
    typed_positive = [r for r in rows if r["failure_type_probabilities"]["generation_invalid"] >= .6]
    predicted_triggers = [r for r in rows if r["controller"]["trigger_source"] == "predicted_failure_type"]
    mapping_rejections = [r for r in rows if "explicit_compositions_required" in (r["controller"]["local_schema_issues"] or [])
                          and "explicit_compositions_required" not in r["schema_replay"]["schema_repaired_controller"]]
    assert all(r["generation_success"] and r["graph_status"] == "unavailable" and
               r["observed_failure_types"]["generation_invalid"] == 0 for r in predicted_triggers)
    assert all(r["predicted_failure_probability"] < .6 and
               all(v is None or v < .6 for v in r["failure_type_probabilities"].values()) for r in supported)
    false_positive = sum(r["observed_failure_types"]["generation_invalid"] == 0 for r in typed_positive)
    true_positive = sum(r["observed_failure_types"]["generation_invalid"] == 1 for r in typed_positive)
    metrics = steps[-1]["metrics"]
    last = 0
    for s in steps:
        current = s["metrics"]["num_newly_discovered_stable"]
        step_rows.append({"arm": name, "step": s["step"], "rpc_id": s["rpc_id"],
            "rpc_response_line": s["response_line"], "candidate_hash": s["candidate_hash"],
            "formula": s["official"]["reduced_formula"],
            "e_above_hull": s["official"]["e_above_hull"],
            "is_stable_at_evaluation": s["official"]["is_stable"],
            "is_newly_discovered": s["official"]["is_newly_discovered"],
            "SUN": current, "SUN_delta": current-last,
            "unique_structure_count": s["metrics"]["novelty_unique_structure_count"],
            "novel_vs_initial_reference_count": s["metrics"]["novelty_novel_structure_count"]})
        last = current
    stats = {"job_id": job["job_id"], "result": a["result"], "data_sources": a["data_sources"],
        "SUN": sun, "mSUN": sun/50, "AUDC": audc,
        "candidate_ORB_attempts": 50, "candidate_ORB_returned": 50,
        "initialization_ORB_attempts": 26,
        "surrogate_MACE_attempts": journal["attempts_by_role_phase"].get("mace|candidate", 0),
        "unclosed_physical_attempts": 0, "scientific_step_errors": 0,
        "LLM_proposals": len(rows), "decision_groups": len(groups),
        "second_proposal_groups": sum(len(v) == 2 for v in groups.values()),
        "controller_retry_request_events": sum(r["controller"]["request_retry"] for r in rows),
        "generation_failures": sum(not r["generation_success"] for r in rows),
        "dispositions": dict(Counter(r["disposition"] for r in rows)),
        "local_schema_issue_events": sum(bool(r["controller"]["local_schema_issues"]) for r in rows),
        "local_schema_issue_counts": dict(Counter(x for r in rows for x in (r["controller"]["local_schema_issues"] or []))),
        "false_mapping_rejection_events": len(mapping_rejections),
        "false_mapping_rejection_ids": [r["local_decision_id"] for r in mapping_rejections],
        "fixed_recovery_actions": len(a["fixed_recovery"]),
        "tool_attempts": len(a["tools"]), "tool_errors": sum(not t["ok"] for t in a["tools"]),
        "tool_counts": dict(Counter(t["action"]["tool"] for t in a["tools"])),
        "generated_crystal_candidates": sum(t["output_summary"].get("generated", 0) for t in a["tools"]),
        "admitted_to_buffer": sum(t["output_summary"].get("accepted", 0) for t in a["tools"]),
        "buffer_rejection_reasons": dict(Counter(x.get("reason") for t in a["tools"] for x in t["output_summary"].get("records", []) if not x.get("accepted"))),
        "graph_succeeded": sum(r["graph_status"] == "succeeded" for r in rows),
        "graph_unavailable": sum(r["graph_status"] == "unavailable" for r in rows),
        "FVU_gate_failures": sum("fidelity gate failed" in (r["graph_error"] or "") for r in rows),
        "valid_generation_missing_graph": sum(r["graph_status"] == "unavailable" for r in valid),
        "valid_generation_scalar_range": [min(r["predicted_failure_probability"] for r in valid), max(r["predicted_failure_probability"] for r in valid)],
        "valid_supported_graph_rows": len(supported),
        "supported_rows_any_NN_head_or_scalar_ge_threshold": 0,
        "predicted_type_retry_events": len(predicted_triggers),
        "predicted_type_retry_ids": [r["local_decision_id"] for r in predicted_triggers],
        "typed_generation_invalid_ge_threshold": {"true_positive": true_positive, "false_positive": false_positive,
            "false_negative": sum(not r["generation_success"] for r in rows)-true_positive,
            "all_false_positives_missing_graph": all(r["graph_status"] == "unavailable" for r in typed_positive if r["observed_failure_types"]["generation_invalid"] == 0)},
        "two_valid_candidate_ranking_groups": sum(len(v) == 2 and sum(x["valid"] for x in v[0]["selection"]["ranking"]) == 2 for v in groups.values()),
        "stable_evaluations": sum(s["official"]["is_stable"] for s in steps),
        "stable_evaluation_fraction": sum(s["official"]["is_stable"] for s in steps)/50,
        "first_time_structure_evaluations": sum(s["official"]["is_newly_discovered"] for s in steps),
        "official_final_unique_structures": metrics["novelty_unique_structure_count"],
        "evaluations_not_adding_unique_structure": 50-metrics["novelty_unique_structure_count"],
        "official_final_novel_vs_initial_reference": metrics["novelty_novel_structure_count"],
        "official_final_stable_unique": metrics["novelty_stable_unique_count"],
        "unique_compositions": metrics["diversity_all_composition_unique_composition_count"],
        "last_SUN_increase_step": max(s["step"] for i,s in enumerate(steps) if s["metrics"]["num_newly_discovered_stable"] > (steps[i-1]["metrics"]["num_newly_discovered_stable"] if i else 0)),
        "costs_as_recorded": a["costs"]}
    summary["arms"][name] = stats

o, r = [summary["arms"][name] for name in arms]
summary["paired_deltas_repaired_minus_original"] = {k:r[k]-o[k] for k in ["SUN", "mSUN", "AUDC", "stable_evaluations", "official_final_unique_structures", "LLM_proposals", "tool_errors", "surrogate_MACE_attempts"]}
summary["interpretation_limits"] = [
    "One paired development seed; no run-to-run variance or population effect claim.",
    "Pure schema replay establishes the local gate difference, not counterfactual crystal outcomes.",
    "Prediction/ranking on rejected candidates has no counterfactual scientific labels.",
    "The 50 candidate ORB budgets match; actual MACE/model/tool/graph costs do not match.",
    "Different subsequent actions alter histories and randomness; downstream SUN gains cannot be decomposed causally from this pair alone.",
    "Near-constant scalar predictions and unused supported-graph thresholds do not prove zero latent predictive information; this pair supplies no positive NN control-benefit evidence.",
    "Support-aware unknown handling is a separate frozen candidate; its benefit is not established by this schema-only pair."]

def row(arm, name):
    return next(r for r in arms[arm]["rows"] if r["local_decision_id"] == name)


def minimal_row(arm, name):
    value = row(arm, name)
    keep = ["local_decision_id", "line", "prefix_hash", "action", "action_sha256", "seed", "generation_success", "generation_failure_code", "disposition", "graph_status", "graph_error", "predicted_failure_probability", "failure_type_probabilities", "observed_failure_types", "failure_type_observation_rpc_ids", "controller", "selection", "schema_replay"]
    return dict({k:value[k] for k in keep}, arm=arm,
                original_decisions_source=arms[arm]["data_sources"]["decisions.jsonl"])


def tool(arm, rpc_id):
    value = next(t for t in arms[arm]["tools"] if t["rpc_id"] == rpc_id)
    error = value["error"]
    result = {k:value[k] for k in ["rpc_id", "request_line", "response_line", "ok", "action"]}
    result["error"] = {k:error[k] for k in ["code", "type", "message"] if k in error} if error else None
    result["candidate_count_at_error"] = (error or {}).get("details", {}).get("counts", {}).get("candidate_oracle_attempts")
    result["output_summary"] = {k:v for k,v in value["output_summary"].items() if k != "records"}
    result["source"] = arms[arm]["data_sources"]["rpc.jsonl"]
    return result


original, repaired = "original_controller", "schema_repaired_controller"
first_o, first_r = row(original, "d0000000-c0"), row(repaired, "d0000000-c0")
for key in ["prefix_hash", "action_sha256", "seed", "predicted_failure_probability", "failure_type_probabilities"]:
    assert first_o[key] == first_r[key]
cases = [
 {"case": "identical_first_proposal_schema_branch", "observed": "The exact same Mapping action passes the frozen adapter but the old controller rejects its shape before execution.",
  "proposals": [minimal_row(original, "d0000000-c0"), minimal_row(repaired, "d0000000-c0")],
  "tools": [tool(original, i) for i in [1, 3, 9]] + [tool(repaired, i) for i in [1, 3, 5]],
  "first_scientific_steps": {name:arms[name]["steps"][0] for name in arms},
  "inference_boundary": "Local rejection cause is identified; the two first evaluated structures are different, not counterfactual evaluations of the same structure."},
 {"case": "valid_missing_graph_still_triggers_wrong_type", "observed": "Repaired d21-c0 is syntactically valid, but FVU rejection yields no graph and generation_invalid=1; it is replaced by a query.",
  "proposals": [minimal_row(repaired, x) for x in ["d0000021-c0", "d0000021-c1"]],
  "tools": [tool(repaired, 43)],
  "next_scientific_step": next(s for s in arms[repaired]["steps"] if s["rpc_id"] == 46),
  "inference_boundary": "Generation-invalid false positive is observable from parsing. The rejected structure was not evaluated here; its potential SUN is unknown."},
 {"case": "buffer_hash_error_persists_at_step47", "observed": "Eight AdapterError tool returns occur while candidate count stays 47. Both final retry candidates lack graphs; high generation-invalid is the wrong diagnosis for valid JSON.",
  "proposals": [minimal_row(repaired, f"d{i:07d}-c0") for i in range(68, 76)] + [minimal_row(repaired, "d0000075-c1")],
  "tools": [tool(repaired, i) for i in range(137, 152, 2)],
  "next_scientific_step": next(s for s in arms[repaired]["steps"] if s["rpc_id"] == 154),
  "inference_boundary": "The tool_failure head on the final chosen proposal is high and agrees with an observed error, but ranking does not avert it. No claim that all NN heads contain no information."}]
write("summary.json", summary)
write("cases.json", cases)
csv_write("oracle_steps.csv", step_rows)
csv_write("decisions.csv", decision_rows)
for key in ["SUN", "mSUN", "AUDC", "candidate_ORB_attempts", "initialization_ORB_attempts", "surrogate_MACE_attempts", "LLM_proposals", "decision_groups", "second_proposal_groups", "controller_retry_request_events", "generation_failures", "local_schema_issue_events", "false_mapping_rejection_events", "fixed_recovery_actions", "tool_attempts", "tool_errors", "generated_crystal_candidates", "admitted_to_buffer", "graph_succeeded", "graph_unavailable", "FVU_gate_failures", "predicted_type_retry_events", "stable_evaluations", "official_final_unique_structures", "evaluations_not_adding_unique_structure", "official_final_novel_vs_initial_reference", "official_final_stable_unique", "unique_compositions"]:
    comparison.append({"metric": key, "original_controller": o[key], "schema_repaired_controller": r[key], "delta_repaired_minus_original": r[key]-o[key]})
csv_write("comparison.csv", comparison)
write("verification.json", {"complete": True, "scope": "CPU-only descriptive audit of two completed development trajectories",
    "input_sha256": {name:sha(HERE/name) for name in ["evidence.json", "additional_evidence.json", "identity_evidence.json"]},
    "verified": {"physical_ORB_candidate_attempts": 100, "physical_ORB_initialization_attempts": 52,
                 "physical_MACE_surrogate_attempts": 31, "closed_RPC_curves": 2,
                 "curve_points_including_zero": 102, "LLM_generation_seeds": 172,
                 "initial_proposal_prefix_action_risk_identical": True,
                 "unclosed_oracle_attempts": 0},
    "not_repeated": ["giant checkpoint rehash", "GPU inference", "oracle execution", "training", "held-out test reading"],
    "extractor_correction": "evidence.json journal_event_counts used event rather than actual kind; ignored in this audit. additional_evidence.json supplies hash-matched kind-based counts; no scientific data changed."})
print(json.dumps({"verified": True, "deltas": summary["paired_deltas_repaired_minus_original"],
                  "arms": {name:{k:v[k] for k in ["SUN", "AUDC", "false_mapping_rejection_events", "second_proposal_groups", "typed_generation_invalid_ge_threshold"]} for name,v in summary["arms"].items()}}, ensure_ascii=False))
