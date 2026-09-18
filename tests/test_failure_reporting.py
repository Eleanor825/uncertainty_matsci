"""Synthetic ordered records only; no model, GPU, environment, or oracle calls."""
import copy
import json

import pytest

from matdiscovery.accounting import file_sha256, fingerprint
from matdiscovery.failure_labels import HEADS, HORIZONS, FailureLabelError
from matdiscovery.failure_reporting import FailureTypeReporter, audit_failure_types_for_job, failure_control_required


def fixture(tmp_path, *, failed_first=False):
    directory = tmp_path / "synthetic_final"
    directory.mkdir()
    job = {"job_id": "final-unit", "benchmark": "made", "model_key": "unit-model", "method": "graph_risk",
           "task_id": "unit-system", "expected_counts": {"candidate_oracle_attempts": 2}}
    stamp = {"state_id": "unit-es-state-4", "checkpoint_hash": "unit-initial-hash", "model_id": "unit-policy",
             "generation": 4, "perturbation_seed": None, "perturbation_sigma": None}
    rows, events, rpc = [], [], []
    def call(op, args, result=None, *, failed=False):
        i = len(rpc) // 2
        response = {"id": i, "ok": False, "error": {"details": result}} if failed else {"id": i, "ok": True, "result": result or {}}
        rpc.extend([{"direction": "request", "payload": {"id": i, "op": op, "args": args}}, {"direction": "response", "payload": response}])
        return i
    call("init", {"benchmark": "made", "budget": 2})
    waiting = []
    # Same read action twice proves matching is chronological, not name-based.
    tools = [("list_compositions", {}), ("list_compositions", {}),
             ("select_for_evaluation", {"composition": "H2", "structure_hash": "unit-1"}),
             ("select_for_evaluation", {"composition": "H2", "structure_hash": "unit-2"})]
    science_count = 0
    for seq, (tool, arguments) in enumerate(tools):
        action = {"tool": tool, "arguments": arguments}
        chosen = None
        for ordinal in range(2 if seq == 0 else 1):
            local = f"d{seq:07d}-c{ordinal}"
            row = {"decision_id": job["job_id"] + ":" + local, "local_decision_id": local,
                "benchmark": "made", "model_key": job["model_key"], "task_id": job["task_id"],
                "split": "test", "group_id": "unit-test-group", "episode_id": "unit-episode", "episode_index": 0,
                "input_ids_file": str(directory / (local + ".pt")), "prefix_hash": fingerprint(local), "policy_stamp": stamp,
                "generation": {"success": True, "parsed_action": action},
                "disposition": "rejected_not_executed" if seq == 0 and ordinal == 0 else "executed",
                "observed_failure_types": {h: 0 if h == "generation_invalid" else None for h in HEADS},
                "failure_type_probabilities": {h: .2 for h in HEADS}, "failure_type_observation_rpc_ids": {h: [] for h in HEADS},
                "failure_type_horizons": dict(HORIZONS), "failure_types_used_for_control": True}
            rows.append(row)
            events.append({"event": "proposed", **{k: row[k] for k in ("decision_id", "policy_stamp", "prefix_hash", "input_ids_file")}})
            if row["disposition"] == "executed":
                chosen = row
        tool_id = call("tool", {"name": tool, "arguments": arguments}, {"tool": tool, "output": {}})
        chosen["observed_failure_types"]["tool_execution_failure"] = 0
        chosen["failure_type_observation_rpc_ids"]["tool_execution_failure"] = [tool_id]
        waiting.append(chosen)
        if tool == "select_for_evaluation":
            science_count += 1
            failed = failed_first and science_count == 1
            # Both successful steps have identical labels, so a future-source-id
            # forgery cannot be detected merely by checking the label's value.
            result = {"counts": {"candidate_oracle_attempts": science_count}} if failed else {
                "observation": {"counts": {"candidate_oracle_attempts": science_count}},
                "official_observation": {"is_stable": True, "is_newly_discovered": False}}
            step_id = call("step", {}, result, failed=failed)
            labels = {"scientific_evaluation_failure": int(failed), "unstable": None if failed else 0, "not_new": None if failed else 1}
            for target in waiting:
                target["observed_failure_types"].update(labels)
                for head in labels:
                    target["failure_type_observation_rpc_ids"][head] = [step_id]
            waiting.clear()
            if not failed:
                continue
        call("observe", {})
    call("close", {})
    selection = {"selected_index": 1, "common_comparison_heads": ["generation_invalid", "tool_execution_failure"],
                 "type_signals_used": True, "fallback": None,
                 "ranking": [{"excluded_from_cross_action_ranking": {"candidate_generation_failure": "not_applicable"}}]}
    for row in rows[:2]:
        row["failure_candidate_selection"] = copy.deepcopy(selection)
    rows[0]["failure_controller"] = {"trigger": "not_new", "controller_action": "retry_with_failure_feedback",
        "request_retry": True, "feedback_for_retry": "Choose another visible candidate."}
    rows[1]["feedback_received_for_this_candidate"] = {"predicted_failure_type": "not_new", "instruction": "Choose another visible candidate."}
    data = {"directory": directory, "job": job, "rows": rows, "events": events, "rpc": rpc,
        "envelope": {"execution_job": {"split": "test", "group_id": "unit-test-group", "selected_generation": 4, "policy_state_id": stamp["state_id"]}}}
    save(data)
    return data


def save(data):
    entries = []
    for name, key in (("decisions.jsonl", "rows"), ("decision_events.jsonl", "events"), ("rpc/rpc.jsonl", "rpc")):
        path = data["directory"] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in data[key]))
        entries.append({"path": str(path), "sha256": file_sha256(path)})
    data["envelope"]["artifacts"] = entries


def audit(data, required=True):
    return audit_failure_types_for_job(data["envelope"], data["directory"], data["job"], required=required)


def test_exact_raw_order_and_es_policy_identity(tmp_path):
    data = fixture(tmp_path)
    result = audit(data)
    assert result["status"] == "audited" and result["scientific_evaluations"] == 2
    assert result["records"][1]["policy_stamp"]["generation"] == 4
    assert result["records"][1]["failure_type_observation_rpc_ids"]["tool_execution_failure"] != result["records"][2]["failure_type_observation_rpc_ids"]["tool_execution_failure"]
    assert len(result["source_files"]) == 3


@pytest.mark.parametrize("mutation", ["forged_tool_id", "later_science_same_label", "changed_label", "rejected_label", "changed_action", "policy_state", "proposal_prefix"])
def test_source_or_label_forgery_rejected_even_if_files_are_rehashed(tmp_path, mutation):
    data = fixture(tmp_path)
    first, last = data["rows"][1], data["rows"][-1]
    if mutation == "forged_tool_id":
        first["failure_type_observation_rpc_ids"]["tool_execution_failure"] = [3]
    elif mutation == "later_science_same_label":
        first["failure_type_observation_rpc_ids"]["not_new"] = last["failure_type_observation_rpc_ids"]["not_new"]
    elif mutation == "changed_label":
        first["observed_failure_types"]["unstable"] = 1
    elif mutation == "rejected_label":
        data["rows"][0]["observed_failure_types"]["not_new"] = 1
    elif mutation == "changed_action":
        data["rpc"][2]["payload"]["args"]["arguments"] = {"invented": True}
    elif mutation == "policy_state":
        data["envelope"]["execution_job"]["policy_state_id"] = "different"
    else:
        data["events"][0]["prefix_hash"] = "changed"
    save(data)
    with pytest.raises(FailureLabelError):
        audit(data)


def test_hash_tamper_fails_before_semantic_audit(tmp_path):
    data = fixture(tmp_path)
    path = data["directory"] / "rpc/rpc.jsonl"
    path.write_text(path.read_text() + "{}\n")
    with pytest.raises(FailureLabelError, match="SHA256 mismatch"):
        audit(data)


def test_unknown_scientific_label_retains_nearest_failed_rpc(tmp_path):
    data = fixture(tmp_path, failed_first=True)
    result = audit(data)
    row = result["records"][1]
    assert row["observed_failure_types"]["scientific_evaluation_failure"] == 1
    assert row["observed_failure_types"]["unstable"] is None
    assert row["failure_type_observation_rpc_ids"]["unstable"] == row["failure_type_observation_rpc_ids"]["scientific_evaluation_failure"]
    data["rows"][1]["failure_type_observation_rpc_ids"]["unstable"] = []
    save(data)
    with pytest.raises(FailureLabelError, match="source RPC IDs"):
        audit(data)


def test_required_missing_head_fails_optional_partial_is_not_recorded(tmp_path):
    data = fixture(tmp_path)
    del data["rows"][1]["failure_type_probabilities"]["not_new"]
    save(data)
    with pytest.raises(FailureLabelError, match="all seven heads"):
        audit(data)
    reporter = FailureTypeReporter()
    reporter.add_job(data["envelope"], data["directory"], data["job"], required=False)
    head = next(g for g in reporter.report()["groups"] if g["head"] == "not_new")
    assert head["counts"]["prediction_not_recorded"] == 1
    assert head["counts"]["prediction_unknown"] == 0
    assert head["counts"]["unscored"] == 1


def test_descriptive_coverage_and_controller_counts_do_not_impute_or_duplicate(tmp_path):
    data = fixture(tmp_path, failed_first=True)
    data["rows"][-1]["failure_type_probabilities"]["not_new"] = None
    save(data)
    reporter = FailureTypeReporter()
    proof = reporter.add_job(data["envelope"], data["directory"], data["job"], required=True)
    report = reporter.report()
    assert report["unknown_labels_or_predictions_imputed"] is False
    assert report["type_improvement_claimed"] is report["paired_significance_estimated"] is False
    assert sum(proof["policy_stamp_counts"].values()) == 5 and len(report["policy_stamps"]) == 1
    heads = {g["head"]: g for g in report["groups"]}
    assert heads["not_new"]["counts"] == {
        "proposals": 5, "executed": 4, "not_executed": 1, "observed": 1, "unknown": 4, "label_not_recorded": 0,
        "available": 4, "prediction_unknown": 1, "prediction_not_recorded": 0, "scored": 0, "unscored": 1, "available_unlabelled": 4}
    assert heads["not_new"]["pooled_descriptive_metrics"] is None
    assert heads["scientific_evaluation_failure"]["pooled_descriptive_metrics"]["brier"] == pytest.approx((3 * .8 ** 2 + .2 ** 2) / 4)
    assert heads["scientific_evaluation_failure"]["counts"]["unknown"] == 1  # Rejected proposal.
    controller = report["controller_groups"][0]
    assert controller["counts"]["selection_decisions"] == 1
    assert controller["counts"]["triggered_proposals"] == 1
    assert controller["counts"]["feedback_received_proposals"] == 1
    assert controller["common_head_sets"] == {"generation_invalid,tool_execution_failure": 1}
    assert controller["excluded_heads"] == {"candidate_generation_failure": 1}
    with pytest.raises(FailureLabelError, match="Duplicate job"):
        reporter.add_job(data["envelope"], data["directory"], data["job"])


def test_scalar_fallback_is_counted_once_per_decision(tmp_path):
    data = fixture(tmp_path)
    for row in data["rows"][:2]:
        row["failure_candidate_selection"].update(common_comparison_heads=[], type_signals_used=False, fallback="scalar_only_no_common_available_types")
    save(data)
    reporter = FailureTypeReporter()
    reporter.add_job(data["envelope"], data["directory"], data["job"], required=True)
    controller = reporter.report()["controller_groups"][0]
    assert controller["counts"]["fallback_decisions"] == 1
    assert controller["fallbacks"] == {"scalar_only_no_common_available_types": 1}


@pytest.mark.parametrize("generation_tool_fails", [False, True])
def test_generation_and_screening_tool_labels_use_exact_response_and_unknown_evidence(tmp_path, generation_tool_fails):
    data = fixture(tmp_path)
    generate = {"tool": "generate_structures", "arguments": {"compositions": ["H2"], "num_candidates": 1}}
    for row in data["rows"][:2]:
        row["generation"]["parsed_action"] = generate
    data["rpc"][2]["payload"]["args"] = {"name": generate["tool"], "arguments": generate["arguments"]}
    data["rpc"][3]["payload"] = {"id": 1, "ok": False, "error": {"code": "unit_tool_error"}} if generation_tool_fails else {
        "id": 1, "ok": True, "result": {"tool": "generate_structures", "output": {"accepted": 0}}}
    row = data["rows"][1]
    row["observed_failure_types"].update(candidate_generation_failure=None if generation_tool_fails else 1,
                                         tool_execution_failure=int(generation_tool_fails))
    row["failure_type_observation_rpc_ids"]["candidate_generation_failure"] = [1]
    scorer = {"tool": "score_buffer", "arguments": {"scorer_name": "oracle"}}
    data["rows"][2]["generation"]["parsed_action"] = scorer
    data["rpc"][6]["payload"]["args"] = {"name": scorer["tool"], "arguments": scorer["arguments"]}
    data["rpc"][7]["payload"]["result"] = {"tool": "score_buffer", "output": [{"scores": [{"nonfinite": "-inf"}]}]}
    data["rows"][2]["observed_failure_types"]["screening_unavailable"] = 0
    data["rows"][2]["failure_type_observation_rpc_ids"]["screening_unavailable"] = [3]
    save(data)
    assert audit(data)["status"] == "audited"
    data["rows"][2]["observed_failure_types"]["screening_unavailable"] = 1
    save(data)
    with pytest.raises(FailureLabelError, match="contradicts raw evidence"):
        audit(data)


def test_legacy_not_recorded_and_cg_unsupported(tmp_path):
    data = fixture(tmp_path)
    for row in data["rows"]:
        for key in ("observed_failure_types", "failure_type_observation_rpc_ids", "failure_type_probabilities"):
            del row[key]
    save(data)
    assert audit(data, required=False)["status"] == "not_recorded"
    with pytest.raises(FailureLabelError, match="requires complete online"):
        audit(data)
    reporter = FailureTypeReporter()
    reporter.add_job(data["envelope"], data["directory"], data["job"])
    report = reporter.report()
    assert all(g["counts"]["label_not_recorded"] == 5 and g["counts"]["unknown"] == 0 for g in report["groups"])
    cg = {**data["job"], "job_id": "cg", "benchmark": "crystalgym"}
    reporter.add_job({}, data["directory"], cg, required=True)
    assert reporter.report()["job_status_counts"]["unsupported_benchmark"] == 1
    assert len(reporter.report()["groups"]) == 7
    assert failure_control_required({"failure_control": {"enabled_benchmarks": ["made", "crystalgym"]}}, "made")
    assert not failure_control_required({"failure_control": {"enabled_benchmarks": ["made", "crystalgym"]}}, "crystalgym")


@pytest.mark.parametrize("invalid", [True, -0.1, 1.1, "0.3"])
def test_invalid_prediction_rejected(tmp_path, invalid):
    data = fixture(tmp_path)
    data["rows"][0]["failure_type_probabilities"]["unstable"] = invalid
    save(data)
    with pytest.raises(FailureLabelError, match="Type probability"):
        audit(data)
