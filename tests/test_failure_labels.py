"""Synthetic CPU data-integrity fixtures only; no scientific model/evaluator runs."""
import copy
import json

import pytest

from matdiscovery.accounting import file_sha256, fingerprint
from matdiscovery.failure_labels import (
    HEADS, FailureLabelError, decision_identity, derive_failure_labels, load_failure_labels,
    row_identity_hash, scientific_failure_labels, screening_evidence, tool_failure_labels,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def action(tool, **arguments):
    return {"tool": tool, "arguments": arguments}


def tool_response(tool, output, ok=True):
    return {"ok": True, "result": {"tool": tool, "output": output}} if ok else {"ok": False, "error": {"code": "synthetic_error"}}


def fixture(tmp_path, *, failed_first=False):
    directory = tmp_path / "unit_collection"
    directory.mkdir()
    job_id = "collection-made-unit-test-only"
    stamp = {"state_id": "unit", "checkpoint_hash": "synthetic-no-model", "model_id": "unit-test-only", "generation": 0}
    job = {"job_id": job_id, "benchmark": "made", "stage": "collection", "method": "baseline", "split": "train", "model_key": "qwen35_4b",
           "task_id": "unit-task", "group_id": "unit-group", "budget": 50,
           "expected_counts": {"candidate_oracle_attempts": 50}, "classification": "unit_test_only"}
    rows, events, rpc, count = [], [], [], 0
    def call(op, args, response):
        identifier = len(rpc) // 2
        rpc.extend([{"direction": "request", "payload": {"id": identifier, "op": op, "args": args}},
                    {"direction": "response", "payload": {"id": identifier, **response}}])
        return identifier
    call("init", {"benchmark": "made", "budget": 50}, {"ok": True, "result": {"observation": {"counts": {"candidate_oracle_attempts": 0}}}})
    specs = [
        (action("generate_structures", compositions=["H2"], num_candidates=2), {"generated": 2, "accepted": 0, "records": [{"accepted": False}, {"accepted": False}]}, False),
        (action("generate_structures", compositions=["H2"], num_candidates=1), {"generated": 1, "accepted": 1, "records": [{"accepted": True}]}, True),
        (action("list_compositions"), {"compositions": ["H2"]}, False),
        (action("list_compositions"), {"compositions": ["H2"]}, False),
        (action("score_buffer", scorer_name="oracle"), [{"scores": [{"nonfinite": "-inf"}, 0.1]}], False),
    ] + [(action("select_for_evaluation", composition="H2", structure_hash=f"unit-{i}"), {"hash": f"unit-{i}"}, False) for i in range(50)]
    for sequence, (chosen_action, output, recovery) in enumerate(specs):
        if recovery:
            events.append({"event": "fixed_failure_recovery", "sequence": sequence, "action": chosen_action})
        else:
            for candidate in range(2 if sequence == 0 else 1):
                local = f"d{sequence:07d}-c{candidate}"
                token_path = directory / "tokens" / (local + ".pt")
                token_path.parent.mkdir(exist_ok=True)
                token_path.write_text("synthetic unit fixture, not tensor/model data")
                row = {"decision_id": job_id + ":" + local, "local_decision_id": local, "benchmark": "made", "model_key": "qwen35_4b",
                    "split": "train", "task_id": "unit-task", "group_id": "unit-group", "episode_id": "unit-episode", "episode_index": 0,
                    "input_ids_file": str(token_path), "prefix_hash": fingerprint(local), "policy_stamp": stamp,
                    "generation": {"success": True, "parsed_action": chosen_action},
                    "disposition": "rejected_not_executed" if sequence == 0 and candidate == 0 else "executed",
                    "features": {"hidden.unit_feature": 0.0}, "label_future_failure": None}
                rows.append(row)
                events.append({"event": "proposed", **{k: row[k] for k in ("decision_id", "policy_stamp", "prefix_hash", "input_ids_file")}})
        call("tool", {"name": chosen_action["tool"], "arguments": chosen_action["arguments"]}, tool_response(chosen_action["tool"], output))
        if chosen_action["tool"] == "select_for_evaluation":
            count += 1
            if count == 1 and failed_first:
                response = {"ok": False, "error": {"code": "oracle_or_environment_exception", "details": {"counts": {"candidate_oracle_attempts": count}}}}
            else:
                response = {"ok": True, "result": {"official_observation": {"is_stable": count != 1, "is_newly_discovered": count != 1},
                            "observation": {"counts": {"candidate_oracle_attempts": count}}}}
            call("step", {}, response)
            if response["ok"]:
                continue
        call("observe", {}, {"ok": True, "result": {"counts": {"candidate_oracle_attempts": count}}})
    call("close", {}, {"ok": True, "result": {"closed": True}})
    write_json(directory / "job.json", job)
    write_json(directory / "episodes.json", [{"complete": True, "costs": {"candidate_oracle_attempts": 50}}])
    write_rows(directory / "decisions.jsonl", rows)
    write_rows(directory / "decision_events.jsonl", events)
    write_rows(directory / "rpc/rpc.jsonl", rpc)
    manifest = {"job_id": job_id, "complete": True, "decision_files": [{"path": "decisions.jsonl", "sha256": file_sha256(directory / "decisions.jsonl")}]}
    write_json(directory / "collection_manifest.json", manifest)
    return {"directory": directory, "manifest_path": directory / "collection_manifest.json", "manifest": manifest, "rows": rows, "events": events, "rpc": rpc}


def test_complete_ordered_join_recovery_repeated_tools_and_posterior_horizons(tmp_path):
    data = fixture(tmp_path)
    before = {str(p): file_sha256(p) for p in data["directory"].rglob("*") if p.is_file()}
    output = tmp_path / "derived" / "labels.json"
    result = derive_failure_labels(data["manifest_path"], output_path=output)
    assert result["scientific_evaluations"] == 50 and result["all_proposals_accounted"] == len(data["rows"])
    assert result["heads"] == list(HEADS) and len(HEADS) == 7
    assert load_failure_labels(output, manifest_path=data["manifest_path"]) == result
    assert derive_failure_labels(data["manifest_path"], output_path=output) == result
    assert before == {str(p): file_sha256(p) for p in data["directory"].rglob("*") if p.is_file()}
    assert result["execution_alignment"][1]["recovery"] is True
    repeated = result["execution_alignment"][2:4]
    assert repeated[0]["action_hash"] == repeated[1]["action_hash"]
    assert repeated[0]["tool_rpc_id"] != repeated[1]["tool_rpc_id"]
    rejected, executed = result["records"][:2]
    assert rejected["labels"]["generation_invalid"] == 0
    assert all(rejected["labels"][h] is None for h in HEADS[1:])
    assert executed["labels"]["candidate_generation_failure"] == 1
    assert executed["labels"]["tool_execution_failure"] == 0
    assert executed["labels"]["unstable"] == executed["labels"]["not_new"] == 1
    assert result["records"][-1]["labels"]["unstable"] == 0
    assert executed["horizons"]["unstable"] == "next_scientific_evaluation"
    assert not result["causal_responsibility_claim"] and result["physical_oracle_calls_by_derivation"] == 0
    assert "features" not in executed
    enriched = {**data["rows"][1], "features": {"graph.new": 9}, "_internal": 10}
    assert row_identity_hash(enriched) == executed["row_identity_hash"]
    assert decision_identity(enriched) == decision_identity(data["rows"][1])


def test_failed_computation_does_not_make_quality_negative_labels(tmp_path):
    data = fixture(tmp_path, failed_first=True)
    result = derive_failure_labels(data["manifest_path"])
    first = result["records"][1]
    assert first["labels"]["scientific_evaluation_failure"] == 1
    assert first["labels"]["unstable"] is first["labels"]["not_new"] is None
    assert first["source_rpc_ids"]["unstable"] == first["source_rpc_ids"]["scientific_evaluation_failure"]
    assert result["records"][-1]["labels"]["scientific_evaluation_failure"] == 0


def test_generation_label_does_not_read_tool_error_or_future_features(tmp_path):
    data = fixture(tmp_path)
    baseline = derive_failure_labels(data["manifest_path"])
    data["rows"][0]["generation"]["success"] = False
    for row in data["rows"]:
        row["label_immediate_error"] = 1
        row["label_future_failure"] = 1
        row["features"]["graph.posterior_must_not_be_read"] = 999999
    write_rows(data["directory"] / "decisions.jsonl", data["rows"])
    data["manifest"]["decision_files"][0]["sha256"] = file_sha256(data["directory"] / "decisions.jsonl")
    write_json(data["manifest_path"], data["manifest"])
    changed = derive_failure_labels(data["manifest_path"])
    assert changed["records"][0]["labels"]["generation_invalid"] == 1
    assert changed["records"][1]["labels"]["generation_invalid"] == 0
    for before, after in zip(baseline["records"], changed["records"]):
        for head in HEADS[1:]:
            assert before["labels"][head] == after["labels"][head]
    assert changed["records"][1]["source_rpc_ids"]["unstable"][0] > changed["records"][1]["source_rpc_ids"]["tool_execution_failure"][0]


def test_executed_generator_exception_retains_unknown_acceptance_and_rpc_evidence(tmp_path):
    data = fixture(tmp_path)
    data["rpc"][3]["payload"] = {"id": 1, "ok": False, "error": {"code": "tool_exception"}}
    write_rows(data["directory"] / "rpc/rpc.jsonl", data["rpc"])
    row = derive_failure_labels(data["manifest_path"])["records"][1]
    assert row["labels"]["generation_invalid"] == 0
    assert row["labels"]["tool_execution_failure"] == 1
    assert row["labels"]["candidate_generation_failure"] is None
    assert row["source_rpc_ids"]["candidate_generation_failure"] == [1]
    assert row["unknown_reasons"]["candidate_generation_failure"] == "generation_tool_failed_output_unknown"


def test_cli_writes_separate_model_job_sidecar(tmp_path):
    import runpy
    from pathlib import Path
    data = fixture(tmp_path)
    entry = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/derive_failure_labels.py"))
    output = tmp_path / "failure_labels"
    assert entry["main"](["--manifest", str(data["manifest_path"]), "--output-root", str(output)]) == 0
    path = output / "qwen35_4b/made/collection-made-unit-test-only.json"
    assert load_failure_labels(path, manifest_path=data["manifest_path"])["complete"]
    assert not (data["directory"] / "failure_labels.json").exists()


@pytest.mark.parametrize("tool,output,label", [
    ("generate_structures", {"accepted": 0}, 1), ("generate_structures", {"accepted": 2}, 0),
    ("generate_structures", {"records": [{"accepted": False}]}, 1),
    ("generate_structures", {"message": "unknown"}, None),
    ("create_structure", {"accepted": True}, 0), ("create_structure", {"accepted": False}, 1),
])
def test_candidate_generation_is_observed_material_acceptance_not_json(tool, output, label):
    labels = tool_failure_labels(action(tool), tool_response(tool, output))
    assert labels["candidate_generation_failure"] == label and labels["tool_execution_failure"] == 0
    failed = tool_failure_labels(action(tool), tool_response(tool, {}, ok=False))
    assert failed["candidate_generation_failure"] is None and failed["tool_execution_failure"] == 1


def test_official_negative_infinity_is_rankable_not_scientific_instability():
    a = action("score_buffer", scorer_name="oracle")
    response = tool_response("score_buffer", [{"scores": [{"nonfinite": "-inf"}], "candidates": [
        {"score": {"nonfinite": "-inf"}, "status": "official_negative_infinity_sentinel", "rankable": True}]}])
    assert tool_failure_labels(a, response)["screening_unavailable"] == 0
    assert screening_evidence(a, response)["score_counts"]["official_negative_infinity_sentinel"] == 1
    assert "unstable" not in tool_failure_labels(a, response)
    for scores in ([], [{"nonfinite": "nan"}], [1.0, {"nonfinite": "inf"}]):
        assert tool_failure_labels(a, tool_response("score_buffer", [{"scores": scores}]))["screening_unavailable"] == 1
    response["result"]["output"][0]["candidates"][0]["rankable"] = False
    with pytest.raises(FailureLabelError, match="rankability"):
        tool_failure_labels(a, response)
    unknown = scientific_failure_labels({"ok": True, "result": {"official_observation": {}}})
    assert unknown == {"scientific_evaluation_failure": 0, "unstable": None, "not_new": None}


@pytest.mark.parametrize("mutation", ["missing_event", "duplicate_event", "event_order", "recovery_collision", "rpc_arguments", "rpc_duplicate_id", "missing_close", "incomplete_manifest", "incomplete_budget"])
def test_alignment_missing_duplicate_or_changed_sources_fail_closed(tmp_path, mutation):
    data = fixture(tmp_path)
    if mutation == "missing_event": data["events"].pop(0)
    elif mutation == "duplicate_event": data["events"].insert(0, copy.deepcopy(data["events"][0]))
    elif mutation == "event_order": data["events"][0:2] = reversed(data["events"][0:2])
    elif mutation == "recovery_collision": data["events"][2]["sequence"] = 0
    elif mutation == "rpc_arguments": data["rpc"][2]["payload"]["args"]["arguments"]["num_candidates"] = 9
    elif mutation == "rpc_duplicate_id": data["rpc"][2]["payload"]["id"] = 0
    elif mutation == "missing_close": data["rpc"] = data["rpc"][:-2]
    elif mutation == "incomplete_manifest": data["manifest"]["complete"] = False
    elif mutation == "incomplete_budget": write_json(data["directory"] / "episodes.json", [{"complete": True, "costs": {"candidate_oracle_attempts": 49}}])
    write_rows(data["directory"] / "decision_events.jsonl", data["events"])
    write_rows(data["directory"] / "rpc/rpc.jsonl", data["rpc"])
    write_json(data["manifest_path"], data["manifest"])
    with pytest.raises(FailureLabelError): derive_failure_labels(data["manifest_path"])


def test_load_rejects_rehashed_label_tampering_and_changed_token_sources(tmp_path):
    data = fixture(tmp_path)
    output = tmp_path / "labels.json"
    result = derive_failure_labels(data["manifest_path"], output_path=output)
    changed = copy.deepcopy(result); changed["records"][1]["labels"]["unstable"] = 0
    changed["sidecar_fingerprint"] = fingerprint({k: v for k, v in changed.items() if k != "sidecar_fingerprint"})
    write_json(output, changed)
    with pytest.raises(FailureLabelError, match="no longer match"):
        load_failure_labels(output, manifest_path=data["manifest_path"])
    write_json(output, result)
    from pathlib import Path
    Path(data["rows"][0]["input_ids_file"]).write_text("changed unit data")
    with pytest.raises(FailureLabelError, match="no longer match"):
        load_failure_labels(output, manifest_path=data["manifest_path"])
    with pytest.raises(FailureLabelError, match="outside"):
        derive_failure_labels(data["manifest_path"], output_path=data["directory"] / "forbidden_labels.json")
