"""Observed MADE failure labels derived without modifying data or calling tools.

Action/RPC association is a complete ordered join, never fuzzy matching. Later
scientific outcomes label earlier EXECUTED prefixes predictively; they do not
assign causal blame. This module contains no model, evaluator, or feature access.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Mapping

from .accounting import file_sha256, fingerprint, write_json_atomic


SCHEMA = "made_observed_stage_failure_labels_v1"
HEADS = ("generation_invalid", "candidate_generation_failure", "tool_execution_failure", "screening_unavailable",
         "scientific_evaluation_failure", "unstable", "not_new")
SCIENTIFIC_HEADS = ("scientific_evaluation_failure", "unstable", "not_new")
IDENTITY_FIELDS = ("decision_id", "local_decision_id", "benchmark", "model_key", "split",
                   "task_id", "group_id", "episode_id", "episode_index", "prefix_hash", "policy_stamp")
HORIZONS = {"generation_invalid": "proposal_generation", "tool_execution_failure": "executed_tool_return",
            "candidate_generation_failure": "executed_candidate_generation_return",
            "screening_unavailable": "executed_score_buffer_return",
            **{head: "next_scientific_evaluation" for head in SCIENTIFIC_HEADS}}
TOOLS = {"generate_structures", "create_structure", "score_buffer", "query_structures",
         "list_compositions", "get_buffer_stats", "select_for_evaluation"}
LABEL_DEFINITIONS = {
    "generation_invalid": "not generation.success; observable for every proposed action",
    "candidate_generation_failure": "successful generate_structures/create_structure returned zero accepted material candidates; unknown for tool exceptions or unobserved acceptance",
    "tool_execution_failure": "not actual tool RPC response.ok; executed actions only",
    "screening_unavailable": "executed score_buffer RPC failure, no returned scores, or at least one NaN/+inf unrankable score; official -inf is rankable-last and is NOT an unavailable score",
    "scientific_evaluation_failure": "not actual scientific step response.ok; predictive next-evaluation horizon for preceding executed prefixes",
    "unstable": "not official_observation.is_stable after successful scientific computation; unknown on computation failure",
    "not_new": "not official_observation.is_newly_discovered after successful scientific computation, without conditioning on stability; unknown on computation failure",
}


class FailureLabelError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise FailureLabelError(message)


def decision_identity(row: Mapping) -> dict:
    """Stable identity only; safe on fitter-enriched records containing features."""
    _require(isinstance(row, Mapping) and all(k in row for k in IDENTITY_FIELDS), "Decision identity is incomplete")
    return {key: row[key] for key in IDENTITY_FIELDS}


def row_identity_hash(row: Mapping) -> str:
    return fingerprint(decision_identity(row))


def _ok(response):
    _require(isinstance(response, Mapping) and type(response.get("ok")) is bool, "Observed RPC response requires boolean ok")
    return response["ok"]


def _score_status(value):
    if type(value) in (int, float):
        if math.isfinite(value):
            return "finite"
        # Strict source JSON rejects bare nonfinite values; pure online helpers
        # also accept native floating values before RPC JSON serialization.
        return "official_negative_infinity_sentinel" if value == -math.inf else "invalid_nan" if math.isnan(value) else "invalid_positive_infinity"
    if isinstance(value, Mapping) and set(value) == {"nonfinite"}:
        names = {"-inf": "official_negative_infinity_sentinel", "inf": "invalid_positive_infinity", "nan": "invalid_nan"}
        _require(value["nonfinite"] in names, "Unknown nonfinite score representation")
        return names[value["nonfinite"]]
    raise FailureLabelError("Scorer output is not a recorded numeric score or explicit nonfinite sentinel")


def screening_evidence(action: Mapping, response: Mapping) -> dict:
    """Match benchmark_adapters.score_rankable/_record_scores, without importing an evaluator."""
    if action.get("tool") != "score_buffer":
        return {"applicable": False, "reason": "not_a_score_buffer_action"}
    if not _ok(response):
        return {"applicable": True, "label": 1, "reason": "score_buffer_rpc_failed", "score_counts": {}}
    result = response.get("result")
    _require(isinstance(result, Mapping) and result.get("tool") == "score_buffer", "Scoring result has wrong tool identity")
    output = result.get("output")
    _require(isinstance(output, list), "score_buffer must return complete per-composition score groups")
    counts = Counter()
    for group in output:
        _require(isinstance(group, Mapping) and isinstance(group.get("scores"), list), "Score group lacks its score list")
        statuses = [_score_status(value) for value in group["scores"]]
        candidates = group.get("candidates")
        if candidates is not None:
            _require(isinstance(candidates, list) and len(candidates) == len(statuses), "Scorer candidate/score lengths differ")
            for status, value, candidate in zip(statuses, group["scores"], candidates):
                _require(isinstance(candidate, Mapping) and _score_status(candidate.get("score")) == status,
                         "Scorer candidate and score-list values disagree")
                if status == "finite":
                    _require(candidate["score"] == value, "Finite candidate score differs from score list")
                _require(candidate.get("status") == status and candidate.get("rankable") is (status in {"finite", "official_negative_infinity_sentinel"}),
                         "Scorer status/rankability contradicts the declared official sentinel semantics")
        counts.update(statuses)
    unavailable = not counts or bool(counts["invalid_nan"] or counts["invalid_positive_infinity"])
    return {"applicable": True, "label": int(unavailable), "score_counts": dict(sorted(counts.items())),
            "reason": "no_scores_returned" if not counts else "unrankable_scores_present" if unavailable else "all_scores_rankable_including_official_sentinels",
            "negative_infinity_is_scientific_instability_label": False}


def _candidate_generation_label(action, response):
    if action.get("tool") not in {"generate_structures", "create_structure"} or not _ok(response):
        return None
    output = response.get("result", {}).get("output")
    if not isinstance(output, Mapping):
        return None
    accepted = output.get("accepted")
    if action["tool"] == "create_structure":
        return int(not accepted) if type(accepted) is bool else None
    _require(accepted is None or (type(accepted) is int and accepted >= 0), "Generated accepted count must be a nonnegative integer")
    records = output.get("records")
    if isinstance(records, list) and all(isinstance(r, Mapping) and type(r.get("accepted")) is bool for r in records):
        counted = sum(r["accepted"] for r in records)
        _require(accepted is None or accepted == counted, "Generated accepted count contradicts candidate records")
        if "generated" in output:
            _require(type(output["generated"]) is int and output["generated"] == len(records), "Generated count contradicts candidate records")
        accepted = counted
    return int(accepted == 0) if accepted is not None else None


def tool_failure_labels(action: Mapping, response: Mapping) -> dict:
    """Pure post-response labels for an actually executed MADE tool action."""
    _require(isinstance(action, Mapping) and action.get("tool") in TOOLS, "Unknown MADE action")
    success = _ok(response)
    if success:
        _require(isinstance(response.get("result"), Mapping) and response["result"].get("tool") == action["tool"],
                 "Tool response belongs to another action")
    screen = screening_evidence(action, response)
    return {"candidate_generation_failure": _candidate_generation_label(action, response),
            "tool_execution_failure": int(not success), "screening_unavailable": screen.get("label")}


def scientific_failure_labels(response: Mapping) -> dict:
    """Pure MADE scientific-return labels; never labels an unexecuted candidate."""
    if not _ok(response):
        return {"scientific_evaluation_failure": 1, "unstable": None, "not_new": None}
    result = response.get("result")
    _require(isinstance(result, Mapping) and isinstance(result.get("official_observation"), Mapping),
             "Successful MADE scientific response lacks official_observation")
    observation = result["official_observation"]
    labels = {"scientific_evaluation_failure": 0}
    for name, field in (("unstable", "is_stable"), ("not_new", "is_newly_discovered")):
        value = observation.get(field)
        _require(value is None or type(value) is bool, "Scientific classification must be an observed boolean or unknown")
        labels[name] = None if value is None else int(not value)
    return labels


def _json(data, context):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"Duplicate JSON key in {context}: {key}")
            result[key] = value
        return result
    def invalid(value):
        raise FailureLabelError(f"Bare nonfinite JSON in {context}: {value}")
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError) as exc:
        raise FailureLabelError(f"Invalid JSON in {context}: {exc}") from exc


def _resolve(name, directory):
    _require(isinstance(name, str) and name, "Source path is missing")
    path = Path(name)
    path = (path if path.is_absolute() else directory / path).resolve()
    _require(directory == path.parent or directory in path.parents, "Source file is outside the collection directory")
    return path


class _Sources:
    def __init__(self):
        self.items = {}
    def read(self, path, kind, expected=None):
        path = Path(path).resolve()
        raw = path.read_bytes()
        import hashlib
        digest = hashlib.sha256(raw).hexdigest()
        _require(expected is None or digest == expected, f"Source SHA256 mismatch: {path}")
        entry = {"path": str(path), "sha256": digest, "kind": kind}
        _require(str(path) not in self.items or self.items[str(path)] == entry, "Source identity changed during derivation")
        self.items[str(path)] = entry
        return raw
    def json(self, path, kind):
        return _json(self.read(path, kind), path)
    def lines(self, path, kind, expected=None):
        raw = self.read(path, kind, expected)
        _require(bool(raw) and raw.endswith(b"\n"), f"Empty or truncated JSONL requires reconciliation: {path}")
        rows = [_json(line, f"{path}:{index}") for index, line in enumerate(raw.splitlines(), 1)]
        _require(all(isinstance(row, dict) for row in rows), "Every JSONL row must be an object")
        return rows


def _rpc_pairs(rows):
    _require(len(rows) % 2 == 0 and rows, "RPC requests/responses are incomplete")
    result = []
    for index in range(0, len(rows), 2):
        request_row, response_row = rows[index:index + 2]
        _require(request_row.get("direction") == "request" and response_row.get("direction") == "response", "RPC log is not a complete serial request/response stream")
        request, response = request_row.get("payload"), response_row.get("payload")
        _require(isinstance(request, dict) and isinstance(response, dict), "RPC payload is missing")
        _require(type(request.get("id")) is int and request["id"] == index // 2 and response.get("id") == request["id"], "Missing, repeated or mismatched RPC id")
        _ok(response)
        _require(isinstance(request.get("args"), dict), "RPC arguments must be an object")
        result.append((request, response))
    return result


def _sequence(row):
    match = re.fullmatch(r"d([0-9]{7,})-c([0-9]+)", row.get("local_decision_id", ""))
    _require(match is not None, "local_decision_id does not encode an exact sequence/candidate pair")
    return tuple(map(int, match.groups()))


def _actions(records, events, job_id):
    indexed, groups = {}, defaultdict(list)
    for row in records:
        decision_identity(row)
        _require(row["decision_id"] == job_id + ":" + row["local_decision_id"] and row["decision_id"] not in indexed, "Missing or duplicate decision identity")
        _require(row["benchmark"] == "made" and row["split"] in {"train", "dev"} and row["episode_index"] == 0,
                 "Only MADE train/dev collection episode zero is supported")
        stamp = row["policy_stamp"]
        _require(isinstance(stamp, Mapping) and type(stamp.get("generation")) is int and stamp["generation"] == 0
                 and stamp.get("perturbation_seed") is None and stamp.get("perturbation_sigma") is None,
                 "Collection labels require the unchanged initial policy, not ES/test trajectories")
        generation = row.get("generation")
        _require(isinstance(generation, dict) and type(generation.get("success")) is bool, "Every proposal needs observed generation.success")
        disposition = row.get("disposition")
        _require(disposition in {"executed", "rejected_not_executed", "invalid_not_executed"}, "Decision has an unfinished/unknown disposition")
        _require(disposition != "executed" or generation["success"], "An invalid generation cannot be an executed policy action")
        seq, candidate = _sequence(row)
        indexed[row["decision_id"]] = row
        groups[seq].append((candidate, row))
    seen, recoveries, event_order = set(), {}, []
    for event in events:
        if event.get("event") == "proposed":
            identifier = event.get("decision_id")
            _require(identifier in indexed and identifier not in seen, "Missing/duplicate proposal event or unflushed decision row")
            row = indexed[identifier]
            _require(all(event.get(k) == row[k] for k in ("policy_stamp", "prefix_hash", "input_ids_file")), "Proposal event identity differs from its decision")
            seen.add(identifier); event_order.append((*_sequence(row), "proposal"))
        elif event.get("event") == "fixed_failure_recovery":
            seq = event.get("sequence")
            _require(type(seq) is int and seq >= 0 and seq not in recoveries and seq not in groups, "Duplicate or ambiguous recovery sequence")
            recoveries[seq] = event.get("action"); event_order.append((seq, -1, "recovery"))
        else:
            raise FailureLabelError("Unknown decision event; cannot reconstruct a complete execution sequence")
    _require(seen == set(indexed), "Every proposal must occur exactly once in decision_events")
    _require(event_order == sorted(event_order), "Decision events are not in chronological sequence/candidate order")
    sequences = sorted(set(groups) | set(recoveries))
    _require(sequences and sequences == list(range(sequences[-1] + 1)), "Missing action-generation/recovery sequence")
    actions = []
    for seq in sequences:
        if seq in recoveries:
            action, identifier = recoveries[seq], None
        else:
            candidates = sorted(groups[seq], key=lambda pair: pair[0])
            _require([c for c, _ in candidates] == list(range(len(candidates))), "Missing/duplicate proposal candidate ordinal")
            chosen = [row for _, row in candidates if row["disposition"] == "executed"]
            _require(len(chosen) <= 1, "Multiple actions executed for one decision sequence")
            if not chosen:
                _require(not any(row["generation"]["success"] for _, row in candidates), "Valid proposal sequence has no selected action")
                continue
            identifier = chosen[0]["decision_id"]; action = chosen[0]["generation"].get("parsed_action")
        _require(isinstance(action, dict) and action.get("tool") in TOOLS and isinstance(action.get("arguments"), dict), "Executed/recovery action lacks exact tool and arguments")
        actions.append({"sequence": seq, "decision_id": identifier, "action": action, "recovery": identifier is None})
    return actions


def derive_failure_labels(manifest_path, *, output_path=None) -> dict:
    """Derive every row of one complete B50 MADE collection; write only an external sidecar."""
    manifest_path = Path(manifest_path).resolve(); directory = manifest_path.parent
    sources = _Sources(); manifest = sources.json(manifest_path, "collection_manifest")
    _require(isinstance(manifest, dict) and manifest.get("complete") is True, "Only an explicitly complete collection manifest can be labelled")
    job = sources.json(directory / "job.json", "job")
    _require(isinstance(job, dict) and isinstance(job.get("job_id"), str)
             and re.fullmatch(r"collection-made-[A-Za-z0-9_-]+", job["job_id"])
             and job.get("model_key") in {"qwen35_4b", "qwen35_9b"}, "Invalid collection job/model identity")
    _require(job.get("job_id") == manifest.get("job_id") and job.get("benchmark") == "made"
             and job.get("stage") == "collection" and job.get("method") == "baseline" and job.get("split") in {"train", "dev"}
             and job.get("budget") == 50 and job.get("expected_counts", {}).get("candidate_oracle_attempts") == 50,
             "Labels require the full B50 MADE train/dev collection job")
    entries = manifest.get("decision_files")
    _require(isinstance(entries, list) and entries, "Manifest must declare all decision files")
    records, locations, paths = [], {}, set()
    for entry in entries:
        _require(isinstance(entry, dict) and isinstance(entry.get("sha256"), str), "Decision file needs a declared SHA256")
        path = _resolve(entry.get("path"), directory)
        _require(path not in paths, "Duplicate decision file entry"); paths.add(path)
        for line, row in enumerate(sources.lines(path, "decisions", entry["sha256"]), 1):
            _require(row.get("model_key") == job.get("model_key") and row.get("split") == job["split"]
                     and row.get("task_id") == job.get("task_id") and row.get("group_id") == job.get("group_id"), "Decision does not belong to the collection job")
            records.append(row)
            locations[row.get("decision_id")] = {"path": str(path), "line": line, "sha256": entry["sha256"]}
            token_path = _resolve(row.get("input_ids_file"), directory)
            sources.read(token_path, "captured_input_ids")
    events = sources.lines(directory / "decision_events.jsonl", "decision_events")
    actions = _actions(records, events, job["job_id"])
    pairs = _rpc_pairs(sources.lines(directory / "rpc/rpc.jsonl", "raw_rpc"))
    _require(pairs[0][0]["op"] == "init" and _ok(pairs[0][1]), "Complete collection requires successful RPC initialization")
    _require(pairs[0][0]["args"].get("budget") == 50 and pairs[0][0]["args"].get("benchmark") == "made", "RPC initialization differs from the B50 MADE job")
    summaries = sources.json(directory / "episodes.json", "episode_summaries")
    _require(isinstance(summaries, list) and len(summaries) == 1 and summaries[0].get("complete") is True
             and summaries[0].get("costs", {}).get("candidate_oracle_attempts") == 50, "Collection episode does not verify the full physical budget")
    outputs = {}
    for row in records:
        outputs[row["decision_id"]] = {
            **decision_identity(row), "row_identity_hash": row_identity_hash(row), "source_decision": locations[row["decision_id"]],
            "labels": {head: int(not row["generation"]["success"]) if head == "generation_invalid" else None for head in HEADS},
            "horizons": dict(HORIZONS), "source_rpc_ids": {head: [] for head in HEADS},
            "unknown_reasons": {head: "not_executed" if row["disposition"] != "executed" else "not_applicable_or_output_unknown" if head in {"screening_unavailable", "candidate_generation_failure"} else "no_observed_event" for head in HEADS[1:]},
            "execution_sequence": _sequence(row)[0], "disposition": row["disposition"], "evidence": {},
        }
    cursor, scientific_count, waiting = 1, 0, []
    alignment = []
    for action in actions:
        _require(cursor < len(pairs), "Executed action is missing its RPC tool request")
        request, response = pairs[cursor]; cursor += 1
        expected = {"name": action["action"]["tool"], "arguments": action["action"]["arguments"]}
        _require(request["op"] == "tool" and request["args"] == expected, "Exact ordered action/RPC tool name or arguments mismatch")
        alignment.append({"sequence": action["sequence"], "decision_id": action["decision_id"], "recovery": action["recovery"], "tool_rpc_id": request["id"], "action_hash": fingerprint(expected)})
        observed = tool_failure_labels(action["action"], response)
        if action["decision_id"] is not None:
            row = outputs[action["decision_id"]]; waiting.append(row)
            for head, value in observed.items():
                row["labels"][head] = value
                if value is not None:
                    row["source_rpc_ids"][head] = [request["id"]]; row["unknown_reasons"].pop(head, None)
                elif head == "candidate_generation_failure" and action["action"]["tool"] in {"generate_structures", "create_structure"}:
                    row["source_rpc_ids"][head] = [request["id"]]
                    row["unknown_reasons"][head] = "generation_tool_failed_output_unknown" if not response["ok"] else "accepted_candidates_not_observed"
            row["evidence"]["screening"] = screening_evidence(action["action"], response)
        if action["action"]["tool"] == "select_for_evaluation" and response["ok"]:
            _require(cursor < len(pairs), "Selected candidate has no scientific response")
            scientific_request, scientific_response = pairs[cursor]; cursor += 1
            _require(scientific_request["op"] == "step" and scientific_request["args"] == {}, "Successful selection must be followed by its scientific step")
            labels = scientific_failure_labels(scientific_response)
            scientific_count += 1
            observation = scientific_response["result"].get("observation", {}) if scientific_response["ok"] else scientific_response.get("error", {}).get("details", {})
            _require(observation.get("counts", {}).get("candidate_oracle_attempts") == scientific_count,
                     "Scientific step lacks an exact one-attempt physical counter increment")
            for row in waiting:
                for head, value in labels.items():
                    row["labels"][head] = value; row["source_rpc_ids"][head] = [scientific_request["id"]]
                    if value is None:
                        row["unknown_reasons"][head] = "scientific_computation_failed" if labels["scientific_evaluation_failure"] else "official_classification_not_observed"
                    else:
                        row["unknown_reasons"].pop(head, None)
                row["evidence"]["scientific_evaluation_trigger"] = {"sequence": action["sequence"], "recovery": action["recovery"], "decision_id": action["decision_id"]}
            waiting.clear()
            alignment[-1]["scientific_rpc_id"] = scientific_request["id"]
            if scientific_response["ok"]:
                continue
        _require(cursor < len(pairs) and pairs[cursor][0]["op"] == "observe" and pairs[cursor][0]["args"] == {} and _ok(pairs[cursor][1]),
                 "Executed action lacks its expected successful observe response")
        cursor += 1
    _require(scientific_count == 50 and not waiting, "Complete collection lacks all scientific outcomes for executed prefixes")
    _require(cursor == len(pairs) - 1 and pairs[cursor][0]["op"] == "close" and pairs[cursor][0]["args"] == {} and _ok(pairs[cursor][1]),
             "Unexpected/unmatched RPC actions or missing clean close boundary")
    rows = sorted(outputs.values(), key=lambda row: _sequence(row))
    online_checks = Counter()
    for original in records:
        derived = outputs[original["decision_id"]]
        for field, target in (("observed_failure_types", "labels"), ("failure_type_observation_rpc_ids", "source_rpc_ids")):
            if field in original:
                _require(fingerprint(original[field]) == fingerprint(derived[target]),
                         f"Online {field} disagrees with independently derived labels/RPC evidence: {original['decision_id']}")
                online_checks[field] += 1
    result = {"schema": SCHEMA, "heads": list(HEADS), "label_definitions": dict(LABEL_DEFINITIONS),
        "source_manifest": {"path": str(manifest_path), "sha256": sources.items[str(manifest_path)]["sha256"]},
        "source_manifest_sha256": sources.items[str(manifest_path)]["sha256"], "job_id": job["job_id"], "model_key": job["model_key"], "benchmark": "made",
        "source_classification": job.get("classification", "recorded_baseline_collection"),
        "source_files": [sources.items[key] for key in sorted(sources.items)], "records": rows, "execution_alignment": alignment,
        "counts": {head: dict(Counter("unknown" if row["labels"][head] is None else str(row["labels"][head]) for row in rows)) for head in HEADS},
        "complete": True, "all_proposals_accounted": len(rows), "scientific_evaluations": scientific_count,
        "online_observation_records_checked": dict(online_checks),
        "physical_oracle_calls_by_derivation": 0, "raw_data_modified": False, "causal_responsibility_claim": False,
        "label_evidence_is_not_a_prediction_feature": True, "deriver_source_sha256": file_sha256(__file__)}
    result["sidecar_fingerprint"] = fingerprint(result)
    _require(all(file_sha256(item["path"]) == item["sha256"] for item in result["source_files"]), "Source changed while labels were being derived")
    if output_path is not None:
        output = Path(output_path).resolve()
        _require(directory != output.parent and directory not in output.parents, "Sidecars must be outside the immutable collection directory")
        if output.exists():
            _require(_json(output.read_bytes(), output) == result, "Existing sidecar differs; use a new output path, never overwrite evidence")
        else:
            write_json_atomic(output, result)
    return result


def load_failure_labels(sidecar_path, *, manifest_path) -> dict:
    """Verify sources and rederive labels; enriched fitter rows compare row_identity_hash only."""
    sidecar_path = Path(sidecar_path).resolve()
    result = _json(sidecar_path.read_bytes(), sidecar_path)
    _require(isinstance(result, dict) and result.get("schema") == SCHEMA
             and result.get("sidecar_fingerprint") == fingerprint({k: v for k, v in result.items() if k != "sidecar_fingerprint"}), "Modified/unknown failure-label sidecar")
    expected = derive_failure_labels(manifest_path)
    _require(result == expected, "Failure labels, source identity, or derivation schema no longer match")
    return result
