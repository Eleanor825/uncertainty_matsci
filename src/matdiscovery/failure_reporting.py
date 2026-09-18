"""Read-only MADE type calibration, with an exact ordered raw-RPC evidence audit.

These are descriptive predictions of observed events, not causal attributions or
paired improvement estimates. Missing labels/predictions remain missing. The
caller must also run the existing final-envelope and complete-budget verifiers;
this additional audit does not replace either of those gates.
"""
from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from collections.abc import Mapping
import math
from pathlib import Path

from .accounting import file_sha256, fingerprint
from .failure_labels import (
    HEADS, HORIZONS, LABEL_DEFINITIONS, TOOLS, FailureLabelError, _Sources,
    _ok, _resolve, _rpc_pairs, _sequence, decision_identity,
    scientific_failure_labels, tool_failure_labels,
)


SCHEMA = "made_failure_type_reporting_v1"
FIELDS = ("observed_failure_types", "failure_type_probabilities", "failure_type_observation_rpc_ids")


def _require(condition, message):
    if not condition:
        raise FailureLabelError(message)


def failure_control_required(protocol, benchmark):
    """CG has no supported type-label schema, even if incorrectly enabled."""
    return benchmark == "made" and benchmark in protocol.get("failure_control", {}).get("enabled_benchmarks", [])


def _read_artifact(sources, envelope, directory, name, kind):
    path = _resolve(name, directory)
    entries = [entry for entry in envelope.get("artifacts", [])
               if _resolve(entry.get("path"), directory) == path]
    _require(len(entries) == 1 and isinstance(entries[0].get("sha256"), str),
             f"Missing or duplicate declared failure evidence artifact: {name}")
    return sources.lines(path, kind, entries[0]["sha256"])


def _actions(rows, events, job, execution_job):
    """Generalized final/test/ES join; no collection-only policy assumptions."""
    indexed, groups = {}, defaultdict(list)
    for row in rows:
        decision_identity(row)
        identifier = row["decision_id"]
        _require(identifier == job["job_id"] + ":" + row["local_decision_id"] and identifier not in indexed,
                 "Missing or duplicate decision identity")
        for key in ("benchmark", "model_key", "task_id"):
            _require(row[key] == job[key], f"Decision {key} differs from its final job")
        for key in ("split", "group_id"):
            if key in execution_job:
                _require(row[key] == execution_job[key], f"Decision {key} differs from executed job")
        _require(row["episode_index"] == 0, "MADE final type audit requires its single episode")
        stamp = row["policy_stamp"]
        _require(isinstance(stamp, Mapping) and type(stamp.get("generation")) is int and stamp["generation"] >= 0
                 and all(isinstance(stamp.get(key), str) and stamp[key] for key in ("state_id", "checkpoint_hash", "model_id")),
                 "Missing actual policy/perturbation identity")
        _require("perturbation_seed" in stamp and "perturbation_sigma" in stamp,
                 "Policy stamp does not record perturbation identity")
        seed, sigma = stamp["perturbation_seed"], stamp["perturbation_sigma"]
        _require(seed is None and sigma is None or type(seed) is int and type(sigma) in (int, float) and math.isfinite(sigma) and sigma > 0,
                 "Incomplete or invalid policy perturbation identity")
        for stamp_key, job_key in (("state_id", "policy_state_id"), ("generation", "selected_generation")):
            if job_key in execution_job:
                _require(stamp[stamp_key] == execution_job[job_key], "Decision actual policy differs from final execution")
        generation = row.get("generation")
        _require(isinstance(generation, Mapping) and type(generation.get("success")) is bool,
                 "Every proposal requires observed generation.success")
        _require(row.get("disposition") in {"executed", "rejected_not_executed", "invalid_not_executed"},
                 "Unfinished proposal disposition")
        _require(row["disposition"] != "executed" or generation["success"], "Invalid generation was executed")
        seq, ordinal = _sequence(row)
        groups[seq].append((ordinal, row)); indexed[identifier] = row
    seen, recoveries, order = set(), {}, []
    for event in events:
        if event.get("event") == "proposed":
            identifier = event.get("decision_id")
            _require(identifier in indexed and identifier not in seen, "Unknown or repeated proposal event")
            row = indexed[identifier]
            _require(all(event.get(key) == row[key] for key in ("policy_stamp", "prefix_hash", "input_ids_file")),
                     "Proposal event identity differs from decision row")
            seen.add(identifier); order.append((*_sequence(row), "proposal"))
        elif event.get("event") == "fixed_failure_recovery":
            seq = event.get("sequence")
            _require(type(seq) is int and seq >= 0 and seq not in groups and seq not in recoveries,
                     "Ambiguous fixed recovery sequence")
            recoveries[seq] = event.get("action"); order.append((seq, -1, "recovery"))
        else:
            raise FailureLabelError("Unknown decision event")
    _require(seen == set(indexed) and order == sorted(order), "Incomplete or unordered proposal events")
    sequences = sorted(set(groups) | set(recoveries))
    _require(sequences and sequences == list(range(sequences[-1] + 1)), "Missing decision sequence")
    actions = []
    for seq in sequences:
        identifier = None
        if seq in recoveries:
            action = recoveries[seq]
        else:
            candidates = sorted(groups[seq], key=lambda item: item[0])
            _require(len(candidates) <= 2 and [i for i, _ in candidates] == list(range(len(candidates))),
                     "Invalid candidate budget/ordinals")
            chosen = [row for _, row in candidates if row["disposition"] == "executed"]
            _require(len(chosen) <= 1, "Multiple executed candidates for one decision")
            # A JSON-valid proposal can still fail the controller's action schema.
            if not chosen:
                continue
            identifier = chosen[0]["decision_id"]
            action = chosen[0]["generation"].get("parsed_action")
        _require(isinstance(action, Mapping) and action.get("tool") in TOOLS and isinstance(action.get("arguments"), dict),
                 "Executed action lacks an exact legal tool/argument identity")
        actions.append((identifier, action))
    return actions


def _derive(rows, actions, pairs, job):
    expected_count = job.get("expected_counts", {}).get("candidate_oracle_attempts")
    _require(type(expected_count) is int and expected_count > 0, "Final job lacks physical-budget denominator")
    _require(pairs[0][0].get("op") == "init" and _ok(pairs[0][1])
             and pairs[0][0]["args"].get("benchmark") == "made"
             and pairs[0][0]["args"].get("budget") == expected_count, "RPC initialization differs from final job")
    derived = {row["decision_id"]: {
        "labels": {head: int(not row["generation"]["success"]) if head == "generation_invalid" else None for head in HEADS},
        "rpc_ids": {head: [] for head in HEADS}} for row in rows}
    cursor, scientific_count, waiting = 1, 0, []
    for identifier, action in actions:
        _require(cursor < len(pairs), "Executed action has no raw tool RPC")
        request, response = pairs[cursor]; cursor += 1
        _require(request.get("op") == "tool" and request["args"] == {"name": action["tool"], "arguments": action["arguments"]},
                 "Exact ordered action/raw tool request mismatch")
        measured = tool_failure_labels(action, response)
        if identifier is not None:
            target = derived[identifier]; waiting.append(target)
            target["labels"].update(measured)
            for head, value in measured.items():
                if value is not None or head == "candidate_generation_failure" and action["tool"] in {"generate_structures", "create_structure"}:
                    target["rpc_ids"][head] = [request["id"]]
        if action["tool"] == "select_for_evaluation" and response["ok"]:
            _require(cursor < len(pairs), "Successful selection has no scientific RPC")
            request, response = pairs[cursor]; cursor += 1
            _require(request.get("op") == "step" and request["args"] == {}, "Selection must precede its next scientific step")
            measured = scientific_failure_labels(response)
            scientific_count += 1
            observation = response["result"].get("observation", {}) if response["ok"] else response.get("error", {}).get("details", {})
            _require(observation.get("counts", {}).get("candidate_oracle_attempts") == scientific_count,
                     "Scientific RPC lacks exact physical-counter increment")
            for target in waiting:
                target["labels"].update(measured)
                for head in measured:
                    # The response is evidence of unknown on computation failure.
                    target["rpc_ids"][head] = [request["id"]]
            waiting.clear()
            if response["ok"]:
                continue
        _require(cursor < len(pairs) and pairs[cursor][0].get("op") == "observe" and pairs[cursor][0]["args"] == {} and _ok(pairs[cursor][1]),
                 "Executed action lacks its expected observe response")
        cursor += 1
    _require(scientific_count == expected_count and not waiting, "Incomplete next-scientific-event coverage")
    _require(cursor == len(pairs) - 1 and pairs[cursor][0].get("op") == "close" and pairs[cursor][0]["args"] == {} and _ok(pairs[cursor][1]),
             "Unmatched raw RPC or missing clean close")
    return derived, scientific_count


def audit_failure_types_for_job(envelope, directory, job, *, required=False):
    """Validate typed rows against immutable raw evidence; return audited records.

    `required` is set from the frozen protocol, not inferred from existing rows.
    A legacy optional missing field is `not_recorded`, distinct from explicit
    unknown (`null`). This function never fills missing online data with labels
    it could derive retrospectively.
    """
    directory = Path(directory).resolve()
    identity = {key: job[key] for key in ("job_id", "benchmark", "model_key", "method")}
    if job["benchmark"] != "made":
        return {**identity, "status": "unsupported_benchmark", "required": False, "records": [], "source_files": []}
    sources = _Sources()
    rows = _read_artifact(sources, envelope, directory, "decisions.jsonl", "decisions")
    typed = any(field in row for row in rows for field in FIELDS)
    _require(not required or typed, "Enabled MADE failure control requires complete online type records")
    if not typed:
        return {**identity, "status": "not_recorded", "required": False, "records": rows,
                "source_files": list(sources.items.values())}
    for row in rows:
        for field in FIELDS:
            value = row.get(field, {})
            _require(isinstance(value, Mapping) and set(value) <= set(HEADS), f"Malformed {field}")
            _require(not required or set(value) == set(HEADS), f"Enabled MADE failure control requires all seven heads in {field}")
        if "failure_type_horizons" in row:
            _require(row["failure_type_horizons"] == HORIZONS, "Type horizon differs from declared event definition")
    events = _read_artifact(sources, envelope, directory, "decision_events.jsonl", "decision_events")
    pairs = _rpc_pairs(_read_artifact(sources, envelope, directory, "rpc/rpc.jsonl", "raw_rpc"))
    actions = _actions(rows, events, job, envelope.get("execution_job", {}))
    derived, count = _derive(rows, actions, pairs, job)
    for row in rows:
        expected = derived[row["decision_id"]]
        observed, ids, probabilities = (row.get(field, {}) for field in (FIELDS[0], FIELDS[2], FIELDS[1]))
        _require(set(observed) == set(ids), "Observed type heads and source-id heads must match")
        for head, value in observed.items():
            _require(value is None or type(value) is int and value in (0, 1), "Observed type must be binary or explicit unknown")
            _require(value == expected["labels"][head], f"Online observed label contradicts raw evidence: {head}")
            _require(isinstance(ids[head], list) and all(type(i) is int for i in ids[head]) and ids[head] == expected["rpc_ids"][head],
                     f"Type source RPC IDs do not match exact tool/nearest scientific evidence: {head}")
        for value in probabilities.values():
            _require(value is None or type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                     "Type probability must be finite in [0,1] or explicit unknown")
    _require(all(file_sha256(item["path"]) == item["sha256"] for item in sources.items.values()), "Failure evidence changed during audit")
    return {**identity, "status": "audited", "required": bool(required), "records": rows,
            "scientific_evaluations": count, "source_files": list(sources.items.values()),
            "audited_online_rows_fingerprint": fingerprint([{key: row.get(key) for key in ("decision_id", "policy_stamp", *FIELDS)} for row in rows])}


COUNT_KEYS = ("proposals", "executed", "not_executed", "observed", "unknown", "label_not_recorded",
              "available", "prediction_unknown", "prediction_not_recorded", "scored", "unscored", "available_unlabelled")
CONTROLLER_COUNT_KEYS = ("proposals", "control_enabled_proposals", "prediction_only_proposals", "control_use_not_recorded",
    "controller_records", "controller_not_recorded", "triggered_proposals", "retry_requested_proposals",
    "feedback_prepared_proposals", "feedback_received_proposals", "selection_decisions", "type_ranked_decisions", "fallback_decisions")


class FailureTypeReporter:
    """Accumulate one audited job at a time; keep scores in compact numeric arrays."""
    def __init__(self):
        self.groups = {}
        self.controllers = {}
        self.jobs = []
        self.stamps = {}
        self._seen = set()

    def add_job(self, envelope, directory, job, *, required=False):
        _require(job["job_id"] not in self._seen, "Duplicate job in type report")
        audited = audit_failure_types_for_job(envelope, directory, job, required=required)
        self._seen.add(job["job_id"])
        proof = {key: value for key, value in audited.items() if key != "records"}
        records = audited["records"]
        proof["proposals"] = len(records)
        stamp_counts = Counter()
        key = (job["model_key"], job["method"])
        if job["benchmark"] != "made":
            self.jobs.append(proof)
            return proof
        controller = self.controllers.setdefault(key, {"counts": Counter({k: 0 for k in CONTROLLER_COUNT_KEYS}), "triggers": Counter(), "feedback_types": Counter(),
            "actions": Counter(), "common_head_sets": Counter(), "fallbacks": Counter(), "excluded_heads": Counter()})
        selections = {}
        for row in records:
            stamp = row.get("policy_stamp")
            if stamp is not None:
                digest = fingerprint(stamp); self.stamps[digest] = stamp; stamp_counts[digest] += 1
            observed, probabilities = row.get(FIELDS[0], {}), row.get(FIELDS[1], {})
            for head in HEADS:
                group = self.groups.setdefault((*key, head), {"counts": Counter({k: 0 for k in COUNT_KEYS}), "y": array("b"), "p": array("d")})
                counts = group["counts"]; counts["proposals"] += 1
                counts["executed" if row.get("disposition") == "executed" else "not_executed"] += 1
                y, p = observed.get(head), probabilities.get(head)
                counts["label_not_recorded" if head not in observed else "unknown" if y is None else "observed"] += 1
                counts["prediction_not_recorded" if head not in probabilities else "prediction_unknown" if p is None else "available"] += 1
                if y is not None and p is not None:
                    counts["scored"] += 1; group["y"].append(y); group["p"].append(p)
                elif y is not None:
                    counts["unscored"] += 1
                elif p is not None:
                    counts["available_unlabelled"] += 1
            counts = controller["counts"]; counts["proposals"] += 1
            if "failure_types_used_for_control" in row:
                counts["control_enabled_proposals" if row["failure_types_used_for_control"] else "prediction_only_proposals"] += 1
            else:
                counts["control_use_not_recorded"] += 1
            plan = row.get("failure_controller")
            if plan is not None:
                _require(isinstance(plan, Mapping), "Malformed controller record")
                counts["controller_records"] += 1
                controller["actions"][str(plan.get("controller_action"))] += 1
                if plan.get("trigger") is not None:
                    counts["triggered_proposals"] += 1; controller["triggers"][str(plan["trigger"])] += 1
                counts["retry_requested_proposals"] += int(plan.get("request_retry") is True)
                counts["feedback_prepared_proposals"] += int(bool(plan.get("feedback_for_retry")))
            else:
                counts["controller_not_recorded"] += 1
            feedback = row.get("feedback_received_for_this_candidate")
            if feedback:
                _require(isinstance(feedback, Mapping), "Malformed received feedback")
                counts["feedback_received_proposals"] += 1
                controller["feedback_types"][str(feedback.get("predicted_failure_type"))] += 1
            selection = row.get("failure_candidate_selection")
            if selection is not None:
                _require(isinstance(selection, Mapping), "Malformed candidate selection")
                sequence = _sequence(row)[0]
                _require(sequence not in selections or selections[sequence] == selection, "Candidates disagree on the same selection audit")
                selections[sequence] = selection
        for selection in selections.values():
            counts = controller["counts"]; counts["selection_decisions"] += 1
            common = selection.get("common_comparison_heads")
            _require(isinstance(common, list) and len(set(common)) == len(common) and set(common) <= set(HEADS), "Malformed common-head selection evidence")
            controller["common_head_sets"][",".join(head for head in HEADS if head in common) or "<none>"] += 1
            counts["type_ranked_decisions"] += int(selection.get("type_signals_used") is True)
            fallback = selection.get("fallback")
            if fallback is not None:
                counts["fallback_decisions"] += 1; controller["fallbacks"][str(fallback)] += 1
            for candidate in selection.get("ranking", []):
                controller["excluded_heads"].update(candidate.get("excluded_from_cross_action_ranking", {}).keys())
        proof["policy_stamp_counts"] = dict(stamp_counts)
        self.jobs.append(proof)
        return proof

    def report(self):
        from .metrics import calibration_metrics
        groups = []
        for (model, method, head), data in sorted(self.groups.items()):
            y, p, counts = data["y"], data["p"], dict(data["counts"])
            metrics = calibration_metrics(y, p) if y else None
            curve = []
            for threshold in (0.0, .1, .2, .4, .6, .8, 1.0):
                retained = [label for label, probability in zip(y, p) if probability <= threshold]
                curve.append({"threshold": threshold, "retained_scored": len(retained),
                    "coverage_of_scored": len(retained) / len(y) if y else None,
                    "observed_risk_on_retained_scored": sum(retained) / len(retained) if retained else None})
            groups.append({"benchmark": "made", "model_key": model, "method": method, "head": head,
                "counts": counts, "scored_fraction_of_observed": counts["scored"] / counts["observed"] if counts["observed"] else None,
                "prediction_availability_of_proposals": counts["available"] / counts["proposals"] if counts["proposals"] else None,
                "pooled_descriptive_metrics": metrics, "risk_coverage_on_scored_only": curve})
        controllers = [{"benchmark": "made", "model_key": key[0], "method": key[1],
                        **{name: dict(sorted(counts.items())) for name, counts in data.items()}}
                       for key, data in sorted(self.controllers.items())]
        return {"schema": SCHEMA, "heads": list(HEADS), "horizons": dict(HORIZONS), "label_definitions": dict(LABEL_DEFINITIONS),
            "descriptive_only": True, "paired_significance_estimated": False, "type_improvement_claimed": False,
            "unknown_labels_or_predictions_imputed": False, "causal_responsibility_claim": False,
            "complete_study_acceptance_claimed": False, "physical_oracle_calls_by_reporting": 0,
            "count_definitions": {"observed": "online label is 0 or 1", "unknown": "online label explicitly null",
                "label_not_recorded": "online label field/head absent in an optional legacy protocol",
                "available": "online prediction is finite in [0,1]", "scored": "both observed and available",
                "unscored": "observed label without an available prediction", "available_unlabelled": "available prediction without an observed label"},
            "unit_warning": "Dependent proposals and repeated next-science labels; pooled complete-case metrics are descriptive. Coverage and unknown counts must accompany every comparison. No new paired intervals or type-specific improvement claim.",
            "controller_count_warning": "Plans/triggers/prepared feedback count proposals; received feedback is separate. Selection/common-head/fallback records are deduplicated per decision. Excluded-head counts count candidate records.",
            "groups": groups, "controller_groups": controllers, "job_status_counts": dict(Counter(job["status"] for job in self.jobs)),
            "jobs": self.jobs, "policy_stamps": self.stamps, "reporter_source_sha256": file_sha256(__file__)}
