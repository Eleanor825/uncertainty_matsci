"""Deterministic MADE failure-type control; no tool, oracle, model or I/O calls.

Only pass the observation actually rendered in the policy's user message.
The output is advisory: the caller owns its existing two-candidate loop and
max_tools_per_query recovery. No request or budget is created here. Head
applicability and ranking are identical across method arms; missing predictions
remain None and never become a predicted zero probability.

Tool contracts audited against rollouts.TOOL_ARGUMENTS and
benchmark_adapters.MADEAdapter._tool. CrystalGym is explicitly unsupported.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any

SCHEMA = "made_failure_type_controller_v1"
MAX_FEEDBACK_CHARS = 240
HEADS = (
    "generation_invalid", "candidate_generation_failure", "tool_execution_failure",
    "screening_unavailable", "scientific_evaluation_failure", "unstable", "not_new",
)
TOOL_ARGUMENTS = {
    "generate_structures": {"compositions", "generator_name", "num_candidates"},
    "create_structure": {"structure", "a", "b", "c", "alpha", "beta", "gamma", "species", "frac_coords"},
    "score_buffer": {"scorer_name", "composition"},
    "query_structures": {"composition", "k", "mode", "scorer_name", "include_structure_details"},
    "list_compositions": set(), "get_buffer_stats": set(),
    "select_for_evaluation": {"composition", "structure_hash", "structure_index", "scorer_name"},
}
HEAD_WEIGHTS = dict(zip(HEADS, (2., 1., 2., 1., 2., 1., 1.)))
HORIZONS = {
    "generation_invalid": "current_proposal_JSON_or_action_schema",
    "candidate_generation_failure": "generate_or_create_zero_accepted_candidates",
    "tool_execution_failure": "proposed_tool_execution",
    "screening_unavailable": "surrogate_screening_availability",
    "scientific_evaluation_failure": "next_scientific_evaluation",
    "unstable": "next_scientific_evaluation",
    "not_new": "next_scientific_evaluation",
}
_FEEDBACK = {
    "generation_invalid": "Repair the JSON/tool/arguments schema. Return one allowed tool with only its documented arguments; do not invent results.",
    "candidate_generation_failure": "Revise the allowed integer composition or valid geometry; use chemeleon or random for generation. Acceptance and stability remain unmeasured.",
    "tool_execution_failure": "Repair the tool arguments. Use an actual visible composition/hash and documented modes or scorers; do not repeat an invalid call.",
    "screening_unavailable": "Do not rank by unavailable/nonfinite oracle-surrogate scores. Query without ranking or use diversity/random; no stability conclusion follows.",
    "scientific_evaluation_failure": "Check candidate geometry and legality; prefer another real buffered hash. Keep the fixed ORB settings and do not add a validation evaluation.",
    "unstable": "Compare candidates using finite existing scores, or propose a different valid candidate. Surrogate scores do not establish final stability.",
    "not_new": "Prefer a different buffered structure_hash or generate a new candidate. A different hash alone does not establish scientific novelty.",
    "overall_risk": "Reconsider this action using only visible evidence and documented tools. Do not invent observations or add an evaluation.",
    "risk_unavailable": "Type/overall confidence is unavailable. Recheck the legal action using visible evidence; unavailable confidence is not a low-risk prediction.",
}
_BUDGET_NOTE = " Use only the existing retry and tool/query limits."
RANK_RULE = "valid_first; common_head_max, common_head_weighted_mean, known_overall_then_probability, original_index"


def _probability(value):
    return float(value) if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1 else None


def _normalize(risk_types):
    if risk_types is None:
        risk_types = {}
    if not isinstance(risk_types, Mapping):
        raise TypeError("risk_types must be a head -> probability/None mapping.")
    values = {head: _probability(risk_types.get(head)) for head in HEADS}
    unavailable = {head: "missing" if risk_types.get(head) is None else "invalid_probability" for head in HEADS if values[head] is None}
    return values, unavailable


def _visible(observation):
    if not isinstance(observation, Mapping):
        return {}
    # Accept the already rendered observation or its user-message wrapper only.
    return observation.get("observation", observation) if isinstance(observation.get("observation", observation), Mapping) else {}


def _entries(observation):
    result = []
    buffer = observation.get("buffer", {})
    if isinstance(buffer, Mapping):
        for composition, rows in buffer.items():
            if not isinstance(composition, str) or not composition or not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                digest = row.get("hash", row.get("structure_hash"))
                if isinstance(digest, str) and digest:
                    result.append((composition, digest, row))
    return sorted(result, key=lambda item: (item[0], item[1]))


def _action_parts(action):
    if not isinstance(action, Mapping):
        return None, {}
    args = action.get("arguments", {})
    return action.get("tool"), dict(args) if isinstance(args, Mapping) else {}


def applicable_heads(action):
    """Generation-invalid is JSON validity for EVERY proposal, not crystal quality."""
    tool, args = _action_parts(action)
    applicable = {"generation_invalid", "tool_execution_failure"}
    if tool in TOOL_ARGUMENTS:
        applicable.update({"scientific_evaluation_failure", "unstable", "not_new"})
    if tool in {"generate_structures", "create_structure"}:
        applicable.add("candidate_generation_failure")
    uses_oracle = tool == "score_buffer" and args.get("scorer_name", "oracle") == "oracle"
    uses_oracle |= tool in {"query_structures", "select_for_evaluation"} and args.get("scorer_name") == "oracle"
    if uses_oracle:
        applicable.add("screening_unavailable")
    return [head for head in HEADS if head in applicable]


def _composition_item_has_shape(value):
    """Match the frozen adapter's string/mapping input shapes, not chemistry.

    MADEAdapter._tool passes each item to pymatgen.Composition and then checks
    allowed elements, integral counts and the atom limit. Keep that authority;
    this pure controller must not reject its supported nonempty mappings or
    invent a second approximate chemistry parser.
    """
    return isinstance(value, (str, Mapping)) and bool(value)


def _action_issues(action):
    """Schema checks only; absent entries in a summarized buffer are not invalid."""
    if not isinstance(action, Mapping) or action.get("tool") not in TOOL_ARGUMENTS or not isinstance(action.get("arguments"), Mapping):
        return ["expected_allowed_tool_and_arguments_object"]
    tool, args = _action_parts(action)
    issues = []
    if set(args) - TOOL_ARGUMENTS[tool]: issues.append("unknown_tool_arguments")
    if tool == "generate_structures":
        formulas = args.get("compositions")
        if isinstance(formulas, str): formulas = [part.strip() for part in formulas.split(",") if part.strip()]
        if not isinstance(formulas, list) or not formulas or any(not _composition_item_has_shape(item) for item in formulas):
            issues.append("explicit_compositions_required")
        if args.get("generator_name", "chemeleon") not in {"chemeleon", "random"}: issues.append("invalid_generator")
        count = args.get("num_candidates", 32)
        if type(count) is not int or not 1 <= count <= 1024: issues.append("invalid_num_candidates")
    if tool in {"query_structures", "select_for_evaluation"} and (not isinstance(args.get("composition"), str) or not args["composition"]):
        issues.append("composition_required")
    if "scorer_name" in args and args["scorer_name"] not in {"oracle", "diversity", "random"}: issues.append("invalid_scorer")
    if tool == "query_structures":
        if args.get("mode", "all") not in {"all", "top", "bottom", "random"}: issues.append("invalid_query_mode")
        if args.get("mode", "all") in {"top", "bottom"} and not args.get("scorer_name"): issues.append("ranked_query_requires_scorer")
        if type(args.get("k", 5)) is not int or args.get("k", 5) < 1: issues.append("invalid_query_k")
    if tool == "select_for_evaluation":
        if "structure_hash" in args and (not isinstance(args["structure_hash"], str) or not args["structure_hash"]): issues.append("invalid_structure_hash")
        if "structure_index" in args and (type(args["structure_index"]) is not int or args["structure_index"] < 0): issues.append("invalid_structure_index")
    if tool == "create_structure":
        if "structure" in args:
            if not isinstance(args["structure"], Mapping): issues.append("structure_must_be_object")
        else:
            for key in ("a", "b", "c"):
                value = args.get(key)
                if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    issues.append("invalid_lattice_lengths"); break
            if not args.get("species") or not args.get("frac_coords"): issues.append("species_and_coordinates_required")
    return issues


def _safe_query(observation, composition=None):
    entries = _entries(observation)
    available = sorted({row[0] for row in entries})
    if not available:
        return {"tool": "get_buffer_stats", "arguments": {}}
    comp = composition if composition in available else available[0]
    return {"tool": "query_structures", "arguments": {"composition": comp, "k": 5, "mode": "all"}}


def _preference(head, action, observation):
    tool, args = _action_parts(action)
    entries = _entries(observation)
    if head in {"scientific_evaluation_failure", "not_new"}:
        alternatives = [item for item in entries if item[1] != args.get("structure_hash")]
        if alternatives:
            composition, digest, _ = alternatives[0]
            return {"tool": "select_for_evaluation", "arguments": {"composition": composition, "structure_hash": digest}}
    if head == "candidate_generation_failure":
        formulas = args.get("compositions") if tool == "generate_structures" else None
        if not formulas and entries: formulas = [entries[0][0]]
        if isinstance(formulas, str): formulas = [part.strip() for part in formulas.split(",") if part.strip()]
        if isinstance(formulas, list) and formulas and all(_composition_item_has_shape(item) for item in formulas):
            count = args.get("num_candidates", 32)
            count = count if type(count) is int and 1 <= count <= 1024 else 32
            return {"tool": "generate_structures", "arguments": {"compositions": list(formulas),
                "generator_name": "random" if args.get("generator_name", "chemeleon") == "chemeleon" else "chemeleon", "num_candidates": count}}
    if head == "unstable" and entries:
        composition = args.get("composition", entries[0][0])
        same = [row for comp, _, row in entries if comp == composition]
        scores = [row.get("scores", {}).get("oracle") if isinstance(row.get("scores"), Mapping) else None for row in same]
        # Only previously visible finite scores are used; never request a new MACE/ORB validation.
        if same and all(isinstance(score, Real) and not isinstance(score, bool) and math.isfinite(score) for score in scores):
            return {"tool": "query_structures", "arguments": {"composition": composition, "k": 5, "mode": "top", "scorer_name": "oracle"}}
    if head == "screening_unavailable":
        # Querying all candidates avoids any dependency on unavailable oracle scores.
        return _safe_query(observation, args.get("composition"))
    return _safe_query(observation, args.get("composition"))


def plan_failure_response(benchmark, proposed_action, risk_types, overall_risk,
                          public_observation, threshold=.6):
    """Return a JSON-safe plan for the remaining EXISTING candidate retry only."""
    threshold = _probability(threshold)
    if threshold is None: raise ValueError("threshold must be a finite probability.")
    values, unavailable = _normalize(risk_types)
    overall = _probability(overall_risk)
    observation = _visible(public_observation)
    applicable = applicable_heads(proposed_action) if benchmark == "made" else []
    available = [head for head in HEADS if values[head] is not None]
    usable = [head for head in applicable if values[head] is not None]
    plan = {"schema": SCHEMA, "benchmark": benchmark, "supported": benchmark == "made",
        "available_heads": available, "applicable_heads": applicable, "available_applicable_heads": usable,
        "unavailable_heads": unavailable, "risk_types": values, "overall_risk": overall,
        "head_coverage": {"available": len(usable), "applicable": len(applicable), "fraction": len(usable) / len(applicable) if applicable else None},
        "head_horizons": dict(HORIZONS), "threshold": threshold, "trigger": None, "triggered_heads": [],
        "controller_action": "keep_candidate", "request_retry": False, "feedback_for_retry": "",
        "optional_tool_preference": None, "reason": "No available risk exceeds the fixed threshold.",
        "extra_physical_budget": 0, "extra_generation_budget": 0, "executes_tools": False,
        "tool_preference_is_advisory": True, "overrides_fixed_recovery": False,
        "retry_scope": "remaining_preexisting_candidate_slots_only", "fallback": None}
    if benchmark != "made":
        plan.update(controller_action="unsupported", trigger="unsupported_benchmark", reason="This controller is enabled only for MADE; retain the existing CrystalGym policy.")
        return plan
    limit, used = observation.get("max_tools_per_query"), observation.get("tools_since_query")
    if type(limit) is int and type(used) is int and limit > 0 and used >= limit:
        plan.update(controller_action="defer_to_fixed_recovery", trigger="existing_tool_limit", reason="The unchanged rollout recovery owns the exhausted tool allowance.")
        return plan
    if observation.get("selected") is True:
        plan.update(controller_action="defer_to_existing_execution", trigger="selection_pending", reason="An existing selected candidate must be handled by the unchanged rollout; do not schedule another tool.")
        return plan
    issues = _action_issues(proposed_action)
    plan["local_schema_issues"] = issues
    triggered = [head for head in usable if values[head] >= threshold]
    plan["triggered_heads"] = sorted(triggered, key=lambda head: (-values[head], HEADS.index(head)))
    if issues:
        trigger = "generation_invalid"
        plan["reason"] = "A current-proposal schema issue is observed; this is not a future scientific label."
        plan["trigger_source"] = "local_schema_validation"
    elif triggered:
        trigger = plan["triggered_heads"][0]
        plan["reason"] = "An applicable predicted failure type exceeds the fixed threshold."
        plan["trigger_source"] = "predicted_failure_type"
    elif overall is not None and overall >= threshold:
        trigger = "overall_risk"
        plan["reason"] = "Overall risk exceeds threshold; no subtype is asserted high."
    elif not usable and overall is None:
        trigger = "risk_unavailable"
        plan["reason"] = "Neither applicable type predictions nor overall confidence are available."
    else:
        if not usable:
            plan.update(controller_action="scalar_fallback", fallback="no_available_applicable_type_heads",
                        reason="Explicit scalar-only fallback; missing type probabilities remain unknown.")
        elif len(usable) != len(applicable):
            plan["fallback"] = "partial_type_coverage_recorded; compare_only_common_available_heads"
        return plan
    plan.update(trigger=trigger, controller_action="retry_with_type_feedback" if trigger in HEADS else "retry_with_fallback_feedback", request_retry=True)
    if not usable: plan["fallback"] = "no_available_applicable_type_heads"
    plan["feedback_for_retry"] = _FEEDBACK[trigger] + _BUDGET_NOTE
    if trigger in HEADS:
        plan["optional_tool_preference"] = _preference(trigger, proposed_action, observation)
    if plan["optional_tool_preference"] is not None and _action_issues(plan["optional_tool_preference"]):
        plan["optional_tool_preference"] = None
    assert len(plan["feedback_for_retry"]) <= MAX_FEEDBACK_CHARS
    return plan


def rank_failure_candidates(benchmark, candidates, public_observation, threshold=.6):
    """Compare only the SAME calibrated event heads across valid proposals.

    A head participates only when applicable to every valid candidate and every
    candidate has a finite prediction. Exclusive heads still drive their own
    retry feedback, but cannot create a cross-action missingness penalty.
    Unknown/NA heads are never imputed. If the common set is empty, explicitly
    fall back to known overall confidence, then original order for unknowns.
    """
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or len(candidates) > 2:
        raise ValueError("Supply only the existing one or two candidate proposals.")
    if benchmark != "made":
        return {"schema": SCHEMA, "supported": False, "selected_index": None, "ranking": [],
                "fallback": "unsupported_benchmark", "extra_physical_budget": 0, "extra_generation_budget": 0}
    records = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping): raise TypeError("Candidates must be mappings.")
        action = candidate.get("action", candidate.get("proposed_action"))
        if "action" in candidate and "proposed_action" in candidate and candidate["action"] != candidate["proposed_action"]:
            raise ValueError("Conflicting action/proposed_action fields.")
        plan = plan_failure_response(benchmark, action, candidate.get("risk_types"), candidate.get("overall_risk"), public_observation, threshold)
        valid = candidate.get("valid", True) is True and not _action_issues(action) and plan["controller_action"] not in {"defer_to_fixed_recovery", "defer_to_existing_execution"}
        records.append({"candidate_index": index, "valid": valid, "plan": plan})
    valid_records = [record for record in records if record["valid"]]
    common = [head for head in HEADS if valid_records and all(head in record["plan"]["available_applicable_heads"] for record in valid_records)]
    for record in records:
        plan = record["plan"]
        excluded = {head: "not_applicable_to_this_action" if head not in plan["applicable_heads"]
                    else "prediction_unavailable_for_this_candidate" if plan["risk_types"][head] is None
                    else "not_available_and_applicable_for_every_valid_candidate"
                    for head in HEADS if head not in common}
        compared = {head: plan["risk_types"][head] for head in common} if record["valid"] else {}
        # Missingness is a separate sort field. The zero below is only an unused
        # numeric placeholder within the missing-overall stratum, never a risk.
        overall = plan["overall_risk"]
        overall_order = [int(overall is None), overall if overall is not None else 0.0]
        if common and record["valid"]:
            maximum = max(compared.values())
            mean = sum(compared[head] * HEAD_WEIGHTS[head] for head in common) / sum(HEAD_WEIGHTS[head] for head in common)
            key = [0, maximum, mean, *overall_order, record["candidate_index"]]
        elif common:
            key = [1, 0.0, 0.0, *overall_order, record["candidate_index"]]
        else:
            key = [int(not record["valid"]), *overall_order, record["candidate_index"]]
        record.update(rank_key=key, compared_head_probabilities=compared, head_coverage=plan["head_coverage"],
                      unknown_applicable_heads=[head for head in plan["applicable_heads"] if plan["risk_types"][head] is None],
                      excluded_from_cross_action_ranking=excluded, missing_probabilities_are_not_imputed=True)
    ranking = sorted(records, key=lambda record: record["rank_key"])
    selected = next((record["candidate_index"] for record in ranking if record["valid"]), None)
    return {"schema": SCHEMA, "supported": True, "selected_index": selected, "ranking": ranking,
            "ranking_rule": RANK_RULE if common else "valid_first; known_overall_then_probability; original_index",
            "common_comparison_heads": common, "type_signals_used": bool(common),
            "fallback": None if common else "scalar_only_no_common_available_types" if valid_records else "no_valid_candidate",
            "missing_probabilities_are_not_imputed": True,
            "extra_physical_budget": 0, "extra_generation_budget": 0, "overrides_fixed_recovery": False}


def select_failure_candidate(benchmark, candidates, public_observation, threshold=.6):
    """Alias returning the same full selection audit, not only an opaque index."""
    return rank_failure_candidates(benchmark, candidates, public_observation, threshold)
