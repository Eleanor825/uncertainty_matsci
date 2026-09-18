"""Pure deterministic CPU tests; no policy, GPU, oracle, tool or network runs."""
import ast
import copy
import json
import math
from pathlib import Path
import unittest

from matdiscovery.failure_controller import (
    HEADS, HORIZONS, MAX_FEEDBACK_CHARS, TOOL_ARGUMENTS,
    applicable_heads, plan_failure_response, rank_failure_candidates,
)


OBS = {"elements": ["Na", "Cl"], "selected": False,
       "buffer": {"NaCl": [{"hash": "known-a", "scores": {"oracle": -.1}, "num_sites": 2},
                            {"hash": "known-b", "scores": {"oracle": -.2}, "num_sites": 2}]}}
SELECT = {"tool": "select_for_evaluation", "arguments": {"composition": "NaCl", "structure_hash": "known-a"}}
GENERATE = {"tool": "generate_structures", "arguments": {"compositions": ["Na1Cl1"], "generator_name": "chemeleon", "num_candidates": 32}}
SCORE = {"tool": "score_buffer", "arguments": {"composition": "NaCl", "scorer_name": "oracle"}}


def risks(**overrides):
    return {**dict.fromkeys(HEADS, .1), **overrides}


def candidate(action=SELECT, overall=.2, **heads):
    return {"action": copy.deepcopy(action), "risk_types": risks(**heads), "overall_risk": overall, "valid": True}


class FailureControllerTests(unittest.TestCase):
    def test_seven_head_order_and_horizons(self):
        self.assertEqual(HEADS, ("generation_invalid", "candidate_generation_failure", "tool_execution_failure", "screening_unavailable", "scientific_evaluation_failure", "unstable", "not_new"))
        self.assertEqual(HORIZONS["generation_invalid"], "current_proposal_JSON_or_action_schema")
        self.assertEqual(HORIZONS["scientific_evaluation_failure"], "next_scientific_evaluation")

    def test_tool_allowlist_exactly_matches_policy_contract_without_importing_rollouts(self):
        tree = ast.parse((Path(__file__).parents[1] / "src/matdiscovery/rollouts.py").read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "TOOL_ARGUMENTS" for target in node.targets))
        actual = {}
        for key, value in zip(assignment.value.keys, assignment.value.values):
            actual[ast.literal_eval(key)] = set(ast.literal_eval(value)) if isinstance(value, ast.Set) else set()
        self.assertEqual(TOOL_ARGUMENTS, actual)

    def test_json_invalid_applies_to_all_proposals_not_just_material_generation(self):
        action = {"tool": "get_buffer_stats", "arguments": {}}
        plan = plan_failure_response("made", action, risks(generation_invalid=.95), .1, OBS)
        self.assertIn("generation_invalid", plan["applicable_heads"])
        self.assertNotIn("candidate_generation_failure", plan["applicable_heads"])
        self.assertEqual(plan["trigger"], "generation_invalid")
        self.assertIn("JSON/tool/arguments", plan["feedback_for_retry"])
        self.assertNotIn("stability", plan["feedback_for_retry"])

    def test_candidate_generation_failure_switches_only_real_generators(self):
        plan = plan_failure_response("made", GENERATE, risks(candidate_generation_failure=.95), .2, OBS)
        self.assertEqual(plan["trigger"], "candidate_generation_failure")
        preference = plan["optional_tool_preference"]
        self.assertEqual(preference["tool"], "generate_structures")
        self.assertEqual(preference["arguments"]["generator_name"], "random")
        self.assertEqual(preference["arguments"]["num_candidates"], 32)
        self.assertNotIn("candidate_generation_failure", applicable_heads(SELECT))
        self.assertNotIn("candidate_generation_failure", applicable_heads(SCORE))

    def test_schema_error_repairs_current_proposal_without_needing_future_labels(self):
        plan = plan_failure_response("made", {"tool": "evaluate_with_new_oracle", "arguments": {}}, {}, None, OBS)
        self.assertEqual(plan["trigger_source"], "local_schema_validation")
        self.assertEqual(plan["trigger"], "generation_invalid")
        self.assertTrue(plan["local_schema_issues"])
        self.assertEqual(plan["available_heads"], [])

    def test_unavailable_surrogate_never_requests_new_oracle_or_uses_nonfinite_ranking(self):
        observation = copy.deepcopy(OBS)
        observation["buffer"]["NaCl"][0]["scores"]["oracle"] = {"nonfinite": "-inf"}
        plan = plan_failure_response("made", SCORE, risks(screening_unavailable=.9), .1, observation)
        preference = plan["optional_tool_preference"]
        self.assertEqual(preference["tool"], "query_structures")
        self.assertEqual(preference["arguments"]["mode"], "all")
        self.assertNotIn("scorer_name", preference["arguments"])
        unstable = plan_failure_response("made", SELECT, risks(unstable=.95), .1, observation)
        self.assertNotIn("scorer_name", unstable["optional_tool_preference"]["arguments"])

    def test_scientific_failure_chooses_a_real_alternate_hash_without_changing_orb(self):
        plan = plan_failure_response("made", SELECT, risks(scientific_evaluation_failure=.95), .1, OBS)
        self.assertEqual(plan["trigger"], "scientific_evaluation_failure")
        self.assertEqual(plan["optional_tool_preference"], {"tool": "select_for_evaluation", "arguments": {"composition": "NaCl", "structure_hash": "known-b"}})
        self.assertIn("fixed ORB", plan["feedback_for_retry"])
        self.assertEqual(plan["extra_physical_budget"], 0)

    def test_novelty_and_instability_produce_distinct_actionable_feedback(self):
        novelty = plan_failure_response("made", SELECT, risks(not_new=.9), .1, OBS)
        instability = plan_failure_response("made", SELECT, risks(unstable=.9), .1, OBS)
        self.assertNotEqual(novelty["feedback_for_retry"], instability["feedback_for_retry"])
        self.assertEqual(novelty["optional_tool_preference"]["arguments"]["structure_hash"], "known-b")
        self.assertEqual(instability["optional_tool_preference"]["arguments"]["mode"], "top")
        self.assertEqual(instability["optional_tool_preference"]["arguments"]["scorer_name"], "oracle")

    def test_type_signal_changes_selection_despite_better_scalar_alternative(self):
        first = candidate(overall=.01, unstable=.95)
        second = candidate(overall=.5, unstable=.2)
        result = rank_failure_candidates("made", [first, second], OBS)
        self.assertEqual(result["selected_index"], 1)
        self.assertTrue(result["type_signals_used"])
        first["risk_types"]["unstable"] = .1
        self.assertEqual(rank_failure_candidates("made", [first, second], OBS)["selected_index"], 0)

    def test_candidate_generation_signal_really_changes_generation_ranking(self):
        first = candidate(action=GENERATE, overall=.01, candidate_generation_failure=.9)
        second = candidate(action=GENERATE, overall=.4, candidate_generation_failure=.2)
        self.assertEqual(rank_failure_candidates("made", [first, second], OBS)["selected_index"], 1)

    def test_none_is_not_zero_and_missing_head_coverage_is_logged(self):
        first = candidate(overall=.001, scientific_evaluation_failure=None)
        second = candidate(overall=.5, scientific_evaluation_failure=.4)
        result = rank_failure_candidates("made", [first, second], OBS)
        self.assertEqual(result["selected_index"], 0)
        row = next(row for row in result["ranking"] if row["candidate_index"] == 0)
        self.assertIsNone(row["plan"]["risk_types"]["scientific_evaluation_failure"])
        self.assertNotIn("scientific_evaluation_failure", result["common_comparison_heads"])
        self.assertNotIn("scientific_evaluation_failure", row["compared_head_probabilities"])
        self.assertLess(row["head_coverage"]["fraction"], 1)
        self.assertTrue(row["missing_probabilities_are_not_imputed"])

    def test_no_types_uses_explicit_scalar_fallback_and_all_unknown_is_stable(self):
        candidates = [{"action": SELECT, "risk_types": dict.fromkeys(HEADS), "overall_risk": probability} for probability in (.6, .2)]
        result = rank_failure_candidates("made", candidates, OBS)
        self.assertEqual(result["selected_index"], 1)
        self.assertEqual(result["fallback"], "scalar_only_no_common_available_types")
        for row in candidates: row["overall_risk"] = None
        result = rank_failure_candidates("made", candidates, OBS)
        self.assertEqual(result["selected_index"], 0)
        self.assertEqual(result["ranking"][0]["plan"]["trigger"], "risk_unavailable")
        self.assertIsNone(result["ranking"][0]["plan"]["overall_risk"])

    def test_invalid_probabilities_remain_missing_and_json_is_finite(self):
        plan = plan_failure_response("made", SELECT, risks(unstable=float("nan"), not_new=True, scientific_evaluation_failure=1.1), float("inf"), OBS)
        self.assertIsNone(plan["risk_types"]["unstable"])
        self.assertIsNone(plan["risk_types"]["not_new"])
        self.assertEqual(plan["unavailable_heads"]["scientific_evaluation_failure"], "invalid_probability")
        json.dumps(plan, allow_nan=False)

    def test_existing_retry_and_fixed_recovery_limits_cannot_be_overridden(self):
        observation = {**OBS, "tools_since_query": 10, "max_tools_per_query": 10}
        plan = plan_failure_response("made", SELECT, risks(unstable=.9), .9, observation)
        self.assertEqual(plan["controller_action"], "defer_to_fixed_recovery")
        self.assertFalse(plan["request_retry"])
        self.assertIsNone(rank_failure_candidates("made", [candidate()], observation)["selected_index"])
        with self.assertRaises(ValueError): rank_failure_candidates("made", [candidate()] * 3, OBS)
        self.assertIsNone(rank_failure_candidates("made", [dict(candidate(), valid=False)], OBS)["selected_index"])

    def test_feedback_budget_and_all_preferences_match_actual_tool_schema(self):
        for head in HEADS:
            action = GENERATE if head == "candidate_generation_failure" else SCORE if head == "screening_unavailable" else SELECT
            with self.subTest(head=head):
                plan = plan_failure_response("made", action, risks(**{head: .99}), .8, OBS)
                self.assertEqual(plan["trigger"], head)
                self.assertLessEqual(len(plan["feedback_for_retry"]), MAX_FEEDBACK_CHARS)
                self.assertEqual(plan["extra_generation_budget"], 0)
                self.assertEqual(plan["extra_physical_budget"], 0)
                self.assertFalse(plan["executes_tools"])
                self.assertFalse(plan["overrides_fixed_recovery"])
                preference = plan["optional_tool_preference"]
                self.assertIn(preference["tool"], TOOL_ARGUMENTS)
                self.assertLessEqual(set(preference["arguments"]), TOOL_ARGUMENTS[preference["tool"]])

    def test_future_labels_and_method_names_cannot_influence_control(self):
        clean = plan_failure_response("made", SELECT, risks(unstable=.9), .2, OBS)
        contaminated = copy.deepcopy(OBS)
        contaminated.update(label_future_failure=0, ground_truth={"is_stable": True}, is_newly_discovered=True)
        self.assertEqual(clean, plan_failure_response("made", SELECT, risks(unstable=.9), .2, contaminated))
        candidates = [candidate(overall=.1, unstable=.9), candidate(overall=.5, unstable=.2)]
        before = rank_failure_candidates("made", candidates, OBS)
        for item in candidates: item.update(method="arbitrary_ablation_arm", label_future_failure=0)
        self.assertEqual(before, rank_failure_candidates("made", candidates, OBS))

    def test_next_science_heads_apply_to_read_tools_and_no_missingness_preference(self):
        heads = dict.fromkeys(HEADS)
        heads.update(generation_invalid=.1, tool_execution_failure=.1)
        choices = [{"action": SELECT, "risk_types": heads, "overall_risk": .05},
                   {"action": {"tool": "get_buffer_stats", "arguments": {}}, "risk_types": heads, "overall_risk": .99}]
        result = rank_failure_candidates("made", choices, OBS)
        self.assertEqual(result["selected_index"], 0)
        self.assertEqual(result["common_comparison_heads"], ["generation_invalid", "tool_execution_failure"])
        self.assertIn("unstable", applicable_heads(choices[1]["action"]))
        self.assertIn("not_new", applicable_heads(choices[1]["action"]))
        self.assertIn("scientific_evaluation_failure", applicable_heads(choices[1]["action"]))

    def test_exclusive_generation_head_triggers_feedback_but_not_cross_action_ranking(self):
        first = candidate(action=GENERATE, overall=.1, candidate_generation_failure=.99)
        second = candidate(action=SELECT, overall=.2)
        result = rank_failure_candidates("made", [first, second], OBS)
        self.assertEqual(result["selected_index"], 0)
        self.assertNotIn("candidate_generation_failure", result["common_comparison_heads"])
        row = next(row for row in result["ranking"] if row["candidate_index"] == 0)
        self.assertEqual(row["plan"]["trigger"], "candidate_generation_failure")
        self.assertIn("candidate_generation_failure", row["excluded_from_cross_action_ranking"])

    def test_crystalgym_remains_explicitly_unsupported(self):
        plan = plan_failure_response("crystalgym", {"action": "Na"}, risks(unstable=.99), .9, {})
        self.assertFalse(plan["supported"])
        self.assertEqual(plan["controller_action"], "unsupported")
        self.assertEqual(plan["feedback_for_retry"], "")
        self.assertIsNone(plan["optional_tool_preference"])
        self.assertIsNone(rank_failure_candidates("crystalgym", [], {})["selected_index"])

    def test_inputs_are_not_mutated_and_ties_preserve_original_order(self):
        args = [candidate(), candidate()]
        original = copy.deepcopy((args, OBS))
        result = rank_failure_candidates("made", args, OBS)
        self.assertEqual(result["selected_index"], 0)
        self.assertEqual((args, OBS), original)


if __name__ == "__main__":
    unittest.main()
