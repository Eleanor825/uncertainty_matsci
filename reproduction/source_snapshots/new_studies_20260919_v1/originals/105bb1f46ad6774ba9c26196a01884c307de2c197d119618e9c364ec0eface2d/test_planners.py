"""Pure CPU planner checks with synthetic history, no oracle evaluations."""
import unittest
import numpy as np
from baselines.planners import LOW, SPAN, official_design, product_utility, propose, propose_with_metadata


class PlannerTests(unittest.TestCase):
    def test_original_lhs_and_random_are_deterministic(self):
        for kind in ("lhs", "random"):
            self.assertEqual(official_design(kind, 7, 5), official_design(kind, 7, 5))
        points = official_design("lhs", 7, 5)
        x = np.asarray([list(p.values()) for p in points])
        cells = np.floor(5 * (x - LOW) / SPAN).astype(int)
        for col in cells.T:
            self.assertEqual(set(col), set(range(5)))

    def test_gp_ei_uses_predictive_uncertainty_and_all_history(self):
        design = official_design("lhs", 11, 8)
        history = [{"parameters": p, "objectives": {"sty": 100.0 + 200.0 * i, "e_factor": 900.0 - 50.0 * i}}
                   for i, p in enumerate(design)]
        result = propose_with_metadata(history, 9, 5)
        self.assertEqual(result["parameters"], propose(history, 9, 5))
        self.assertEqual(result["planner"]["phase"], "gp_expected_improvement")
        self.assertEqual(result["planner"]["history_count"], 8)
        self.assertGreater(result["planner"]["selected_predictive_sd"], 0)
        self.assertFalse(result["planner"]["official_summit_sobo"])
        self.assertEqual(product_utility({"sty": 26000, "e_factor": 0}), 2.0)


if __name__ == "__main__":
    unittest.main()
