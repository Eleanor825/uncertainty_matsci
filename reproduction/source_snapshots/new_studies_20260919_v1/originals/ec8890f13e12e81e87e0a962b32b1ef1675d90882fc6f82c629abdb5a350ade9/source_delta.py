"""Exact admitted changes to the original frozen rollout, default untouched."""
def expected_rollout_source(original):
    pairs = [
        ('risk_model=None, attributor=None):\n        self.project',
         'risk_model=None, attributor=None, support_aware=False):\n        if type(support_aware) is not bool: raise ValueError("Explicit support mode required")\n        self.support_aware = support_aware\n        self.project'),
        ('                if typed_control:\n                    visible =',
         '                if self.support_aware:\n                    if not typed_control or method not in {"graph_risk", "esopt_graph_risk"}: raise ValueError("Support contract only for registered MADE graph control")\n                    from .support_control import apply_support\n                    risk = apply_support(row, risk)\n                if typed_control:\n                    visible ='),
        ('                    type_plan = plan_failure_response(benchmark, generated.parsed_action,\n                        row["failure_type_probabilities"], row["predicted_failure_probability"],\n                        visible, threshold=self.settings.risk_threshold)',
         '                    if self.support_aware:\n                        from .support_control import plan_with_support\n                        type_plan = plan_with_support(benchmark, generated.parsed_action,\n                            row["failure_type_probabilities"], row["predicted_failure_probability"],\n                            visible, threshold=self.settings.risk_threshold, support=row["uncertainty_support"])\n                    else:\n                        type_plan = plan_failure_response(benchmark, generated.parsed_action,\n                            row["failure_type_probabilities"], row["predicted_failure_probability"],\n                            visible, threshold=self.settings.risk_threshold)'),
        ('            selection = rank_failure_candidates(benchmark, [',
         '            ranker = rank_failure_candidates\n            support_args = {}\n            if self.support_aware:\n                from .support_control import rank_with_support\n                ranker = rank_with_support\n                support_args = {"supports": [c[1]["uncertainty_support"] for c in candidates]}\n            selection = ranker(benchmark, ['),
        ('                for candidate in candidates], visible, threshold=self.settings.risk_threshold)',
         '                for candidate in candidates], visible, threshold=self.settings.risk_threshold, **support_args)')]
    for before, after in pairs:
        if original.count(before) != 1:
            raise ValueError('Frozen rollout patch anchor changed: ' + before[:80])
        original = original.replace(before, after)
    return original
