# Closed DiscoveryWorld development audit, 2026-09-22 18:47 CST

No downstream improvement is established. This audit reads only the two completed, source-hash-verified development episodes; it adds zero model forwards, graph extractions, environment calls or optimizer updates. Q75 and Common2 use different policy seeds and are not a paired treatment comparison. All selected-action risks and all twenty primary labels were available; no counterfactual outcome was assigned to unexecuted proposals.

| Condition | Returned actions | Action failures | Task score | Second proposals | Different proposed actions | Actual selected-action changes vs first | Consecutive identical failed actions, max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q75, policy 304 | 10 | 10 | 0 | 3 | 0 | 0 | 9 |
| Common2, policy 305 | 10 | 1 | 0 | 10 | 6 | 0 | 1 |

Q75 triggered three real revision requests. All three second proposals parsed to exactly the first proposal's action. Two recorded neural rank changes therefore changed candidate identity but not the action dispatched to the environment. Steps 2–10 executed the same failing PICKUP action. This directly identifies a proposal-diversity/repair failure in this episode; it is not evidence that every possible alternative would fail.

At a retrospective fixed probability threshold of 0.5, the cached risks of Q75's executed actions flag 9 of its 10 failures (Brier 0.040686, NLL 0.145225). The actual Q75 controller used its original calibration quantile, not this diagnostic threshold. These ten actions all failed, so this episode cannot measure discrimination between successful and failing actions. It does show that many repeated failures had high predicted risk while effective action revision was absent.

Common2's nine successful actions all had zero immediate official task-score increase; its one failed USE action had risk 0.044265 and was missed at 0.5. Low action-failure risk is not a predictor of scientific progress in this audit. No utility or completion head was fitted in the current production risk model. Success without immediate score increase does not prove the action has no eventual value; the entire observed B10 episode nevertheless ended at score zero.

Both observations motivate testing the already registered grounded-feedback repair and explicitly measuring actual action changes and task progress. They do not establish that the new repair works, that graph features cause better decisions, or that the original model is fully trained. The separate training-curve audit found late overconfidence on its small development split. The two issues can coexist.

Evidence: [twenty selected-action records and source hashes](audit.json), [metric table](metrics.csv), [behavior counts](behavior.csv), and [portable validation](validate.py). The raw operational receipt is private and is represented only by its SHA256 and byte count in [provenance](provenance.json). The portable validator independently recomputes published metric and count arithmetic; it does not replace the original source-hash and raw-log join audit. No confidence interval is estimated from dependent steps within these two episodes. These are the same two trajectories in the 18:33 development snapshot, not additional evaluations.
