# Observed Full action path before the p401 meter pickup

This is a **postepisode mechanism audit** of the already published world 3 / policy seed 401 / B30 Native–Full pair, observed at **2026-09-23 04:43:23 CST**. The original endpoints remain Native score 0 / 29 failed actions and Full G1 score 0.125 / 22 failed actions; both task-success flags are false. No experiment or outcome is added by this audit. [Original result](../../test/20260923T032828_CST/report.md).

The sole Full score increase occurs at **step 24**, when `PICKUP(33276)` takes the proteomics meter from table 42983 into agent 63296. The preceding two steps show actual changes following risk-triggered revision:

| Step | First proposal | First failure risk | Actual action | Revised packet differs from first? | Returned success | Score increase |
|---:|---|---:|---|---|---|---:|
| 22 | PICKUP(33276) | 0.880321 | TELEPORT_TO_OBJECT(33276), proposal 1 | Yes | Yes | 0 |
| 23 | PICKUP(33276) | 0.928685 | TELEPORT_TO_OBJECT(33276), proposal 1 | Yes | Yes | 0 |
| 24 | PICKUP(33276) | 0.878076 | PICKUP(33276); both proposals identical | No | Yes | 0.125 |
| 25 | TELEPORT_TO_LOCATION("Instrument Table") | 0.023726 | Same single proposal | No revision requested | Yes | 0 |

All four selected steps, all seven generated candidates, source hashes, official outcomes and available public object observations are retained in [steps.csv](steps.csv) and [mechanism.json](mechanism.json). The window was selected after observing the score transition: two preceding steps and one following step. It is not a representative sample for estimating error or calibration rates.

At steps 22 and 23 the recorded first-candidate risk triggered a neural revision. The only generated packet matching the executed teleport is proposal 1, so the changed action is identified from the candidate and official action records without reading the large policy-attempt log. This is evidence that risk revision changed the executed path before the meter pickup. **It does not prove either teleport was necessary or that the unexecuted pickup would have failed.** Step 23 repeats the same object-directed teleport; its public object distance is 1 both before and after.

Step 22 encodes object UUID as the string `"33276"`; step 23 uses integer `33276`. The frozen official `actionTeleportAgentToObject` converts its UUID with `int(...)`. Their semantic actions are therefore the same. Raw packets and raw hashes are preserved, alongside a comparison-only canonical form; this type difference is not counted as action novelty. [Official source/version and evidence provenance](provenance.json).

At the scoring step 24, both candidates have exactly the same PICKUP packet. The small evidence cannot identify which identical candidate index was selected, and this step supplies no distinct alternative-action rejection to credit to the NN. The first-candidate alarm exceeds the registered 0.5 threshold even though that same packet subsequently succeeds: a **false-positive alarm at the observed action-packet level**. One such outcome does not establish miscalibration of its probability, and no separate outcome is assigned to an unexecuted candidate generation.

Native and Full have the same recorded initial public-observation hash; this does not establish full private-state/RNG equality. After their paths diverge, the later observations are not matched counterfactual states. This audit adds trajectory evidence for a plausible revision-to-progress pathway, while retaining repetition, the successful-action alarm, both unsuccessful task endpoints and the absence of a causal necessity estimate. Weight changes, the controller and extra candidate/graph computation remain combined in this pair.

The read-only probe consumed 1,639,870 bytes of bounded evidence. It read neither the 20.7-GB checkpoint nor the 108-MB policy-attempt log and made zero model, graph, environment or fitting calls. Empty target-object extracts at string UUIDs or named locations reflect the probe's integer-ID extraction scope, not proof that an object was invisible. [Validator](validate.py) checks the complete four-step window, packet normalization, original source hashes and publication scope.
