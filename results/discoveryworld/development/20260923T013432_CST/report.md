# Four fixed B30 development conditions and separate TRAIN diagnostics

**All four world 2 / policy seed 307 B30 trajectories are closed and accepted. Both Internal conditions reached task score 0.125; Common2 and OutputAction remained at 0. All four task-success flags are false.** This is one matched-world/seed development cohort, not four independent replications or a Native-versus-Full held-out test. Native1 is absent. The preselected combined condition remains the primary Internal condition; base84 is not promoted after seeing results.

| Condition | Score at B10 | Score at B30 | Failed actions /30 | NN index changes | NN changes to actual action packet | Longest identical failed-action run |
|---|---:|---:|---:|---:|---:|---:|
| Common2 fixed hash | 0 | 0 | 7 | 0 | 0 | 2 |
| Internal-base84 | 0.125 | 0.125 | 5 | 14 | 10 | 2 |
| Internal-combined | 0 | 0.125 | 21 | 13 | 3 | 15 |
| OutputAction-combined | 0 | 0 | 26 | 16 | 3 | 24 |

The B30 endpoint was fixed in advance. B10 is a postepisode prefix report, not an adaptive continuation decision. Each step requested exactly two candidates; no extra neural revision was triggered. Of 240 candidate graphs, 229 passed the unchanged graph support gate and 11 were unavailable; those failures remain recorded. Risk comparisons with a missing candidate graph fall back to the fixed common-hash selection. [All four results](metrics.csv), [all 120 executed steps and both proposed packets](actions.csv), and [source-bound data](cohort.json) preserve the full cohort.

## What actually produced the 0.125 milestones

Public object 33276 is the **proteomics meter**. It moved from table 42983 into agent 63296 at Internal-base84 attempt 3 and Internal-combined attempt 25. Both changes coincide with the only positive score transition in each trajectory, 0→0.125, matching the official “take meter” one-of-eight score component. Neither trajectory completed the scientific task or gained further score by B30. [Observed object identities and locations](public_object_identity.json); [official scoring-source reference](milestone_source.json).

The executed paths do **not** support saying that the NN rejected a bad action at the scoring step:

- Internal-base84 first diverged from Common2 at attempt 1: it selected `PICKUP(33276)` instead of `TELEPORT_TO_OBJECT(45538)`, and that pickup actually failed. At attempt 2 both packets were the same teleport to 42983; at attempt 3 both packets were the same pickup 33276. The model changed indices at attempts 2/3, but did not change the action packet. The successful pickup at 3 therefore has no distinct alternative-action rejection to credit. Its two base84 risks were0.439534/0.099639 despite identical parsed packets.
- Internal-combined first diverged from Common2 at attempt 2, choosing feed updates instead of reading 45538. It later had a long failed-pickup sequence. At attempt 24 both packets were the same teleport to 33276. At25 it selected pickup 33276 by **common-hash fallback** because the alternative USE packet lacked a supported graph. The unexecuted USE outcome is unknown.
- Common2 and OutputAction never obtained the meter in these trajectories. OutputAction made only 3 actual packet changes despite 16 index changes, and its longest identical failed-action run was 24. That is an observed behavior limitation, not evidence that any unexecuted alternative would have succeeded.

All four initial public-observation hashes are identical. This proves equality of the recorded public observation only, not complete private simulator/RNG equivalence. After actions diverge, subsequent candidates and states diverge. These observations identify real task progress and different behavior, but do not identify a local causal benefit of one risk score or establish general superiority.

## Three shadow models on the same executed actions

The three frozen heads can all score 113 of the 120 actually executed actions (57 failures). On exactly these same rows, the independently recomputed descriptive results are:

| Frozen shadow head | AUROC | Brier |
|---|---:|---:|
| base84/Internal | 0.921366 | 0.124354 |
| combined/Internal | 0.918860 | 0.127342 |
| combined/OutputAction | 0.879073 | 0.178023 |

These are 113 correlated actions from 4 dependent adaptive trajectories, 1 world and 1 policy seed. There is no independence-based CI, significance claim, new training, threshold selection or unexecuted-action label. At 58 same-parsed-packet candidate pairs, mean absolute risk gaps are 0.088900/0.057455/0.072439 respectively. Different generation prefixes can yield different hidden/graph/sampling features even with identical parsed packets; the gaps describe sensitivity, not proof the probabilities are wrong. [Matched executed rows](selected_shadow_rows.csv), [all shadow diagnostics](shadow_diagnostic.json).

## Separate fixed-TRAIN numerical diagnostics

These diagnostics did not produce the B30 actions above and are not efficacy tests.

**SFT numerical path v2:** one fixed original TRAIN row, 13 response tokens. Checkpointed versus uncheckpointed teacher forcing matched raw logits, CE and all 426 text-parameter gradients exactly; weights remained unchanged. Teacher forcing versus forced cached decoding retained all 13 argmax tokens but failed the original strict numerical tolerance: max raw-logit difference 9.9182e−5 (532,689/3,228,160 components outside tolerance); max target-logprob difference 1.3113e−5 (2/13 outside). The diagnostic is complete and resource-closed, but `qualified=false` and `no_main_admission=true` remain. It performed 15 top-level forwards, 2 backwards, 0 optimizer updates, 0 generations and 0 environment calls; the prior failed attempt’s 1 forward intent / 0 returns remains counted separately. One short row cannot validate every SFT/cache path or explain online behavior. [Exact numerical outcomes](numerical_path_diagnostic.json).

**Action-influence node selection:** on 3 predeclared TRAIN prefixes (p101/a2,p102/a47,p103/a100), the new 32-feature selector passed all 8 original qualification prefixes, including 32-layer fidelity and the sampled native FP32 finite-difference checks. It selected 19/19/18 layers instead of 2/2/2; after masks, 21/18/20 features across 14/10/11 layers remained instead of 3/4/2 features from 1 layer. The retained-feature share of direct absolute action-score attribution rose from 1.31%/0.91%/0.79% to 4.97%/4.44%/5.16%. These are fractions within each constructed local graph, **not causal coverage of the model**. The selector jointly changes influence ranking, action-prediction positions and per-layer cap 2, so this does not isolate any one change. Bank/constants remained on CPU; this was not combined with GPU-residency changes. All 11 additional selector backwards (8 qualification + 3 graph) are counted, 85.329 seconds in total; each new main graph used 34 backwards including selection, within 42. Old graphs were reused, with 0 NN calls, 0 new policy generations and 0 environment calls. No production NN or online result automatically adopts these new graph features. [Six old/new graph records and hashes](action_influence_graphs.csv), [full diagnostic projection](action_influence_diagnostic.json).

The graph construction remains our 32-feature budget-limited adaptation, not an exact reproduction of original CRV’s much larger circuit extraction. Historical old/new graph latencies are retained as provenance, not a controlled speedup claim. The small TRAIN diagnostics motivate further registered checks; they do not turn the four development trajectories into a robust efficacy result.

MADE remains 1079 valid results plus 1 retained technical failure; its original results are unchanged. No model, environment, graph extraction or fitting call was added by this export. Run [validate.py](validate.py) to recompute cohort counts, step/progress joins, same-row shadow statistics and all file hashes.
