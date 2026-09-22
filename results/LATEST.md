# Latest verified experimental results

**Latest verified DiscoveryWorld status — September23,05:46 CST:** TRAIN36 is complete; the frozen support gate returned **0utility fits/0updates**. A separate CPU audit found2/4 positive-R8 roots below0.5, which is **not evidence of actual hard rejection**. P332 remains2/3closed: Native1 score0/29failures, Common2 score0.125/9failures, preselected Internal-combined12/30actions with11failures so far and no final score. NoGraph passed8qualification prefixes/3TRAIN checks and fully closed; p335 then loaded G0 and entered its own qualification with0environment actions/results. [Current report, all36 audit rows and exact statuses](discoveryworld/progress/20260923T054626_CST/report.md). Earlier results below remain unchanged.

MADE remains **1079 valid trajectories, 1 technical failure and 0 pending**. One original DiscoveryWorld Native–Full B30 test pair is now closed: Full scored **0.125 versus Native 0**, with **22 versus 29 failed actions**. Both task-success flags are false. This single world/seed result does not establish general improvement or isolate an NN/ES effect.

## MADE — Qwen3.5-4B

| Budget | Accepted / planned | Failed | Pending |
|---|---:|---:|---:|
| B10 | 360/360 | 0 | 0 |
| B30 | 360/360 | 0 | 0 |
| B50 | 359/360 | 1 | 0 |

The four arms are Native, uncertainty-only, ESOpt-only and Full. All three budgets contain 30 chemical systems and evaluation seeds 2, 3 and 4. The original seed1 study remains separate. B50 has Native 89/90, uncertainty-only 90/90, ESOpt-only 90/90 and Full 90/90 valid trajectories. Collection is closed with one Native technical failure retained; this is not a complete 1080-valid-result matrix, and the missing result is not imputed as zero.

| Full minus Native | Matched pairs | Mean SUN difference | Mean AUDC difference |
|---|---:|---:|---:|
| B10 | 90 | -0.433333 | -0.034556 |
| B30 | 90 | -0.600000 | -0.013901 |
| B50 | 89 | -1.573034 | -0.028054 |

The completed matched results do **not** show an overall Full-method improvement: Full minus Native is negative for both reported metrics at every budget. All positive, tied and negative cases remain in the export. B50 comparisons have 89 matched pairs because one Native trajectory failed; B10 and B30 have 90. The two ES policies were trained separately, so these comparisons do not isolate a causal NN contribution.

- [Completed-result report](made_components/incremental/20260922T101850_CST/report.md)
- [All systems / budgets / arms / seeds](made_components/incremental/20260922T101850_CST/all_system_results.csv)
- [Per-system seed mean, sample variance and SD](made_components/incremental/20260922T101850_CST/seed_statistics.csv)
- [Matched pair differences](made_components/incremental/20260922T101850_CST/paired_results.csv)
- [Discovery curves](made_components/incremental/20260922T101850_CST/completed_curves.csv)

Variance is across evaluation seeds within the same system, budget and arm (sample variance, ddof=1). Missing/failed runs stay blank; n<2 variance is unavailable and incomplete seed sets are marked. These are not repeated training runs.

A new [historical MADE cause audit and incumbent-selection replay](development/made_underperformance_20260923T012020_CST/report.md) preserves these 1079 results. On one previously used development seed, official AUDC is G0 **0.68**, G1 **0.62**, G2 **0.64**: the original G1/G2-only rule selects G2, while the implemented candidate rule retaining G0 selects G0. Original scalar-risk development AUROC is **0.516260**; its calibration compression and inactive threshold are distinct from causal evidence. This adds **no held-out experiment, model/environment call or production replacement**, and does not establish superiority to Native.

## DiscoveryWorld — Qwen3.5-4B, Proteomics Normal

New original test comparison, audited **2026-09-23T03:28:28.657255+08:00**: **world 3 / policy seed 401 / B30**.

| Method | Task score | Failed actions /30 | Task success |
|---|---:|---:|---|
| Native (earlier unchanged Baseline run) | 0 | 29 | No |
| Full, dev-selected G1 | 0.125 | 22 | No |

The two real ES updates and the selected checkpoint's actual test reload are verified. Original raw action events, official summaries, model/decoding and environment/evaluator contracts were checked. B30 matches environment actions, while Full uses additional candidate/graph computation. There is one paired episode per method; this is not a multi-seed estimate, task completion or component-level causal evidence. [Complete pair, selection trace, source hashes and validator](discoveryworld/test/20260923T032828_CST/report.md).

A subsequent p401 mechanism audit records risk-triggered revisions at steps 22/23: proposed PICKUP became an actually executed successful TELEPORT_TO_OBJECT. Step 24 then picked up the meter for 0.125, but both candidates were identical PICKUP packets and the first risk was 0.878 despite success. The two teleports address the same object (string/integer UUID differences are normalized); step 23 is a repetition. This is trajectory evidence preceding progress, not proof either revision was necessary or that an unexecuted pickup would fail. All four steps 22–25 remain in the audit. [Small-evidence mechanism report](discoveryworld/test_mechanisms/20260923T044323_CST/report.md).

Latest closed cohort: **world 2 / policy seed 307, four conditions × B30**.

| Condition | Task score | Failed actions /30 | Actual NN packet changes |
|---|---:|---:|---:|
| Common2 | 0 | 7 | 0 |
| Internal-base84 | 0.125 | 5 | 10 |
| Internal-combined (preselected) | 0.125 | 21 | 3 |
| OutputAction-combined | 0 | 26 | 3 |

Both Internal trajectories earned their sole 0.125 milestone by picking up the proteomics meter. At base84's scoring step the two parsed candidate actions were identical; combined's scoring step used common-hash fallback because the alternative graph was unavailable. These steps do not prove that NN rejection of a different action caused progress. All four failed to complete the task; base84 does not replace the preselected combined variant. This is one world/seed cohort, not independent replications or a Native/Full test.

Three shadow heads scored the same 113 executed actions: Internal-base84 / Internal-combined / OutputAction AUROC 0.921366 / 0.918860 / 0.879073 and Brier 0.124354 / 0.127342 / 0.178023. These correlated action-level metrics have no causal or significance claim and assign no outcome to unexecuted alternatives.

Separate fixed-TRAIN diagnostics: the checkpointed/uncheckpointed SFT gradient comparison matched, but strict teacher-forcing/cache numerical parity failed; no admission was granted. The action-influence selector passed 8 qualification prefixes and produced 3 wider-layer graphs, with 11 extra selector backwards recorded, 0 NN fits and 0 environment calls. Neither diagnostic is an online performance result.

[Full report, all 120 action rows, source hashes and independent validator](discoveryworld/development/20260923T013432_CST/report.md).

Latest completed policy-only comparison: **9/9 world2 B30 episodes**, three sampling seeds. Mean official task scores: Baseline 0.000000 / SFT 0.000000 / CreditSFT 0.000000. Error rates and behavior remain separate; this is not Internal/Full efficacy evidence. [Complete fixed9 report](discoveryworld/development/20260923T002200_CST/report.md).

Earlier p308 policy-only B30 pair: **world2/policy308, both task scores0**. Native has29/30 action failures; fixed final32 SFT has0/30, but performs28 rotations and2 moves without score progress. This pair contains0 NN-controller calls and0 candidate graphs. It does not establish Full-method efficacy. [All60 action rows and separate training-window audit](discoveryworld/development/20260922T231329_CST/report.md).

Earlier verified snapshot: **September 22, 21:16 CST**. Grounded repair is complete at world 2 / policy 306 / B10: GenericRisk and GroundedRisk each have 7 failures; GroundedCommon2 has 0 and GroundedInternal2 has 5. All task scores are zero. These mechanism comparisons show no task improvement and are not Native-versus-Full tests.

The original Full ES completed one genuine update to 723 parameter tensors. Its G1 development rollout reduced failures from 10 to 9, with task score remaining zero; G2 has started. At that earlier snapshot, the held-out Full pair remained incomplete. The 24-graph supplement and 2,525-update CV2 diagnostic are complete. Three missed successful milestone graphs are also complete. The earlier v1 refresh failed before fitting and remains recorded. The independent corrected v2 refresh now completed eight fits and 500 updates in 67.124 seconds, with two exact base84 reproductions; see the separate 21:29 update below.

- [Latest grounded-repair, CV2 and ES-update report](discoveryworld/development/20260922T211645_CST/report.md)
- [Latest terminal per-arm metrics](discoveryworld/development/20260922T211645_CST/V6_metrics.csv)
- [Earlier matched-proposal pair: 1 versus 2 failures, both score zero](discoveryworld/development/20260922T192407_CST/report.md)
- [Earlier threshold pair: both 10 failures, no actual action changes](discoveryworld/development/20260922T185915_CST/report.md)
- [Original feature/training diagnostics](discoveryworld/diagnostics/20260922T172256_CST/report.md)

Latest CPU refresh snapshot: **September 22, 21:29:53 CST**. The fixed data variants have **84/96/87/99** fitting rows. Combined Internal Brier worsens from **0.068870 to 0.084463**; combined OutputAction improves from **0.110689 to 0.082908**. All eight models are reported without selecting a development winner. The same 40 reused development rows contain 10 failures; no new online result or LLM parameter update is part of this refresh.

- [Eight-fit report and all variants](discoveryworld/development/20260922T212953_CST/report.md)
- [Exact eight-model metrics](discoveryworld/development/20260922T212953_CST/metrics.csv)

The single original Full/Native pair above is now complete; robust online Full-method improvement has not been established. Actions within a trajectory are dependent; diagnostic fits reusing these actions are not independent validation trials. The closed p307 B30 cohort above adds development evidence, not a held-out Full/Native test pair.

## Scope and provenance

Publication reuses completed records only and adds no model, simulator, graph, NN-fitting or ES calls. The feature diagnostic records four fits/800 updates; the separate training diagnostic records another four NN fits/800 updates plus four classical fits. Neither adds graphs or environment calls. These repeated fits reuse the same observations and must not be counted as independent validation trials. Current exploration concerns MADE and DiscoveryWorld; existing historical9B, original seed1 MADE and SnAr exports remain separate. Negative results, unavailable heads, gate failures and source hashes remain preserved. The single-pair positive task-score difference is reported above; there is no general Full-method improvement or larger-benchmark completion claim.
