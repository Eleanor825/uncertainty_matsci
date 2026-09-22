# Latest verified experimental results

MADE remains **1079 valid trajectories, 1 technical failure and 0 pending**. The latest DiscoveryWorld grounded-repair comparisons and first actual Full-ES update are reported below. Scientific-task improvement is not established.

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

Latest completed policy-only comparison: **9/9 world2 B30 episodes**, three sampling seeds. Mean official task scores: Baseline 0.000000 / SFT 0.000000 / CreditSFT 0.000000. Error rates and behavior remain separate; this is not Internal/Full efficacy evidence. [Complete fixed9 report](discoveryworld/development/20260923T002200_CST/report.md).

Earlier p308 policy-only B30 pair: **world2/policy308, both task scores0**. Native has29/30 action failures; fixed final32 SFT has0/30, but performs28 rotations and2 moves without score progress. This pair contains0 NN-controller calls and0 candidate graphs. It does not establish Full-method efficacy. [All60 action rows and separate training-window audit](discoveryworld/development/20260922T231329_CST/report.md).

Latest verified snapshot: **September 22, 21:16 CST**. Grounded repair is complete at world 2 / policy 306 / B10: GenericRisk and GroundedRisk each have 7 failures; GroundedCommon2 has 0 and GroundedInternal2 has 5. All task scores are zero. These mechanism comparisons show no task improvement and are not Native-versus-Full tests.

The original Full ES completed one genuine update to 723 parameter tensors. Its G1 development rollout reduced failures from 10 to 9, with task score remaining zero; G2 has started. The held-out Full pair remains incomplete. The 24-graph supplement and 2,525-update CV2 diagnostic are complete. Three missed successful milestone graphs are also complete. The earlier v1 refresh failed before fitting and remains recorded. The independent corrected v2 refresh now completed eight fits and 500 updates in 67.124 seconds, with two exact base84 reproductions; see the separate 21:29 update below.

- [Latest grounded-repair, CV2 and ES-update report](discoveryworld/development/20260922T211645_CST/report.md)
- [Latest terminal per-arm metrics](discoveryworld/development/20260922T211645_CST/V6_metrics.csv)
- [Earlier matched-proposal pair: 1 versus 2 failures, both score zero](discoveryworld/development/20260922T192407_CST/report.md)
- [Earlier threshold pair: both 10 failures, no actual action changes](discoveryworld/development/20260922T185915_CST/report.md)
- [Original feature/training diagnostics](discoveryworld/diagnostics/20260922T172256_CST/report.md)

Latest CPU refresh snapshot: **September 22, 21:29:53 CST**. The fixed data variants have **84/96/87/99** fitting rows. Combined Internal Brier worsens from **0.068870 to 0.084463**; combined OutputAction improves from **0.110689 to 0.082908**. All eight models are reported without selecting a development winner. The same 40 reused development rows contain 10 failures; no new online result or LLM parameter update is part of this refresh.

- [Eight-fit report and all variants](discoveryworld/development/20260922T212953_CST/report.md)
- [Exact eight-model metrics](discoveryworld/development/20260922T212953_CST/metrics.csv)

No robust online Full-method improvement or completed held-out Full/Native pair has been established. Actions within a trajectory are dependent; diagnostic fits reusing these actions are not independent validation trials. New fixed B30 training-refresh comparisons are prospective.

## Scope and provenance

Publication reuses completed records only and adds no model, simulator, graph, NN-fitting or ES calls. The feature diagnostic records four fits/800 updates; the separate training diagnostic records another four NN fits/800 updates plus four classical fits. Neither adds graphs or environment calls. These repeated fits reuse the same observations and must not be counted as independent validation trials. Current exploration concerns MADE and DiscoveryWorld; existing historical9B, original seed1 MADE and SnAr exports remain separate. Negative results, unavailable heads, gate failures and source hashes remain preserved. There is no completed Full-method gain or larger-benchmark completion claim.
