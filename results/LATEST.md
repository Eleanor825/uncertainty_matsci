# Latest verified experimental results

MADE remains **1079 valid trajectories, 1 technical failure and 0 pending** (September22 10:18 CST). DiscoveryWorld adds **completed CPU diagnostics through September22 17:22 CST**: four feature-family fits, a separate training/classifier diagnostic, and fixed0.5 cached-prediction analysis. These do not establish online improvement or a completed Full-method comparison.

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

## DiscoveryWorld — Qwen3.5-4B, Proteomics Normal

The completed CPU diagnostic used **60 fit /60 calibration /40 development rows**, with development clustered in **2 episodes from1 world**. Four fixed200-update fits completed in46.02s. The original Internal and OutputAction checkpoint/metric controls reproduce exactly; the production predictor is unchanged.

| Features | Raw fields | AUROC | Brier | NLL |
|---|---:|---:|---:|---:|
| OutputAction | 10 | 0.9600 | 0.08107 | 0.25350 |
| NoGraph | 140 | 0.9600 | 0.06871 | 0.23688 |
| GraphOnly | 153 | 0.9133 | 0.08545 | 0.42457 |
| Internal | 293 | 0.9700 | 0.06444 | 0.19994 |

GraphOnly is weaker than OutputAction on pooled metrics. Internal minus NoGraph is +0.01 AUROC and −0.00428 Brier; model capacities differ. These small, clustered development results do not establish significance or a causal contribution to scientific planning.

A separate completed53.88s diagnostic reports4 NN fits plus4 classical fits and16 NN states at0/50/100/200 updates. Internal's raw fit NLL falls0.00920→0.000144 from50→200 updates, while raw development NLL rises0.16915→0.33678 and calibrated development Brier rises0.05535→0.06444; AUROC stays0.9700. This is consistent with late overfitting/overconfidence on this split, not proof that50 is a generalizable optimum or that graph signal is useless. Step0 already includes label-fitted error-prototype preprocessing. The production200-step NN remains unchanged. Full-feature GBC/L2-logistic development AUROCs0.9417/0.9633 do not exceed NN0.9700; these are classifier probes, not a full CRV reproduction.

On the **same cached probabilities**, fixed0.5 flags22/26 calibration failures,10/10 development failures and6/7 G0 failures, versus15/26,1/10 and0/7 under the original0.9867 threshold. Fixed0.5 also adds2 calibration and3 development false-positive flags. These are offline flagging counts, not prevented failures or observed benefits from revising actions. No threshold search or predictor change was made for this calculation.

The prior **160/160 V4 graphs** and frozen-risk completion remain unchanged. V4's accepted G0 world2/policy303/B10 had score0 and7 failures, and the activity gate stopped it before any ES parameter update. No completed new online operating-point pair, matched-proposal control or Full ES result is added here. Held-out Baseline1/Full0 and completed pairs0 remain the completed-result record; G0 cannot be compared with the different world/seed/B30 Baseline as an effect estimate.

- [Completed CPU feature/operating-point report](discoveryworld/diagnostics/20260922T172256_CST/report.md)
- [Training curves and classifier audit](discoveryworld/diagnostics/20260922T172256_CST/training_diagnostic_report.md)
- [All108 training metric rows](discoveryworld/diagnostics/20260922T172256_CST/training_metrics.csv)
- [Four-family metrics](discoveryworld/diagnostics/20260922T172256_CST/feature_metrics.csv)
- [Portable validator](discoveryworld/diagnostics/20260922T172256_CST/validate.py)
- [Prior V4 graph/risk/G0 record](discoveryworld/progress/20260922T155718_CST/report.md)

## Scope and provenance

Publication reuses completed records only and adds no model, simulator, graph, NN-fitting or ES calls. The feature diagnostic records four fits/800 updates; the separate training diagnostic records another four NN fits/800 updates plus four classical fits. Neither adds graphs or environment calls. These repeated fits reuse the same observations and must not be counted as independent validation trials. Current exploration concerns MADE and DiscoveryWorld; existing historical9B, original seed1 MADE and SnAr exports remain separate. Negative results, unavailable heads, gate failures and source hashes remain preserved. There is no completed Full-method gain or larger-benchmark completion claim.
