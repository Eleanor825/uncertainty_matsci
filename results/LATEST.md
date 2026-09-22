# Latest verified experimental results

Observations: **MADE 2026-09-22 10:18 CST; DiscoveryWorld 2026-09-22 15:57 CST**. MADE remains at **1079 valid trajectories, 1 technical failure and 0 pending**. DiscoveryWorld has **160/160 V4 graphs**, a development-admitted next-action-failure risk head and **one accepted G0 development trajectory**. Full stopped before its first ES parameter perturbation; held-out tests remain **Baseline 1, Full 0, completed pairs 0**.

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

## DiscoveryWorld — Qwen3.5-4B

All **160 selected graphs** from eight preparatory episodes are complete. Each of four groups passed the same **8/8 fixed-prefix suite under the declared V4 local-Jacobian mathematical reference**. Original native FP32 finite-difference results and failure flags remain preserved; this does not mean every original native FP32 check passed. The full 32-feature / 42-backward-target graph procedure is unchanged.

The **next-action-failure** head passed its development admission on **40 rows from 2 episodes in 1 world**:

| Features | AUROC | Brier score | NLL |
|---|---:|---:|---:|
| Internal | 0.970000 | 0.064438 | 0.199944 |
| OutputAction | 0.960000 | 0.081075 | 0.253498 |

These are development statistics with clustered rows, not a held-out task-performance comparison or evidence of online improvement. Both feature-family NNs were fitted; the generation-invalid and terminal-noncompletion heads were not fitted because support was insufficient.

The **Full G0 development** trajectory (world 2, policy seed 303, B10) was accepted with **normalized score 0 and 7/10 failed attempts**. Its maximum supported risk, **0.916376948**, stayed below the frozen threshold **0.986700532**. All ten attempts used one candidate, with **0 neural revisions and 0 rank changes**. The registered controller-activity gate failed, stopping the V4 branch before the first ES parameter perturbation. There is no completed parameter-updating Full search or Full held-out test.

The existing **Baseline held-out** trajectory (world 3, policy seed 401, B30) remains accepted with normalized score 0 and 29/30 failed attempts. It cannot be paired with G0 because their worlds, policy seeds, budgets and development/test roles differ. A completed Baseline–Full effect estimate is still unavailable.

- [DiscoveryWorld scientific status through 15:57](discoveryworld/progress/20260922T155718_CST/report.md)
- [Machine-readable scientific projection](discoveryworld/progress/20260922T155718_CST/snapshot.json)
- [Earlier 10:44 state and numerical diagnosis](discoveryworld/progress/20260922T104452_CST/report.md)
- [Historical Native preparation, earlier ES-only branch and failed MADE-bank transfer](discoveryworld/progress/20260922T052401_CST/report.md)

The V4 branch is stopped at the activity gate. Released MADE resources remain assigned to DiscoveryWorld; resource allocation is not counted as completed science.

## Scope and provenance

This update re-exports existing records only. No model, simulator, graph extraction, NN fitting or ES job was run to generate these reports. MADE uses the unchanged 10:18 export; DW preserves each source observation time and ends at 15:57. Prior snapshots overlap and must not be added together. Existing 9B collections, original seed1 MADE and SnAr results remain separate. Negative results, unavailable heads, controller inactivity, failure records, source hashes, validation scripts and export manifests are retained.
