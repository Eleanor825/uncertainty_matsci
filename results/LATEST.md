# Latest verified experimental results

Observations: **MADE 2026-09-22 10:18 CST; DiscoveryWorld 2026-09-22 10:44 CST**. MADE has **1079 valid trajectories, 1 technical failure and 0 pending** out of 1080 registered evaluations. DiscoveryWorld has **1 Baseline trajectory, 0 Full trajectories and 0 completed matched pairs**.

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

The fixed **Baseline** test at world seed 3, policy seed 401 and B30 completed with **normalized score 0, task success false and 29/30 failed action attempts**. Full has no accepted held-out result. One completed Baseline trajectory does not constitute a completed Baseline–Full comparison.

The newly trained DW-domain transcoder bank passed development fidelity for **32/32 layers** at the unchanged FVU limit of 0.5. The original attribution check at epsilon 0.001 failed. A separate diagnostic on one TRAIN prefix passed all four selected checks at epsilon 0.01, but explicitly remains **unqualified**; this is not the required eight-prefix formal qualification. At this snapshot there are **0 formally qualified graph rows, 0 fitted/calibrated risk NNs, 0 Full ES episodes and 0 Full test results**.

The earlier MADE-to-DW bank transfer failure (layer 0 FVU 0.84356 > 0.5) is retained. The [05:24 historical report](discoveryworld/progress/20260922T052401_CST/report.md) preserves eight Native B100 preparatory trajectories and six of seven earlier ES-only B100 episodes at their original observation time; those are not held-out Full evaluations. The intended Full subset still includes a risk NN and full-parameter Agentic ESOpt. No method-effectiveness claim is supported yet.

- [DiscoveryWorld scientific status through 10:44](discoveryworld/progress/20260922T104452_CST/report.md)
- [Machine-readable scientific projection](discoveryworld/progress/20260922T104452_CST/snapshot.json)

With the registered MADE collection closed, released resources are assigned to DiscoveryWorld. Additional benchmarks and seeds remain deferred. Resource allocation and later V3 graph work are not counted as completed science in this snapshot.

## Scope and provenance

This update re-exports existing records only. No model, simulator or materials oracle was run to generate these reports. MADE uses the already published 10:18 export; DW preserves each source observation time and ends at 10:44. Prior snapshots overlap and must not be added together. Existing 9B collections, original seed1 MADE and SnAr results remain in the historical index and are not relabelled as new tests. Negative results, failure records, source hashes, metric recomputation scripts and export manifests are retained.
