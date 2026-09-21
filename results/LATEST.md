# Latest verified experimental results

Observation: **2026-09-22 05:24 CST**. MADE **1068/1080** accepted trajectories; DiscoveryWorld **0/2** held-out paired trajectories.

## MADE — Qwen3.5-4B

| Budget | Accepted / planned | Failed | Pending |
|---|---:|---:|---:|
| B10 | 360/360 | 0 | 0 |
| B30 | 360/360 | 0 | 0 |
| B50 | 348/360 | 1 | 11 |

The four arms are Native, uncertainty-only, ESOpt-only and Full. All three budgets contain 30 chemical systems and evaluation seeds 2, 3 and 4. The original seed1 study remains separate. B50 currently has Native89/90, uncertainty90/90, ESOpt79/90 and Full90/90. The Native technical failure remains recorded and is not imputed as zero.

| Full minus Native | Matched pairs | Mean SUN difference | Mean AUDC difference |
|---|---:|---:|---:|
| B10 | 90 | -0.433333 | -0.034556 |
| B30 | 90 | -0.600000 | -0.013901 |
| B50 | 89 | -1.573034 | -0.028054 |

The completed matched results do **not** show an overall Full-method improvement. All positive, tied and negative cases remain in the export. B50 comparisons are incomplete because one Native trajectory failed; matching denominators also differ for the unfinished ES-only arm.

- [Completed-result report](made_components/incremental/20260922T052401_CST/report.md)
- [All systems / budgets / arms / seeds](made_components/incremental/20260922T052401_CST/all_system_results.csv)
- [Per-system seed mean, sample variance and SD](made_components/incremental/20260922T052401_CST/seed_statistics.csv)
- [Matched pair differences](made_components/incremental/20260922T052401_CST/paired_results.csv)
- [Discovery curves](made_components/incremental/20260922T052401_CST/completed_curves.csv)

Variance is across evaluation seeds within the same system, budget and arm (sample variance, ddof=1). Missing/failed runs stay blank; n<2 variance is unavailable and incomplete seed sets are marked. These are not repeated training runs.

## DiscoveryWorld — Qwen3.5-4B

Eight preparatory Native trajectories are complete (800 action attempts). The earlier, separate ES-only B100 branch has six of seven episodes complete. The new Full-method subset still has zero valid graph rows, no fitted risk NN and no held-out result. The MADE-trained transcoder bank failed the unchanged fidelity threshold (FVU0.84356 >0.5); its failed attempts are retained. The original parameter-updating Full method remains the intended subset, but it is not yet an efficacy result.

- [DiscoveryWorld progress, source scores and failure record](discoveryworld/progress/20260922T052401_CST/report.md)
- [Machine-readable scientific projection](discoveryworld/progress/20260922T052401_CST/snapshot.json)

The active scheduling priority is to finish existing MADE jobs and move released resources to DiscoveryWorld. Additional benchmarks and seeds are deferred. This statement records priority, not proof that every allocated GPU is computing.

## Scope and provenance

This update re-exports existing records only. No model, simulator or materials oracle was run to generate these reports. Prior snapshots overlap and must not be added together. Existing 9B collections, original seed1 MADE and SnAr results remain in the historical index and are not relabelled as new tests. Source hashes, metric recomputation scripts and export manifests are retained.
