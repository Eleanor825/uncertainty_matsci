# Time-stamped MADE results

These are **derived exports of observed experiment results**, including negative outcomes and incomplete snapshots. Latest observation: **2026-09-19 01:10:39 CST / September18 17:10:39 UTC**. No experiment was rerun to create this release.

The [latest full report](budget_sweep/20260919T010939_CST/report.md) and [180-row planned matrix](budget_sweep/20260919T010939_CST/all_system_results.csv) cover B10/B30/B50 for every one of the30 systems and both methods. B10/B30 are complete. B50 has34/60 completed trajectories and4/30 completed pairs; final missing metrics are blank. B50's4 paired systems have SUN63→39 and mean AUDC0.3539→0.2285, with1 improvement and3 declines on both metrics. The full B50 arm has completed all30 systems, but its all30 aggregate must not be compared against only4 baselines.

The following table preserves the earlier observations and is not a live progress table.

| Snapshot / cohort | Budget | Completed trajectories | Complete pairs | SUN total, baseline → full | Mean AUDC, baseline → full |
|---|---:|---:|---:|---:|---:|
| Original two systems, Sep17 16:32 CST | 10 | 4/4 | 2/2 | 6 → 6 | 0.34000 → 0.25000 |
| Original five systems, accepted Sep17; read back Sep18 | 10 | 10/10 | 5/5 | 12 → 6 | 0.29600 → 0.10000 |
| Sep18 01:04:38 CST, partial all30 | 10 | 53/60 | 23/30 | 22 → 31 | 0.11652 → 0.14130 |
| Sep18 15:32:27 CST | 10 | **60/60** | **30/30** | **42 → 54** | **0.15467 → 0.18867** |
| Sep18 15:32:27 CST, partial | 30 | **45/60** | **15/30** | **110 → 68** | **0.26578 → 0.16370** |
| Sep18 15:32:27 CST | 50 | 0/60 | 0/30 | unavailable | unavailable |
| Sep18 18:44:26 CST | 10 | **60/60** | **30/30** | **42 → 54** | **0.15467 → 0.18867** |
| Sep18 18:44:26 CST | 30 | **60/60** | **30/30** | **136 → 97** | **0.16081 → 0.12419** |
| Sep18 18:44:26 CST | 50 | 0/60 | 0/30 | unavailable | unavailable |

Arm comparisons use **only complete system/seed pairs**. The15:32 B30 comparison uses15 baseline and15 full trajectories, while all45 completed trajectories remain in its historical CSV. The18:44 B30 comparison uses all30 complete pairs. Both the earlier partial decline and the completed B30 negative result are retained.

B10 and B30 data are complete:60 real trajectories and30 pairs each, checked through result/receipt, fixed-profile and RPC evidence. The three-budget global receipt remains pending; this is not completion of B50 or of the original multi-model/multi-seed study. At18:44, B50 had0 completed,10 claimed unfinished/running and50 unclaimed. Its missing metrics are NA. Cross-system sample variance (ddof=1) and SD use seed1 only; a smaller variance is not evidence of better uncertainty calibration or across-seed repeatability.

## Files

- Latest per-system tables: [B10](budget_sweep/20260919T010939_CST/systems_B10.md), [B30](budget_sweep/20260919T010939_CST/systems_B30.md), [B50](budget_sweep/20260919T010939_CST/systems_B50.md). [Mean/variance/SD data](budget_sweep/20260919T010939_CST/statistics.csv) and [recomputation script](budget_sweep/20260919T010939_CST/recompute.py) cover the same observation.
- [index.json](index.json): historical snapshot index plus the dated statistical-export index.
- [statistics/b10_b30_complete_20260918T184426_CST](statistics/b10_b30_complete_20260918T184426_CST/README.md): complete B10/B30 pairs, mean/variance/SD tables, source hashes and a stdlib-only reproducible `analyze.py`.
- `snapshots/<id>/trajectories.csv`: every completed test trajectory in that snapshot; model/method/system/seed/budget and SUN/AUDC/mSUN.
- `paired_systems.csv`: complete matched pairs, including all negative and tied outcomes.
- `curves.csv`: cached original cumulative SUN curves; sufficient to independently recompute the exported SUN/AUDC/mSUN.
- `timings_and_costs.csv`: observed rollout, initialization, graph and verification times/counters. Blank means unavailable, **not zero**.
- `summary.json`: snapshot time, expected/completed counts, publication status, wins/losses/ties and matched-pair means.
- `provenance.json`: original source-cache/result/RPC hashes and scope fingerprints.
- [manifest.json](manifest.json): separate hashes of the **derived published files**.
- [validate_exports.py](validate_exports.py): stdlib validation of hashes, curves, metrics, pairing and repeated-trajectory identity.

The snapshots overlap. Original trajectories are reused read-only; **do not add trajectory or cost totals across snapshots**. The15:32 curve snapshot contains105 distinct completed test trajectories (60 B10 +45 B30), accounting for1950 candidate evaluations in completed trajectories. The18:44 statistical export covers120 distinct completed trajectories (60 B10 +60 B30), accounting for2400 candidate evaluations in completed trajectories. This excludes unfinished trajectories and all training/development/diagnostic costs; it is not the end-to-end research cost.

Detailed interpretation, method/source identities and limitations: [docs/results.md](../docs/results.md).
