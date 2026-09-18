# Time-stamped MADE results

Latest audit: **2026-09-19 05:21:38 CST / September18 21:21:38 UTC**. These are derived exports of observed results, retaining negative outcomes and all earlier incomplete snapshots. No experiment was rerun for publication.

The [complete report](budget_sweep/20260919T052138_CST/report.md) and [180-row matrix](budget_sweep/20260919T052138_CST/all_system_results.csv) cover all30 systems, two methods and independent B10/B30/B50 episodes. **180/180 results are complete:179 original runner acceptances plus one independently derived GaPt baseline acceptance.**

| Budget | Complete pairs | SUN total baseline → full | Mean AUDC baseline → full | SUN W/T/L | AUDC W/T/L |
|---|---:|---:|---:|---:|---:|
| B10 | 30/30 | 42 → 54 | 0.154667 → 0.188667 | 7/18/5 | 8/16/6 |
| B30 | 30/30 | 136 → 97 | 0.160815 → 0.124185 | 5/17/8 | 4/16/10 |
| B50 | 30/30 | 233 → 147 | 0.158840 → 0.112787 | 9/10/11 | 10/10/10 |

**Only seed1 is available; repeated-seed variance is not estimable.** Cross-system variance in the secondary statistics files is a different quantity. The [GaPt reconciliation](budget_sweep/20260919T052138_CST/ga_pt_tm_reconciliation.json) preserves the official nonmonotone curve and original failure. Original global publication is not claimed. Completed evaluation candidate calls total5,400; the5,580 curve points permit independent SUN/AUDC recomputation. Earlier snapshots overlap and must not be added as new execution costs.

The previous [01:10 partial B50 snapshot](budget_sweep/20260919T010939_CST/report.md) remains unchanged.

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

The table above is historical. The latest snapshot has all180 complete trajectories, including one explicitly reconciled GaPt result, while the original global publication receipt is not asserted. At the historical18:44 observation B50 was still unfinished; those missing values remain NA in that snapshot. Cross-system sample variance (ddof=1) and SD use seed1 only; a smaller variance is not evidence of better uncertainty calibration or across-seed repeatability.

## Files

- Latest per-system tables: [B10](budget_sweep/20260919T052138_CST/systems_B10.md), [B30](budget_sweep/20260919T052138_CST/systems_B30.md), [B50](budget_sweep/20260919T052138_CST/systems_B50.md). [Mean/variance/SD data](budget_sweep/20260919T052138_CST/statistics.csv) and [recomputation script](budget_sweep/20260919T052138_CST/recompute.py) cover the same observation.
- [index.json](index.json): historical snapshot index plus the dated statistical-export index.
- [statistics/b10_b30_complete_20260918T184426_CST](statistics/b10_b30_complete_20260918T184426_CST/README.md): complete B10/B30 pairs, mean/variance/SD tables, source hashes and a stdlib-only reproducible `analyze.py`.
- `snapshots/<id>/trajectories.csv`: every completed test trajectory in that snapshot; model/method/system/seed/budget and SUN/AUDC/mSUN.
- `paired_systems.csv`: complete matched pairs, including all negative and tied outcomes.
- `curves.csv`: cached original current official SUN curves (including dynamic-hull decreases); sufficient to independently recompute the exported SUN/AUDC/mSUN.
- `timings_and_costs.csv`: observed rollout, initialization, graph and verification times/counters. Blank means unavailable, **not zero**.
- `summary.json`: snapshot time, expected/completed counts, publication status, wins/losses/ties and matched-pair means.
- `provenance.json`: original source-cache/result/RPC hashes and scope fingerprints.
- [manifest.json](manifest.json): separate hashes of the **derived published files**.
- [validate_exports.py](validate_exports.py): stdlib validation of hashes, curves, metrics, pairing and repeated-trajectory identity.

The snapshots overlap. Original trajectories are reused read-only; **do not add trajectory or cost totals across snapshots**. The15:32 curve snapshot contains105 distinct completed test trajectories (60 B10 +45 B30), accounting for1950 candidate evaluations in completed trajectories. The18:44 statistical export covers120 distinct completed trajectories (60 B10 +60 B30), accounting for2400 candidate evaluations in completed trajectories. This excludes unfinished trajectories and all training/development/diagnostic costs; it is not the end-to-end research cost.

Detailed interpretation, method/source identities and limitations: [docs/results.md](../docs/results.md).
