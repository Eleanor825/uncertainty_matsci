# MADE B10/B30/B50 results

Fresh B50 observation: 2026-09-19T01:10:39.978473+08:00. B10/B30 reuse their immutable completed audits; provenance retains their original observation times. Qwen3.5-4B, seed1, fixed base versus G2 full method.

## Completed and pending denominators

| Budget | Baseline complete/30 | Full complete/30 | Complete pairs | Active | Pending |
|---:|---:|---:|---:|---:|---:|
| 10 | 30 | 30 | 30 | 0 | 0 |
| 30 | 30 | 30 | 30 | 0 | 0 |
| 50 | 4 | 30 | 4 | 10 | 16 |

## Matched-system comparisons only

| Budget | Paired n | SUN mean base→full | AUDC mean base→full | SUN wins/ties/losses | AUDC wins/ties/losses |
|---:|---:|---:|---:|---:|---:|
| 10 | 30 | 1.4→1.8 | 0.154667→0.188667 | 7/18/5 | 8/16/6 |
| 30 | 30 | 4.53333→3.23333 | 0.160815→0.124185 | 5/17/8 | 4/16/10 |
| 50 | 4 | 15.75→9.75 | 0.3539→0.2285 | 1/0/3 | 1/0/3 |

The B50 complete-pair set currently has one positive system and three negative systems in both metrics. It is not the full30-system estimate. Full-only B50 has30 completed trajectories; its separate denominator and descriptive statistics are in summary.json/statistics.csv.

## B50 completed pairs

| System | SUN base→full | ΔSUN | AUDC base→full | ΔAUDC |
|---|---:|---:|---:|---:|
| Al-Li-V | 22→3 | -19 | 0.4776→0.0844 | -0.3932 |
| Al-V-Zn | 22→17 | -5 | 0.5000→0.3940 | -0.1060 |
| Co-Dy-W | 1→0 | -1 | 0.0340→0.0000 | -0.0340 |
| Ga-Ho-Lu | 18→19 | 1 | 0.4040→0.4356 | 0.0316 |

## Files and interpretation

- trajectories.csv:154 complete trajectories only, with original result/RPC/receipt or ledger hashes.
- all_system_results.csv:all180 registered budget/system/method rows; incomplete final metrics stay empty.
- curves.csv:all4254 curve points for the154 complete trajectories; AUDC can be recomputed independently.
- paired_systems.csv and paired_B*.csv:only identical complete-pair sets.
- statistics.csv: n,mean,sample variance(ddof1),SD for SUN,mSUN,AUDC; each completed arm, matched arm, and paired delta are explicitly separated.
- timings_and_costs.csv:recorded complete-trajectory costs; no new execution is attributed to this export.
- provenance.json distinguishes original raw hashes from derived file hashes; raw experiment assets are not published.

Sample variance describes cross-system dispersion at one seed. No across-seed stability, statistical significance, or general effectiveness is established. All negative/zero outcomes are retained. No scientific calls, model calls, parameter changes, or checkpoint reselection occurred to produce this export.

## Per-system readable tables

- [B10:all30 systems](systems_B10.md)
- [B30:all30 systems](systems_B30.md)
- [B50:all30 systems, including active/pending](systems_B50.md)

The completed trajectories represented here consumed4100 candidate oracle attempts (B10:600;B30:1800;B50:1700). This excludes active partial trajectories, initialization, surrogate scoring, training and development. The export itself performed zero scientific/model calls.
