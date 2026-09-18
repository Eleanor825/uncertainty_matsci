# MADE B10/B30/B50 results

Observed 2026-09-19T04:53:46.838107+08:00 to 2026-09-19T05:21:38.188817+08:00. Qwen3.5-4B, evaluation seed1, baseline theta0 versus the fixed G2 full method.

**180/180 trajectories have audited complete results.** Of these, 179 retain original runner acceptance and one Ga-Pt-Tm B50 baseline uses independently derived acceptance. No pending result is filled with zero.

| Budget | Baseline/30 | Full/30 | Complete pairs | SUN total baseline→full (paired) | AUDC mean baseline→full (paired) | SUN W/T/L | AUDC W/T/L |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 30 | 30 | 30 | 42→54 | 0.154666667→0.188666667 | 7/18/5 | 8/16/6 |
| 30 | 30 | 30 | 30 | 136→97 | 0.160814815→0.124185185 | 5/17/8 | 4/16/10 |
| 50 | 30 | 30 | 30 | 233→147 | 0.158840000→0.112786667 | 9/10/11 | 10/10/10 |

The full method is better on the complete B10 mean and worse on the complete B30 mean. B50 is now complete across all30 paired systems and its full-method mean is lower on both SUN and AUDC. These are descriptive results for the fixed model and seed1; negative and zero outcomes are retained.

[All B10 systems](systems_B10.md) · [All B30 systems](systems_B30.md) · [All B50 systems](systems_B50.md)

Ga-Pt-Tm baseline B50 finished all50 physical evaluations and close. Official SUN fell from9 at step25 to7 at step26 when the hull changed, then ended at18 (AUDC0.3664). The old monotonic-count verifier rejected this valid trajectory. The independent corrected audit preserves the original curve and records root adoption; it does not rewrite the old failure or claim an original runner success. See [the reconciliation evidence](ga_pt_tm_reconciliation.json).

**Repeated-seed variance is unavailable.** Every configuration has only seed1. `statistics.csv` gives secondary cross-system sample variance/SD (ddof1); it must not be presented as run-to-run variance. `seed_coverage.csv` records the missing repeat-seed statistics explicitly.

Audited completed evaluation candidate calls: **5400**. `cost_summary.csv` separates initialization, surrogate evaluations and recorded wall times. Graph wall is included inside rollout wall and must not be added twice. These evaluation totals do not include earlier training/collection or offline diagnostics.

`curves.csv` has 5580 points for all 180 complete trajectories. `trajectories.csv` binds result/RPC/acceptance hashes; `all_system_results.csv` covers all180 planned rows. `paired_systems.csv` and `paired_B*.csv` contain only complete pairs. Source result-envelope SHA and derived export SHA are distinct. GaPt's legacy raw_result fields refer to its labelled derived envelope, not an invented original file.

Original global acceptance/publication is not asserted by this export. It is a read-only combined result audit, including the explicit GaPt reconciliation. No scientific calls, model calls, parameter edits, checkpoint selection or experimental reruns occurred to produce this export.
