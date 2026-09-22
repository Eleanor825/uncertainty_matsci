# Latest verified experimental results

MADE remains **1079 valid trajectories, 1 technical failure and 0 pending** (September 22, 10:18 CST). DiscoveryWorld now has a **completed same-world/seed operating-point pair**, verified at 18:59: both Q75 and fixed-0.5 Posterior05 failed all 10 actions and scored 0. Lowering the threshold increased candidate graph calls from 13 to 19 without changing the executed action sequence. This is frozen-G0 development, not a Full/Native test; earlier snapshots remain unchanged.

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

**The Q75 versus Posterior05 operating-point pair is complete** at world 2, policy seed 304, B10. It uses the same frozen G0 weights, 200-update risk NN and temperature. The observed paired differences in failed actions, task score and fitness are all zero.

| Condition | Threshold | Failed actions | Task score | Fitness | Neural revisions | Recorded rank changes | Actual action changes | Candidate graph returns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q75 | 0.9867005 | 10/10 | 0 | −0.10 | 3 | 2 | 0 | 13 |
| Posterior05 | 0.5 | 10/10 | 0 | −0.10 | 9 | 7 | 0 | 19 |

All second proposals repeat their first action. Both executed sequences match exactly and contain a nine-step exact run of the same failing action. Lowering the threshold caused six additional candidate graph calls but no observed improvement in this pair. The two activity gates pass, yet rank changes only change candidate identity. This is one development world/seed, not a general null-effect estimate and **not Full versus Native**. One model load and one actual eight-prefix qualification were shared at unchanged G0; 13/19 candidate graph counts exclude shared qualification cost. The pair made zero parameter updates and no held-out test calls. Q75 is reused from its earlier publication, not counted twice.

The separately registered Full ES continuation had started G0 qualification in the **18:56 observer**, with two of eight prefix results and no parameter update or completed Full test confirmed. Qualification-in-progress is not qualification passed. At that same snapshot, Internal2/p305 had five returned actions with zero failures and remained incomplete. Its Common2/p305 partner was complete with one failure, nine successful actions and task score zero. The p305 comparison remains unfinished and is not compared with p304.

The earlier closed-action audit remains useful: Q75's cached executed-action risks flagged 9/10 failures at retrospective 0.5, while Common2's one failure was missed. Good alarm scores did not produce successful repair; Common2's nine successful actions had no immediate task-score increase. Such actions could still have later value. No outcomes are assigned to unexecuted proposals.

CV-v1 retains **0 fits and 0 optimizer updates** after one training fold had 33 positive / 7 negative rows, below the unchanged ten-per-class gate. At the published label-feasibility snapshot, 24 fixed extra training positions were prospective, with zero extra graphs extracted and no 84-row dataset ready. CPU feature and training diagnostics are unchanged; the production predictor is not replaced by a development-selected checkpoint.

The prior V4 G0/p303 failure at its activity gate remains recorded. The held-out record remains Baseline 1 / Full 0, with zero completed Full-method pairs. No online Full-method improvement is established.

- [Completed operating-point pair and portable checks](discoveryworld/development/20260922T185915_CST/report.md)
- [Paired metrics](discoveryworld/development/20260922T185915_CST/paired_metrics.csv)
- [18:56 continuation status](discoveryworld/development/20260922T185915_CST/related_progress.json)
- [Earlier closed-action risk and diversity audit](discoveryworld/diagnostics/20260922T184705_CST/report.md)
- [Immutable 18:33 partial/development and CV snapshot](discoveryworld/development/20260922T183307_CST/report.md)
- [CPU feature/training diagnostics](discoveryworld/diagnostics/20260922T172256_CST/report.md)

## Scope and provenance

Publication reuses completed records only and adds no model, simulator, graph, NN-fitting or ES calls. The feature diagnostic records four fits/800 updates; the separate training diagnostic records another four NN fits/800 updates plus four classical fits. Neither adds graphs or environment calls. These repeated fits reuse the same observations and must not be counted as independent validation trials. Current exploration concerns MADE and DiscoveryWorld; existing historical9B, original seed1 MADE and SnAr exports remain separate. Negative results, unavailable heads, gate failures and source hashes remain preserved. There is no completed Full-method gain or larger-benchmark completion claim.
