# Latest verified experimental results

MADE remains **1079 valid trajectories, 1 technical failure and 0 pending** (September 22, 10:18 CST). DiscoveryWorld adds an **18:47 read-only diagnosis of 20 actions from two already completed development trajectories**: Q75's three revisions changed no executed action; Common2's nine successful actions earned no immediate task score. This adds no trajectories, model updates or completed Full-method benefit. The dated 18:33 status snapshot remains unchanged.

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

The latest **closed-trajectory audit** reuses Q75/p304 and Common2/p305; it does not complete their partners. Q75 made three second proposals, all identical to their first action. Two recorded neural rank changes therefore changed candidate identity but not executed action; its longest identical failing-action run was nine. Common2 produced six different second actions in ten attempts, but its three second-candidate selections still changed no executed action relative to the first candidate. All nine successful actions had zero immediate task-score increase; neither trajectory completed the task.

On the executed actions' **cached** risks, retrospective threshold 0.5 flags 9/10 Q75 failures (Brier 0.040686, NLL 0.145225) and misses Common2's only failure (Brier 0.099204, NLL 0.372739; 0 false alarms among 9 successes). The actual Q75 threshold remains unchanged. Q75 contains no successful actions, so it cannot assess two-class discrimination. Different policy seeds preclude a paired method comparison; action success without immediate score does not prove absence of future value. All 20 primary labels and selected risks were available. No outcome is assigned to an unexecuted proposal.

[Closed-action diagnosis and portable recomputation](discoveryworld/diagnostics/20260922T184705_CST/report.md) · [20 selected-action records](discoveryworld/diagnostics/20260922T184705_CST/audit.json)

The newest completed rows are **development conditions**, not Native/Full held-out tests:

| Condition | Policy seed | Returned actions | Failed / successful actions | Task score | Fitness |
|---|---:|---:|---:|---:|---:|
| Q75 original quantile-risk controller | 304 | 10 | 10 / 0 | 0 | −0.10 |
| Common2 fixed two-proposal hash selection | 305 | 10 | 1 / 9 | 0 | −0.01 |

Q75's supported controller-activity gate is true despite every action failing. Common2's successful actions did not earn task score. **The rows use different seeds and are not an effect comparison.** Each trajectory is one clustered observation from world 2.

At 18:33, Posterior05/p304 had 7 returned actions, all failed, and Internal2/p305 had 1 returned successful action; both were incomplete. Neither pair supports a final improvement estimate. Each warm pair's eight-prefix qualification passed. Full ES still waited for its operating-point prerequisite, with no new parameter update confirmed. Grounded V6 was registered and waiting; resource/process metadata is not scientific execution.

**CV-v1 did not run: 0 fits and 0 optimizer updates.** One episode-held-out training fold has 33 positive / 7 negative rows, below the unchanged minimum of 10 rows in each class. No fold was dropped or gate relaxed. A training-only label audit identifies 24 fixed early positions that could supplement 60 fitting rows to at most 84, but **0 additional graphs were extracted and the 84-row dataset was not ready** at this snapshot.

The prior completed CPU feature diagnostic remains: 60 fit / 60 calibration / 40 development rows, with development in 2 episodes from 1 world. GraphOnly AUROC 0.9133 is below OutputAction/NoGraph 0.9600; Internal is 0.9700. Different capacities and the small clustered split preclude a causal or significant graph-benefit claim. Training diagnostics show increasing late development probability loss despite declining fit loss; the production 200-step NN and temperature remain unchanged. Cached fixed-0.5 flagging is not online failure prevention.

The prior V4 160/160 graph dataset and frozen-risk admission are unchanged. Its accepted G0/p303 B10 scored 0 with 7 failures and stopped at the activity gate before ES updates. Held-out Baseline 1 / Full 0 and completed Full-method pairs 0 remain the result record; different development worlds, seeds and budgets cannot be pooled into that test comparison.

- [18:33 development status, CV stop and label-only feasibility](discoveryworld/development/20260922T183307_CST/report.md)
- [Two completed development rows](discoveryworld/development/20260922T183307_CST/completed_development_metrics.csv)
- [Portable status validator](discoveryworld/development/20260922T183307_CST/validate.py)
- [Completed CPU feature and cached-risk report](discoveryworld/diagnostics/20260922T172256_CST/report.md)
- [Training curves and classifier diagnostic](discoveryworld/diagnostics/20260922T172256_CST/training_diagnostic_report.md)
- [Prior V4 graph/risk/G0 record](discoveryworld/progress/20260922T155718_CST/report.md)

## Scope and provenance

Publication reuses completed records only and adds no model, simulator, graph, NN-fitting or ES calls. The feature diagnostic records four fits/800 updates; the separate training diagnostic records another four NN fits/800 updates plus four classical fits. Neither adds graphs or environment calls. These repeated fits reuse the same observations and must not be counted as independent validation trials. Current exploration concerns MADE and DiscoveryWorld; existing historical9B, original seed1 MADE and SnAr exports remain separate. Negative results, unavailable heads, gate failures and source hashes remain preserved. There is no completed Full-method gain or larger-benchmark completion claim.
