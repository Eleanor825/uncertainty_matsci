# Incremental MADE completed-result update

Observed **2026-09-20T10:08:49.292415+08:00**. **522/1080** registered evaluations have original completed receipts. This is an overlapping snapshot, not additional experiments.

| Arm | Accepted | Failed | Claimed, unfinished | Reserved, unknown | Unclaimed |
|---|---:|---:|---:|---:|---:|
| baseline_reference | 117 | 0 | 1 | 0 | 152 |
| uq_only_support_aware | 129 | 0 | 8 | 0 | 133 |
| es_only_independent | 120 | 0 | 2 | 0 | 148 |
| full_support_aware | 156 | 0 | 8 | 0 | 106 |

Only completed matched task/budget/seed pairs enter comparisons. Pending, failed and unknown rows remain without metrics. Within-configuration sample variance uses evaluation seeds; incomplete seed sets are labelled and n<2 variance is blank. Negative results and original failure hashes are retained.

| Comparison | Matched pairs | Mean SUN difference | Mean AUDC difference |
|---|---:|---:|---:|
| uq_only_support_aware@B10 minus baseline | 83 | -0.07228915662650602 | 0.00554216867469879 |
| uq_only_support_aware@B30 minus baseline | 20 | -0.1 | 0.011222222222222217 |
| uq_only_support_aware@B50 minus baseline | 1 | -1.0 | 0.024400000000000005 |
| es_only_independent@B10 minus baseline | 90 | -0.5111111111111111 | -0.042888888888888886 |
| es_only_independent@B30 minus baseline | 17 | -0.9411764705882353 | -0.030326797385620913 |
| es_only_independent@B50 minus baseline | 2 | 0.0 | -0.023600000000000003 |
| full_support_aware@B10 minus baseline | 84 | -0.2976190476190476 | -0.021547619047619048 |
| full_support_aware@B30 minus baseline | 21 | -0.19047619047619047 | -0.008253968253968257 |
| full_support_aware@B50 minus baseline | 0 | None | None |

These current matching sets vary across arms and snapshots. Their means are not full-matrix effects or evidence of a causal NN contribution. The two ES policies were trained separately. The original seed1 180-run study and SnAr are separate.

Verification checks fixed production registration hashes, original receipt seals, exact result/claim hashes, registered job identity, complete curves and candidate counts. It does not reload models or rerun original raw-trajectory acceptance. Reading is sequential while other jobs continue.

Recompute this export with `python3 -B recompute.py`. Original completed-result hashes and all1080 status rows are retained; no credentials or process/operator records are published.
