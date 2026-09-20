# Incremental MADE completed-result update

Observed **2026-09-21T01:20:08.602963+08:00**. **737/1080** registered evaluations have original completed receipts. This is an overlapping snapshot, not additional experiments.

| Arm | Accepted | Failed | Claimed, unfinished | Reserved, unknown | Unclaimed |
|---|---:|---:|---:|---:|---:|
| baseline_reference | 150 | 0 | 1 | 0 | 119 |
| uq_only_support_aware | 194 | 0 | 8 | 0 | 68 |
| es_only_independent | 159 | 0 | 2 | 0 | 109 |
| full_support_aware | 234 | 0 | 8 | 0 | 28 |

Only completed matched task/budget/seed pairs enter comparisons. Pending, failed and unknown rows remain without metrics. Within-configuration sample variance uses evaluation seeds; incomplete seed sets are labelled and n<2 variance is blank. Negative results and original failure hashes are retained.

| Comparison | Matched pairs | Mean SUN difference | Mean AUDC difference |
|---|---:|---:|---:|
| uq_only_support_aware@B10 minus baseline | 83 | -0.07228915662650602 | 0.00554216867469879 |
| uq_only_support_aware@B30 minus baseline | 44 | -0.5909090909090909 | -0.015000000000000003 |
| uq_only_support_aware@B50 minus baseline | 12 | -0.08333333333333333 | -0.0183 |
| es_only_independent@B10 minus baseline | 90 | -0.5111111111111111 | -0.042888888888888886 |
| es_only_independent@B30 minus baseline | 41 | -1.0731707317073171 | -0.030189701897018972 |
| es_only_independent@B50 minus baseline | 10 | -2.9 | -0.06436 |
| full_support_aware@B10 minus baseline | 84 | -0.2976190476190476 | -0.021547619047619048 |
| full_support_aware@B30 minus baseline | 45 | -0.6666666666666666 | -0.02162962962962963 |
| full_support_aware@B50 minus baseline | 10 | -1.9 | -0.03284 |

These current matching sets vary across arms and snapshots. Their means are not full-matrix effects or evidence of a causal NN contribution. The two ES policies were trained separately. The original seed1 180-run study and SnAr are separate.

Verification checks fixed production registration hashes, original receipt seals, exact result/claim hashes, registered job identity, complete curves and candidate counts. It does not reload models or rerun original raw-trajectory acceptance. Reading is sequential while other jobs continue.

Recompute this export with `python3 -B recompute.py`. Original completed-result hashes and all1080 status rows are retained; no credentials or process/operator records are published.
