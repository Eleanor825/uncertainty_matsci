# Incremental MADE completed-result update

Observed **2026-09-21T18:30:10.128907+08:00**. **956/1080** registered evaluations have original completed receipts. This is an overlapping snapshot, not additional experiments.

| Arm | Accepted | Failed | Claimed, unfinished | Reserved, unknown | Unclaimed |
|---|---:|---:|---:|---:|---:|
| baseline_reference | 224 | 1 | 4 | 0 | 41 |
| uq_only_support_aware | 257 | 0 | 7 | 0 | 6 |
| es_only_independent | 205 | 0 | 2 | 0 | 63 |
| full_support_aware | 270 | 0 | 0 | 0 | 0 |

Only completed matched task/budget/seed pairs enter comparisons. Pending, failed and unknown rows remain without metrics. Within-configuration sample variance uses evaluation seeds; incomplete seed sets are labelled and n<2 variance is blank. Negative results and original failure hashes are retained.

| Comparison | Matched pairs | Mean SUN difference | Mean AUDC difference |
|---|---:|---:|---:|
| uq_only_support_aware@B10 minus baseline | 90 | -0.15555555555555556 | -0.003111111111111116 |
| uq_only_support_aware@B30 minus baseline | 90 | -0.3333333333333333 | -0.010123456790123457 |
| uq_only_support_aware@B50 minus baseline | 44 | -0.8636363636363636 | -0.0226909090909091 |
| es_only_independent@B10 minus baseline | 90 | -0.5111111111111111 | -0.042888888888888886 |
| es_only_independent@B30 minus baseline | 90 | -0.32222222222222224 | -0.003197530864197535 |
| es_only_independent@B50 minus baseline | 25 | -2.08 | -0.039552000000000004 |
| full_support_aware@B10 minus baseline | 90 | -0.43333333333333335 | -0.034555555555555555 |
| full_support_aware@B30 minus baseline | 90 | -0.6 | -0.013901234567901237 |
| full_support_aware@B50 minus baseline | 44 | -2.5 | -0.04623636363636364 |

These current matching sets vary across arms and snapshots. Their means are not full-matrix effects or evidence of a causal NN contribution. The two ES policies were trained separately. The original seed1 180-run study and SnAr are separate.

Verification checks fixed production registration hashes, original receipt seals, exact result/claim hashes, registered job identity, complete curves and candidate counts. It does not reload models or rerun original raw-trajectory acceptance. Reading is sequential while other jobs continue.

Recompute this export with `python3 -B recompute.py`. Original completed-result hashes and all1080 status rows are retained; no credentials or process/operator records are published.
