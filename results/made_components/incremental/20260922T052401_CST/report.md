# Incremental MADE completed-result update

Observed **2026-09-22T05:24:01.115881+08:00**. **1068/1080** registered evaluations have original completed receipts. This is an overlapping snapshot, not additional experiments.

| Arm | Accepted | Failed | Claimed, unfinished | Reserved, unknown | Unclaimed |
|---|---:|---:|---:|---:|---:|
| baseline_reference | 269 | 1 | 0 | 0 | 0 |
| uq_only_support_aware | 270 | 0 | 0 | 0 | 0 |
| es_only_independent | 259 | 0 | 9 | 0 | 2 |
| full_support_aware | 270 | 0 | 0 | 0 | 0 |

Only completed matched task/budget/seed pairs enter comparisons. Pending, failed and unknown rows remain without metrics. Within-configuration sample variance uses evaluation seeds; incomplete seed sets are labelled and n<2 variance is blank. Negative results and original failure hashes are retained.

| Comparison | Matched pairs | Mean SUN difference | Mean AUDC difference |
|---|---:|---:|---:|
| uq_only_support_aware@B10 minus baseline | 90 | -0.15555555555555556 | -0.003111111111111116 |
| uq_only_support_aware@B30 minus baseline | 90 | -0.3333333333333333 | -0.010123456790123457 |
| uq_only_support_aware@B50 minus baseline | 89 | -0.449438202247191 | -0.010022471910112362 |
| es_only_independent@B10 minus baseline | 90 | -0.5111111111111111 | -0.042888888888888886 |
| es_only_independent@B30 minus baseline | 90 | -0.32222222222222224 | -0.003197530864197535 |
| es_only_independent@B50 minus baseline | 78 | -1.5256410256410255 | -0.030733333333333335 |
| full_support_aware@B10 minus baseline | 90 | -0.43333333333333335 | -0.034555555555555555 |
| full_support_aware@B30 minus baseline | 90 | -0.6 | -0.013901234567901237 |
| full_support_aware@B50 minus baseline | 89 | -1.5730337078651686 | -0.02805393258426966 |

These current matching sets vary across arms and snapshots. Their means are not full-matrix effects or evidence of a causal NN contribution. The two ES policies were trained separately. The original seed1 180-run study and SnAr are separate.

Verification checks fixed production registration hashes, original receipt seals, exact result/claim hashes, registered job identity, complete curves and candidate counts. It does not reload models or rerun original raw-trajectory acceptance. Reading is sequential while other jobs continue.

Recompute this export with `python3 -B recompute.py`. Original completed-result hashes and all1080 status rows are retained; no credentials or process/operator records are published.
