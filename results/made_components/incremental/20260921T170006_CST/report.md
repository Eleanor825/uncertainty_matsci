# Incremental MADE completed-result update

Observed **2026-09-21T17:00:06.189530+08:00**. **940/1080** registered evaluations have original completed receipts. This is an overlapping snapshot, not additional experiments.

| Arm | Accepted | Failed | Claimed, unfinished | Reserved, unknown | Unclaimed |
|---|---:|---:|---:|---:|---:|
| baseline_reference | 220 | 1 | 4 | 0 | 45 |
| uq_only_support_aware | 252 | 0 | 7 | 0 | 11 |
| es_only_independent | 199 | 0 | 2 | 0 | 69 |
| full_support_aware | 269 | 0 | 1 | 0 | 0 |

Only completed matched task/budget/seed pairs enter comparisons. Pending, failed and unknown rows remain without metrics. Within-configuration sample variance uses evaluation seeds; incomplete seed sets are labelled and n<2 variance is blank. Negative results and original failure hashes are retained.

| Comparison | Matched pairs | Mean SUN difference | Mean AUDC difference |
|---|---:|---:|---:|
| uq_only_support_aware@B10 minus baseline | 90 | -0.15555555555555556 | -0.003111111111111116 |
| uq_only_support_aware@B30 minus baseline | 90 | -0.3333333333333333 | -0.010123456790123457 |
| uq_only_support_aware@B50 minus baseline | 40 | -0.35 | -0.013380000000000003 |
| es_only_independent@B10 minus baseline | 90 | -0.5111111111111111 | -0.042888888888888886 |
| es_only_independent@B30 minus baseline | 87 | -0.3793103448275862 | -0.005862068965517245 |
| es_only_independent@B50 minus baseline | 22 | -2.3636363636363638 | -0.04494545454545455 |
| full_support_aware@B10 minus baseline | 89 | -0.4044943820224719 | -0.03101123595505618 |
| full_support_aware@B30 minus baseline | 90 | -0.6 | -0.013901234567901237 |
| full_support_aware@B50 minus baseline | 40 | -1.4 | -0.0252 |

These current matching sets vary across arms and snapshots. Their means are not full-matrix effects or evidence of a causal NN contribution. The two ES policies were trained separately. The original seed1 180-run study and SnAr are separate.

Verification checks fixed production registration hashes, original receipt seals, exact result/claim hashes, registered job identity, complete curves and candidate counts. It does not reload models or rerun original raw-trajectory acceptance. Reading is sequential while other jobs continue.

Recompute this export with `python3 -B recompute.py`. Original completed-result hashes and all1080 status rows are retained; no credentials or process/operator records are published.
