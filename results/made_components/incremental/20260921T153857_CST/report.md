# Incremental MADE completed-result update

Observed **2026-09-21T15:38:57.813514+08:00**. **920/1080** registered evaluations have original completed receipts. This is an overlapping snapshot, not additional experiments.

| Arm | Accepted | Failed | Claimed, unfinished | Reserved, unknown | Unclaimed |
|---|---:|---:|---:|---:|---:|
| baseline_reference | 213 | 0 | 5 | 0 | 52 |
| uq_only_support_aware | 248 | 0 | 7 | 0 | 15 |
| es_only_independent | 195 | 0 | 1 | 0 | 74 |
| full_support_aware | 264 | 0 | 1 | 0 | 5 |

Only completed matched task/budget/seed pairs enter comparisons. Pending, failed and unknown rows remain without metrics. Within-configuration sample variance uses evaluation seeds; incomplete seed sets are labelled and n<2 variance is blank. Negative results and original failure hashes are retained.

| Comparison | Matched pairs | Mean SUN difference | Mean AUDC difference |
|---|---:|---:|---:|
| uq_only_support_aware@B10 minus baseline | 90 | -0.15555555555555556 | -0.003111111111111116 |
| uq_only_support_aware@B30 minus baseline | 90 | -0.3333333333333333 | -0.010123456790123457 |
| uq_only_support_aware@B50 minus baseline | 33 | -0.5757575757575758 | -0.021103030303030307 |
| es_only_independent@B10 minus baseline | 90 | -0.5111111111111111 | -0.042888888888888886 |
| es_only_independent@B30 minus baseline | 85 | -0.2823529411764706 | -0.0018300653594771274 |
| es_only_independent@B50 minus baseline | 20 | -2.85 | -0.05906 |
| full_support_aware@B10 minus baseline | 84 | -0.2976190476190476 | -0.021547619047619048 |
| full_support_aware@B30 minus baseline | 90 | -0.6 | -0.013901234567901237 |
| full_support_aware@B50 minus baseline | 33 | -1.9696969696969697 | -0.033078787878787874 |

These current matching sets vary across arms and snapshots. Their means are not full-matrix effects or evidence of a causal NN contribution. The two ES policies were trained separately. The original seed1 180-run study and SnAr are separate.

Verification checks fixed production registration hashes, original receipt seals, exact result/claim hashes, registered job identity, complete curves and candidate counts. It does not reload models or rerun original raw-trajectory acceptance. Reading is sequential while other jobs continue.

Recompute this export with `python3 -B recompute.py`. Original completed-result hashes and all1080 status rows are retained; no credentials or process/operator records are published.
