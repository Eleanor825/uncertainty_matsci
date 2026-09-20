# Incremental MADE completed-result update

Observed **2026-09-20T08:08:47.198599+08:00**. **483/1080** registered evaluations have original completed receipts. This is an overlapping snapshot, not additional experiments.

| Arm | Accepted | Failed | Claimed, unfinished | Reserved, unknown | Unclaimed |
|---|---:|---:|---:|---:|---:|
| baseline_reference | 114 | 0 | 1 | 0 | 155 |
| uq_only_support_aware | 116 | 0 | 8 | 0 | 146 |
| es_only_independent | 114 | 0 | 2 | 0 | 154 |
| full_support_aware | 139 | 0 | 8 | 0 | 123 |

Only completed matched task/budget/seed pairs enter comparisons. Pending, failed and unknown rows remain without metrics. Within-configuration sample variance uses evaluation seeds; incomplete seed sets are labelled and n<2 variance is blank. Negative results and original failure hashes are retained.

| Comparison | Matched pairs | Mean SUN difference | Mean AUDC difference |
|---|---:|---:|---:|
| uq_only_support_aware@B10 minus baseline | 83 | -0.07228915662650602 | 0.00554216867469879 |
| uq_only_support_aware@B30 minus baseline | 17 | -0.11764705882352941 | 0.013202614379084961 |
| uq_only_support_aware@B50 minus baseline | 0 | None | None |
| es_only_independent@B10 minus baseline | 90 | -0.5111111111111111 | -0.042888888888888886 |
| es_only_independent@B30 minus baseline | 14 | -1.1428571428571428 | -0.036825396825396824 |
| es_only_independent@B50 minus baseline | 0 | None | None |
| full_support_aware@B10 minus baseline | 84 | -0.2976190476190476 | -0.021547619047619048 |
| full_support_aware@B30 minus baseline | 18 | -0.2222222222222222 | -0.009629629629629634 |
| full_support_aware@B50 minus baseline | 0 | None | None |

These current matching sets vary across arms and snapshots. Their means are not full-matrix effects or evidence of a causal NN contribution. The two ES policies were trained separately. The original seed1 180-run study and SnAr are separate.

Verification checks fixed production registration hashes, original receipt seals, exact result/claim hashes, registered job identity, complete curves and candidate counts. It does not reload models or rerun original raw-trajectory acceptance. Reading is sequential while other jobs continue.

Recompute this export with `python3 -B recompute.py`. Original completed-result hashes and all1080 status rows are retained; no credentials or process/operator records are published.
