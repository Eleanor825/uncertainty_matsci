# Incremental MADE completed-result update

Observed **2026-09-20T08:38:44.808611+08:00**. **497/1080** registered evaluations have original completed receipts. This is an overlapping snapshot, not additional experiments.

| Arm | Accepted | Failed | Claimed, unfinished | Reserved, unknown | Unclaimed |
|---|---:|---:|---:|---:|---:|
| baseline_reference | 115 | 0 | 1 | 0 | 154 |
| uq_only_support_aware | 123 | 0 | 8 | 0 | 139 |
| es_only_independent | 115 | 0 | 2 | 0 | 153 |
| full_support_aware | 144 | 0 | 8 | 0 | 118 |

Only completed matched task/budget/seed pairs enter comparisons. Pending, failed and unknown rows remain without metrics. Within-configuration sample variance uses evaluation seeds; incomplete seed sets are labelled and n<2 variance is blank. Negative results and original failure hashes are retained.

| Comparison | Matched pairs | Mean SUN difference | Mean AUDC difference |
|---|---:|---:|---:|
| uq_only_support_aware@B10 minus baseline | 83 | -0.07228915662650602 | 0.00554216867469879 |
| uq_only_support_aware@B30 minus baseline | 18 | -0.1111111111111111 | 0.01246913580246913 |
| uq_only_support_aware@B50 minus baseline | 1 | -1.0 | 0.024400000000000005 |
| es_only_independent@B10 minus baseline | 90 | -0.5111111111111111 | -0.042888888888888886 |
| es_only_independent@B30 minus baseline | 15 | -1.0666666666666667 | -0.03437037037037037 |
| es_only_independent@B50 minus baseline | 0 | None | None |
| full_support_aware@B10 minus baseline | 84 | -0.2976190476190476 | -0.021547619047619048 |
| full_support_aware@B30 minus baseline | 19 | -0.21052631578947367 | -0.009122807017543862 |
| full_support_aware@B50 minus baseline | 0 | None | None |

These current matching sets vary across arms and snapshots. Their means are not full-matrix effects or evidence of a causal NN contribution. The two ES policies were trained separately. The original seed1 180-run study and SnAr are separate.

Verification checks fixed production registration hashes, original receipt seals, exact result/claim hashes, registered job identity, complete curves and candidate counts. It does not reload models or rerun original raw-trajectory acceptance. Reading is sequential while other jobs continue.

Recompute this export with `python3 -B recompute.py`. Original completed-result hashes and all1080 status rows are retained; no credentials or process/operator records are published.
