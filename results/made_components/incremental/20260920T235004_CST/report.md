# Incremental MADE completed-result update

Observed **2026-09-20T23:50:04.441263+08:00**. **717/1080** registered evaluations have original completed receipts. This is an overlapping snapshot, not additional experiments.

| Arm | Accepted | Failed | Claimed, unfinished | Reserved, unknown | Unclaimed |
|---|---:|---:|---:|---:|---:|
| baseline_reference | 148 | 0 | 1 | 0 | 121 |
| uq_only_support_aware | 191 | 0 | 8 | 0 | 71 |
| es_only_independent | 152 | 0 | 2 | 0 | 116 |
| full_support_aware | 226 | 0 | 8 | 0 | 36 |

Only completed matched task/budget/seed pairs enter comparisons. Pending, failed and unknown rows remain without metrics. Within-configuration sample variance uses evaluation seeds; incomplete seed sets are labelled and n<2 variance is blank. Negative results and original failure hashes are retained.

| Comparison | Matched pairs | Mean SUN difference | Mean AUDC difference |
|---|---:|---:|---:|
| uq_only_support_aware@B10 minus baseline | 83 | -0.07228915662650602 | 0.00554216867469879 |
| uq_only_support_aware@B30 minus baseline | 42 | -0.6428571428571429 | -0.01637566137566138 |
| uq_only_support_aware@B50 minus baseline | 11 | -0.6363636363636364 | -0.02461818181818182 |
| es_only_independent@B10 minus baseline | 90 | -0.5111111111111111 | -0.042888888888888886 |
| es_only_independent@B30 minus baseline | 39 | -1.1794871794871795 | -0.03304843304843305 |
| es_only_independent@B50 minus baseline | 7 | -0.42857142857142855 | -0.0242857142857143 |
| full_support_aware@B10 minus baseline | 84 | -0.2976190476190476 | -0.021547619047619048 |
| full_support_aware@B30 minus baseline | 43 | -0.7674418604651163 | -0.02338501291989664 |
| full_support_aware@B50 minus baseline | 10 | -1.9 | -0.03284 |

These current matching sets vary across arms and snapshots. Their means are not full-matrix effects or evidence of a causal NN contribution. The two ES policies were trained separately. The original seed1 180-run study and SnAr are separate.

Verification checks fixed production registration hashes, original receipt seals, exact result/claim hashes, registered job identity, complete curves and candidate counts. It does not reload models or rerun original raw-trajectory acceptance. Reading is sequential while other jobs continue.

Recompute this export with `python3 -B recompute.py`. Original completed-result hashes and all1080 status rows are retained; no credentials or process/operator records are published.
