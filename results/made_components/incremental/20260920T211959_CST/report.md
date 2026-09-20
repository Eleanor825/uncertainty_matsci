# Incremental MADE completed-result update

Observed **2026-09-20T21:19:59.077470+08:00**. **688/1080** registered evaluations have original completed receipts. This is an overlapping snapshot, not additional experiments.

| Arm | Accepted | Failed | Claimed, unfinished | Reserved, unknown | Unclaimed |
|---|---:|---:|---:|---:|---:|
| baseline_reference | 143 | 0 | 2 | 0 | 125 |
| uq_only_support_aware | 185 | 0 | 7 | 0 | 78 |
| es_only_independent | 146 | 0 | 1 | 0 | 123 |
| full_support_aware | 214 | 0 | 8 | 0 | 48 |

Only completed matched task/budget/seed pairs enter comparisons. Pending, failed and unknown rows remain without metrics. Within-configuration sample variance uses evaluation seeds; incomplete seed sets are labelled and n<2 variance is blank. Negative results and original failure hashes are retained.

| Comparison | Matched pairs | Mean SUN difference | Mean AUDC difference |
|---|---:|---:|---:|
| uq_only_support_aware@B10 minus baseline | 83 | -0.07228915662650602 | 0.00554216867469879 |
| uq_only_support_aware@B30 minus baseline | 38 | -0.7631578947368421 | -0.017807017543859653 |
| uq_only_support_aware@B50 minus baseline | 11 | -0.6363636363636364 | -0.02461818181818182 |
| es_only_independent@B10 minus baseline | 90 | -0.5111111111111111 | -0.042888888888888886 |
| es_only_independent@B30 minus baseline | 35 | -1.2 | -0.031936507936507944 |
| es_only_independent@B50 minus baseline | 6 | 0.6666666666666666 | -0.005466666666666682 |
| full_support_aware@B10 minus baseline | 84 | -0.2976190476190476 | -0.021547619047619048 |
| full_support_aware@B30 minus baseline | 39 | -0.8974358974358975 | -0.025555555555555557 |
| full_support_aware@B50 minus baseline | 10 | -1.9 | -0.03284 |

These current matching sets vary across arms and snapshots. Their means are not full-matrix effects or evidence of a causal NN contribution. The two ES policies were trained separately. The original seed1 180-run study and SnAr are separate.

Verification checks fixed production registration hashes, original receipt seals, exact result/claim hashes, registered job identity, complete curves and candidate counts. It does not reload models or rerun original raw-trajectory acceptance. Reading is sequential while other jobs continue.

Recompute this export with `python3 -B recompute.py`. Original completed-result hashes and all1080 status rows are retained; no credentials or process/operator records are published.
