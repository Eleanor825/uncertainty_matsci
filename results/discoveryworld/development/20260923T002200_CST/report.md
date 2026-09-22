# Fixed three-policy development comparison — 2026-09-23 00:22:00 CST

**All 9 completed task scores are zero, with no task success.** Task score and task success are the primary endpoints. Lower action-error rates alone do not establish scientific progress.

| Policy | Task score mean ± sample SD | Score sample variance | Task success | Action-failure rate mean ± sample SD | Failure-rate sample variance |
|---|---:|---:|---:|---:|---:|
| Original G0 Native1 | 0.000000 ± 0.000000 | 0.00000000 | 0/3 | 0.433333 ± 0.466667 | 0.21777778 |
| Action-type SFT32 Native1 | 0.000000 ± 0.000000 | 0.00000000 | 0/3 | 0.000000 ± 0.000000 | 0.00000000 |
| Progress-credit SFT32 Native1 | 0.000000 ± 0.000000 | 0.00000000 | 0/3 | 0.011111 ± 0.019245 | 0.00037037 |

All three policies use the same three evaluation seeds in one fixed Proteomics Normal world. Sample variance uses denominator n−1=2 and describes sampling variation only, not cross-world or repeated-training variation.

| Policy seed | Policy | Task score | Task success | Failed actions / actual attempts | USE successes | PICKUP successes | Rotate/move actions | Longest identical-packet run |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 309 | Original G0 Native1 | 0.000000 | false | 3/30 | 0 | 0 | 5/30 | 4 |
| 309 | Action-type SFT32 Native1 | 0.000000 | false | 0/30 | 0 | 0 | 30/30 | 2 |
| 309 | Progress-credit SFT32 Native1 | 0.000000 | false | 1/30 | 0 | 0 | 30/30 | 3 |
| 310 | Original G0 Native1 | 0.000000 | false | 29/30 | 0 | 0 | 0/30 | 28 |
| 310 | Action-type SFT32 Native1 | 0.000000 | false | 0/30 | 0 | 0 | 30/30 | 3 |
| 310 | Progress-credit SFT32 Native1 | 0.000000 | false | 0/30 | 0 | 0 | 29/30 | 3 |
| 311 | Original G0 Native1 | 0.000000 | false | 7/30 | 1 | 0 | 0/30 | 3 |
| 311 | Action-type SFT32 Native1 | 0.000000 | false | 0/30 | 0 | 0 | 30/30 | 2 |
| 311 | Progress-credit SFT32 Native1 | 0.000000 | false | 0/30 | 0 | 0 | 30/30 | 2 |

| Paired policy minus G0 | Score difference mean ± sample SD | Score-difference sample variance | Failure-rate difference mean ± sample SD | Failure-rate-difference sample variance |
|---|---:|---:|---:|---:|
| Action-type SFT32 Native1 | 0.000000 ± 0.000000 | 0.00000000 | -0.433333 ± 0.466667 | 0.21777778 |
| Progress-credit SFT32 Native1 | 0.000000 ± 0.000000 | 0.00000000 | -0.422222 ± 0.478810 | 0.22925926 |

Both trained policies start independently from the same original Qwen3.5-4B G0. They use the same 106 actually successful world0 training actions, original token targets, fixed 32×4 schedule (128 training examples processed), full-text-parameter SGD, gradient clipping and G0 regularization. Ordinary SFT uses action-type weights. Progress-credit SFT changes only the response-loss multiplier: 0.8 normalized discounted positive-score credit over the current and next seven action transitions (γ=0.9), plus 0.2 normalized original type weights. Every original row keeps positive weight. A successful API action is not necessarily a scientific milestone; future credit is a training heuristic, not a proven causal contribution label. No world1, world2 or test row enters this training.

The checkpoints are their fixed final32 outputs, selected before these outcomes. Both training runs had their eight-prefix numerical graph qualification checked before evaluation. This verifies the numerical graph interface for those states; it does not prove calibration or task-solving ability. The comparison itself uses **Native1, zero risk-controller calls and zero candidate graphs**. It isolates a policy-training diagnostic and provides no direct Internal/Full-method efficacy evidence.

Each policy seed309,310,311 runs world2/B30 in fixed G0→ordinary SFT→credit SFT order with independent environments and histories. One warm model process strictly loads each source-bound state; initial public prompts and shared-step generation seeds match within each seed. All three prescribed seeds continue independently of results. Fixed-order wall times are descriptive and are separated from model/state loading and graph qualification. The earlier p308 two-arm observation remains separate and is not pooled here.

This is iterative development on one task instance, not held-out or broad benchmark validation. Three sampling seeds do not establish generalization, training robustness or statistical significance. Complete per-action parsed packets and official outcomes are exported; their exact repetition counts do not imply identical hidden world states. Recorded score changes follow the action plus the official tick and are not necessarily direct causal action credit. No unexecuted-candidate label is manufactured.

[All episode endpoints](episode_metrics.csv), [actual action rows](actions.csv), [action-type distributions](action_types.csv), [paired seed differences](paired_differences.csv), [mean/SD/sample variance](statistics.csv), [compact audited source projection](audit_projection.json), [source provenance](provenance.json). Raw prompts, host identities and private runtime logs are omitted. Publication performs no new model, NN, graph, environment or optimizer calls.
