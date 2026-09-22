# Completed DiscoveryWorld operating-point pair — 22 September 2026, 18:59 CST

**Changing only the risk threshold did not improve this paired development case.** Q75 and Posterior05 both completed ten returned actions, all failed, with task score 0 and fitness −0.1. The observed paired difference is zero for failures, task score and fitness. This is one Proteomics Normal world, one policy seed and one trajectory per condition, not evidence of a universal null effect.

Both conditions use Qwen3.5-4B, world 2, policy seed 304, B10, the same unchanged G0 tensors, frozen 200-update risk NN and temperature, public prompt/decoder/parser/evaluator, and at most one revision. Q75 retains its calibration quantile threshold 0.9867005323054212; Posterior05 uses the prospectively fixed 0.5. This is an actual online operating-point pair, unlike the earlier retrospective cached-flag calculation. **It is not Full versus Native**: the original jobs retain an internal `arm=Full` collector label, but this pair performs zero ES parameter updates and no held-out tests.

| Condition | Threshold | Failed actions | Task score | Fitness | Neural revisions | Recorded rank changes | Different second actions | Actual action changes | Proposals / candidate graph returns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q75 | 0.9867005 | 10/10 | 0 | −0.10 | 3 | 2 | 0 | 0 | 13 / 13 |
| Posterior05 | 0.5 | 10/10 | 0 | −0.10 | 9 | 7 | 0 | 0 | 19 / 19 |

The lower threshold produced six additional revisions and six additional candidate graph evaluations, but none of the second proposals changed the parsed action. Candidate rank changes therefore did not change what the environment executed. Both action sequences are equal at every step and both contain an exact nine-step run of the same failing action. The first step uses a string argument and later steps an integer argument; the exact-record repetition count does not silently normalize that difference. Both supported activity gates are true, illustrating why activity is a mechanism check rather than proof of effective repair or task progress.

The warm pair records one model load and one complete eight-prefix qualification suite shared only at the unchanged G0. The 13/19 graph counts cover generated candidates and **exclude that shared qualification work**. They are counts, not GPU-time or wall-time estimates. Unexecuted alternatives receive no outcome labels. Q75 was already published as a closed individual trajectory; this release newly closes its Posterior05 partner and must not count Q75 twice.

## Separately observed continuation status

The 18:56 observer confirms that the independently registered Full ES v5 continuation started its G0 qualification: two of eight prefix results were observed, with no ES parameter update or completed Full test confirmed. That incomplete suite is not marked passed. The registration SHA is 53fa5513f801461f3e2b636b7679e5afc13abdc0d5a676a54731f5ac10052027; this is a registration hash, not a trained-parameter result. Internal2 on policy seed 305 had five returned actions with zero failures and remained incomplete. It is not compared with the p304 pair or assigned a final score/fitness.

This result supports the narrow diagnosis that more threshold-triggered calls, using the existing generic feedback and candidate selector, failed to produce different actions in this case. It does not establish that grounded feedback will help, that no possible threshold can help, or that the full method is complete. Earlier negative, partial and diagnostic snapshots remain immutable.

[Closed source projection](closed_pair.json), [paired metrics](paired_metrics.csv), [summary](summary.json), [continuation status](related_progress.json), [provenance](provenance.json), and [portable validator](validate.py). Exporting adds zero model, graph, environment or optimizer calls.
