# One completed Native–Full DiscoveryWorld test comparison

Audited as of **2026-09-23T03:28:28.657255+08:00**. This is one original **Proteomics / Normal / world 3 / policy seed 401 / B30** comparison with Qwen3.5-4B. The earlier Native result is reused unchanged; publication adds no model, environment, graph, fitting or ES calls.

| Method | Official normalized task score | Failed actions /30 | Task completed successfully | Generated candidates | Candidate graphs |
|---|---:|---:|---|---:|---:|
| Native (original Baseline record) | 0 | 29 | No | 30 | 0 |
| Full, development-selected G1 | 0.125 | 22 | No | 55 | 55 |

Full has **+0.125 task score and 7 fewer failed actions** in this pair. Its 22/30 failure rate remains high, and neither method completes the task. Both episodes are accepted, have 30 actual action/tick returns and closed environment RPCs; episode closure is distinct from task success. Official fitness (score + task-success indicator − 0.1 × failure fraction) is -0.096666667 for Native and 0.051666667 for Full. [Metrics](metrics.csv), [machine-readable comparison](comparison.json).

The audit rereads the original action events and official summaries, verifies their hashes and recomputes these counts and fitness. The committed Native and Full registrations have the same backbone/model configuration, original prompt/decoding contract, environment/evaluator/runtime contract and B30 action budget (apart from output locations). **This is not equal inference or graph compute:** Native allows one proposal per action; Full permits a second proposal and computes candidate graphs. Actual counts are shown above. [Original scientific evidence and source hashes](source_evidence.json).

The Full ES stage closed all seven registered B10 evaluations: one unchanged G0 evaluation reused, six new evaluations and two full-parameter updates over 4,539,265,536 parameters. Each update changed 723 parameter tensors; the before/after tensor hashes form a continuous G0→G1→G2 chain. Selection was made over G0/G1/G2 using the registered development rule, retaining G0 and earliest-generation ties. The selected checkpoint is **G1**; its tensor hash differs from G0. The audit rehashed the selected checkpoint bytes and matched the held-out runtime's actual selected-state reload record to that checkpoint. The selected state passed its original eight-prefix V4 qualification before this test. [Selection, all checkpoint metadata and actual update evidence](ES_trace.json).

Qualification uses the explicitly declared V4 local-Jacobian mathematical reference where eligible; it does not relabel the earlier native-FP32 finite-difference failures as passes. Neither the prior numerical failures nor earlier all-zero development/task-success results are removed. The update and qualification evidence establish what was run, not which component caused the score difference.

This is **one world and one policy seed**, with a historical Native run and a later Full run. No variance, confidence interval, statistical significance or general improvement claim is supported. Actions within each trajectory are dependent. Weights, uncertainty control, graph use and proposal computation differ together; this pair does not isolate an NN, graph or ES effect, and no outcome is assigned to an unexecuted alternative. The earlier p307 development cohort and p332 follow-up have different seeds and are not substituted into this pair.

The p333 source/metadata registration receipt records zero scientific calls; this release does not infer its later execution or completion. GPU occupancy and deployed source alone are not experimental results. Existing MADE **1079 valid trajectories + 1 technical failure**, including negative Full-minus-Native aggregate results, are unchanged. [Prospective metadata status](prospective_status.json).

The selected-versus-G0 weight comparison is recomputed from actual tensor hashes. [Provenance](provenance.json). Run [validate.py](validate.py) to check arithmetic, exact hashes, selected-state consistency and public-data scope.

A later [four-step mechanism audit](../../test_mechanisms/20260923T044323_CST/report.md) records the two risk-triggered PICKUP→TELEPORT changes before step24's successful identical-PICKUP candidates. It retains the successful-action alarm and repeated teleport; no local necessity or unexecuted outcome is inferred. The numerical endpoints above are unchanged.
