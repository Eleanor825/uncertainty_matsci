# Completed matched-proposal development pair — 22 September 2026, 19:24 CST

**Frozen-NN ranking did not improve this matched development case.** Common2 had 1 failed action in 10; Internal2 had 2. Both task scores were zero and neither completed the task. Fitness was −0.01 versus −0.02, an Internal2−Common2 difference of −0.01. This is one paired case; it does not establish a general effect or statistical significance.

Both conditions use Qwen3.5-4B at the same initial G0 weights, Proteomics Normal world 2, policy seed 305 and B10. Each requests exactly two proposals per step with the same public recheck feedback, proposal-seed scheme and parser. Both collect graph features and frozen-NN shadow risks. Common2 selects through the registered common hash rule; Internal2 selects by the frozen NN risk when supported and otherwise uses the same fallback. This isolates the registered ranking strategy under matched proposal allocation, rather than the conditional threshold trigger. The frozen predictor was not refitted for the comparison.

| Condition | Returned actions | Failed actions | Successful actions | Task score | Fitness | Candidate graph returns |
|---|---:|---:|---:|---:|---:|---:|
| Common2 | 10 | 1 | 9 | 0 | −0.01 | 20 |
| Internal2 | 10 | 2 | 8 | 0 | −0.02 | 20 |

The observed differences are +1 failed action, +0.10 failure fraction, zero task-score difference, −0.01 fitness, and zero candidate-graph-count difference. Each trajectory has ten action returns and ten tick returns; the pair's eight-prefix qualification passed. Candidate graph counts exclude qualification work and are not GPU-time estimates. The pair records zero parameter updates. No unexecuted proposal receives an invented outcome label.

This is a **development mechanism comparison, not Native versus Full**. It is separate from the policy-304 Q75/Posterior05 threshold pair and is not pooled with it. Common2 was already published individually; this release newly closes Internal2 and does not count Common2 again as new evaluation work. The earlier cached-risk/action-diversity audit describes Common2 only and must not be read as an Internal2 action audit. This increment does not infer its revision diversity or effective action changes from terminal outcomes.

The result preserves a negative observation for internal-risk ranking. It motivates further development diagnosis, but does not establish that grounded repair works, that the graph has no predictive information, or that the complete ES method has finished. Runtime readiness, model-load intents and incomplete training trajectories remain progress evidence outside this result table. No six-GPU scientific-computation claim is inferred from them.

[Original terminal scientific projection](terminal_source_projection.json), [paired metrics](paired_metrics.csv), [summary](summary.json), [provenance and source hashes](provenance.json), [portable checks](validate.py). The validator checks exported terminal arithmetic and counts; the underlying raw environment events are represented by their original hashes, not independently replayed here. Publishing adds zero model, graph, environment or optimizer calls. All prior dated snapshots remain unchanged.
