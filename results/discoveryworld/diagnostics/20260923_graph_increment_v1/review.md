# Independent review

**PASS.** All110 candidate rows,90 executed-action labels and20 unexecuted candidates were retained. No unexecuted candidate has an outcome label. The selected-G1 NN-off cohort keeps all30 actions with zero graph/prediction coverage.

The reviewer used a rank-sum AUROC implementation rather than the diagnostic's positive-negative pair loop, plus independent loss arithmetic and candidate ranking reconstruction. All81 metrics and paired differences agree with maximum absolute error0. The full90-row decision table and all coverage denominators agree.

A separate read-only remote audit independently verified218 source/artifact hashes, both committed protocol blobs, all three frozen model checkpoint hashes, and reconstructed90 selected-action labels from the original official action/response/tick journals. Every reconstructed label and selected-candidate index matches the CSV. Official failure counts are4/30 for G0_UQ,16/30 for Full, and18/30 for selected G1 without NN. All source episodes have zero serialization-repair events, so these labels match the frozen risk head's eligibility definition.

The transport's text reader normalized CSV line endings. Restoring the original CRLF format reproduces both remote CSV hashes exactly; no data values changed. The original report bytes also match their source hash. The review did not rerun risk inference, fit a model, reconstruct a graph, generate a policy action or call an environment.

**Interpretation remains restricted:** this is a post-outcome, exploratory prediction audit, one episode per cohort at a single world/seed, with G1 parameter/distribution shift. No pooling, row-independent significance, causal online improvement, or general superiority is supported. Internal outperforms NoGraph on both recorded trajectories, but GraphOnly outperforms Internal on G1; that result and the missing-graph cohort are published in full.
