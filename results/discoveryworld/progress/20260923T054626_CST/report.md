# TRAIN36 support gate and p332 partial development snapshot

Verified through **September23,05:46:26 CST**, with each component timestamp retained below. TRAIN36 finished, but the prospectively fixed utility-support gate rejected fitting: **36/36 eligible windows,0fits,0NN updates, no utility models sealed**. Separately, p332 is still **2/3 arms closed**; its preselected Internal-combined arm is pending. These facts supersede older running/queued status, while all previous exports remain unchanged.

## Closed TRAIN36 collection; no utility fit

| Original world0 source | Closed / eligible windows | Positive task-gain windows |
|---|---:|---:|
| policy101 |12/12|0|
| policy102 |12/12|4|
| policy103 |12/12|0|

All36 independent windows and their original resource are closed. Actual work:36 root generations +252 continuation generations,36 root graphs,288 executed window attempts,834 prefix-replay actions and36 bootstrap ticks/environment initializations. These are training windows, not36 independent worlds or test episodes.

Positive task gain is supported by only **one** original source trajectory. The fixed gate required at least two; additionally, the leave-source102-out training fold has no target variation after duplicate-input purging. The registered result is `not_run_support`. It is a valid fail-closed data-support result, **not a failed fitted NN or evidence that a trained utility model is ineffective**. All optimizer/NN-forward/NN-backward counts are0. The gate, labels and split were not loosened, and no utility-online result exists.

## p332: two closed arms, primary Internal arm pending

World2 / policy332 / B30; frozen G0. Snapshot time05:35:44 CST.

| Condition | Status | Actions | Official task score | Failed actions | Task success |
|---|---|---:|---:|---:|---|
| Native1 |Closed|30/30|0|29/30|No|
| Common2 |Closed|30/30|0.125|9/30|No|
| Internal-combined (preselected) |Pending|12/30|Not final|11/12 so far|Not final|

Common2's observed score exceeds Native1 by0.125 in this one case, with20 fewer action failures. Common2 uses the original two-proposal/hash control and public feedback; its NN evaluations are shadows, not neural ranking decisions. This does **not** establish an Internal/Full NN benefit, task completion, statistical significance or heldout generalization. It is not a same-state counterfactual. The primary Internal-combined row remains unclosed; no final score, F, task-success flag or final failure rate is imputed for it, and Common2 is not promoted to the primary.

Both closed rows have accepted original RPC closure. Native1 used30 proposals/0candidategraphs; Common2 used60proposals/60candidategraphs. Internal-combined currently has26generated candidates and25graph returns from26intents, with only12executed actions. The group and resource have not completed. All three planned arms remain in the export. A single world/seed pair provides no seed-variance estimate; action steps are not independent trials.

## Completed CPU v2: failure probability is not long-horizon value

At05:42:25 CST, the corrected independent CPU audit scored all36 actually executed TRAIN roots in one batch, with0updates/0LLM calls/0environment calls and no GPU initialization. The original v1 failure remains preserved. All32 zero-return windows and all4 positive windows are exported; no successful-looking subset replaces the corpus.

| Positive-R8 window (world0/source102) | Observed R8 | Frozen V4 failure probability | Below0.5 |
|---|---:|---:|---|
| after11, replicaA |0.125|0.7513903975|No|
| after11, replicaB |0.125|0.7409842014|No|
| after18, replicaB |0.125|0.0117520243|Yes|
| after70, replicaA |0.125|0.1997977197|Yes|

Thus2/4 positive-return roots lie below the hypothetical strict0.5 cutoff. This does **not** mean an existing controller hard-rejected the other two: p335 uses0.5 to trigger an extra proposal and keeps the first candidate selectable. This audit uses the original V4 Internal failure predictor, not p335's separate NoGraph checkpoint. A root action may fail while later continuation produces progress; observed8-step return is not root-action success or causal advantage. The four positives are dependent anchors/replicas from one original trajectory, so50% is descriptive coverage, not a population estimate or evidence of online improvement. It reinforces the distinction between immediate failure risk and long-horizon task value without changing either controller or the frozen support gate.

## NoGraph preflight completed; p335 is at its own qualification

At05:46:26 CST, NoGraph preflight and its resource were complete: original8/8 qualification and all3 fixed TRAIN comparisons passed, with9candidate forward returns,0candidate assembly/backwards and no model updates. P335 had started its scientific process and loaded G0, but its own required qualification had1intent/0returned prefixes,0environment actions and0closed arms. No p335 outcome is reported.

| Fixed TRAIN prefix | NoGraph extraction seconds | Historical full-graph seconds |
|---|---:|---:|
| p101/action2 |31.1284|144.0409|
| p102/action47 |32.0687|159.0847|
| p103/action100 |36.3630|182.1544|

These are historical per-candidate microtimings, **not a controlled end-to-end speedup**. NoGraph retains3full forwards and32TC fidelity work per candidate; it skips candidate graph assembly/backwards. Cold load/8prefix qualification and later online costs are not included in this table. Passing these3fixed TRAIN comparisons is not online efficacy or full-graph equivalence.

[Machine-readable snapshot](snapshot.json), [p332 status CSV](p332_status.csv), [TRAIN source-support CSV](TRAIN36_support.csv), [all36 gate-audit rows including zeros](gate_audit_all36.csv), [audit definitions](gate_audit.json), [provenance](provenance.json), and [portable validator](validate.py). Raw prompts, process/host identifiers and authentication material are excluded; scientific raw artifacts are represented by source SHA references. Publication adds0new scientific calls. All earlier MADE and DiscoveryWorld exports remain unchanged.
