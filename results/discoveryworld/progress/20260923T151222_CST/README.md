# DiscoveryWorld progress — 23 September, 15:12 CST

Two frozen original-method B30 conditions have closed on world4 / policy seed336, Qwen3.5-4B:

| Condition | Score | Failed actions | Task success |
|---|---:|---:|---|
| G0 + original uncertainty controller | 0 | 4/30 | false |
| Selected ES-updated checkpoint, controller off | 0 | 18/30 | false |

The selected-checkpoint condition uses the checkpoint trained with the Full procedure; it is **not an independently ES-only-trained policy**. The corresponding Full condition is still in progress: 26/30 attempts and score0 at this snapshot. These data do not establish improvement by the original Full method.

The independent ten-episode B10 diagnostic has been recovered on six H20 GPUs. All six have fresh matching native CUDA process/PID-birth evidence and approximately32124MiB memory used per GPU. Each group returned its first of eight numerical-qualification prefixes. No benchmark episode has started at this snapshot; GPU allocation, model/qualification computation and completed benchmark results remain distinct. Raw technical attempts were retained; the first gang had zero scientific reservations or calls before termination.

The previously [completed NoGraph B30 diagnostic](../../diagnostics/20260923T125245_CST/README.md) shows a local positive equal-two-candidate contrast, with two-seed mean score0.125 for Internal2 versus0.0625 for Common2, on one world. All task-success flags are false, and the score advantage occurs at one seed only. This is separate from original Full/ES efficacy.

Component checks distinguish implementation from empirical utility. The new bounded-lock/lifetime recovery has9 passing tests and placement compatibility has10. Earlier coverage missed simultaneous metadata-lock contention and platform master-first gang teardown. Risk-network optimization itself fits its small training sample; development probability errors worsen with longer Internal-network fitting. In the [p336 conditional-controller audit](../../diagnostics/20260923T123632_CST/report.md), all28 explicit-revision alternatives fail support gating and none is selected. Thus successful proposal parsing and controller triggering do not establish a changed executed action. Failure-risk prediction and scientific progress also remain different targets; a causal explanation of the original Full outcome awaits its completed comparison.
