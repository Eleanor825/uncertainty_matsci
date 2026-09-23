# DiscoveryWorld progress — 23 September, 14:55 CST

The original frozen-G0 uncertainty-controller condition (`G0_UQ`, Qwen3.5-4B, world4, policy seed336, B30) has closed: **30 agent attempts, normalized task score0, 4 failed actions, task success false**. This is one closed trajectory, not the complete factorial comparison or evidence of successful task completion.

The corresponding Full condition using the selected ES-updated checkpoint was at **24/30 attempts, score0** at this snapshot. This is in progress. The selected-checkpoint-with-controller-off condition remains part of the same continuing developer-node job. No new ES updates or risk-network fitting occur during this frozen evaluation.

The additional independent B10 ten-trajectory matrix has **no scientific results yet**. Its six-Pod launch encountered a metadata-lock contention bug; four groups stopped before scientific admission, then CloudML removed the other two Pods after the master exited. A read-only audit found no scientific reservations, group outputs, episode outputs, model-load intents, qualification calls, or environment calls for all six groups. Original technical-attempt records remain intact. A resource-only recovery is being prepared; no B10 improvement is claimed.

This progress report supplements the [completed seeded NoGraph B30 diagnostic](../../diagnostics/20260923T125245_CST/README.md). Do not pool the NoGraph diagnostic with the original Full evaluation: their observation and controller procedures differ.
