# Error-free actions do not establish scientific progress — 22 September 2026, 23:13:29 CST

**Both completed B30 development trajectories have task score0 and no task success.** SFT reduces official action failures from29/30 to0/30 in this fixed comparison, but its30 actions comprise28 rotations and2 moves. The observed result is a change toward error-free actions without recorded scientific progress, not successful task solving or a Full-method gain.

| Policy | Action failures | Task score | Task success | Exact unique action packets | Longest identical-packet run | Episode seconds |
|---|---:|---:|---|---:|---:|---:|
| Original G0 Native1 | 29/30 | 0 | false | 3 | 28 | 86.498 |
| Fixed final32 SFT Native1 | 0/30 | 0 | false | 6 | 2 | 76.553 |

The frozen pair uses the same Qwen3.5-4B, Proteomics Normal world2, policy seed308, original public prompt/parser/decoding, one proposal per action and B30. The original G0 runs first and the fixed32-update SFT checkpoint second with one warm model process and independent environments/histories. Initial public prompts and shared-step sampling seeds match. Both raw episodes and the process/subtree/CUDA resource closure are verified. There are **zero risk-controller calls and zero candidate graphs** in this comparison; no NN contribution is demonstrated. The SFT checkpoint's earlier8-prefix numerical qualification is separate from these graph-free episodes.

Native performs one successful teleport followed by29 failed PICKUP actions;28 have the exact same parsed packet consecutively. SFT rotates north12 times, west9, east5 and south2, and moves east/north once each. SFT therefore does **not** repeat one identical packet for30 steps: its maximum exact-packet run is2. Nevertheless all30 successful API actions have zero observed task-score change. Different directions are preserved as different packets; no identical hidden-state or information-gain claim is made.

Episode wall times exclude separate model/checkpoint loading and closure/tensor audits. This fixed-order, one-world/one-seed result does not establish a speed advantage, statistical significance, generalization, or a benefit from the risk NN. The two originally failed metadata consumers and their corrected reference handling remain separate from the scientifically completed training and pair. No trajectory or training was replayed for this export.

## Separate original training behavior audit

The original world0/policy102 training episode has81 successful and19 failed actions over100 attempts. Only steps12,19,71 increase official score, each by.125, giving final score.375. The successful actions within the eight **preceding** steps are:

| Milestone | Successful preceding steps (action type) | Milestone action |
|---|---|---|
| 12 | 4,5,7,8,9,11: TELEPORT_TO_OBJECT | PICKUP |
| 19 | 11,13: TELEPORT_TO_OBJECT;12: PICKUP;14: USE;15–18: TELEPORT_TO_LOCATION | USE |
| 71 | 63,65–70: TELEPORT_TO_LOCATION;64: MOVE_DIRECTION | USE |

Steps72–100 contain29 actions:23 successes,6 failures and no positive score transition. They include17 location teleports,10 USE actions, one move and one object teleport. This is additional evidence that action success alone can reward behavior with no observed task-score progress; it does not show that every safe action is useless. These are existing **training** rows, separate from the60 development actions. Window membership overlaps and is not a new independent sample. A score increment follows action plus world tick, so future-credit weights would be a prospective training hypothesis, not proven causal action contributions. No cross-state DPO pair or unexecuted-candidate label is constructed.

[All60 development action rows](pair_actions.csv), [pair metrics](pair_metrics.csv), [separate100-step training audit](training_p102_actions.csv), [all27 training-window memberships](training_milestone_windows.csv), [complete compact behavior projection](behavior_projection.json), [source hashes](provenance.json). No raw prompts or private runtime identities are published. Publication adds no model, graph, environment or optimizer calls.
