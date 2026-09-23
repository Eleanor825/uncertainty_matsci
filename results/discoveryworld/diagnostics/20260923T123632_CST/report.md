# Why p336 revisions did not change the selected action

**Recorded-evidence diagnostic, observed 2026-09-23 12:36:32 CST.** This audit covers the two already closed seeded world 4 / policy seed 336 / B30 NoGraph arms. It explains their recorded selection paths; it is not a new method evaluation.

| Closed condition | Generated alternatives | Parsed alternatives | TC fidelity gate failures | Supported risk comparisons | Alternatives selected |
|---|---:|---:|---:|---:|---:|
| NoGraphRisk | 28 | 28 | 6: layer 13 × 1, layer 16 × 5 | 22 | 0 |
| ExplicitRepeatRisk | 28 | 28 | 28: layer 12 × 28 | 0 | 0 |

Both arms made no revision at their first two attempts and selected candidate 0 at every one of their 30 attempts. In ExplicitRepeatRisk, all 28 alternatives failed the transcoder fidelity gate and followed the recorded incomplete-support fallback. **No supported risk comparison of these alternatives occurred.** Their missing risk values remain null; they were not recomputed.

NoGraphRisk has a different mechanism: 6 alternatives followed the same support fallback, while all 22 supported comparisons assigned the alternative a higher recorded risk and retained candidate 0. The 6 recorded gate FVU values span 0.500707–0.506281; ExplicitRepeatRisk’s 28 span 0.501880–0.524826. These are observed rejection values under the frozen gate, not a justification for relaxing it.

All 56 alternative generations parsed successfully. The saved `parsed_action_changed` flags and related counts compare JSON structures, **not environment-normalized action semantics**: string and integer object IDs may normalize to the same action. Those flags therefore do not establish distinct executed actions, useful revisions, or missed successful alternatives. No unexecuted alternative was given an outcome label, and this audit cannot say that selecting it would have succeeded.

The conclusion is confined to this NoGraph procedure and these two completed episodes. Original full-graph ES behavior must be reconstructed and tested with its own evidence; this diagnostic does not establish its failure mechanism or efficacy. Evaluation of a revised method has not been executed. No FVU threshold, controller, NN, model, or scientific registration was changed.

[Actual recorded evidence, including all 60 decisions](evidence.json) · [Exact read-only audit source](audit_source.py) · [Source validation metadata](source_validation.json) · [Portable evidence validation](validate.py) · [Earlier 9/10 partial progress snapshot](../../progress/20260923T122847_CST/README.md)

The source receipt records stable reads of 236 existing files / 57,083,614 bytes, with zero new LLM, NN, environment, optimizer or GPU-query calls. The public evidence contains existing risks, support outcomes, hashes and parsed actions; raw prompts, tokens, activations and private task answers/maps are excluded. The audit source is retained for provenance and is not executed by the portable validator. Earlier dated results and the original incomplete Full status remain intact.
