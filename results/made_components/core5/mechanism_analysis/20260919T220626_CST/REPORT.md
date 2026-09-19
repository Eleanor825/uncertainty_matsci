# MADE core60: observed controller paths, tool choices and limits of attribution

The complete fixed set is **4 arms × 5 core systems × seeds2/3/4, all B10**:60 accepted trajectories,600 candidate ORB steps,1,237 proposals and1,174 executed policy tools. The read-only extraction has no gaps. This is not the earlier17-system/seed2 comparison. All results below use the full60; no negative/positive case was removed and no threshold, checkpoint or model was selected from these test outcomes.

The principal observed distinction is: **no valid scalar NN probability reaches0.6**; UQ has no actual second proposal caused by a neural trigger and no valid-pair neural reranking. Full does produce8 extra proposals from typed NN triggers, but keeps the original candidate in all8. This does **not** establish that neural computation has no causal effect or that all gains come from schema rules. The traces are not a complete intervention disabling NN/graphs, and even identical initial G0 actions sometimes produce different recorded Chemeleon candidate pools.

## Complete results and execution counts

The actual profiles confirm baseline/UQ use θ0/G0 (`6a308686…6516`); Full uses the new G2 (`d97a518f…4b85`); independent ES-only selects G1 (`d5204b69…8df8`). These are MADE identities, distinct from SnAr's selected-G0 result.

| Arm | Trajectories | SUN sum | Mean AUDC | Proposals / executed tools | Failed select calls | Generate / score / query calls | Recorded MACE attempts |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 15 | 22 | 0.156000 | 330 / 319 | 26 | 108 / 31 / 3 | 1,787 |
| Independent ES-only | 15 | 17 | 0.131333 | 303 / 292 | 33 | 91 / 18 / 0 | 1,003 |
| UQ support-aware | 15 | 25 | 0.192667 | 339 / 314 | 13 | 106 / 25 / 12 | 1,560 |
| Full support-aware | 15 | 20 | 0.144000 | 265 / 249 | 24 | 63 / 5 / 5 | 190 |

Each arm has150 candidate ORB and189 initialization ORB attempts. MACE entries are separately counted existing `oracle_evaluation/role=mace` journal events. Their counts match every present cost field; the absent surrogate-cost field remains null in3/2/4/11 baseline/ES/UQ/Full trajectory records. Zero journal events in such a trajectory is independent observed evidence, not an imputed missing cost. No additional oracle was called in this analysis.

| System, all3 seeds | Baseline SUN / meanAUDC | ES-only | UQ | Full |
|---|---:|---:|---:|---:|
| Al-Li-V | 3 / .076667 | 1 / .050000 | 1 / .030000 | 2 / .093333 |
| Al-V-Zn | 15 / .616667 | 12 / .480000 | 15 / .616667 | 11 / .410000 |
| Au-K-Tb | 4 / .086667 | 3 / .123333 | 6 / .206667 | 4 / .146667 |
| Co-Dy-W | 0 / .000000 | 0 / .000000 | 3 / .110000 | 1 / .043333 |
| Co-Mg-Na | 0 / .000000 | 1 / .003333 | 0 / .000000 | 2 / .026667 |

| Seed, all5 systems | Baseline SUN / meanAUDC | ES-only | UQ | Full |
|---|---:|---:|---:|---:|
| 2 | 8 / .152 | 6 / .160 | 10 / .224 | 6 / .100 |
| 3 | 7 / .154 | 6 / .160 | 11 / .238 | 7 / .194 |
| 4 | 7 / .162 | 5 / .074 | 4 / .116 | 7 / .138 |

`trajectories.csv`, `per_system_arm.csv`, `per_seed_arm.csv` and `paired_results.csv` retain every job, system/seed result and sample variance. The arm-level variance mixes systems and is descriptive; it is not an independent-action confidence interval.

## NN values, graph support and actual retries

Baseline's319 scalar values are logging-only entropy NN predictions, not controller inputs. ES-only records no scalar NN. UQ has325 valid-generation raw NN values in[0.4989052713,0.5012179017]; Full has258 in[0.4986876249,0.5012679100]. Every valid supported scalar is below the registered0.6 threshold. Invalid-generation rule values of1 are excluded from this NN statement, including one Full invalid generation with a technically successful graph.

Graph status is UQ309 succeeded /30 unavailable of339 proposals; Full220/45 of265. Baseline and ES have no graph route, so their absent graph statuses are not graph failures. Unavailable support masks decision-time scalar/types and prevents risk-based cross-candidate preference; raw predictions remain diagnostic. The `type_signals_used` flag can be true when there is only one candidate or one schema-valid candidate. That flag alone is not an intervention or changed selection.

| Controller event | UQ | Full |
|---|---:|---:|
| Retry request records | 28 | 17 |
| Local schema requests | 25 | 8 |
| Typed NN requests | 3 (`not_new`) | 9 (`not_new`7, `unstable`1, `tool_execution_failure`1) |
| Actual two-proposal sequences | 24 | 16 |
| Actual second proposal after local schema trigger | 24 | 8 |
| Actual second proposal after typed NN trigger | 0 | 8 |
| Executed second candidate | 23 | 8 |
| Executed second after first schema-invalid | 23 | 8 |
| Changed choice away from first valid candidate in a valid pair | 0 | 0 |

UQ's three `not_new` requests occur **at candidate1, with no remaining slot**: Al-V-Zn seed2 `support-final-50481d7de104501120b84fb5:d0000003-c1`, Al-V-Zn seed3 `support-final-5d1cc6924d5efee9cbf9f4a0:d0000002-c1`, and Co-Dy-W seed3 `support-final-97076161f2eb1dff20b02ffe:d0000002-c1`. They did not produce a third proposal. One UQ two-proposal sequence has two invalid generations and no executed action. All23 executed second candidates follow a genuinely schema-invalid first action; none replaces an explicitly schema-valid first candidate. Unknown schema metadata is not treated as false.

Full's8 typed NN retries have two schema-valid proposals. In7, the second graph fails the current-policy layer13 FVU gate, so the registered support contract disables the comparison and chooses the first valid candidate by original order. The remaining double-supported pair (Au-K-Tb seed2 sequence5) also ranks candidate0 first. All8 extra candidates remain unexecuted. They consumed8 LLM calls,17,972 prompt tokens,917 completion tokens,90.688 seconds of recorded generation timing and139.209 seconds of recorded graph timing. These timing fields are not a measured whole-run counterfactual speedup. The extra forward/graph work may affect state/RNG/context or timing; retaining candidate0 does not prove that disabling the extra computation yields an identical future trajectory.

Every case, candidate index, requested versus actual retry, graph status, selected index and decision ID is retained in `sequence_effects.csv`, `controller_requests.json` and `neural_extra_candidates.json`.

## What candidate2 actually receives

All24 UQ candidate1 records receive the same fixed schema-repair instruction, `predicted_failure_type="generation_invalid"`, an advisory tool computed from the visible buffer/action, and source/“not measured” metadata. They contain **no scalar or type-probability numbers**. The frozen prompt builder adds exactly this dictionary at `decision_risk_feedback`; it does not add the row's diagnostic NN vector. For the local-schema trigger, the suggested tool is produced by deterministic `_safe_query`, not by a numerical NN score. Counts/compositions in that suggested tool are action arguments, not neural probabilities.

The Full NN-triggered second candidates receive typed feedback corresponding to the observed threshold trigger. Thus neural predictions do affect those8 local generation prompts even though the original candidate is ultimately retained. The evidence permits exact statements about these recorded paths, not the causal claim “UQ gains come only from rules.” A prospective schema-only control with the same θ0 and runtime remains necessary.

## Failed selection and concrete positive/negative cases

Full has24 failed select calls, versus UQ13, baseline26 and ES-only33. Full's selection errors are23 “selected hash is not in this composition buffer” plus1 nonexistent composition. UQ has8 absent-hash,4 nonexistent-composition and1 out-of-range-index errors. These are actual failed tool calls, distinct from local schema rejection before a tool runs. They consume tools/time, not a new candidate ORB step. Each arm still completes150 physical steps. `tool_failures.csv` records the exact tool request and later associated outcome; that later outcome is not assigned causally to the failed action.

**Al-V-Zn Full deficit is not uniform across metrics.** Full versus UQ for seeds2/3/4 is SUN4/3/4 versus5/7/3, and AUDC.42/.47/.34 versus.71/.73/.41. Full loses AUDC at all three seeds, but has one more terminal SUN at seed4. Across all3 seeds Full never calls score_buffer and has0 MACE events; UQ has2 score calls/67 MACE events, all at seed2. UQ's seed3 improvement also occurs without MACE, so absence of screening alone cannot explain the Full deficit.

- **Seed2:** Full `support-final-ca740c106b65301c3c20f121` plateaus atSUN1 on steps2–4. `d0000004-c0` and `d0000005-c0` select structures that are stable but not newly discovered. UQ `support-final-50481d7de104501120b84fb5` reachesSUN3 by step3 (`d0000009-c0`) andSUN5 by step6 (`d0000013-c0`). These are different generated/selected structures, not outcomes for interchangeable counterfactual actions.
- **Seed3:** Full `support-final-c3b9e4053e380cd376460042` reachesSUN3 at step4 (`d0000010-c0`) and never increases thereafter. Later `d0000013-c0` and `d0000017-c0` are stable but not new. UQ `support-final-5d1cc6924d5efee9cbf9f4a0` reachesSUN7 by step9 (`d0000013-c0`), with distinct accepted structure hashes even when formulas repeat.
- **Seed4 negative UQ case:** UQ `support-final-b8ea68c4cba1f93c238e3acd` endsSUN3/AUDC.41, versus baseline `component-final-70adf56898316f73ea4720ab` SUN6/.66. UQ has3 failed selects versus baseline1; UQ `d0000007-c0` is stable but not new, and `d0000011-c0` is neither stable nor new. Across all5 systems, seed4 UQ totalsSUN4/.116 versus baseline7/.162; its gains at Co-Dy-W do not erase the Al-Li-V/Al-V-Zn losses.
- **Positive UQ examples remain included:** Au-K-Tb seed3 (`support-final-2b6cfd7c97c158c305971c45`) finds3 versus baseline0; its directly verified discoveries include `d0000004-c0`, `d0000006-c0`, `d0000021-c0`. Co-Dy-W seed2 (`support-final-5eccea7f5a24da2f5b31b4ab`) finds2 versus baseline0, at `d0000004-c0` and `d0000019-c0`. Co-Dy-W seed4 finds1 at `support-final-c16688941b047acff551e984:d0000003-c0`. Their presence does not identify a neural causal contribution.

`direct_material_outcomes.csv` contains all600 direct select→ORB associations with formula, candidate hash, stable/new flags and the actual SUN increment. A future-label horizon and a change in the current discovery curve are different quantities; an earlier generate/query/failed select may share the later physical label. No outcome is invented for an unexecuted proposal.

## Identical initial G0 action does not establish paired physical randomness

Al-V-Zn seed4 baseline and UQ have the same initial `d0000000-c0` full action, generated-text SHA, complete-prefix hash, LLM seed40000076 and generation configuration. Both issue the same Chemeleon request and receive generated20/accepted20, but their recorded candidate-hash lists already differ (first baseline `af3b0686…`, UQ `97927d60…`). This precedes later action divergence and any retry in this UQ trajectory. Candidate identifiers are the observed fact; this comparison does not yet isolate seeding, geometry, cache, nondeterminism, or a specific bug.

Across the15 baseline/UQ pairs,9 initial full actions match;7 pairs both return candidate-hash lists, and those lists differ in all7. Complete-prefix hashes match in5 pairs. Therefore paired evaluation seed does not, on this evidence, mean an identical realized material-generation stream. The fixed-seed results remain valid descriptive outcomes, but causal controller attribution must account for this additional pairing limit. `initial_G0_pairing.json` preserves exact IDs, seeds, configuration equality and hash lists without exporting geometry.

## Scope and verification

The analysis pins raw gzip `6430dde5…19a05`, extraction summary `7e46c8a2…1ae7`, all60 source/receipt references, and the actual frozen controller/rollout source hashes in `source_contract.json`. It verifies the60-member matrix, each direct tool/step association, proposal/sequence counts, original no-replay flags, every B10 SUN endpoint and normalized trapezoidal AUDC. It performs no new model, graph, oracle, fit, source modification or public push.

These are heldout case descriptions. Any new model, threshold, schema-only control, or collection choice must be registered from train/dev reasoning; the present positive/negative test cases cannot be used to tune them. No claim of neural benefit, neural irrelevance or “rules explain all gains” is established by this audit.
