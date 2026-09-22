# Grounded repair and training diagnosis — 22 September 2026, 21:16 CST

**No scientific-task improvement is established.** The four grounded-repair development trajectories are complete. They use the same initial Qwen3.5-4B, Proteomics Normal world 2, policy seed 306, B10 and frozen original risk predictor.

| Condition | Failed actions / 10 | Task score | Fitness | Candidate graphs |
|---|---:|---:|---:|---:|
| GenericRisk | 7 | 0 | −0.07 | 17 |
| GroundedRisk | 7 | 0 | −0.07 | 17 |
| GroundedCommon2 | 0 | 0 | 0 | 20 |
| GroundedInternal2 | 5 | 0 | −0.05 | 20 |

GenericRisk/GroundedRisk use the same conditional risk trigger; GroundedRisk adds explicit public error and prerequisite feedback. Both make zero actual changes from first-proposal actions. GroundedCommon2/GroundedInternal2 both allocate two proposals and the same grounded feedback; their selection rules differ. Internal-risk ranking is worse in this paired case. Zero failed actions in Common2 does not imply task progress. These are two mechanism pairs, not four Native/Full comparisons. Neither cross-seed pooling nor statistical significance is claimed. All four graph qualifications passed; the experiment updates no LLM parameters.

## Training and parameter updates

The fixed early-prefix supplement completed 24 graphs, forming 84 fitting rows. Train-episode CV2 completed 12 fold fits and 2 final fits, totaling 2,525 optimizer updates in 86.26 seconds. It selected Internal 25 updates / weight decay 0.0001 and OutputAction 100 / 0.01 using training episodes. On the original 40 development rows, calibrated Internal AUROC/Brier/NLL is 0.960 / 0.068870 / 0.220769; OutputAction is 0.910 / 0.110689 / 0.355796. This is not better than the original 60-row Internal fit (0.970 / 0.064438 / 0.199944), and both training data and model-selection procedure changed. The small development sample contains only two episodes in one world.

The original Full ES run separately completed its first actual parameter update: 723 tensors changed, total L2 change 23.8206066. Both training-member task scores were zero; fitness −0.02 versus −0.04 came only from action failures. Its completed G1 development trajectory has 9/10 failed actions versus G0 10/10, while both task scores remain zero. This one-action difference is not evidence of robust task improvement. G2 has entered its first training member. There is no completed held-out Full/Native pair at this snapshot.

Three previously missed, genuinely score-increasing training actions now have qualified graphs. A new eight-fit data-coverage comparison did start its CPU preparation, but stopped before registration or fitting because the consumer assumed a dense counter field that the producer omits at zero. This technical failure is preserved; new optimizer updates are zero. A corrected independent run is being prepared. No new risk checkpoint has been substituted into the original Full run.

The next planned development comparison uses four fixed B30 arms: Common2, Internal-base84, Internal-combined and OutputAction-combined. It is not yet a result. B30 is chosen before these runs because the first observed training score milestone occurred at step 12; B10 can test immediate failures but may miss longer task progress.

[Terminal metrics](V6_metrics.csv), [scientific source projection](terminal_source_projection.json), [training/update summary](summary.json), [receipt hashes](provenance.json). Publication introduces no model, graph, environment or optimizer calls. Historical positive, tied, negative and failed records remain unchanged.
