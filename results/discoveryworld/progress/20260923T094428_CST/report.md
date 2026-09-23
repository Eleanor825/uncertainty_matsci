# P335 completed: a positive exploratory score contrast, with limits

Verified through **2026-09-23T09:44:28.518382+08:00**. All five p335 arms and their resources are closed. The **predeclared primary** ExplicitRepeatRisk − Native1 official task-score difference is **+0.125**. The predeclared equal-nominal-compute ranking contrast, ExplicitRepeatInternal2 − ExplicitRepeatCommon2, is **+0.250**. These are two separately specified contrasts; the best of five is not promoted to the primary result. **None of the five completed the task successfully.**

The same update retains a neutral/negative control result: p332 Internal-combined tied Common2 at0.125 task score and incurred20 versus9 failures. Separately, an expanded TRAIN corpus now supported utility fitting, but both selected utility networks lost to the constant predictor in TRAIN selection CV. No utility-online evaluation was admitted or run.

## All five p335 outcomes

World2, policy seed335, B30, frozen original G0 and saved NoGraph next-action-failure predictor; no new NN, policy or ES training in this online experiment.

| Condition | Official task score | Action failures | Task success | Proposals | NoGraph captures | Episode seconds |
|---|---:|---:|---|---:|---:|---:|
| Native1 | 0.000 | 4/30 | No | 30 | 0 (0 supported) | 117.43 |
| NoGraphRisk | 0.000 | 3/30 | No | 34 | 34 (30 supported) | 1126.70 |
| ExplicitRepeatRisk | 0.125 | 4/30 | No | 33 | 33 (30 supported) | 1176.23 |
| ExplicitRepeatCommon2 | 0.000 | 3/30 | No | 60 | 60 (53 supported) | 2091.39 |
| ExplicitRepeatInternal2 | 0.250 | 4/30 | No | 60 | 60 (49 supported) | 2062.24 |

All150 official action attempts,217 proposals and187 capture records are retained. Captures produced536 forward returns;162/187 candidates were supported. Unsupported capture records are not called available full graphs. There were0candidate graph assembly/backward calls and0parameter updates. The shared cold setup, including G0/NN/bank load and the run's own8prefix qualification, took696.63 seconds and is reported separately. Episode times include the original postepisode audit and serialization. Fixed warm order and differing subsequent histories prevent a controlled latency or end-to-end speedup claim.

The immutable analysis plan was recorded at2026-09-22 21:37:17 UTC, before this reviewer's access to any p335 outcome. Its SHA is `465801d80dacb354b778fd5cefe6c619d1ce8e02d09f611938fb48fb7e1e33aa`; the [public projection](analysis_plan_public_projection.json) preserves its endpoints. The planned feedback-package contrast ExplicitRepeatRisk − NoGraphRisk is also+0.125, but **causal attribution to feedback is not established**. Both conditional arms executed a packet different from the first proposal on0steps, despite4 and3 supported revision triggers and3 and2 different generated alternatives, respectively. The positive score difference does not show that an explicitly revised action caused progress. A separate mechanism audit is needed; same initial public prompt and proposal seeds do not certify identical hidden state, RNG evolution or entire subsequent trajectories.

The fixed-two arms each made60proposals and computed shadow risks, while the common selector used the registered hash and the internal selector used lower frozen failure risk when supported. Each executed4packets different from its first proposal. Their+0.25 score contrast is descriptive evidence for this one registered comparison, not proof of a general internal-signal effect. All shared proposal0 seeds, and proposal1 seeds wherever both were generated, matched. The0.5 threshold in conditional arms triggers another proposal; it is not hard exclusion. Explicit feedback requests a distinct packet but does not enforce semantic deduplication.

This is **one development world and one policy seed**. Variance is unavailable; no p-value, confidence interval, action-level pseudo-replication or cross-world generalization is claimed. NoGraph uses hidden/sampling/action statistics and three forwards for a supported capture; it is not the original CRV graph, a long-horizon utility controller or Full ES method. Fewer errors alone would not satisfy the primary task-score endpoint.

## P332 final control result, kept separate

The previous pending Internal-combined arm is now closed. World2/policy332 used the full graph backend and a different frozen predictor/controller comparison; these results are not pooled with p335.

| Condition | Task score | Action failures | Task success |
|---|---:|---:|---|
| Native1 |0.000|29/30|No|
| Common2 |0.125|9/30|No|
| Internal-combined (preselected) |0.125|20/30|No|

Internal-combined − Common2 is **0 task-score gain and+11 failures**, despite10 supported rank changes. Its F is0.0583333 versus Common2's0.095. All3arms/90attempts/150proposals/120candidate graphs and the resource are closed. This update supersedes the older pending status and retains the unfavorable comparison.

## New TRAIN data: all seven trajectories and all78 windows

Seven additional native G0 world0 trajectories (policy104–110) completed700 actions and700 generations. Their42 prescribed NoGraph captures returned126forwards, with no candidate assembly, NN inference, fitting or policy updates in the producer. These are TRAIN trajectories, not online validation of the utility model.

| TRAIN policy seed | Final trajectory score | Action failures | Task success |
|---|---:|---:|---|
| 104 | 0.000 | 95/100 | No |
| 105 | 0.000 | 95/100 | No |
| 106 | 0.375 | 25/100 | No |
| 107 | 0.250 | 32/100 | No |
| 108 | 0.375 | 30/100 | No |
| 109 | 0.375 | 43/100 | No |
| 110 | 0.125 | 97/100 | No |

The combined corpus preserves original36 separately executed observed-continuation windows plus42subwindows of the new native trajectories. All78 have eligible known labels. There are12positive R8 rows and66zero rows across10source episodes; positive windows occur in5sources, satisfying the prospectively fixed multi-source gate. The7new sources alone contribute4positive-window sources (106–109). Policy110 finishes at0.125 but has no positive R8 among its six prescribed slots: trajectory outcome and sampled-window coverage are different quantities.

| Source policy seed | Eligible windows | Positive R8 | Zero R8 |
|---|---:|---:|---:|
| 101 | 12 | 0 | 12 |
| 102 | 12 | 4 | 8 |
| 103 | 12 | 0 | 12 |
| 104 | 6 | 0 | 6 |
| 105 | 6 | 0 | 6 |
| 106 | 6 | 2 | 4 |
| 107 | 6 | 2 | 4 |
| 108 | 6 | 1 | 5 |
| 109 | 6 | 3 | 3 |
| 110 | 6 | 0 | 6 |

Original anchors/replicas and overlapping same-trajectory windows remain dependent. Validation groups by original source episode and purges duplicate prompt/token inputs. Loss and metrics give equal weight to source episodes, then distinct input clusters, then observations within each cluster; repeated inputs are not treated as extra independent weight. No hidden-state counterfactual, pairwise preference or unexecuted candidate outcome is fabricated. The old36-only experiment's `not_run_support` and0fits remain valid historical results; new data enabled a separately registered fit without rewriting that failed support gate.

## Utility fit passed support but did not beat the constant

CPU-only execution completed **42fits,8250optimizer updates,8370NN forwards and8250backwards**, with0LLM/environment/policy-update calls. Two final networks were sealed. The target is observed8-step task return, not next-action failure. The original failure predictor was not replaced.

The TRAIN-only,10-source macro-MSE of the per-fold fitted **constant** is **0.0022798152971**. The fixed selection rule picked NoGraph steps200/decay0.1 (MSE0.0023626563488756) and OutputAction steps50/decay0.1 (MSE0.0024845785619476). **Both are worse than the constant**. Selection used the same finite grid and source folds; these selected-CV scores are optimistic selection estimates, not independent generalization results. No dev/test data selected these hyperparameters and no online utility outcome exists.

| Family | Weight decay | Steps | Macro-source CV MSE | Selected by fixed rule | Beats constant |
|---|---:|---:|---:|---|---|
| NoGraph | 0.01 | 50 | 0.0026705047 | No | No |
| NoGraph | 0.01 | 100 | 0.0024954421 | No | No |
| NoGraph | 0.01 | 200 | 0.0023638331 | No | No |
| NoGraph | 0.1 | 50 | 0.0026695348 | No | No |
| NoGraph | 0.1 | 100 | 0.0024966531 | No | No |
| NoGraph | 0.1 | 200 | 0.0023626563 | Yes | No |
| OutputAction | 0.01 | 50 | 0.0024860493 | No | No |
| OutputAction | 0.01 | 100 | 0.0027569832 | No | No |
| OutputAction | 0.01 | 200 | 0.0026431065 | No | No |
| OutputAction | 0.1 | 50 | 0.0024845786 | Yes | No |
| OutputAction | 0.1 | 100 | 0.0027540553 | No | No |
| OutputAction | 0.1 | 200 | 0.0026344133 | No | No |

All12grid candidates,120fold metrics, individual out-of-fold predictions, the10constant-fold records and all78labels are exported. This current result is **training completed but failure to outperform a simple predictive baseline**, distinct from the earlier **insufficient support,0fit** result. It does not justify activating either utility model.

[P335 five rows](p335_results.csv), [all150 action audit rows](p335_actions_all150.csv), [source-bound P335 evidence](p335_evidence.json), [P332 final rows](p332_final_results.csv), [all7 TRAIN outcomes](native7_all_results.csv), [all78 TRAIN rows](TRAIN78_all_rows.csv), [source support](TRAIN78_source_support.csv), [all12 CV choices](utility_all12_grid.csv), [all120 fold metrics](utility_all120_fold_metrics.csv), [all predictions and model seals](utility_all_predictions_and_seals.json), [snapshot](snapshot.json), [provenance](provenance.json), and [portable validator](validate.py).

Publication created0new scientific calls. Raw prompts, host/process identities and authentication material are excluded; source hashes identify retained private artifacts. All earlier results remain available unchanged.
