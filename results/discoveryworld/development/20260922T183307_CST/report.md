# DiscoveryWorld development status — 22 September 2026, 18:33 CST

Two separate development conditions have closed. **Neither within-seed pair is complete**, so this snapshot does not establish a controller benefit. Both completed trajectories finish with zero task score. All results concern Qwen3.5-4B, Proteomics Normal, world 2 and a ten-action budget.

| Development condition | Policy seed | Actions returned | Failed actions | Successful actions | Task score | Task completed | Trajectory fitness |
|---|---:|---:|---:|---:|---:|---|---:|
| Q75, original calibrated-quantile controller | 304 | 10 | 10 | 0 | 0 | No | −0.10 |
| Common2, two proposals with hash selection | 305 | 10 | 1 | 9 | 0 | No | −0.01 |

Q75 uses the original calibration 75th-percentile risk threshold. It is **not Native and not a Full-method test**. Its supported controller-activity gate is true, but every executed action failed. Activity alone does not imply successful repair. Common2 makes exactly two proposals and selects using the frozen common hash rule; successful individual actions did not produce task-score progress in this trajectory. The fitness formula remains score + task-completion indicator −0.1 × action-failure fraction, with calibration penalty zero.

The two rows use different policy seeds and are **not an effect comparison**. They are development observations from one scenario/world, not held-out benchmark results. Individual actions within a trajectory are not independent evaluation replicates.

At the same snapshot, fixed-0.5 Posterior05 on policy seed 304 had 7 returned actions, all failed, and had not completed its B10 episode. Internal2 on policy seed 305 had 1 returned successful action and had not completed its episode. These prefixes are reported only as partial status; no final score, final fitness, pairwise improvement or confidence interval is assigned. Each warm pair's actual eight-prefix graph qualification had passed. Full ES v5 still waited for the operating-point prerequisite, with no new parameter update confirmed. Grounded V6 was registered and waiting for its resource predecessor; process launch and waiting metadata are not counted as GPU scientific execution.

## Cross-validation did not run

The preregistered episode-held-out CV-v1 stopped with `not_run_insufficient_fold_support`, before any fitting: **0 fold fits, 0 final fits and 0 optimizer updates**. Holding out the policy-102 training episode leaves 33 positive and 7 negative training rows. The original gate requires at least 10 rows and 2 episodes in each class. No failing fold was removed and no gate was relaxed. The other two training folds contain 15/25 and 20/20 positive/negative rows. Held-out single-episode fold scores, had they been computed, would have been diagnostics rather than formal controller admission.

## Label-only feasibility is not graph or fit completion

A separate read-only reconstruction examined only the existing world-0 training trajectories for policies 101/102/103. It fixes the same eight additional early positions per episode: 2, 3, 4, 5, 7, 8, 9 and 10. This would supplement the original 60 fitting rows with at most 24 rows; the unchanged calibration set is separate. The positional rule includes every listed position regardless of its label. Label counts suggest the previously insufficient fold could reach 42 positive/14 negative rows **if all required graph records are successfully extracted and eligible**.

At this snapshot, **0 of those 24 extra graphs had been extracted, no enriched 84-row dataset was ready, and CV-v1 had not been retried**. No new environment call, policy generation or NN update was performed by the feasibility audit. The production 200-update predictor and temperature remain frozen. This prospective supplement is not evidence that cross-validation succeeded.

The export adds no scientific calls. Portable checks cover the published arithmetic, statuses, support counts and hashes; they do not independently reconstruct trajectories from the private raw environment logs. [Metrics](completed_development_metrics.csv), [status](status.json), [CV stop record](cv_not_run.json), [label-only support](training_prefix_feasibility.json) and [provenance](provenance.json) retain the unfavorable and unavailable outcomes.
