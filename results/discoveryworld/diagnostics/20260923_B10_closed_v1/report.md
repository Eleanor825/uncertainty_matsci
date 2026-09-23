# DiscoveryWorld world4 B10 NoGraph extension

All ten normalized scores were **0**, and none of the ten trajectories achieved task success. NoGraphRisk and ExplicitRepeatRisk matched Native1 in score and failure count at both seeds. In the registered fixed-two comparison, Internal2 matched Common2 at seed 336 but had **4 more failed actions at seed 337**, giving F differences of **0** and **−0.04**. The mean paired F difference was **−0.02**, with sample variance **0.0008**.

All 10 registered trajectories closed: five conditions at policy seeds 336 and 337, on Proteomics Normal world seed 4 using Qwen3.5-4B. No condition or negative result was excluded.

This budget extension was chosen after observing B30 on the same world and policy seeds. The B10 episodes are fresh executions in a separate namespace, but they are not a new holdout, unused-seed confirmation, or a pre-B30 hypothesis test. NoGraph conditions are not the original Full method and do not estimate the incremental value of native attribution graphs.

F = normalized score + task-success indicator - 0.1 × failure fraction. Parser rejections and returned action failures remain in the denominator. A closed execution is distinct from successful task completion.

| Condition | Policy seed | Score | F | Task success | Failures / attempts | Proposals | NoGraph forwards |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Native1 | 336 | 0 | -0.09 | false | 9 / 10 | 10 | 0 |
| NoGraphRisk | 336 | 0 | -0.09 | false | 9 / 10 | 18 | 48 |
| ExplicitRepeatRisk | 336 | 0 | -0.09 | false | 9 / 10 | 18 | 46 |
| ExplicitRepeatCommon2 | 336 | 0 | -0.05 | false | 5 / 10 | 20 | 52 |
| ExplicitRepeatInternal2 | 336 | 0 | -0.05 | false | 5 / 10 | 20 | 52 |
| Native1 | 337 | 0 | -0.05 | false | 5 / 10 | 10 | 0 |
| NoGraphRisk | 337 | 0 | -0.05 | false | 5 / 10 | 15 | 41 |
| ExplicitRepeatRisk | 337 | 0 | -0.05 | false | 5 / 10 | 15 | 40 |
| ExplicitRepeatCommon2 | 337 | 0 | -0.02 | false | 2 / 10 | 20 | 60 |
| ExplicitRepeatInternal2 | 337 | 0 | -0.06 | false | 6 / 10 | 20 | 55 |


Condition summaries over the two policy seeds (sample variance, ddof=1):

| Condition | Score mean | Score variance | F mean | F variance | Failure-count mean | Failure-count variance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native1 | 0 | 0 | -0.07 | 0.0008 | 7 | 8 |
| NoGraphRisk | 0 | 0 | -0.07 | 0.0008 | 7 | 8 |
| ExplicitRepeatRisk | 0 | 0 | -0.07 | 0.0008 | 7 | 8 |
| ExplicitRepeatCommon2 | 0 | 0 | -0.035 | 0.00045 | 3.5 | 4.5 |
| ExplicitRepeatInternal2 | 0 | 0 | -0.055 | 5e-05 | 5.5 | 0.5 |

All normalized-score variances are zero. The registered ExplicitRepeatRisk-minus-Native1 F contrast has mean and variance zero. For Internal2-minus-Common2, the failure-count difference is 0 and +4 (mean +2, sample variance 8). These two-seed summaries retain the null and negative findings.

Contrasts use right minus left. The registered primary comparisons are ExplicitRepeatRisk minus Native1 and ExplicitRepeatInternal2 minus ExplicitRepeatCommon2. The other two contrasts are explicitly descriptive. Every sign, including null and negative changes, is retained.

| Contrast | Seed | Score difference | F difference | Failure-count difference |
| --- | ---: | ---: | ---: | ---: |
| NoGraphRisk - Native1 | 336 | 0 | 0 | 0 |
| ExplicitRepeatRisk - NoGraphRisk | 336 | 0 | 0 | 0 |
| ExplicitRepeatRisk - Native1 | 336 | 0 | 0 | 0 |
| ExplicitRepeatInternal2 - ExplicitRepeatCommon2 | 336 | 0 | 0 | 0 |
| NoGraphRisk - Native1 | 337 | 0 | 0 | 0 |
| ExplicitRepeatRisk - NoGraphRisk | 337 | 0 | 0 | 0 |
| ExplicitRepeatRisk - Native1 | 337 | 0 | 0 | 0 |
| ExplicitRepeatInternal2 - ExplicitRepeatCommon2 | 337 | 0 | -0.04 | 4 |

Sample variances use ddof=1 over exactly two policy seeds. These are descriptive sampling-seed summaries within one previously used world; no population confidence interval, statistical significance, or cross-world generalization is claimed.

Completed-episode costs: 100 agent attempts, 166 generated proposals, 146 NoGraph captures, and 394 returned NoGraph forwards. Candidate graph assemblies, candidate backwards, and policy-parameter updates were zero.

Qualification is separate from episode costs: six policy loads, six full eight-prefix qualification suites (48 prefix qualifications). Qualification inference/forward costs are not inferred from these suite counts. Historical failed infrastructure launches are preserved separately and their startup costs are not folded into these completed-episode totals.

The export invokes the original aggregate.aggregate raw-closure gate, verifies the exact committed B10 registry and pinned source files, and additionally recomputes action-failure and post-closure score-progress audits. It does not rerun the launch-time training/checkpoint contract, instantiate a model/environment, query a GPU, or write a scientific completion seal. Source references and SHA-256 hashes are recorded in provenance.json.
