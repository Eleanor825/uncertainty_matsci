# P335 mechanism qualification: environment diverged before controller action changes

Authenticated read-only audit observed **2026-09-23T09:49:18.594827+08:00**. All previously reported scores and the predeclared contrasts remain unchanged, but they are **observational differences, not isolated causal controller benefits**.

Across all four audited pairings, the first13 executed actions, prompt tokens, sampling seeds, proposal0 completion tokens and recorded token log-probabilities were identical. Attempt13 was the same successful `TELEPORT_TO_LOCATION` to `Possible Animal Area 1`. At the prompt for attempt14, public nearby animals/agents or accessible-object state already differed while the prior executed-action prefix still matched. This public-state difference existed before the next differing action could be selected or dispatched.

| Audited pair | Identical prefix actions | First public/prompt difference | First executed-action difference | Different completions under identical input/seed/runtime |
|---|---:|---:|---:|---:|
| Native1 vs NoGraphRisk |13|14|14|0|
| NoGraphRisk vs ExplicitRepeatRisk |13|14|14|0|
| Native1 vs ExplicitRepeatRisk |13|14|15|0|
| ExplicitRepeatCommon2 vs ExplicitRepeatInternal2 |13|14|14|0|

The primary Native1/ExplicitRepeatRisk pair still executed the same action at14 and first changed at15. The environment mismatch therefore precedes that primary action difference. The fixed-two pair's public mismatch likewise precedes any selector-induced change to its first proposal: Common2 changed its own proposal0 packet on17/19/24/25; Internal2 on21/23/25/28. Both conditional-risk arms selected the first candidate on all30attempts, so no direct executed second-proposal revision pathway was observed.

No pair showed different generations when recorded prompt tokens, seed, G0 and runtime matched. This audit does not establish an LLM numerical/RNG bug. It also does not identify the environment's exact divergence origin: initial hidden-state differences and later stochastic/NPC dynamics remain possibilities. **No Python-hash bug is claimed.** Equal world seed and initial public observation did not verify identical evolving environment state.

Consequently the preserved+0.125 primary score difference and+0.25 fixed-two ranking difference cannot isolate feedback or risk-selection effects from environmental divergence. All five task-success flags remain false. The neutral/unfavorable p332 control and utility networks' failure to beat the constant remain visible in the [complete results report](report.md). This finding strengthens the interpretation limits; it does not change the endpoint, choose another winner, alter scores or discard a result.

Further world4 runs are on hold pending the environment check; prepared sources are not reported as executed tests. Source audit and CPU replay are separate follow-up work, with no replay result assumed here. This note used0new environment, LLM, NN, optimizer or GPU calls. The [small machine-readable projection](mechanism_audit.json) records all four comparisons and source hashes; raw prompts and token values are excluded.
