# Seeded world 4: complete ten-episode diagnostic

Observed **2026-09-23 12:52:45 CST**. **All 10 registered B30 episodes and both resource groups are closed and validated.** The unchanged exporter reports `all_ten_complete=true` and `publication_ready=true`. This is the seeded NoGraph diagnostic on one Proteomics Normal world (world seed 4), using Qwen3.5-4B and policy sampling seeds 336/337. It is separate from the original full-graph/Agentic-ESOpt follow-up.

The conditional NoGraphRisk and ExplicitRepeatRisk arms matched all 30 Native1 actions at each seed and scored 0. Equal-two-proposal Internal2 scored 0.25 versus Common2's 0.125 at seed 336, with 10 versus 9 failed actions; both scored 0 at seed 337, with 4 versus 21 failures. **All 10 task-success flags are false.** The positive two-seed mean score contrast is descriptive evidence in this world; it does not establish robust improvement, generalization or a complete full-method result.

[Unmodified source-bound snapshot](snapshot.json) · [Original renderer report](report.md) · [Renderer paired statistics](complete_two_seed_statistics.json) · [All arm and paired statistics](descriptive_statistics.json) · [Portable validation](validate.py)

## All closed outcomes and recorded costs

Each episode made 30 agent attempts. `F = scoreNormalized - 0.1 × failure_fraction`. Failed actions are action-return failures, not task-success flags. Counts below are recorded costs of the completed experiments; this publication made zero scientific calls.

| Policy seed | Condition | Final score | Failed actions | F | Task success | Generated proposals | NoGraph forward returns |
|---|---|---:|---:|---:|---|---:|---:|
| 336 | Native1 | 0 | 29/30 | -0.0966666666667 | false | 30 | 0 |
| 336 | NoGraphRisk | 0 | 29/30 | -0.0966666666667 | false | 58 | 168 |
| 336 | ExplicitRepeatRisk | 0 | 29/30 | -0.0966666666667 | false | 58 | 146 |
| 336 | ExplicitRepeatCommon2 | 0.125 | 9/30 | 0.095 | false | 60 | 155 |
| 336 | ExplicitRepeatInternal2 | 0.25 | 10/30 | 0.216666666667 | false | 60 | 155 |
| 337 | Native1 | 0 | 2/30 | -0.00666666666667 | false | 30 | 0 |
| 337 | NoGraphRisk | 0 | 2/30 | -0.00666666666667 | false | 30 | 90 |
| 337 | ExplicitRepeatRisk | 0 | 2/30 | -0.00666666666667 | false | 30 | 90 |
| 337 | ExplicitRepeatCommon2 | 0 | 21/30 | -0.07 | false | 60 | 159 |
| 337 | ExplicitRepeatInternal2 | 0 | 4/30 | -0.0133333333333 | false | 60 | 164 |

Across all 10 episodes: **300 agent actions, 476 generated proposals and 1127 recorded NoGraph forward returns**. Candidate graph assembly/backward calls are zero. Native1 and conditional-risk generation counts differ, whereas Common2 and Internal2 each use exactly 60 proposals at each seed. Thus Internal2/Common2 is the equal-two-proposal ranking comparison; supplementary comparisons to Native1 also change proposal/control conditions.

## Two sampling seeds: means, sample variances and SDs

Every row below has **n = 2** policy sampling seeds, **one world**. Sample variance uses `sum((x - mean)^2)/(n - 1)`, with denominator 1. SD is the square root of that sample variance. These are not standard errors, confidence intervals or repeated-world estimates. JSON retains the full numerical precision; displayed decimals use 12 significant digits.

| Condition | Mean score | Score sample variance | Score sample SD | Mean failed actions | Failure sample variance | Failure sample SD |
|---|---:|---:|---:|---:|---:|---:|
| Native1 | 0 | 0 | 0 | 15.5 | 364.5 | 19.091883092 |
| NoGraphRisk | 0 | 0 | 0 | 15.5 | 364.5 | 19.091883092 |
| ExplicitRepeatRisk | 0 | 0 | 0 | 15.5 | 364.5 | 19.091883092 |
| ExplicitRepeatCommon2 | 0.0625 | 0.0078125 | 0.0883883476483 | 15 | 72 | 8.48528137424 |
| ExplicitRepeatInternal2 | 0.125 | 0.03125 | 0.176776695297 | 7 | 18 | 4.24264068712 |

Paired differences are computed within the same sampling seed before summarizing. Higher score and fewer failures are favorable. The Common2/Native1 and Internal2/Native1 rows are supplementary descriptive comparisons, not a replacement for the registered equal-two-proposal comparison.

| Right minus left | Metric | Seed 336 | Seed 337 | Mean difference | Sample variance | Sample SD |
|---|---|---:|---:|---:|---:|---:|
| NoGraphRisk - Native1 | scoreNormalized | 0 | 0 | 0 | 0 | 0 |
| NoGraphRisk - Native1 | failure_count | 0 | 0 | 0 | 0 | 0 |
| ExplicitRepeatRisk - Native1 | scoreNormalized | 0 | 0 | 0 | 0 | 0 |
| ExplicitRepeatRisk - Native1 | failure_count | 0 | 0 | 0 | 0 | 0 |
| ExplicitRepeatInternal2 - ExplicitRepeatCommon2 | scoreNormalized | 0.125 | 0 | 0.0625 | 0.0078125 | 0.0883883476483 |
| ExplicitRepeatInternal2 - ExplicitRepeatCommon2 | failure_count | 1 | -17 | -8 | 162 | 12.7279220614 |
| ExplicitRepeatCommon2 - Native1 | scoreNormalized | 0.125 | 0 | 0.0625 | 0.0078125 | 0.0883883476483 |
| ExplicitRepeatCommon2 - Native1 | failure_count | -20 | 19 | -0.5 | 760.5 | 27.5771644663 |
| ExplicitRepeatInternal2 - Native1 | scoreNormalized | 0.25 | 0 | 0.125 | 0.03125 | 0.176776695297 |
| ExplicitRepeatInternal2 - Native1 | failure_count | -19 | 2 | -8.5 | 220.5 | 14.8492424049 |

Internal2 minus Common2 therefore has mean score difference **+0.0625**, sample variance **0.0078125**, SD **0.0883883476483**; its failure difference has mean **−8**, sample variance **162**, SD **12.7279220614**. The score gain occurs only at seed 336, where Internal2 also incurs one more failure. At seed 337 there is no score gain; Internal2 has 17 fewer failures than Common2 but two more than Native1. Relative to Native1, Internal2's mean score difference is **+0.125** (variance **0.03125**, SD **0.176776695297**), and mean failure difference is **−8.5** (variance **220.5**, SD **14.8492424049**). All null and unfavorable results remain included.

## Recorded mechanism and reproducibility limits

Both conditional arms selected candidate 0 on every action at both seeds. Their reported revision counters do not establish a supported risk comparison for every alternative; the separate [seed 336 gate/ranking audit](../20260923T123632_CST/report.md) documents that distinction. Internal2 recorded 3 supported ranking changes at seed 336 and 7 at seed 337. Common2 and Internal2 shared the first 19 executed actions at seed 336 and the first 3 at seed 337. The first observed score increase at seed 336 occurred in both arms after attempt 14; Internal2 additionally increased after attempt 22. These transitions include the action and world tick and do not label outcomes of unexecuted candidates.

The eight exported within-seed pair audits contain **210 aligned snapshot comparisons including initial states**. For all three pairwise comparisons among Native1, NoGraphRisk and ExplicitRepeatRisk, the same-action prefix covers all 30 actions in each seed. For Common2/Internal2 it covers 19 and 3 actions respectively. At every aligned snapshot, the public UI, recorded global Python RNG fingerprint, and object-construction count/order fingerprints are available and equal. Available next-candidate-0 prompt/token hashes and sampling seeds also match. The portable validator recomputes prefix lengths from executed-action hashes and checks RNG/construction and next-proposal hashes against the snapshot records; public-UI equality is the source-bound exporter's recorded comparison.

After the first action difference, later states are not tested as a common prefix. The online sidecar did not record current per-object, world or UUID RNG state, so these results do not claim complete hidden-state equivalence or repeat the separate fixed-13 replay's five-gate claim. The observation is not atomic across arms, but all arms and resource groups have verified closure.

Historical unseeded p335 results are excluded from all aggregates. No controller, checkpoint, threshold or seed was selected using these outcomes. NoGraph diagnostics do not establish full-graph or ES efficacy. The original Full follow-up remains a separate scientific result requiring its own closure and validation.

## Reproduction and preservation

Run `python3 -B validate.py` in this directory. It checks exported file hashes, all ten closed metrics, original action-return failure counts, resource closures, within-seed contrasts, prefix audit coverage, recorded costs, unchanged renderer output, and every n=2 descriptive mean/variance/SD. The renderer source and its report/statistics are preserved byte for byte; its report retains local-render wording. This README and the publication manifest identify the completed public diagnostic release.

The source receipt hash, exporter hash, renderer hash and file manifest preserve provenance. This release adds no environment, LLM, NN, graph, optimizer or GPU calls, reads no large checkpoint/activation artifact, and preserves all earlier dated results and failures.
