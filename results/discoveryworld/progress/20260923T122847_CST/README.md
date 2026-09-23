# Seeded world 4 — partial progress at 12:28:47 CST

Observed **2026-09-23 12:28:47 CST**. **9 of 10 B30 episodes are closed; this is a partial progress publication.** The exporter’s `publication_ready=false` and `all_ten_complete=false` remain unchanged. Its complete-results gate has not been passed or changed. The observation is not atomic across arms.

[All ten slots and the same-prefix audit](report.md) · [Hash-bound snapshot](snapshot.json) · [Portable local validation](validate.py)

Within each of policy seeds 336 and 337, Native1, NoGraphRisk and ExplicitRepeatRisk executed exactly the same 30 actions. Their final scores are all 0; the failed-action counts are 29/30 at seed 336 and 2/30 at seed 337. The conditional arms show no score or action improvement at either sampling seed.

At seed 336, ExplicitRepeatCommon2 closed with score 0.125 and 9/30 failed actions. ExplicitRepeatInternal2 has only 6 returned actions, 14 generation returns and 13 of 14 capture returns; its final score, failure metric and Common2/Internal2 contrast are unavailable. Common2’s positive score is not evidence of an NN contribution.

At seed 337, ExplicitRepeatCommon2 and ExplicitRepeatInternal2 both have score 0, with 21/30 and 4/30 failed actions respectively. Internal2 has **17 fewer failures than Common2, but 2 more than Native1**. This is one completed equal-two-proposal comparison with no task-score gain. Every closed episode’s task-success flag is false.

The matching executed prefixes are 30 actions for all three pairwise comparisons among Native1, NoGraphRisk and ExplicitRepeatRisk in each seed, 6 observed actions for seed 336 Common2/Internal2, and 3 for seed 337 Common2/Internal2. Available aligned public-state, recorded global-RNG, object-construction, and next-proposal0 prompt/token/seed fingerprints agree. The online sidecar does not capture full current per-object/world/UUID RNG state, so complete hidden-state equivalence is not established. After an action diverges, later states are not compared as a common prefix.

No two-seed aggregate is reported: `complete_two_seed_statistics.json` is intentionally empty. These are two sampling seeds in one world. Historical unseeded p335 is excluded from pooling and causal evidence. NoGraph is not the original full-graph ES method; this snapshot contains no results from the separate original Full follow-up.

The existing renderer was used unchanged; `report.md` retains its local-render wording. The publication manifest distinguishes this partial public release from readiness for a complete scientific result. The snapshot and renderer’s report/statistics are unmodified. Publication performs no new environment, model, graph, fitting or optimizer calls, and prior dated results and diagnostics remain available.
