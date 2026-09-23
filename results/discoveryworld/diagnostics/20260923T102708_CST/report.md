# Fixed-prefix environment replay: supplemental seeding passes a bounded check

Three original-seeding replays diverged after the same 13th action. Three separately registered controlled-seeding replays matched at every public snapshot from initial state through action 13. This identifies uncontrolled environment RNG as a real reproducibility issue in the earlier p335 comparison; it **does not establish feedback or NN-selection benefit**. All historical results, including unfavorable outcomes and the original p335 scores, remain unchanged.

The diagnostic reused the already executed first 13 p335 actions in fresh **world 2 / policy 335 / Proteomics Normal** environments. Replicas A/B used the same original Native1 episode specification; C used the original ExplicitRepeatRisk specification. No new action was generated. The public B30 budget and original adapter/action/tick/evaluator behavior were retained; the diagnostic closed after 13 actions without claiming B30 or task completion.

| Seeding mode | Pair | First public divergence | Global Python RNG | Per-object RNG | World RNG | UUID RNG |
|---|---|---|---|---|---|---|
| original | A/B | after action 13 | different | different | equal | equal |
| original | A/C | after action 13 | different | different | equal | equal |
| controlled | A/B | none at 0–13 | equal | equal | equal | equal |
| controlled | A/C | none at 0–13 | equal | equal | equal | equal |

Each mode completed 3 fresh loads, 39 action requests/returns, 39 action ticks and 3 bootstrap ticks. Across both modes this is **6 loads, 78 actions and 6 bootstrap ticks**, with 0 LLM, 0 NN and 0 GPU calls. All 6 child processes and adapters closed. The 56 pair-step comparisons are included in [replay_evidence.json](replay_evidence.json); their flags and all exported file hashes can be checked with `python3 validate.py`.

The earlier p335 audit compared the next prompt at attempt 14, whereas this replay compares public state immediately **after action 13**; these identify the same transition boundary. Global and per-object RNG fingerprints already differed in original replicas, so the first visible public divergence is not evidence that hidden-state divergence began only on that action.

## Why supplement seeding

The frozen official source initializes the world RNG from the scenario seed, but `Pathfinding.runWander` draws from module-level Python `random`, and the base `Object` constructor creates a separate `random.Random()` without an explicit seed. Original world/UUID RNG agreement therefore did not ensure agreement of the random streams used by NPC/object behavior.

The new, separately registered seed protocol calls `random.seed(world_seed)` before environment load. It then seeds each object's RNG with the integer represented by SHA256 of compact JSON `[world_seed, original_uuid, original_object_type]`. The hook runs after the original base constructor and before subclass construction continues. The frozen base constructor was checked to assign its RNG once and never read it, so no base-constructor draw is skipped. The original world and UUID RNGs, evaluator, action dynamics, budgets, model weights and prompt construction rules are unchanged. The protocol uses no policy seed, arm label, episode ID or outcome.

The controlled replicas matched all five predeclared comparisons—public view and global/world/object/UUID RNG fingerprints—at every recorded step. This supports the **combined supplemental-seeding protocol on this fixed prefix**. It does not separate the effects of global versus object seeding, establish complete private-state equality, or guarantee determinism in other worlds or longer trajectories. It is a new environment-seeding variant; historical experiments are not retroactively relabeled as controlled runs.

## Instrument limits and preserved failure

NumPy was neither reseeded nor an admission gate; its recorded fingerprints differed in both modes. No claim is made that NumPy or every possible stochastic source is controlled.

The optional NPC-position observer selected the exact class name `NPCMovingAnimal`, while the actual animals have subclass names. Its position arrays were empty, making their equality vacuous. **NPC-position equality is not used as evidence or an admission gate.** The separate recursive per-object RNG traversal includes subclasses and recorded 1188 reachable objects per snapshot. This limitation is retained with the executed source; no extra replay was added to repair a non-gating field.

The first controlled metadata preparation failed because a required source-reference mirror was missing. It made 0 environment calls. Recovery copied 5 exact source files from the already locked installed runtime, preserved the failed attempt, and then performed the first controlled 3 runs. It did not modify the frozen diagnostic/seeding source or rerun the original 3.

## Follow-up status, not an outcome

At **2026-09-23T10:23:47.717907+08:00**, all 10 world 4 B30 jobs (the same 5 NoGraph conditions at policy seeds 336/337) were registered under the supplemental protocol, and a CPU resource owner had started. That receipt establishes neither actual GPU scientific execution nor any completed outcome; this report includes **0 world 4 outcome rows**. World 2's fixed-prefix check does not itself validate world 4 hidden-state equality. The follow-up must retain its own environment fingerprints and all outcomes.

The earlier [p335 scores and mechanism qualification](../../progress/20260923T094428_CST/report.md) remain published. Their positive score differences are observational and confounded by the preceding environmental divergence. This publication adds no model, simulator, NN-fitting, ES or GPU calls. Source/registration hashes and the unchanged official source commit are recorded in the portable evidence; private host/process details, credentials, raw prompts and token sequences are excluded.
