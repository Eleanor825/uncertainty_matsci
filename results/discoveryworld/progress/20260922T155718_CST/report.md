# DiscoveryWorld: V4 graphs complete; Full search stopped at G0

Observed **2026-09-22T15:57:18.551619+08:00**. Model: **Qwen3.5-4B**; scenario: **Proteomics Normal**.

**All 160 selected graphs and the supported next-action-failure risk models are complete. The Full branch stopped after its G0 development trajectory, before the first ES parameter perturbation. Full held-out tests and completed Baseline–Full pairs remain zero.**

| Stage | Result | Scientific scope |
|---|---|---|
| Selected graphs | 160/160; 4 groups; 8 source episodes | Fixed 20 attempts per original preparatory episode |
| V4 qualification | Each group returned 8/8 fixed prefixes, 0 qualification failures | Declared local-Jacobian mathematical reference |
| Primary risk head | Internal and OutputAction models fitted and development-admitted | 40 development rows, 2 episodes, 1 world |
| Two auxiliary risk heads | Not fitted/admitted | Fixed support gates failed |
| Full G0 development | 1 accepted B10 trajectory | No controller revision or rank change |
| Full ES parameter perturbations | 0 | Stopped at the registered G0 activity gate |
| Held-out results | Baseline 1; Full 0; complete pairs 0 | No Full-effect estimate |

## V4 graph qualification

The original eight preparatory episodes supplied 160 graphs at the fixed 20 attempt indices. Six episodes supply fitting/calibration data (worlds 0 and 1), and two supply development data (world 2). Graph extraction generated no new actions and made no new environment calls.

Each of four fresh graph processes passed the same eight-prefix suite under the declared V4 local-Jacobian mathematical-reference procedure. These are repeated checks of eight fixed prefixes, not 32 independent qualification examples. For eligible final-layer feature-to-action-score edges, V4 uses an independently checked FP64 mathematical tail reference. Original native FP32 finite-amplitude FD results and failure flags remain recorded. This completion does not claim that every original native FP32 FD check passed, or establish causality for environment failures.

The full graph computation retains the original 32-feature / 42-backward-target limits. The DW bank remains the separately trained 32-layer bank that passed development fidelity; the earlier MADE-to-DW transfer failure and prior numerical failures remain in historical snapshots.

## Risk-model development results

The next-action-failure head has 10 positive and 30 negative development rows across two episodes in world 2. Internal features and output/action features were evaluated on the same rows. The frozen admission record passed its numerical and support gates.

| Model/control | AUROC | Brier score | NLL |
|---|---:|---:|---:|
| Internal | 0.970000 | 0.064438 | 0.199944 |
| OutputAction | 0.960000 | 0.081075 | 0.253498 |
| Calibration-partition prior | 0.500000 | 0.221111 | 0.635050 |
| Fitting-partition prior | 0.500000 | 0.287778 | 0.769182 |
| Constant 0.5 | 0.500000 | 0.250000 | 0.693147 |

AUROC is higher-is-better; Brier score and NLL are lower-is-better. These are development/admission statistics, not held-out task-performance estimates. Forty rows within two episodes are not 40 independent trials. No significance claim or demonstrated online improvement follows from the small Internal–OutputAction difference.

Both next-action-failure models have changed NN parameters after 200 optimizer updates each (400 total). Fitting used world 0; temperature and threshold calibration used world 1. The fit seal records no world-2 development or held-out test read during fitting. The generation-invalid and terminal-noncompletion heads were not fitted because their fitting/calibration labels lacked both classes. Their unavailable outputs are not reported as trained predictions.

## G0 activity gate and stopped Full search

| World seed | Policy seed | Budget | Normalized score | Failed attempts | Fitness F = J |
|---:|---:|---:|---:|---:|---:|
| 2 | 303 | 10 | 0.0 | 7/10 | -0.070000 |

The accepted G0 development episode did not solve the task. Its largest supported risk was **0.916376948**, below the frozen threshold **0.986700532** (the 45th of 60 world-1 calibration risks, quantile 0.75). All ten attempts retained one candidate; supported neural revisions and rank changes were both zero.

The registered controller-activity gate failed and stopped the V4 branch before the first ES parameter perturbation. The risk NNs were trained, but the language model had not undergone an ES search update. The accepted G0 record and gate failure are both retained. There is no completed parameter-updating Full search or Full held-out trajectory in this snapshot.

The previously accepted Baseline test used world 3, policy seed 401 and B30, with normalized score 0 and 29/30 failed attempts. It cannot be paired with G0: the world, policy seed, budget and development/test role differ. Baseline and G0 scores therefore do not provide a Full-versus-Baseline effect estimate.

## History and validation

[The 10:44 snapshot](../20260922T104452_CST/report.md) preserves the earlier single-prefix diagnosis and zero-completion state at that time. [The 05:24 snapshot](../20260922T052401_CST/report.md) preserves the preparatory Native episodes, earlier ES-only branch and failed bank transfer. Later snapshots overlap these records; counts must not be added together.

MADE remains unchanged at 1079 valid evaluations, one retained technical failure and zero pending. Its Full-minus-Native mean SUN and AUDC differences remain negative at B10, B30 and B50.

`snapshot.json` contains only explicitly selected scientific fields, relative artifact references and source hashes. Run `python3 -B validate.py` to verify the projection seal, counts, identities, G0 risk/threshold arithmetic, report rendering, privacy field constraints and bundle hashes. Development AUROC/Brier/NLL values are original reported metrics; this export does not reconstruct them from raw predictions or repeat original simulator acceptance. No model, graph extraction, simulator, NN fitting or ES work was run to create the export.
