# Uncertainty-guided materials discovery

Research code for a white-box language-model agent that proposes materials-discovery actions, estimates failure risk from internal representations, and adapts policy weights with Agentic ESOpt. The implemented interfaces cover MADE and CrystalGym. Published results include **MADE, Qwen3.5-4B, seed1, baseline versus a fixed G2 graph-risk policy**, and a separately accepted seven-arm Summit SnAr study with five evaluation seeds.

This repository contains the actual frozen server code used by the current budget-sweep workspace, its tests and configuration references, and dated result exports. It is not an export of the server's older top-level working copy. The original161 source/script/test files match the original full-CPU validation inventory byte for byte. That server validation recorded **840 passed, 3 skipped, 0 failed**; skipped GPU checks are not counted as passed experiments.

## Results and scope

The [new-method source snapshot](reproduction/source_snapshots/new_studies_20260919_v1/README.md) adds **61 original-byte scientific source, validation and configuration payloads** for the current MADE four-arm/support-aware study and SnAr. Its [registered-source mapping](reproduction/source_snapshots/new_studies_20260919_v1/source_mapping.json) covers226 binding rows:215 source files can be reconstructed from existing and new payloads, while11 external resource/artifact bindings remain explicitly missing. Four derived registration projections retain separate original/export hashes. The [CPU verifier and source-layout tool](reproduction/source_snapshots/new_studies_20260919_v1/README.md#verify-without-loading-a-model) do not run experiments. External weights/data and a separately reviewed execution adapter are still needed; the snapshot does not make the original server CLIs directly runnable on another machine. Existing scientific source and results are unchanged.

The [fixed-five MADE B10 report, September19 at20:34:17 CST](results/made_components/core5/20260919T203417_CST/report.md) now has all three evaluation seeds for baseline, full and ES-only (15 runs each); UQ is11/15. Baseline→full total SUN is **22→20**, and mean AUDC is **.156→.144**. Four of five system-mean AUDCs improve, but the larger Al–V–Zn loss leaves the fixed-cohort aggregate negative. Within-system and paired seed variances (ddof=1) are available; UQ's incomplete15-run aggregates remain blank. These56 completed core runs are a subset of the observed254/1,080 study evaluations, not additional experiments.

The [exact SnAr candidate diagnosis](results/summit_snar/candidate_analysis/20260919T202943_CST/report.md) adds portable scientific inputs for750 already executed queries,248 full candidates and125 train/dev feature rows. The selected-action failure AUROC is .791284, but Brier is .677841 and mean predicted failure .175444 versus observed .973214. All23 retry queries had zero HV gain, while six gains came from single proposals; this does not identify a causal retry effect. The24 unexecuted candidates remain without invented outcomes. This analysis reuses the accepted35-episode study and does not change its negative aggregate comparison or G0 selection.

A separate [single-development-seed G0 control](results/development/G0_comparison/20260919T200818_CST/report.md) reports Al–Pd–Sm B10 AUDC **G0 .68 / G1 .62 / G2 .64**. Only the G0 diagnostic added10 candidate calls; G1/G2 are existing references. This previously used development seed is outside both test matrices, and the frozen G2 selection remains unchanged.

A new [SnAr case analysis](results/summit_snar/case_analysis/20260919T2023_CST/report.md) explains the existing five-seed comparison without adding experiments: full/UQ has two positive pairs, two negative pairs and one tie against Qwen base, and loses to GP-EI in every seed. Its weighted selected-action Brier is .677841 versus .097142 on development. The [label/prior audit](results/summit_snar/case_analysis/20260919T2023_CST/label_prior_check.md) finds consistent label direction and a declared shift from episode-local cold-start history to a frozen 550-row test prior; that difference is a mechanism hypothesis, not proof of causation or evidence of test-time recalibration. Both ES branches selected G0. Recomputable case tables, scientific source excerpts and remaining evidence gaps are included.

A new [CPU-only grouped scalar-risk diagnostic](results/development/grouped_scalar_risk/20260919T192455_CST/report.md) completed on September 19 at 19:24:55 CST. Training-only episode cross-validation selected 8 updates for internal features and 32 for external action/sampling features, but original-dev AUROC was **0.495935 and 0.430894**, respectively. Both calibrated models had zero probabilities above the fixed 0.6 threshold and did not beat the constant-0.5 Brier score. The release includes all 54 candidate states’ recorded metrics, 166 new prediction rows, 386-row coverage, byte-exact scientific training code and a portable checker. This was a predictive development diagnostic; no controller, G2 checkpoint or primary experiment result changed.

The [component-study snapshot at September 19, 18:50:04 CST](results/made_components/20260919T185004_CST/report.md) has **205/1,080 accepted evaluations**. It reports actual within-system evaluation-seed means, sample variances (ddof=1), SDs and missing seeds. The matched full comparison remains negative: SUN45→33 and mean AUDC .192917→.154583 across24 task-seed pairs. Only one full configuration has all three evaluation seeds; no complete all30 full seed exists yet, so its benchmark-level seed variance remains unavailable. This partial study is separate from the original seed1 matrix.

See [the latest per-system and aggregate report](results/budget_sweep/20260919T052138_CST/report.md), [all 180 planned trajectory rows](results/budget_sweep/20260919T052138_CST/all_system_results.csv), [the results chronology](docs/results.md), and [the export manifest](results/manifest.json). The final observation is September19 at05:21:38 CST: B10, B30 and B50 each have60/60 completed trajectories, for180 total and90 matched comparisons. B50 includes59 original accepted trajectories and one explicitly reconciled Ga–Pt–Tm trajectory; its original failed verification is preserved. B10 improved in aggregate for this single seed, while B30 and B50 declined. Historical snapshots and all negative or tied outcomes are retained. Repeated-seed variance and general efficacy are not established by this matrix.

The registered expansion preserves the prior60 B10 trajectories and adds120 independent B30/B50 trajectories over the same30 official MADE test systems. It reuses the selected G2 policy and frozen uncertainty/transcoder models; it does not retrain on these test results. The scheduler reserves all B30 jobs before it can reserve B50 jobs. This is a **claim-order** guarantee, not a guarantee that every B30 trajectory finishes before B50 starts.

A subsequent [paired development mechanism audit](results/development/schema_repair_seed2/20260919T074355_CST/report.md) identifies an adapter-compatible composition Mapping that the original controller incorrectly rejected. Correcting that gate changed one Al–Pd–Sm seed2 B50 rollout from SUN9 to15, with31 additional MACE screening evaluations. The neural models were unchanged, and this pair does not establish an independent benefit from neural uncertainty. These development results and costs are separate from the original180 test trajectories; the ongoing multi-seed component study will assess the corrected method.

An [offline audit of the frozen neural risk models](results/development/nn_information_audit/20260919T080602_CST/report.md) rechecks the original 386 train/dev proposals and 200 scientific events, with reproducible predictions and event-level statistics. The sole original dev episode was already used for epoch selection and temperature calibration; these are **development diagnostics, not held-out test scores**. The report distinguishes weak predictive evidence, an already encoded parsing label, and the absence of valid-action/missing-graph training examples.

A [source-bound support-method mechanism audit](results/development/support_mechanism_audit/20260919T174119_CST/REPORT.md) examines the fixed 17-system comparison and all six new full-ES train/dev trajectories. Both ES updates recorded 723 nonzero parameter blocks, but none of those six trajectories used a neural retry, threshold crossing or risk-ranking change. Deduplicated dev AUROC values (.75 and .7083 over ten events each) coexist with Brier scores near .25. All negative cases and contrary MACE examples are retained; these observations do not establish a neural or screening causal benefit. G0 was not evaluated by the original selection rule, and the report records only the missing-control design at its observation time.

The [accepted Summit SnAr results](results/summit_snar/20260919T162548_CST/report.md) cover seven arms and 35 test episodes: 550 shared adaptation calls plus 1,750 held-out calls, reconciled as 2,300 unique physical attempts. Independent CPU acceptance passed at September 19, 16:25:48 CST after an import-only report failure; the original failure is preserved. Full has a negative mean paired gain relative to Qwen base, and its development rule selected G0. Five-seed sample variances, the common prior, raw curves and all negative outcomes are reported; these results do not establish unseen-function generalization. This is a result export; the existing frozen source inventory above remains scoped to the MADE budget-sweep code.

The [component-study snapshot at September 19, 17:26:03 CST](results/made_components/20260919T172603_CST/progress.md) contains 172/1,080 accepted new evaluations. Its [complete four-arm case comparison](results/made_components/20260919T172603_CST/report.md) uses the same 17 B10/seed 2 systems: full SUN is 27 versus baseline 34, with mean AUDC .174706 versus .202353. All positive, negative and tied cases remain visible; this partial subset is not a repeated-seed variance estimate.

The larger multi-model, multi-scale and CrystalGym study remains deferred. These exports do not establish a general improvement, an independent causal effect of uncertainty, or performance for model sizes other than the evaluated4B checkpoint.

## Code map

| Component | Main implementation |
| --- | --- |
| Native HF policy, generation, traces and state identity | `src/matdiscovery/policy.py` |
| TopK transcoders and train-only scalar-RMS conditioning | `transcoders.py`, `normalized_transcoders.py` |
| Native local Jacobian graphs and action-token targets | `native_attribution.py`, `action_targets.py` |
| Failure-risk networks, typed heads and control | `uncertainty.py`, `failure_risk.py`, `failure_controller.py` |
| Full-policy ES updates and training driver | `esopt.py`, `es_training.py`, `core_es.py` |
| Scientific tool adapters and audited oracle attempts | `benchmark_adapters.py`, `mace_parallel.py`, `rpc.py` |
| Independent trajectories, claims and budget reports | `made_budget_runner.py`, `made_budget_sweep.py` |
| Source, artifact and cost verification | `collection_provenance.py`, `training_jobs.py`, `accounting.py`, `core_report.py` |

The scientific policy uses actual FP32 native HF computation, CPU embedding/output-head placement, and attention recomputation with recorded runtime settings. The main graph rule excludes mathematically unreachable final-MLP positions while preserving the complete causal prefix and original graph budget. Native finite-difference/parity and transcoder fidelity gates remain explicit; missing graphs remain recorded observations.

The accepted normalized transcoder recipe estimates centering/RMS statistics on training data, trains64 epochs, exports an actual raw-space FP32 TopK model, and selects on raw development error. Hard-TopK near-tie rounding means the exported FP32 model is not claimed to be pointwise identical to the normalized computation. Typed risk heads without required class support are explicitly unavailable, not assigned invented probabilities. Evaluators, transcoders and uncertainty networks are kept outside the ES policy parameter manifest.

## Install and run tests

Python3.12 is the observed scientific runtime; package metadata requires Python>=3.11. From a fresh environment:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,policy]'
PYTHONPATH=src python -B -m pytest -q
```

`pyproject.toml` pins the policy stack's PyTorch2.9.1 and Transformers5.17.0. Some tests require the optional policy dependencies or CUDA; synthetic CPU tests do not constitute material-discovery outcomes or replace actual model numerical qualification.

The publication checkout additionally passed63 targeted CPU tests, including the budget-claim contracts and normalized-transcoder reference comparisons. The two small source-bound reference scripts needed by those comparisons are retained under `technical_not_main/tc_scale_diagnostic_v2/`; their diagnostic outputs and tensors are not included.

For a short CPU contract check:

```sh
PYTHONPATH=src python -B -m pytest -q \
  tests/test_esopt.py tests/test_transcoders.py tests/test_action_targets.py \
  tests/test_made_budget_sweep.py tests/test_made_budget_runner.py
```

For result-export consistency without model loading:

```sh
python3 results/validate_exports.py
python3 results/budget_sweep/20260919T052138_CST/recompute.py
```

## Scientific execution

Read [the execution guide](docs/execution.md) before running a scientific job. Models, datasets, private environments, full activation/graph/checkpoint artifacts and external benchmark repositories are not included. Upstream commits are pinned in [configs/vendor_lock.json](configs/vendor_lock.json); each upstream project and model retains its own license.

The production commands accept an independently prepared, sealed workspace and verify its source, asset, model, controller and claim provenance before a physical action. For example:

```sh
python -m matdiscovery.made_budget_runner --workspace /absolute/SEALED_WORKSPACE --actor-id actor_00
python -m matdiscovery.made_budget_runner --workspace /absolute/SEALED_WORKSPACE --aggregate
```

Do not treat a code checkout or a derived public summary as an executable sealed study. Existing claims are not recycled, failed physical jobs are not automatically replayed, and a published plan is not evidence of completed evaluation.

## Provenance

- [Server source manifest](provenance/server_source_manifest.json): per-file SHA256 and the original validation inventory.
- [Older-root comparison](provenance/server_root_comparison.json): the files that distinguish the active frozen code from the older server working copy.
- [Registered-study overview](provenance/registered_study_overview.json): a public descriptive derivation, not a replacement registration.
- [Effective core protocol](provenance/active_core_main_protocol.json): the core settings inherited by the fixed G2 evaluation.

`configs/` contains the full-CPU validation snapshot's configuration templates, including historical full-study defaults. They are deliberately not relabeled as the executed reduced-core configuration. Some historical source/config references retain the original server layout because those bytes are part of provenance; they are not credentials or portable local paths.

The byte-preserved active core protocol also contains an inherited `candidate_training_budget` description mentioning50. The explicit FAST registration overrides that description: ES used budget10, two generations/two population members, four training plus two development episodes, totaling60 candidate evaluations. The publication does not edit that historical JSON to conceal the inherited text.

No model weights, raw benchmark datasets, virtual environments, credential files, authentication/browser-control tools or large runtime artifacts are included.
