# Uncertainty-guided materials discovery

Research code for a white-box language-model agent that proposes materials-discovery actions, estimates failure risk from internal representations, and adapts policy weights with Agentic ESOpt. The implemented interfaces cover MADE and CrystalGym. Published results include **MADE, Qwen3.5-4B, seed1, baseline versus a fixed G2 graph-risk policy**, and a separately accepted seven-arm Summit SnAr study with five evaluation seeds.

This repository contains the actual frozen server code used by the current budget-sweep workspace, its tests and configuration references, and dated result exports. It is not an export of the server's older top-level working copy. All161 exported source/script/test files match the original full-CPU validation inventory byte for byte. That server validation recorded **840 passed, 3 skipped, 0 failed**; skipped GPU checks are not counted as passed experiments.

## Results and scope

See [the latest per-system and aggregate report](results/budget_sweep/20260919T052138_CST/report.md), [all 180 planned trajectory rows](results/budget_sweep/20260919T052138_CST/all_system_results.csv), [the results chronology](docs/results.md), and [the export manifest](results/manifest.json). The final observation is September19 at05:21:38 CST: B10, B30 and B50 each have60/60 completed trajectories, for180 total and90 matched comparisons. B50 includes59 original accepted trajectories and one explicitly reconciled Ga–Pt–Tm trajectory; its original failed verification is preserved. B10 improved in aggregate for this single seed, while B30 and B50 declined. Historical snapshots and all negative or tied outcomes are retained. Repeated-seed variance and general efficacy are not established by this matrix.

The registered expansion preserves the prior60 B10 trajectories and adds120 independent B30/B50 trajectories over the same30 official MADE test systems. It reuses the selected G2 policy and frozen uncertainty/transcoder models; it does not retrain on these test results. The scheduler reserves all B30 jobs before it can reserve B50 jobs. This is a **claim-order** guarantee, not a guarantee that every B30 trajectory finishes before B50 starts.

A subsequent [paired development mechanism audit](results/development/schema_repair_seed2/20260919T074355_CST/report.md) identifies an adapter-compatible composition Mapping that the original controller incorrectly rejected. Correcting that gate changed one Al–Pd–Sm seed2 B50 rollout from SUN9 to15, with31 additional MACE screening evaluations. The neural models were unchanged, and this pair does not establish an independent benefit from neural uncertainty. These development results and costs are separate from the original180 test trajectories; the ongoing multi-seed component study will assess the corrected method.

An [offline audit of the frozen neural risk models](results/development/nn_information_audit/20260919T080602_CST/report.md) rechecks the original 386 train/dev proposals and 200 scientific events, with reproducible predictions and event-level statistics. The sole original dev episode was already used for epoch selection and temperature calibration; these are **development diagnostics, not held-out test scores**. The report distinguishes weak predictive evidence, an already encoded parsing label, and the absence of valid-action/missing-graph training examples.

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
