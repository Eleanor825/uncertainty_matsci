# Execution and reproduction boundaries

## What is directly reproducible from this checkout

The Python implementation and synthetic contract tests are present. The exported files under `src/`, `scripts/` and `tests/` are byte-identical to the active server budget workspace. Configurations come from the matching full-CPU validation snapshot; the effective reduced-core protocol and source-registration fingerprints are documented separately under `provenance/`.

The result CSV/JSON exports can be validated independently with `python results/validate_exports.py`. They reference original artifact hashes without distributing private server directories or claiming that a source hash alone reproduces a physical trajectory.

## Assets required for new scientific execution

1. Obtain the benchmark/method repositories at the commits in `configs/vendor_lock.json`. They were not silently copied into this repository.
2. Prepare the exact official benchmark assets, MP snapshots and model weights. `configs/model_manifest.json` records model IDs/revisions and file evidence; it contains metadata, not weights. The evaluated methods require white-box access to the complete policy parameters and native activations.
3. Establish separate policy and scientific-tool environments. The original policy runtime used FP32, explicit decoder settings and recorded placement/kernel flags. MADE's final evaluator is its official ORB oracle; MACE is the declared screening surrogate. CrystalGym would require its real DFT backend and separate validation.
4. Create a new declared workspace with fixed train/dev/test tasks, seeds, budgets and source hashes. Do not edit an existing frozen workspace or reuse a completed/partial physical claim as a new trajectory.
5. Fit representations and uncertainty on the declared training/development corpus only. The normalized raw-FP32 transcoder must pass the same development-fidelity and native attribution checks. Unsupported failure heads remain unavailable. Test outcomes are not model-selection or threshold-tuning data.
6. Run the declared training and evaluation entrypoints only after their source/asset admission passes. Original full-scope ES training, core ES, all30 B10 and the B30/B50 expansion are separate registered scopes; the current expansion uses fixed G2 weights selected earlier at B10.

The registered execution adapters intentionally reject missing evidence, changed sources, duplicate actor/job claims and unsupported backend changes. A fresh Git clone does not contain the original complete provenance graph, original material datasets or checkpoint tensors, so supplying one of the example commands alone cannot recreate the original sealed run.

For the executed FAST ES scope, the explicit registered execution budget is10: four population-training episodes plus two development episodes total60 candidate evaluations. The preserved protocol's inherited `candidate_training_budget` text still mentions50; that text does not override the validated FAST registration. Historical representation collection remained B50.

## Entrypoints

- `scripts/train_representations.py`: raw collection/representation pipeline.
- `python -m matdiscovery.core_representation`: declared core representation/risk stages.
- `scripts/train_es_condition.py`: full registered ES condition driver.
- `python -m matdiscovery.core_es`: separately registered reduced-core ES.
- `python -m matdiscovery.core_final`: core held-out evaluation.
- `python -m matdiscovery.made_all30_runner`: B10 coverage expansion.
- `python -m matdiscovery.made_budget_runner`: independent B30/B50 actors and CPU aggregation.

Use each entrypoint's `--help` and its corresponding registration/provenance requirements. Physical evaluation is not part of a default test or install. Administrative deployment/authentication helpers from the live servers are deliberately outside this source export.

## Validation interpretation

The imported server validation was840 passed/3 skipped/0 failed. The publication check records its own local tests separately; neither substitutes for actual policy, transcoder, native-gradient or oracle validation. Report tests, numerical diagnostics and empirical outcomes under their distinct scopes. Full-method versus baseline comparisons do not isolate uncertainty from ES adaptation, and a single seed/selected checkpoint does not establish a scale law.
