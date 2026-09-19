# SnAr warm-start risk repair: preparation only

This is a new training/development experiment. It does not alter original SnAr sources, 550 adaptation rows, old test results, selected G0 histories, old NN, or current GPU queues. No production registration or scientific execution has occurred at this package freeze.

The original label remains **no positive HV increment for the next executed query** (increment ≤1e-12). It is not an ODE-error or reasoning-error label. New trajectories actually receive the original fixed **550-row prior**. The earlier zero-call60/90-row archive rescoring is diagnostic only and is not used for fitting.

## Registered preparation scope

- Three G0 training episodes, seeds6101/6102/6103; two independent development episodes, seeds6201/6202. Prepare checks these against the original train/dev/population/mutation/test seeds plus the registered5201–5203 evaluation extension.
- Every episode has30 actual official Summit SnAr queries, including5 paired LHS initialization queries. **5 jobs /150 new physical calls =90 training+60 development.** There are at most125 non-LHS chosen feature rows, not150 NN examples.
- Original verified Qwen3.5-4B FP32/eager, CPU vocabulary, full-prefix2048, allocator.35, CPU2; original noise0 official ODE oracle and source/runtime/config unchanged. No4B/9B mixture.
- Original `Pipeline.episode` runs `qwen_base`: valid first proposal executes; a second is only for failed generation/schema. Every successful proposal gets a separate diagnostic native graph using the existing SnAr32-TC bank, unchanged per-prefix FVU≤.5 and four original FD checks. No new TC training. The original `candidate.json` retains `risk=None, features=None`; `warm_graph.json` cannot enter retry/advice/rank.
- All raw failures, tokens, prefixes, intents, physical returns, graphs and costs remain in the new job directory. A failed graph is recorded and does not erase the material outcome. No old collection or test episode is replayed or symlinked into writable output.

The original `audit_episodes` selects nonempty prior only for old `role='test'`. This package uses **real train/dev identities**, with a private, source-bound function copy changing exactly `p=prior if spec.get('role')=='test' else []` to `p=prior`. All its physical/token/prompt/label/summary checks are otherwise identical. The private `_oracle_evidence` catalog is exactly these five new jobs, with independent scope names; no old2300-call membership is forged. Separate original `_graph` validation checks the diagnostic graphs. CPU tests execute the real original audit on a tiny warm train fixture and reject altered labels.

## Four CPU fits, train-only selection and calibration

All fitting rows come from these five new trajectories. The original550-row prior only supplies a frozen observed context, never NN fitting rows. The285-feature schema is inherited; old fitting rows/test outcomes are not appended. Initial LHS, invalid fallback, graph failures and exact train/dev prefix overlaps retain explicit exclusion counts. Original graph availability threshold is.9. Every training episode and fitting fold must contain both labels, as must pooled new development. Otherwise a permanent `data_insufficient` report records zero fits; no automatic data expansion, relabeling, threshold weakening or retry occurs.

The only model-selection amendment is explicit: three leave-training-episode-out paths, each100 epochs, followed by one fresh full-training fit at the selected epoch. Original64×2 GELU RiskMLP, AdamW lr.001/wd.0001, seed1729, batch256, train-only imputation/normalization/error prototype and episode loss weights are reused. Each100-epoch fold records all epoch tensors/logits. The earliest epoch improving macro held-out-training-episode BCE by1e-8 is selected. Positive temperature is fitted on those selected-epoch **training OOF** logits, with original logT bounds[-4,4]. Final NN weights/normalizer/epoch/temperature are sealed before new dev scoring. **At most4 fits /400 optimizer steps**, and no LLM/GPU/graph/oracle in the fitting process. This is episode CV on one reaction function, not cross-chemistry generalization.

New dev reports raw/calibrated AUROC/Brier/NLL, per-episode metrics, .5 threshold counts, and the fixed training-failure-rate constant. A predictive gate requires strictly lower dev Brier and NLL than that constant plus AUROC>.5. This is a small-data descriptive gate, not significance or proof of policy benefit. Single-class per-episode metrics remain undefined. No threshold is tuned; no controller is deployed and no ES or new heldout job is registered here. Negative/no-improvement results remain final. Further neural-vs-schema behavior validation and fresh full ES require their own concrete registration; passing this package is not completion of the method.

The CPU audit loads only small saved NN states, recomputes all300 fold epoch logits, train-only preprocessing, OOF duration/temperature and final-model report exactly in canonical JSON. It never calls optimizer.step. Calibration objective reconstruction during audit creates no deployed parameter update.

## CLI after review and deployment

Use the actual registered policy interpreter; the normal namespace is:

```
P=/mnt/ai-material-data/made_crystalgym_crv_esopt_20260915_164607/benchmark_extensions/summit_snar_warm_risk_repair_20260920_v1
PY=/mnt/ai-material-data/made_crystalgym_crv_esopt_20260915_164607/environments/policy-py312/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PY" -B "$P/runner.py" prepare
```

Prepare is CPU-only and one-shot. It pins the original completed parent,550-prior/TC assets, new plan/source and actual CPU package versions. No checkpoint tensor loading, model forward or material call occurs. `REG=$P/runs/v1/registration.json`; `SHA` is the **actual returned registration SHA**, not a placeholder.

```
"$PY" -B "$P/runner.py" status --registry "$REG" --expected-sha256 "$SHA"
"$PY" -B "$P/runner.py" worker --registry "$REG" --expected-sha256 "$SHA" --worker-id warm_00 --gpu-uuid ACTUAL_GPU_UUID
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PY" -B "$P/runner.py" fit --registry "$REG" --expected-sha256 "$SHA"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PY" -B "$P/runner.py" audit --registry "$REG" --expected-sha256 "$SHA"
```

The worker acquires the original source-derived two GPU leases, then keeps them across original009a `launch_and_drain` children; an existing owner can instead pass `--lease-fds FD1,FD2`. It verifies the original resource gate before dispatch and again before the child's scientific claim. Every material job has its own permanent dispatch claim and independent scientific directory; another worker can consume different unclaimed jobs. Failed/unknown claimed work is never replayed. A nonzero scientific child stops that worker and propagates failure; original009a waits for its whole subtree and actual GPU closure before returning. The parent has no model. It neither kills other processes nor changes pod lifecycle. Collection receipt explicitly does not claim GPU closure; the009a dispatch exit is that separate evidence.

`fit` returns2/pending with no fit claim if any collection receipt is absent, and3 for an explicit failed dependency. It does not wait on or reserve a GPU. Complete immutable collection data is audited before creating the permanent NN claim. Single-class/support failure returns a truthful completed-data-insufficient result with0 fits. Unknown/failed NN claims are never retrained automatically. Root must inspect both technical completion and predictive gate rather than infer improvement from exit0.

Source-level CPU contracts cover real tiny four-fit/audit execution, exact original warm-prior physical label validation, development exclusion from selection, graph/class failure preservation, permanent claims and unknown-dispatch non-replay. Mocked resources/Popen in those tests are not production hardware qualification.
