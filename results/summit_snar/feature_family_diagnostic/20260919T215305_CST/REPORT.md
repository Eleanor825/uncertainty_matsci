# SnAr stored-feature diagnostic v2: unchanged G0 weights, ranking and calibration separated

The fixed 12-model CPU diagnostic completed **746 optimizer steps** and scored the same **224 previously executed actions**. The full 285-feature models have observed mean AUROC **0.7661**, versus **0.6478** for sampling-only, **0.6544** for hidden-only, and **0.4743** for graph-only. However, full-feature mean calibrated Brier is **0.5956**, far worse than the fixed training-prior predictor's **0.02766**. These results do not establish graph contribution, reliable calibrated uncertainty, or a policy/HV benefit. No model is selected for deployment.

## Exact reconstruction and preserved failure

The original prepare and run both exited 0. The original audit exited 1 at its final report comparison and remains recorded as failed. Three count dictionaries used Python integer keys 0/1; JSON storage converted those keys to strings. The failed code directly compared the reloaded report to the in-memory dictionary. This was an audit serialization defect, not a difference in trained model values.

A separately source-bound, read-only reconstruction completed at **2026-09-19 21:53:05.531281 CST / 13:53:05.531281 UTC**. It re-ran the original source/data/runtime/checkpoint checks, original reproduction gate, and complete report computation from saved small models. The only three raw Python differences were `cohort_counts/{training,development,scored_test}_label_counts` key types. **The entire reconstructed original publish-format JSON was byte-for-byte equal** to the stored report, SHA `b358c7dfb107884f5c471bff327d8043747e74a9ddc345308c4609a08609ac5c`; canonical differences were 0. All 59 observed original package/run/failure files retained their SHA and mtime.

The original all-feature/seed1729 checkpoint was reproduced with exact parameter tensors, preprocessing, config, training history, selected epoch, provenance, and temperature. All 224 saved action probabilities matched with maximum difference **0.0**. The previously registered gate's tolerance was not changed, and the full-report comparison added no numerical tolerance. The reconstruction performed no fitting, LLM, graph, oracle or policy execution, and did not edit the frozen source, model, report, failed audit or invocation.

`derived_acceptance.json` is a **separate technical acceptance of exact report reconstruction** based on the pinned actual readback. It does not relabel the original audit as passed or write its absent audit.json. The original failed exit, traceback and stopped-pipeline artifacts remain explicitly referenced. The v2 archive now includes the 17 byte-verified original JSON artifacts (1,111,076 bytes), including all 2,688 prediction records. Its portable Python-standard-library verifier independently recomputes AUROC with ties, average precision, Brier, NLL, ECE, risk-coverage metrics, fixed-threshold counts, all five policy-seed strata, 50-row development metrics and seed summaries. All 1,870 individual metric comparisons agree within a maximum arithmetic difference of 8.8818×10⁻¹⁶. Its declared 10⁻¹² cross-implementation roundoff bound is separate from—and does not weaken—the original byte-exact reconstruction acceptance.

## Fixed populations and analysis

The original NN target is **no positive hypervolume increment for the next executed query**. No-HVI is label1; an HVI improvement is label0. Original fitting uses 75 training rows (70 no-HVI / 5 improvements) from three collection seeds and 50 development rows (45/5) from two seeds. Development selects the epoch and temperature, so its metrics are selection/calibration in-sample. No new data, hyperparameter search, threshold search, or test-based checkpoint/family selection was performed.

There are 248 stored full-policy test candidates: 224 selected successful candidates with complete features and saved scores, 15 scored but unselected candidates, 8 generation failures and 1 successful candidate without graph features. The same original 250 queries contain 25 paired-LHS initial-design queries, 224 scored selected actions and 1 explicit invalid fallback. The latter 26 queries are excluded from this risk-scoring cohort, and unselected candidates receive no invented outcomes.

The 224 scored actions have **218 no-HVI outcomes and only 6 improvements**, across five original policy seeds. Seed 5105 has no improvement, so its AUROC is undefined and remains null. These are dependent actions along adaptive trajectories, not 224 independent scientific replicates. The diagnostic was designed after the physical study and is explicitly posthoc exploratory; freezing its computational settings before these refits does not make it a prospective policy experiment.

All families use the original CPU RiskMLP/preprocessing/AdamW/early stopping/temperature code and hyperparameters. Features are removed **before** fitting normalization, missing indicators and error prototypes, preventing hidden/graph information from leaking into sampling-only preprocessing. Seeds 1729/1730/1731 retain the original seed effects on initialization and training permutation. All 12 models were sealed before scoring test outcome labels.

## Results on the same224 actions

Means and sample variances below are across the three NN training seeds, **not confidence intervals or repeated-physics variance**. A higher AUROC is better; lower Brier and NLL are better. Positive-class AUPRC concerns the very common no-HVI event, with a constant-predictor baseline 0.9732.

| Feature family | Features / parameters | Mean AUROC | AUROC sample variance | Mean calibrated Brier | Brier sample variance | Mean calibrated NLL |
|---|---:|---:|---:|---:|---:|---:|
| Sampling-only | 4 / 4,993 | 0.647808 | 0.00778340 | 0.0279218 | 0.0000009430 | 0.135578 |
| Hidden-only | 128 / 20,865 | 0.654434 | 0.00134903 | 0.4288191 | 0.00450004 | 1.080179 |
| Graph-only | 153 / 24,065 | 0.474261 | 0.00338679 | 0.0444400 | 0.0000269017 | 0.198865 |
| All features | 285 / 40,961 | 0.766055 | 0.00346550 | 0.5955723 | 0.03577581 | 1.726120 |
| Constant train prior70/75 | — | 0.500000 | — | 0.0276587 | — | 0.139682 |
| Constant0.5 | — | 0.500000 | — | 0.2500000 | — | 0.693147 |

The full-feature AUROCs by seed are0.791284,0.808104,0.698777; calibrated Briers are0.677841,0.729651,0.379225. Every full-feature Brier exceeds even the constant 0.5 baseline. Hidden-only calibrated Briers also all exceed0.25. Sampling-only has low Brier primarily on a cohort overwhelmingly labelled no-HVI; its mean Brier is slightly worse than the fixed training-prior constant, while mean NLL is slightly better. That is a mixed comparison, not evidence of a replacement winner.

Graph-only AUROCs are0.436544,0.541284,0.444954. They do not support reliable standalone graph discrimination in this cohort. Its lower Brier than full/hidden does not establish information value, because it is still worse than the fixed high-prior constant. Full features outperform the isolated families in observed AUC, but these four comparisons do not isolate the graph's incremental contribution: a sampling+hidden model is absent, interactions differ, parameter counts differ, and only 6 improvement events support the ranking calculation.

| Family | Selected epochs, seeds1729/1730/1731 | Executed optimizer steps | Temperatures | Actions at p≥0.5, out of 224 |
|---|---|---|---|---|
| Sampling | 57 / 60 / 53 | 72 / 75 / 68 | 0.987738 / 0.992830 / 0.989022 | 224 / 224 / 224 |
| Hidden | 40 / 42 / 47 | 55 / 57 / 62 | 1.127011 / 1.106236 / 1.191720 | 25 / 27 / 20 |
| Graph | 47 / 48 / 46 | 62 / 63 / 61 | 0.946481 / 0.911481 / 0.948026 | 224 / 224 / 223 |
| All | 44 / 40 / 42 | 59 / 55 / 57 | 1.158375 / 1.098503 / 1.240846 | 10 / 13 / 40 |

These threshold counts are retrospective classifications of fixed actions; they are not simulated retry counts or results of a new controller. For original all-feature seed1729, only 10 of 224 actions score at least0.5 despite 218 no-HVI outcomes, and 85 no-HVI actions score at most0.1. The other two full-feature seeds also underestimate no-HVI risk at the fixed threshold. The temperature fit slightly improves Brier (full-family raw mean0.628881 → calibrated0.595572) but does not repair the large error. Temperatures here are near1, unlike the previously diagnosed MADE scalar model; no high-temperature explanation is inferred for this SnAr result.

The original all-feature seed1729 model has development AUROC 0.608889/Brier 0.097142 on its 50 development rows, compared with test-cohort 0.791284/0.677841. This describes a large difference in probability performance. Both fitting and final full-policy evaluation use the original G0/θ0 weights. Full selected G0, and independent ES-only also selected G0; full/UQ share the selected weights and observed HV curves, as do ES-only/base. The previous v1 phrase “evolved full policy” incorrectly implied a changed-weight explanation and is superseded here. The original collection episodes start with an empty archive, whereas heldout evaluation starts from the frozen 550-row adaptation prior, followed by its own within-episode history. Controller-driven candidate selection, this prior/history context and the different action populations are possible sources of distribution change. ES weight drift is not an explanation for this result. These remain hypotheses, not causes isolated by this posthoc comparison.

## Permissible conclusions and next evidence

There is observed relative ranking signal in the all-feature model on these fixed selected actions, together with severe probability error. Graph-only contributes no convincing standalone ranking signal here. The analysis neither proves that internal features cause better discoveries nor that graph features are useless in every regime. It provides no basis for selecting a new family, dropping calibration, moving the0.5 threshold, or replacing the original controller using these test outcomes.

The original all-feature policy chose these 224 actions, frequently by selecting among proposals using that same risk model. This creates a selected-policy cohort. All alternative models are scored on the identical cohort for comparability, but their own counterfactual selections and outcomes are unobserved. Three NN training seeds do not remove this bias or replace new independent train/dev validation and prospective policy evaluation.

The complete local archive includes the byte-exact report, registration, model set, original reproduction gate, completion and twelve fit metadata files. `sources/*.gz` contains these scientific JSON sources with deterministic gzip encoding; `source_manifest.json` records compressed and original SHA/bytes. No checkpoint tensors or credentials are included. The original fitting source is copied for provenance, while `recompute.py` is a portable stdlib-only verifier requiring no original machine paths or ML packages. This reviewed v2 archive is now published as a separate dated diagnostic; original scientific artifacts and the failed audit remain unchanged. Publication metadata is recorded separately.

## Fixed calibration bins and paired descriptive checks

`calibration_bins.csv` contains **1,440 rows**: 12 models × raw/calibrated × pooled plus five policy-seed strata × 10 bins. Bins are fixed equal-width [0.0,0.1), …, [0.9,1.0]; each reports count, mean predicted no-HVI risk and observed no-HVI fraction. Empty bins retain unknown means. No bins, thresholds or models were selected from outcomes, and no confidence intervals treat actions as independent. The original all-feature seed1729 calibrated example is:

| Risk bin | Actions | Mean predicted no-HVI | Observed no-HVI |
|---|---:|---:|---:|
| [0.0,0.1) | 90 | 0.070131 | 0.944444 |
| [0.1,0.2) | 73 | 0.148238 | 0.986301 |
| [0.2,0.3) | 26 | 0.231225 | 1.000000 |
| [0.3,0.4) | 13 | 0.343744 | 1.000000 |
| [0.4,0.5) | 12 | 0.444026 | 1.000000 |
| [0.5,0.6) | 4 | 0.558947 | 1.000000 |
| [0.6,0.7) | 3 | 0.652227 | 1.000000 |
| [0.7,0.8) | 3 | 0.721646 | 1.000000 |
| [0.8,0.9) | 0 | undefined | undefined |
| [0.9,1.0] | 0 | undefined | undefined |

Pairing the same NN training seed is descriptive bookkeeping, not independent physics or a confidence interval. Full minus sampling AUROC is **+0.120795, +0.257645, −0.023700** across seeds1729/1730/1731: two positive differences and one negative, not three independent positive claims. Hidden minus sampling is negative/positive/negative; graph minus sampling is negative in all three. Every corresponding full/hidden/graph Brier difference versus sampling is positive (worse). There are only 224 unique executed actions underlying all 2,688 prediction records. The unchanged full=UQ and ES-only=base physical curves are not counted as separate mechanism successes.

The missing sampling+hidden union prevents isolation of incremental graph information in this four-family study. Any subsequently registered union analysis must remain a separate posthoc diagnostic; it cannot retroactively change this matrix, select the deployed controller, or imply prospective policy gain.
