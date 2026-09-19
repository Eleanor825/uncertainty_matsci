# Grouped scalar-risk CPU diagnostic — completed September 19, 19:24:55 CST

**This diagnostic did not demonstrate improved risk prediction or controller benefit.** Both prescribed models completed training and saved-state audit. On the original83 development rows, full internal features give AUROC0.495935 and external action/sampling features give0.430894. Original calibrated NN AUROC is0.516260. No model replaced the original NN, and no materials call, LLM generation, GPU workload or new materials result was produced by this CPU study.

Completion: 2026-09-19T19:24:55.039408+08:00. Read-only observation: 2026-09-19T19:28:21.417572+08:00. Eight model initializations and 1,240 optimizer updates actually ran; the zero-fit counter of the separate observation/export must not be read as zero study training. The run plus its in-process saved-state verification took 20.110s after invocation; this excludes preparation and the separate audit.

## All prescribed models and controls

| Model/scoring | Selected updates | Dev AUROC | Brier | BCE/NLL | Predictions ≥0.6 |
|---|---:|---:|---:|---:|---:|
| full_internal / raw | 8 | 0.495935 | 0.265422749 | 0.726910417 | 23/83 |
| full_internal / calibrated | 8 | 0.495935 | 0.250075533 | 0.693298246 | 0/83 |
| external_only_control / raw | 32 | 0.430894 | 0.264134108 | 0.722517278 | 41/83 |
| external_only_control / calibrated | 32 | 0.430894 | 0.250045530 | 0.693238242 | 0/83 |
| constant_half / constant |  | 0.500000 | 0.250000000 | 0.693147181 | 0/83 |
| constant_train_prior / constant |  | 0.500000 | 0.286147982 | 0.771797538 | 83/83 |
| original_scalar / original_temperature |  | 0.516260 | 0.250006473 | 0.693160126 | 0/83 |

The two positive temperatures are54.597891 and54.597932, near the registered upper bound exp(4)=54.598150. Temperature scaling preserves ranks and AUROC. It drives the poor raw predictions toward0.5; both calibrated Brier scores remain slightly worse than the constant0.5 control. Their positive NLL derivative at the smallest allowed inverse temperature verifies that the upper temperature boundary is optimal within this registered one-parameter calibration family. Raw threshold crossings23/83 and41/83 are not evidence that those triggers are useful. Calibrated crossings are0/83; the threshold was not tuned to create crossings.

## Training-only duration selection

The three folds leave out one original Al-Au-Hf collection episode at a time. Both models use the same nine registered durations (0,2,4,8,16,32,64,128,200 updates), original64/64 GELU architecture, optimizer and training seed1729. External columns are selected before fold-local preprocessing and prototype fitting. The only duration criterion is the unweighted macro mean of three held-out training-episode raw BCEs, with the earlier step winning a tie. Original dev rows do not choose either duration.

| Feature condition | Columns | Selected updates | Selected macro CV BCE | Fold-train-prior constant | Constant0.5 |
|---|---:|---:|---:|---:|---:|
| full_internal | 298 | 8 | 0.665039 | 0.636932 | 0.693147 |
| external_only_control | 15 | 32 | 0.628878 | 0.636932 | 0.693147 |

Internal features do not beat the fold-training-prevalence control on the selection score. External-only features slightly beat it on these folds but fail on the original development chemistry. All54 candidate states are retained in [fold_candidates.csv](fold_candidates.csv), including step0 and later overfitting; [duration_selection.csv](duration_selection.csv) contains all18 macro scores. These are related folds of one training chemistry, not three independently sampled chemical domains.

## Coverage and dependence

The386 original proposals are302 train and84 dev. Preserved future-failure labels admit293 train rows (204 positive,89 negative) and83 dev rows (42 positive,41 negative). Ten labels are unknown, including9 train and1 dev; they are excluded, not relabelled negative. Nine of all386 proposals lack graph support (8 train,1 dev); none of the376 fitted/scored rows lacks a graph. The [386-row coverage table](coverage.csv) preserves every proposal, original episode, chemical group, label/mask and graph availability. Its exact ordered identity hash matches the newly audited production accounting.

Three original training episodes use Al-Au-Hf, and the single original dev episode uses Al-Pd-Sm. These are the project’s original train/dev collection designations; the official MADE benchmark defines the30 test systems. Repeated proposal/event labels are preserved by the original scalar-training target;83 rows are not83 independent episodes. No episode-level or chemical-generalization confidence interval is claimed. Dev temperature calibration is in-sample, and this dev data had already been used by the original NN selection/calibration. The two feature conditions have different input parameter counts. Training seed1729 is fixed; no run-to-run variance is estimated.

## Reproduction and provenance

Run `python3 recompute.py` here. It checks all published hashes and regenerates the tables from [evidence.json](evidence.json), including independent AUROC/Brier/BCE/ECE/AP/risk-coverage calculations from the166 new saved logits and labels. It verifies fold membership and duration selection, calibration boundaries, constant controls and the386-row accounting. The original scalar control is checked against previously published float32 predictions within2e-7; the newly recorded control metrics are retained rather than rounded to those cached values.

The production audit separately loaded all54 saved candidate tensors and both final models and reproduced their predictions. This public checker verifies that audit’s recorded evidence and state hashes but does not re-create unpublished tensors or re-fit a model. Raw input/evaluation/receipt hashes remain distinct from derived export hashes in [provenance.json](provenance.json) and [data_source_refs.json](data_source_refs.json).

Byte-exact scientific [runner.py](diagnostic_source/runner.py), [engine.py](diagnostic_source/engine.py), [original proposal](diagnostic_source/proposal.json), and [executed source manifest](diagnostic_source/executed_source_manifest.json) are included. The original proposal records its earlier planning status; actual completion comes from the later registration/report/audit evidence. Historical absolute scientific storage/runtime paths in these sources require adaptation and access to the pinned inputs; they are not portable standalone launch commands. Operator/observer/authentication/process-control files and.pt tensors are excluded.

An [independent arithmetic review](independent_verified_results.json) agrees on all selected durations and new raw/calibrated development metrics; the public checker verifies this agreement. The study supplies a negative predictive diagnostic, not a materials-discovery comparison. It did not alter the original NN, G2, controller, or primary results. No new scientific call or NN fit was made to produce this export.
