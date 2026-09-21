# DiscoveryWorld: preparatory results and incomplete comparison

Observed **2026-09-21T21:24:01.963647+00:00**. Model: **Qwen3.5-4B**; scenario: **Proteomics Normal**.

**There is no completed held-out Baseline-versus-Full comparison and no demonstrated method improvement in this export.**

| Stage | Completed | Interpretation |
|---|---:|---|
| Native data collection | 8/8 episodes, 800 action attempts | Training/calibration/development sources only |
| Earlier ES-only branch | 6/7 episodes | Separate B100 branch, not the Full method |
| Valid subset graph records | 0/160 | Cross-domain bank fidelity failed |
| Risk NN fit | 0 | Not started |
| Subset Full ES branch | 0/7 episodes | Not started |
| Paired held-out evaluation | 0/2 episodes | Baseline and Full, B30; pending |

The intended subset retains internal graph features, a trained/calibrated failure-risk NN, bounded proposal revisions, and full-parameter Agentic ESOpt. Its Full branch has two ES generations, population two, four B10 candidate trajectories and three B10 development trajectories including G0. The held-out pair uses world seed 3 and policy seed 401. Extra seeds, other worlds and the larger study are deferred.

## Failed bank transfer

The MADE-trained transcoder bank did not pass the original DiscoveryWorld qualification gate: layer 0 output FVU was **0.843559980392456**, above the unchanged **0.5** limit. No successful graph was produced. These attempts share the same bank and fixed prefix suite; they are not independent efficacy trials. The failed records and hashes are retained. A DW-domain bank must pass the original fidelity and attribution checks before the Full pipeline can proceed.

| Graph group | Claimed | Failed | Complete |
|---|---|---|---|
| group0 | True | True | False |
| group1 | True | True | False |
| group2 | True | True | False |
| group3 | True | True | False |

## Native source trajectories

Scores below belong to preparatory B100 collection episodes. They are not paired held-out baseline results.

| World seed | Policy seed | Role | Attempts | Final normalized score | Task success |
|---:|---:|---|---:|---:|---|
| 0 | 101 | NN fit / TC training | 100 | 0.0 | False |
| 0 | 102 | NN fit / TC training | 100 | 0.375 | False |
| 0 | 103 | NN fit / TC training | 100 | 0.0 | False |
| 1 | 101 | NN calibration / TC development | 100 | 0.0 | False |
| 1 | 102 | NN calibration / TC development | 100 | 0.25 | False |
| 1 | 103 | NN calibration / TC development | 100 | 0.5 | False |
| 2 | 201 | NN development | 100 | 0.375 | False |
| 2 | 202 | NN development | 100 | 0.5 | False |

## Earlier ES-only preparation

These completed B100 training/development episodes are separate from the new B10 Full search. Do not pool their scores or count them as held-out Full evaluations.

| Stage | Generation | Member | World / policy seed | Attempts | Normalized score | Failure fraction | Fitness |
|---|---:|---:|---|---:|---:|---:|---:|
| es_dev | 0 | — | 2 / 303 | 100 | 0.0 | 0.87 | -0.08700000000000001 |
| es_dev | 1 | — | 2 / 303 | 100 | 0.0 | 0.99 | -0.099 |
| es_train | 1 | 0 | 0 / 301 | 100 | 0.125 | 0.06 | 0.119 |
| es_train | 1 | 1 | 0 / 301 | 100 | 0.0 | 0.5 | -0.05 |
| es_train | 2 | 0 | 1 / 302 | 100 | 0.0 | 0.99 | -0.099 |
| es_train | 2 | 1 | 1 / 302 | 100 | 0.125 | 0.22 | 0.103 |

## Provenance

`snapshot.json` contains a scientific-field projection of existing records, source hashes and project-relative references. No prompts, credentials, hostnames or process records are included. Acceptance here reads original completed records; it does not rerun the simulator or raw acceptance procedure. The snapshot is sequential while other work may continue.

Run `python3 -B validate.py` to verify identities, counts, metric arithmetic, file hashes and this rendered report.
