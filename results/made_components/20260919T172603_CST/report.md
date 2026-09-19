# UQ control versus ES: fixed 17-case descriptive audit

Snapshot: 2026-09-19T17:26:03.436872+08:00; SHA256 `6b6cab9beeff97ef12d65246a7cf5d034821b27cf746ba881c7a6afa276491cd`.

All 17 complete four-arm B10/seed2 cases are retained. This report does not select a favorable subset or estimate repeated-seed variance.

| Arm | Final SUN sum | Mean AUDC | Candidate ORB calls | LLM calls | Graph time, seconds |
|---|---:|---:|---:|---:|---:|
| Baseline | 34 | 0.202353 | 170 | 390 | 0.0 |
| UQ only | 40 | 0.252941 | 170 | 390 | 20283.4 |
| ES only | 24 | 0.209412 | 170 | 370 | 0.0 |
| Full | 27 | 0.174706 | 170 | 323 | 13169.8 |

Full minus UQ is -13 final SUN in total and -0.078235 mean AUDC. Full minus baseline is -7 final SUN and -0.027647 mean AUDC. These comparisons use the same tasks, budget and evaluation seed.

## All cases

| System | Baseline SUN | UQ SUN | ES SUN | Full SUN | Full minus UQ AUDC | Full minus baseline AUDC |
|---|---:|---:|---:|---:|---:|---:|
| Al-Li-V | 0 | 0 | 1 | 0 | +0.000 | +0.000 |
| Al-V-Zn | 4 | 5 | 4 | 4 | -0.290 | -0.080 |
| Au-Cr-Cs-Dy | 0 | 0 | 4 | 0 | +0.000 | +0.000 |
| Au-K-Tb | 4 | 3 | 1 | 0 | -0.250 | -0.260 |
| Au-Tb-V-Y | 1 | 1 | 1 | 0 | -0.110 | -0.110 |
| Ba-Be-Hf-Li | 0 | 0 | 0 | 0 | +0.000 | +0.000 |
| Ba-Nd-Ni-W | 0 | 0 | 0 | 0 | +0.000 | +0.000 |
| Ca-Pd-Sn-W | 0 | 0 | 0 | 0 | +0.000 | +0.000 |
| Ce-Er-Pb-Rh | 3 | 1 | 1 | 3 | +0.020 | -0.080 |
| Ce-Ir-Pt-Sn | 1 | 4 | 0 | 0 | -0.160 | -0.070 |
| Co-Dy-W | 0 | 2 | 0 | 0 | -0.160 | +0.000 |
| Co-Mg-Na | 0 | 0 | 0 | 2 | +0.080 | +0.080 |
| Co-Pd-Tl | 0 | 0 | 0 | 0 | +0.000 | +0.000 |
| Ga-Ho-Lu | 4 | 6 | 3 | 3 | -0.270 | +0.130 |
| Ga-Pt-Tm | 5 | 3 | 2 | 5 | +0.080 | -0.060 |
| Hf-Ni-Zr | 6 | 5 | 3 | 6 | +0.090 | -0.120 |
| Mg-Sn-Sr | 6 | 10 | 4 | 4 | -0.360 | +0.100 |

## Verified trajectory differences

- **Mg-Sn-Sr:** UQ adds one SUN at every query, ending at10. Full reaches4 at query4 and stays there through query10; ES-only has the same SUN curve. Baseline ends at6. Full nevertheless has higher AUDC than baseline (.64 versus .54), so final discovery count and discovery speed give different comparisons. The plateau alone does not identify duplicates, instability or a causal reason.
- **Au-K-Tb:** full remains at0 throughout all10 queries, while baseline reaches4 and UQ reaches3. This is a sustained discovery failure in this seed, rather than a completed result inferred from an unfinished trajectory.
- **Co-Mg-Na:** full is the positive counterexample: it discovers its first SUN at query7 and ends at2; the other three arms remain at0. It is retained alongside every negative case.
- **Ga-Ho-Lu:** full ends with3 versus baseline4, but has higher AUDC (.45 versus .32). UQ reaches6 and has AUDC.72. A blanket label of better or worse would hide the metric tradeoff.

## Mechanism questions requiring raw decisions

Several full trajectories omit the optional surrogate-call counter, whereas the corresponding UQ trajectories report MACE calls (Al-V-Zn67, Au-K-Tb52, Co-Mg-Na32, Mg-Sn-Sr89). These missing fields are not yet treated as measured zero. Reconcile tool requests and the actual oracle journals before asserting that ES reduced screening. Separately count revisions triggered by neural risk, parser/tool validation and unavailable graphs; a change in controller behavior is not automatically evidence that internal features predict future scientific failure.

Inspect the same distributions on the pre-existing training/development trajectories to decide whether any subsequent version is justified. These held-out cases cannot be used to choose thresholds, select a checkpoint or repeat only poor seeds.

## Evidence and limits

[All cases](all17_cases.csv), [all episodes and costs](all68_episodes_and_costs.csv), [all curves](all68_discovery_curves.csv), [machine-readable summary](summary.json). Every episode row retains its original result/receipt path and SHA256 from the source-bound snapshot. No original scientific file is edited and no experiment is replayed.
