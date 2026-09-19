# MADE component study: evaluation-seed statistics at 18:50:04 CST

Observed September 19, 2026 at 10:50:04 UTC / 18:50:04 CST. This is a frozen partial snapshot, separate from the original 180 seed1 trajectories and the accepted SnAr results. **205/1,080 new evaluations are accepted; 7 are claimed without a completed result, 0 failed, and 868 are unclaimed.** All 205 accepted trajectories are B10. B30/B50 have no accepted new results in this snapshot. An open claim is not a measurement of live GPU occupancy.

| Arm | Accepted / 270 | Open claims | Failed | Unclaimed | Fixed configurations with all 3 seeds |
|---|---:|---:|---:|---:|---:|
| Baseline | 68/270 | 1 | 0 | 201 | 8 |
| UQ-only | 43/270 | 2 | 0 | 225 | 0 |
| ES-only | 70/270 | 1 | 0 | 199 | 10 |
| Full | 24/270 | 3 | 0 | 243 | 1 |

## Variance means repeated evaluation seeds within the same setting

For each fixed chemical system, budget and arm, we use only its completed seeds 2/3/4. The sample variance uses **ddof=1**; SD is its square root. With n=1, variance and SD are NA. With n=2 they are computable but explicitly partial because the planned set contains three seeds. Missing runs are never recorded as zero. These seeds repeat evaluation with the same trained weights/controller; they do not repeat model training or checkpoint selection.

[All fixed-configuration mean/variance/SD values](seed_statistics.csv) include SUN, mSUN and AUDC, observed and missing seed lists, even for configurations with no results. [Paired delta variances](paired_seed_statistics.csv) use only matching completed seeds within that same system/budget. No cross-system variance or bootstrap interval is substituted for run-to-run variance.

All 19 configurations that have completed all three evaluation seeds are shown below; this table is not restricted to favorable outcomes. Each row is B10, seeds 2/3/4, n=3.

| System | Arm | SUN mean | SUN sample variance | AUDC mean | AUDC sample variance |
|---|---|---:|---:|---:|---:|
| Al-Li-V | Baseline | 1 | 1 | 0.0766667 | 0.00563333 |
| Al-Li-V | ES-only | 0.333333 | 0.333333 | 0.05 | 0.0075 |
| Al-Li-V | Full | 0.666667 | 1.33333 | 0.0933333 | 0.0261333 |
| Al-V-Zn | Baseline | 5 | 1 | 0.616667 | 0.0104333 |
| Al-V-Zn | ES-only | 4 | 1 | 0.48 | 0.0199 |
| Au-K-Tb | Baseline | 1.33333 | 5.33333 | 0.0866667 | 0.0225333 |
| Au-K-Tb | ES-only | 1 | 0 | 0.123333 | 0.00973333 |
| Co-Dy-W | Baseline | 0 | 0 | 0 | 0 |
| Co-Dy-W | ES-only | 0 | 0 | 0 | 0 |
| Co-Mg-Na | Baseline | 0 | 0 | 0 | 0 |
| Co-Mg-Na | ES-only | 0.333333 | 0.333333 | 0.00333333 | 3.33333e-05 |
| Co-Pd-Tl | Baseline | 0 | 0 | 0 | 0 |
| Co-Pd-Tl | ES-only | 0 | 0 | 0 | 0 |
| Ga-Ho-Lu | Baseline | 4.66667 | 0.333333 | 0.433333 | 0.0352333 |
| Ga-Ho-Lu | ES-only | 3 | 1 | 0.463333 | 0.00803333 |
| Ga-Pt-Tm | Baseline | 4 | 3 | 0.486667 | 0.0124333 |
| Ga-Pt-Tm | ES-only | 4 | 7 | 0.46 | 0.0343 |
| Hf-Ni-Zr | ES-only | 3.33333 | 0.333333 | 0.426667 | 0.00663333 |
| Mg-Sn-Sr | ES-only | 6 | 3 | 0.673333 | 0.00243333 |

Full currently has one completed three-seed configuration: Al–Li–V. Its SUN values are [0, 0, 2], mean 0.666667, sample variance 1.333333 and SD 1.154701. Baseline on those same seeds is [0, 2, 1]. Full-minus-baseline SUN differences [0, −2, 1] have mean −0.333333 and variance 2.333333. The paired AUDC difference is positive on average (+0.016667), illustrating that SUN and discovery timing can disagree. This one system does not establish general repeatability or efficacy. Zero variance in a repeatedly zero-SUN system is not a positive discovery result.

## Comparisons use exactly matched task, budget and seed

| Arm versus baseline | Matched B10 task-seed pairs | SUN total baseline → arm | Mean AUDC baseline → arm | SUN W/T/L | AUDC W/T/L |
|---|---:|---:|---:|---|---|
| UQ-only | 43 | 79 → 83 | 0.193721 → 0.21186 | 11/22/10 | 12/22/9 |
| ES-only | 68 | 108 → 74 | 0.169412 → 0.130882 | 11/36/21 | 15/32/21 |
| Full | 24 | 45 → 33 | 0.192917 → 0.154583 | 3/14/7 | 5/10/9 |

Full has 22 completed seed2 tasks plus Al–Li–V seeds3/4: its 24 pairs represent 22 distinct systems, not 24 independent systems or three complete benchmark repetitions. **The partial full comparison is negative in both total SUN and mean AUDC.** Different arms above have different completed pair sets; their means should not be compared directly between arms. [Every matched pair](paired_results.csv) retains negative, tied and positive values.

The identical four-arm intersection has 23 task-seed sets across 22 systems (the extra set is Al–Li–V seed3):

| Arm | SUN total on same 23 sets | Mean AUDC |
|---|---:|---:|
| Baseline | 44 | 0.194783 |
| UQ-only | 45 | 0.211739 |
| ES-only | 26 | 0.164348 |
| Full | 31 | 0.14913 |

## Core5 and whole-benchmark seed coverage

Core5 is the original fixed set Al–Li–V, Al–V–Zn, Au–K–Tb, Co–Dy–W and Co–Mg–Na. It was not chosen from the current outcomes.

| B10 accepted core5 systems | Baseline | UQ-only | ES-only | Full |
|---|---:|---:|---:|---:|
| seed 2 | 5/5 | 5/5 | 5/5 | 5/5 |
| seed 3 | 5/5 | 5/5 | 5/5 | 1/5 |
| seed 4 | 5/5 | 0/5 | 5/5 | 1/5 |

Core5 has **47/60 B10 trajectories** and 47/180 across B10/B30/B50. Baseline and ES-only each have all three complete core5 seeds; UQ has two and full has one. B30/B50 core5 cells remain unstarted. [Full core5 matrix](core5_matrix.csv).

For a benchmark-level variance, the script first averages over every registered system in a complete seed cohort, then calculates variance across those seed means. It never averages different available subsets and calls them equivalent repetitions. For all30 B10, baseline and ES-only each have two complete seed cohorts, UQ has one, and full has none. Therefore **full has no all30 aggregate seed mean/variance estimate yet**, and no full core5 variance estimate (n=1). These NA values are not zero. See [per-seed cohort means](aggregate_seed_means.csv), [aggregate seed statistics](aggregate_seed_statistics.csv) and [paired aggregate-seed differences](paired_aggregate_seed_statistics.csv).

## Evidence, costs and recomputation

All 205 accepted curves are included (2,255 points including each initial point). The completed new evaluations account for **2,050 candidate ORB calls**. Initialization, MACE screening, LLM calls, tokens, graph time and wall time are separately retained in [timings_and_costs.csv](timings_and_costs.csv), with missing values blank and missing-row counts in summary.json. This excludes unfinished trajectories, independent training/development and earlier studies; it is not total research cost.

The original read-only observer checked each registered job identity, receipt fingerprint/result SHA, candidate budget and curve length. This export checks the exact returned snapshot SHA, recomputes all SUN/mSUN/AUDC curves and statistics, and retains source references. It does not repeat the expensive full remote scientific acceptance. The [scientific input projection](study_inputs.json) excludes operational and authentication records; [provenance](provenance.json) distinguishes raw source hashes from exported-file hashes.

```sh
python3 analyze.py
```

No scientific calls, model calls, fits, checkpoint changes or modifications to original results occurred to produce this export. The earlier dated snapshots, including their incomplete and negative outcomes, remain unchanged.
