# MADE Tiny20: 20 accepted canonical trajectories out of 20

This fixed export ends at **23 September 2026, 22:30:15 CST (14:30:15 UTC)**. It contains **20 accepted trajectories and 0 unfinished jobs**: **B5 10/10; B15 10/10**. 12 accepted results come from the original attempt, and 8 from separately validated recovery adoptions. All 20 registered jobs are now accepted. This is the final snapshot for the registered Tiny20 batch; results from other experiments are not mixed into it.

The canonical matrix remains two official systems (Mg–Sn–Sr and Au–K–Tb) × B5/B15 × five arms × seed1 = 20 jobs. All Native trajectories are fresh for this Tiny20 batch. The planned candidate-ORB budget is 200; accepted trajectories account for 200 candidate ORB attempts. These two systems and budgets were selected after earlier results were known; this is neither a new holdout nor full-benchmark or repeated-seed evidence.

## All systems, budgets and methods

Each cell shows **SUN / AUDC**. All cells have accepted canonical results; zeros are measured outcomes.

| System | Budget | Native | G0GraphRisk | G2ControllerOff | G2Full | Hidden145Fixed2 |
|---|---:|---:|---:|---:|---:|---:|
| Mg-Sn-Sr | 5 | 5 / 1 | 5 / 1 | 4 / 0.8 | 4 / 0.96 | 4 / 0.8 |
| Mg-Sn-Sr | 15 | 11 / 0.8578 | 11 / 0.7867 | 11 / 0.8044 | 10 / 0.7467 | 10 / 0.7822 |
| Au-K-Tb | 5 | 2 / 0.64 | 4 / 0.64 | 2 / 0.56 | 0 / 0 | 1 / 0.2 |
| Au-K-Tb | 15 | 10 / 0.7644 | 2 / 0.2489 | 4 / 0.1956 | 0 / 0 | 4 / 0.3111 |

## Descriptive Native-paired means

Each alternative arm is compared with Native on the same four system–budget blocks (two systems × two budgets × seed1). Means give each block equal weight. The systems repeat across budgets; these are not four independently sampled systems. SUN means combine two different budgets, so the budget-specific table below is needed for interpretation. No repeated-seed uncertainty or significance test is estimated. Hidden145Fixed2 remains a distinct second-version method and is not pooled with G2Full.

| Arm | Matched blocks | Native mean SUN | Arm mean SUN | Mean ΔSUN | Native mean AUDC | Arm mean AUDC | Mean ΔAUDC |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0GraphRisk | 4 | 7 | 5.5 | -1.5 | 0.8156 | 0.6689 | -0.1467 |
| G2ControllerOff | 4 | 7 | 5.25 | -1.75 | 0.8156 | 0.59 | -0.2256 |
| G2Full | 4 | 7 | 3.5 | -3.5 | 0.8156 | 0.4267 | -0.3889 |
| Hidden145Fixed2 | 4 | 7 | 4.75 | -2.25 | 0.8156 | 0.5233 | -0.2922 |

## Separate B5 and B15 summaries

Each row is the arithmetic mean over Mg–Sn–Sr and Au–K–Tb at the specified budget and seed1. Δ values use the matched Native trajectories from those same systems. All five arms have 2/2 accepted results at each budget.

| Budget | Arm | Systems | Mean SUN | Mean mSUN | Mean AUDC | Mean ΔSUN vs Native | Mean ΔAUDC vs Native |
|---|---|---:|---:|---:|---:|---:|---:|
| 5 | Native | 2 | 3.5 | 0.7 | 0.82 | 0 | 0 |
| 5 | G0GraphRisk | 2 | 4.5 | 0.9 | 0.82 | +1 | 0 |
| 5 | G2ControllerOff | 2 | 3 | 0.6 | 0.68 | -0.5 | -0.14 |
| 5 | G2Full | 2 | 2 | 0.4 | 0.48 | -1.5 | -0.34 |
| 5 | Hidden145Fixed2 | 2 | 2.5 | 0.5 | 0.5 | -1 | -0.32 |
| 15 | Native | 2 | 10.5 | 0.7 | 0.8111 | 0 | 0 |
| 15 | G0GraphRisk | 2 | 6.5 | 0.4333 | 0.5178 | -4 | -0.2933 |
| 15 | G2ControllerOff | 2 | 7.5 | 0.5 | 0.5 | -3 | -0.3111 |
| 15 | G2Full | 2 | 5 | 0.3333 | 0.3733 | -5.5 | -0.4378 |
| 15 | Hidden145Fixed2 | 2 | 7 | 0.4667 | 0.5467 | -3.5 | -0.2644 |

Exact means, including paired mSUN differences, are in [comparison_summaries.json](comparison_summaries.json). These descriptive summaries do not establish a benefit beyond the two selected systems.

## Every paired contrast against Native

Differences are method minus Native within the same system, budget and seed. Both trajectories must be accepted. All recorded contrasts, including negative differences, are retained.

| System | Budget | Arm | ΔSUN | ΔAUDC | Pair status |
|---|---:|---|---:|---:|---|
| Mg-Sn-Sr | 5 | G0GraphRisk | 0 | 0 | both_accepted |
| Mg-Sn-Sr | 5 | G2ControllerOff | -1 | -0.2 | both_accepted |
| Mg-Sn-Sr | 5 | G2Full | -1 | -0.04 | both_accepted |
| Mg-Sn-Sr | 5 | Hidden145Fixed2 | -1 | -0.2 | both_accepted |
| Mg-Sn-Sr | 15 | G0GraphRisk | 0 | -0.0711 | both_accepted |
| Mg-Sn-Sr | 15 | G2ControllerOff | 0 | -0.0533 | both_accepted |
| Mg-Sn-Sr | 15 | G2Full | -1 | -0.1111 | both_accepted |
| Mg-Sn-Sr | 15 | Hidden145Fixed2 | -1 | -0.0756 | both_accepted |
| Au-K-Tb | 5 | G0GraphRisk | +2 | 0 | both_accepted |
| Au-K-Tb | 5 | G2ControllerOff | 0 | -0.08 | both_accepted |
| Au-K-Tb | 5 | G2Full | -2 | -0.64 | both_accepted |
| Au-K-Tb | 5 | Hidden145Fixed2 | -1 | -0.44 | both_accepted |
| Au-K-Tb | 15 | G0GraphRisk | -8 | -0.5156 | both_accepted |
| Au-K-Tb | 15 | G2ControllerOff | -6 | -0.5689 | both_accepted |
| Au-K-Tb | 15 | G2Full | -10 | -0.7644 | both_accepted |
| Au-K-Tb | 15 | Hidden145Fixed2 | -6 | -0.4533 | both_accepted |

G2Full has lower SUN and AUDC than Native in **4/4 completed matched blocks**. At Mg-Sn-Sr B15, Full has SUN 10 versus Native 11, and AUDC 0.7467 versus 0.8578. At Au-K-Tb B15, Full has SUN 0 versus Native 10, and AUDC 0 versus 0.7644. These are descriptive comparisons from one policy seed on two previously examined systems; no significance or generalization claim follows. The same systems occur at both budgets, so the four blocks are not four independently sampled systems. Budget steps are not independent samples. [paired_comparisons.json](paired_comparisons.json) preserves exact values for all 16 completed pairs.

## Accepted measurements and recorded costs

| System | Budget | Arm | Source | SUN | mSUN | AUDC | LLM calls | Graph seconds | Rollout wall seconds |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| Mg-Sn-Sr | 5 | Native | original | 5 | 1 | 1 | 16 | 0 | 280.34 |
| Mg-Sn-Sr | 5 | G0GraphRisk | recovery | 5 | 1 | 1 | 26 | 893.09 | 1385.28 |
| Mg-Sn-Sr | 5 | G2ControllerOff | recovery | 4 | 0.8 | 0.8 | 16 | 0 | 301.22 |
| Mg-Sn-Sr | 5 | G2Full | original | 4 | 0.8 | 0.96 | 23 | 418.58 | 804.95 |
| Mg-Sn-Sr | 5 | Hidden145Fixed2 | original | 4 | 0.8 | 0.8 | 36 | 0 | 634.23 |
| Au-K-Tb | 5 | Native | recovery | 2 | 0.4 | 0.64 | 11 | 0 | 256.97 |
| Au-K-Tb | 5 | G0GraphRisk | original | 4 | 0.8 | 0.64 | 12 | 543.35 | 830.54 |
| Au-K-Tb | 5 | G2ControllerOff | original | 2 | 0.4 | 0.56 | 8 | 0 | 109.64 |
| Au-K-Tb | 5 | G2Full | recovery | 0 | 0 | 0 | 28 | 469.16 | 949.3 |
| Au-K-Tb | 5 | Hidden145Fixed2 | original | 1 | 0.2 | 0.2 | 26 | 0 | 455.45 |
| Mg-Sn-Sr | 15 | Native | recovery | 11 | 0.7333 | 0.8578 | 30 | 0 | 579.32 |
| Mg-Sn-Sr | 15 | G0GraphRisk | original | 11 | 0.7333 | 0.7867 | 54 | 2418.04 | 3674.36 |
| Mg-Sn-Sr | 15 | G2ControllerOff | original | 11 | 0.7333 | 0.8044 | 22 | 0 | 408.87 |
| Mg-Sn-Sr | 15 | G2Full | recovery | 10 | 0.6667 | 0.7467 | 48 | 1609.36 | 2505.03 |
| Mg-Sn-Sr | 15 | Hidden145Fixed2 | original | 10 | 0.6667 | 0.7822 | 72 | 0 | 1394.94 |
| Au-K-Tb | 15 | Native | recovery | 10 | 0.6667 | 0.7644 | 35 | 0 | 1029.52 |
| Au-K-Tb | 15 | G0GraphRisk | original | 2 | 0.1333 | 0.2489 | 40 | 2166.54 | 3132.94 |
| Au-K-Tb | 15 | G2ControllerOff | original | 4 | 0.2667 | 0.1956 | 22 | 0 | 395.54 |
| Au-K-Tb | 15 | G2Full | recovery | 0 | 0 | 0 | 49 | 1465.14 | 2403.65 |
| Au-K-Tb | 15 | Hidden145Fixed2 | original | 4 | 0.2667 | 0.3111 | 48 | 0 | 811.54 |

SUN is the final stable, unique and novel discovery count. mSUN = SUN / B. Normalized AUDC = 2 × trapezoidal area under the recorded SUN curve / B². Every accepted curve includes the origin and B recorded budget steps; no smoothing or cumulative-maximum correction is applied. [curves.json](curves.json) contains every point. [accepted_metrics.json](accepted_metrics.json) retains all exported costs and exact result/completion/adoption references. Missing cost keys remain missing, not zero. Recorded rollout costs do not include all prior model/profile loading or failed-initialization costs and are not total experiment costs.

Native is original G0. G0GraphRisk adds the original graph-risk controller to G0. G2ControllerOff uses the original Full-trained selected generation 2 with its controller disabled; it is not independently trained ES-only. G2Full is the original generation-2 full method. Hidden145Fixed2 combines G0, the frozen MADE hidden145 risk head, exactly two proposals and explicit public feedback; no new ES or risk-head training occurs here. With no Common2 arm, this batch cannot isolate the hidden NN's contribution from extra proposals and feedback. The model remains Qwen/Qwen3.5-4B at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. This Tiny20 configuration is kept separate from the historical all-30-system V1 and later component experiments.

## All 20 canonical statuses

| System | Budget | Arm | Status at export | Original init failure retained |
|---|---:|---|---|---|
| Mg-Sn-Sr | 5 | Native | `accepted_original` | no |
| Mg-Sn-Sr | 5 | G0GraphRisk | `accepted_recovery` | yes |
| Mg-Sn-Sr | 5 | G2ControllerOff | `accepted_recovery` | yes |
| Mg-Sn-Sr | 5 | G2Full | `accepted_original` | no |
| Mg-Sn-Sr | 5 | Hidden145Fixed2 | `accepted_original` | no |
| Au-K-Tb | 5 | Native | `accepted_recovery` | yes |
| Au-K-Tb | 5 | G0GraphRisk | `accepted_original` | no |
| Au-K-Tb | 5 | G2ControllerOff | `accepted_original` | no |
| Au-K-Tb | 5 | G2Full | `accepted_recovery` | yes |
| Au-K-Tb | 5 | Hidden145Fixed2 | `accepted_original` | no |
| Mg-Sn-Sr | 15 | Native | `accepted_recovery` | yes |
| Mg-Sn-Sr | 15 | G0GraphRisk | `accepted_original` | no |
| Mg-Sn-Sr | 15 | G2ControllerOff | `accepted_original` | no |
| Mg-Sn-Sr | 15 | G2Full | `accepted_recovery` | yes |
| Mg-Sn-Sr | 15 | Hidden145Fixed2 | `accepted_original` | no |
| Au-K-Tb | 15 | Native | `accepted_recovery` | yes |
| Au-K-Tb | 15 | G0GraphRisk | `accepted_original` | no |
| Au-K-Tb | 15 | G2ControllerOff | `accepted_original` | no |
| Au-K-Tb | 15 | G2Full | `accepted_recovery` | yes |
| Au-K-Tb | 15 | Hidden145Fixed2 | `accepted_original` | no |

`accepted_original` and `accepted_recovery` each count once by canonical job ID. `recovery_pending` means no adopted accepted result existed at export; `claimed_or_running` means a claim folder existed without completion; neither status proves a live process. `unclaimed` means no attempt folder existed. No canonical row is unfinished in this final export. All metrics are accepted observations; zero scores are measured outcomes.

8 original hfeno attempts failed before environment initialization completed because the ORB cache asset was missing. Their original failures and claims remain intact. Recovery required a pinned zero-physics audit, original process-tree closure and all four original asset hashes, then wrote a separate attempt and adoption with the same canonical job, seed, budget and method. 8 adoptions are accepted here; 0 remain pending adoption. These infrastructure failures are not zero-score outcomes or agent behavioral failures. Failure rates are not inferred from their count. Earlier 3/20, 9/20 and 19/20 publications overlap this snapshot and must not be added to it.

## Evidence and checks

- [snapshot.json](snapshot.json): the fixed exported data, all 20 statuses and nulls unchanged.
- [accepted_metrics.json](accepted_metrics.json), [curves.json](curves.json) and [paired_comparisons.json](paired_comparisons.json): accepted measurements, complete recorded curves and every matched contrast.
- [summary.json](summary.json): counts, budget split and exact export timestamp.
- [comparison_summaries.json](comparison_summaries.json): the four alternative arms paired with Native and separate five-arm B5/B15 means.
- [scientific_sha256.json](scientific_sha256.json): source-envelope/payload hashes, original registry and per-result/completion/adoption SHA references, plus the verification boundary.
- [manifest.json](manifest.json) and [validate.py](validate.py): public bundle hashes and numerical/identity checks; run `python3 -B validate.py` from this directory.

The source-bound CPU exporter checked original acceptance, canonical registration/profile identity, raw result hashes, candidate budgets and recorded metrics. Recovery rows additionally required the adoption and original-failure/completion bindings. Publication rechecked the fixed payload and exporter hashes and recomputed every displayed accepted curve metric and paired delta. Original raw server artifacts remain in the research archive; this bundle does not repeat a full raw-envelope validation or any scientific execution. No later experiment status is substituted for this snapshot.
