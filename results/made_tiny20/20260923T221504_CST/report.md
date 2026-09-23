# MADE Tiny20: 19 accepted canonical trajectories out of 20

This fixed export ends at **23 September 2026, 22:15:04 CST (14:15:04 UTC)**. It contains **19 accepted trajectories and 1 unfinished job**: **B5 10/10; B15 9/10**. 11 accepted results come from the original attempt, and 8 from separately validated recovery adoptions. This is an interim snapshot; later observations are not mixed into it.

The canonical matrix remains two official systems (Mg–Sn–Sr and Au–K–Tb) × B5/B15 × five arms × seed1 = 20 jobs. All Native trajectories are fresh for this Tiny20 batch. The planned candidate-ORB budget is 200; accepted trajectories account for 185 candidate ORB attempts. Incomplete work is excluded from that accepted-only cost total. These two systems and budgets were selected after earlier results were known; this is neither a new holdout nor full-benchmark or repeated-seed evidence.

## All systems, budgets and methods

Each cell shows **SUN / AUDC**. A dash means no accepted canonical result at the export time, never a zero imputation.

| System | Budget | Native | G0GraphRisk | G2ControllerOff | G2Full | Hidden145Fixed2 |
|---|---:|---:|---:|---:|---:|---:|
| Mg-Sn-Sr | 5 | 5 / 1 | 5 / 1 | 4 / 0.8 | 4 / 0.96 | 4 / 0.8 |
| Mg-Sn-Sr | 15 | 11 / 0.8578 | 11 / 0.7867 | 11 / 0.8044 | 10 / 0.7467 | 10 / 0.7822 |
| Au-K-Tb | 5 | 2 / 0.64 | 4 / 0.64 | 2 / 0.56 | 0 / 0 | 1 / 0.2 |
| Au-K-Tb | 15 | 10 / 0.7644 | — | 4 / 0.1956 | 0 / 0 | 4 / 0.3111 |

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
| Au-K-Tb | 15 | G0GraphRisk | — | — | incomplete_pair |
| Au-K-Tb | 15 | G2ControllerOff | -6 | -0.5689 | both_accepted |
| Au-K-Tb | 15 | G2Full | -10 | -0.7644 | both_accepted |
| Au-K-Tb | 15 | Hidden145Fixed2 | -6 | -0.4533 | both_accepted |

G2Full has lower SUN and AUDC than Native in **4/4 completed matched blocks**. At Mg-Sn-Sr B15, Full has SUN 10 versus Native 11, and AUDC 0.7467 versus 0.8578. At Au-K-Tb B15, Full has SUN 0 versus Native 10, and AUDC 0 versus 0.7644. These are descriptive comparisons from one policy seed on two previously examined systems; no significance or generalization claim follows. The same systems occur at both budgets, so the four blocks are not four independently sampled systems. Do not rank all methods using unequal completed subsets. Budget steps are not independent samples. [paired_comparisons.json](paired_comparisons.json) preserves exact values and all incomplete pairs.

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
| Au-K-Tb | 15 | G0GraphRisk | `claimed_or_running` | no |
| Au-K-Tb | 15 | G2ControllerOff | `accepted_original` | no |
| Au-K-Tb | 15 | G2Full | `accepted_recovery` | yes |
| Au-K-Tb | 15 | Hidden145Fixed2 | `accepted_original` | no |

`accepted_original` and `accepted_recovery` each count once by canonical job ID. `recovery_pending` means no adopted accepted result existed at export; `claimed_or_running` means a claim folder existed without completion; neither status proves a live process. `unclaimed` means no attempt folder existed. All 1 unfinished rows retain null metrics in [snapshot.json](snapshot.json), never imputed zeros.

8 original hfeno attempts failed before environment initialization completed because the ORB cache asset was missing. Their original failures and claims remain intact. Recovery required a pinned zero-physics audit, original process-tree closure and all four original asset hashes, then wrote a separate attempt and adoption with the same canonical job, seed, budget and method. 8 adoptions are accepted here; 0 remain pending adoption. These infrastructure failures are not zero-score outcomes or agent behavioral failures. Failure rates are not inferred from their count. Earlier 3/20 and 9/20 publications overlap this snapshot and must not be added to it.

## Evidence and checks

- [snapshot.json](snapshot.json): the fixed exported data, all 20 statuses and nulls unchanged.
- [accepted_metrics.json](accepted_metrics.json), [curves.json](curves.json) and [paired_comparisons.json](paired_comparisons.json): accepted measurements, complete recorded curves and every matched contrast.
- [summary.json](summary.json): counts, budget split and exact export timestamp.
- [scientific_sha256.json](scientific_sha256.json): source-envelope/payload hashes, original registry and per-result/completion/adoption SHA references, plus the verification boundary.
- [manifest.json](manifest.json) and [validate.py](validate.py): public bundle hashes and numerical/identity checks; run `python3 -B validate.py` from this directory.

The source-bound CPU exporter checked original acceptance, canonical registration/profile identity, raw result hashes, candidate budgets and recorded metrics. Recovery rows additionally required the adoption and original-failure/completion bindings. Publication rechecked the fixed payload and exporter hashes and recomputed every displayed accepted curve metric and paired delta. Original raw server artifacts remain in the research archive; this bundle does not repeat a full raw-envelope validation or any scientific execution. No later experiment status is substituted for this snapshot.
