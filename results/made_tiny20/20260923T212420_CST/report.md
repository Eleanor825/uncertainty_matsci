# MADE Tiny20: 9 accepted canonical trajectories out of 20

This fixed export ends at **23 September 2026, 21:24:20 CST (13:24:20 UTC)**. It contains **9 accepted trajectories and 11 unfinished jobs**: **B5 7/10; B15 2/10**. Six accepted results come from the original attempt, and three from separately validated recovery adoptions. This is an interim snapshot; later observations are not mixed into it.

The canonical matrix remains two official systems (Mg–Sn–Sr and Au–K–Tb) × B5/B15 × five arms × seed1 = 20 jobs. All Native trajectories are fresh. The planned candidate-ORB budget is 200; the nine accepted trajectories account for 65 candidate ORB attempts. The other jobs' incomplete work is not included in that cost total. These two systems and budgets were selected after earlier results were known; this is neither a new holdout nor full-benchmark or repeated-seed evidence.

## Accepted results

| System | Budget | Arm | Source | SUN | mSUN | AUDC | LLM calls | Graph seconds | Rollout wall seconds |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| Mg-Sn-Sr | 5 | Native | original | 5 | 1 | 1 | 16 | 0 | 280.34 |
| Mg-Sn-Sr | 5 | G2Full | original | 4 | 0.8 | 0.96 | 23 | 418.58 | 804.95 |
| Mg-Sn-Sr | 5 | Hidden145Fixed2 | original | 4 | 0.8 | 0.8 | 36 | 0 | 634.23 |
| Au-K-Tb | 5 | Native | recovery | 2 | 0.4 | 0.64 | 11 | 0 | 256.97 |
| Au-K-Tb | 5 | G0GraphRisk | original | 4 | 0.8 | 0.64 | 12 | 543.35 | 830.54 |
| Au-K-Tb | 5 | G2ControllerOff | original | 2 | 0.4 | 0.56 | 8 | 0 | 109.64 |
| Au-K-Tb | 5 | Hidden145Fixed2 | original | 1 | 0.2 | 0.2 | 26 | 0 | 455.45 |
| Mg-Sn-Sr | 15 | Native | recovery | 11 | 0.7333 | 0.8578 | 30 | 0 | 579.32 |
| Au-K-Tb | 15 | Native | recovery | 10 | 0.6667 | 0.7644 | 35 | 0 | 1029.52 |

SUN is the final stable, unique and novel discovery count. mSUN = SUN / B. Normalized AUDC = 2 × trapezoidal area under the recorded SUN curve / B². Every accepted curve includes the origin and B recorded budget steps; no smoothing or cumulative-maximum correction is applied. [curves.json](curves.json) contains every point. [accepted_metrics.json](accepted_metrics.json) retains all exported costs and exact result/completion/adoption references. Missing cost keys remain missing, not zero. Recorded rollout costs do not include all prior model/profile loading or failed-initialization costs and are not total experiment costs.

The observed comparisons are mixed. At Mg–Sn–Sr B5, G2Full has SUN 4 versus Native 5 and AUDC 0.96 versus 1.00; Hidden145Fixed2 has SUN 4 and AUDC 0.80. At Au–K–Tb B5, G0GraphRisk has SUN 4 versus Native 2, but both have AUDC 0.64; G2ControllerOff has SUN 2 and AUDC 0.56, and Hidden145Fixed2 has SUN 1 and AUDC 0.20. Au–K–Tb G2Full is unfinished in this snapshot. B15 has only Native accepted results, so no B15 method contrast is yet available. Do not rank all methods using unequal completed subsets. System/seed blocks are the comparison units; budget steps are not independent samples.

Native is original G0. G0GraphRisk adds the original graph-risk controller to G0. G2ControllerOff uses the original Full-trained selected generation 2 with its controller disabled; it is not independently trained ES-only. G2Full is the original generation-2 full method. Hidden145Fixed2 combines G0, the frozen MADE hidden145 risk head, exactly two proposals and explicit public feedback; no new ES or risk-head training occurs here. With no Common2 arm, this batch cannot isolate the hidden NN's contribution from extra proposals and feedback. The model remains Qwen/Qwen3.5-4B at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

## All 20 canonical statuses

| System | Budget | Arm | Status at export | Original init failure retained |
|---|---:|---|---|---|
| Mg-Sn-Sr | 5 | Native | `accepted_original` | no |
| Mg-Sn-Sr | 5 | G0GraphRisk | `recovery_pending` | yes |
| Mg-Sn-Sr | 5 | G2ControllerOff | `recovery_pending` | yes |
| Mg-Sn-Sr | 5 | G2Full | `accepted_original` | no |
| Mg-Sn-Sr | 5 | Hidden145Fixed2 | `accepted_original` | no |
| Au-K-Tb | 5 | Native | `accepted_recovery` | yes |
| Au-K-Tb | 5 | G0GraphRisk | `accepted_original` | no |
| Au-K-Tb | 5 | G2ControllerOff | `accepted_original` | no |
| Au-K-Tb | 5 | G2Full | `recovery_pending` | yes |
| Au-K-Tb | 5 | Hidden145Fixed2 | `accepted_original` | no |
| Mg-Sn-Sr | 15 | Native | `accepted_recovery` | yes |
| Mg-Sn-Sr | 15 | G0GraphRisk | `claimed_or_running` | no |
| Mg-Sn-Sr | 15 | G2ControllerOff | `claimed_or_running` | no |
| Mg-Sn-Sr | 15 | G2Full | `recovery_pending` | yes |
| Mg-Sn-Sr | 15 | Hidden145Fixed2 | `claimed_or_running` | no |
| Au-K-Tb | 15 | Native | `accepted_recovery` | yes |
| Au-K-Tb | 15 | G0GraphRisk | `claimed_or_running` | no |
| Au-K-Tb | 15 | G2ControllerOff | `claimed_or_running` | no |
| Au-K-Tb | 15 | G2Full | `recovery_pending` | yes |
| Au-K-Tb | 15 | Hidden145Fixed2 | `unclaimed` | no |

`accepted_original` and `accepted_recovery` each count once by canonical job ID. `recovery_pending` means no adopted accepted result existed at export; `claimed_or_running` means a claim folder existed without completion; neither status proves a live process. `unclaimed` means no attempt folder existed. All 11 unfinished rows retain null metrics in [snapshot.json](snapshot.json), never imputed zeros.

Eight original hfeno attempts failed before environment initialization completed because the ORB cache asset was missing. Their original failures and claims remain intact. Recovery required a pinned zero-physics audit, original process-tree closure and all four original asset hashes, then wrote a separate attempt and adoption with the same canonical job, seed, budget and method. Three adoptions are accepted here; five are pending. These infrastructure failures are not zero-score outcomes or agent behavioral failures. Failure rates are not inferred from their count. The prior 3/20 publication overlaps this snapshot and must not be added to it.

## Evidence and checks

- [snapshot.json](snapshot.json): the fixed exported data, all 20 statuses and nulls unchanged.
- [accepted_metrics.json](accepted_metrics.json) and [curves.json](curves.json): accepted measurements and complete recorded curves.
- [summary.json](summary.json): counts, budget split and exact export timestamp.
- [scientific_sha256.json](scientific_sha256.json): source-envelope/payload hashes, original registry and per-result/completion/adoption SHA references, plus the verification boundary.
- [manifest.json](manifest.json) and [validate.py](validate.py): public bundle hashes and numerical/identity checks; run `python3 -B validate.py` from this directory.

The source-bound CPU exporter checked original acceptance, canonical registration/profile identity, raw result hashes, candidate budgets and recorded metrics. Recovery rows additionally required the adoption and original-failure/completion bindings. Publication rechecked the fixed payload and exporter hashes and recomputed every displayed accepted curve metric. Original raw server artifacts remain in the research archive; this bundle does not repeat a full raw-envelope validation or any scientific execution. No later experiment status is substituted for this snapshot.
