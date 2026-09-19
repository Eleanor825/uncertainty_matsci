# Time-stamped research results

A separate [SnAr feature-family diagnostic](summit_snar/feature_family_diagnostic/20260919T215305_CST/REPORT.md) compares12 small risk NNs (four feature families × three NN seeds) on the same224 previously executed actions. All-feature mean AUROC is **.766055**, but mean calibrated Brier is **.595572**, versus **.027659** for the fixed training-prior predictor; graph-only mean AUROC is **.474261**. This posthoc comparison does not isolate incremental graph value or establish a policy gain. Both physical ES branches selected G0: full/UQ and ES-only/base have duplicate physical curves. The original audit exit1 is preserved alongside a separate byte-exact report reconstruction; the portable checker verifies1,870 metric comparisons without model loading or fitting.

The [complete fixed-core MADE ablation, September19 at21:27:12 CST](made_components/core5/20260919T212712_CST/report.md) now has **60/60 results: five systems × three evaluation seeds × four arms**. Baseline/UQ-controller/ES-only/full total SUN is **22/25/17/20**, with mean AUDC **.156/.192667/.131333/.144**. UQ/controller improves the core mean but loses to baseline on seed4; full remains below baseline. Per-system and paired seed means, sample variances and SDs are available. The two ES policies were trained separately, so these are algorithm-level contrasts and do not isolate the internal-feature NN contribution. These60 runs belong to the ongoing279/1,080 study; the earlier56/60 snapshot remains unchanged.

The [fixed-five MADE B10 report, September19 at20:34:17 CST](made_components/core5/20260919T203417_CST/report.md) now has all three evaluation seeds for baseline, full and ES-only (15 runs each); UQ is11/15. Baseline→full total SUN is **22→20**, and mean AUDC is **.156→.144**. Four of five system-mean AUDCs improve, but the larger Al–V–Zn loss leaves the fixed-cohort aggregate negative. Within-system and paired seed variances (ddof=1) are available; UQ's incomplete15-run aggregates remain blank. These56 completed core runs are a subset of the observed254/1,080 study evaluations, not additional experiments.

The [exact SnAr candidate diagnosis](summit_snar/candidate_analysis/20260919T202943_CST/report.md) adds portable scientific inputs for750 already executed queries,248 full candidates and125 train/dev feature rows. The selected-action failure AUROC is .791284, but Brier is .677841 and mean predicted failure .175444 versus observed .973214. All23 retry queries had zero HV gain, while six gains came from single proposals; this does not identify a causal retry effect. The24 unexecuted candidates remain without invented outcomes. This analysis reuses the accepted35-episode study and does not change its negative aggregate comparison or G0 selection.

A separate [single-development-seed G0 control](development/G0_comparison/20260919T200818_CST/report.md) reports Al–Pd–Sm B10 AUDC **G0 .68 / G1 .62 / G2 .64**. Only the G0 diagnostic added10 candidate calls; G1/G2 are existing references. This previously used development seed is outside both test matrices, and the frozen G2 selection remains unchanged.

A new [SnAr case analysis](summit_snar/case_analysis/20260919T2023_CST/report.md) explains the existing five-seed comparison without adding experiments: full/UQ has two positive pairs, two negative pairs and one tie against Qwen base, and loses to GP-EI in every seed. Its weighted selected-action Brier is .677841 versus .097142 on development. The [label/prior audit](summit_snar/case_analysis/20260919T2023_CST/label_prior_check.md) finds consistent label direction and a declared shift from episode-local cold-start history to a frozen 550-row test prior; that difference is a mechanism hypothesis, not proof of causation or evidence of test-time recalibration. Both ES branches selected G0. Recomputable case tables, scientific source excerpts and remaining evidence gaps are included.

A new [CPU-only grouped scalar-risk diagnostic](development/grouped_scalar_risk/20260919T192455_CST/report.md) completed on September 19 at 19:24:55 CST. Training-only episode cross-validation selected 8 updates for internal features and 32 for external action/sampling features, but original-dev AUROC was **0.495935 and 0.430894**, respectively. Both calibrated models had zero probabilities above the fixed 0.6 threshold and did not beat the constant-0.5 Brier score. The release includes all 54 candidate states’ recorded metrics, 166 new prediction rows, 386-row coverage, byte-exact scientific training code and a portable checker. This was a predictive development diagnostic; no controller, G2 checkpoint or primary experiment result changed.

The [component-study snapshot at September 19, 18:50:04 CST](made_components/20260919T185004_CST/report.md) has **205/1,080 accepted evaluations**. It reports actual within-system evaluation-seed means, sample variances (ddof=1), SDs and missing seeds. The matched full comparison remains negative: SUN45→33 and mean AUDC .192917→.154583 across24 task-seed pairs. Only one full configuration has all three evaluation seeds; no complete all30 full seed exists yet, so its benchmark-level seed variance remains unavailable. This partial study is separate from the original seed1 matrix.

Completed original MADE seed 1 audit: **2026-09-19 05:21:38 CST / September18 21:21:38 UTC**. These are derived exports of observed results, retaining negative outcomes and all earlier incomplete snapshots. No experiment was rerun for publication.

The [complete report](budget_sweep/20260919T052138_CST/report.md) and [180-row matrix](budget_sweep/20260919T052138_CST/all_system_results.csv) cover all30 systems, two methods and independent B10/B30/B50 episodes. **180/180 results are complete:179 original runner acceptances plus one independently derived GaPt baseline acceptance.**

| Budget | Complete pairs | SUN total baseline → full | Mean AUDC baseline → full | SUN W/T/L | AUDC W/T/L |
|---|---:|---:|---:|---:|---:|
| B10 | 30/30 | 42 → 54 | 0.154667 → 0.188667 | 7/18/5 | 8/16/6 |
| B30 | 30/30 | 136 → 97 | 0.160815 → 0.124185 | 5/17/8 | 4/16/10 |
| B50 | 30/30 | 233 → 147 | 0.158840 → 0.112787 | 9/10/11 | 10/10/10 |

**Only seed1 is available; repeated-seed variance is not estimable.** Cross-system variance in the secondary statistics files is a different quantity. The [GaPt reconciliation](budget_sweep/20260919T052138_CST/ga_pt_tm_reconciliation.json) preserves the official nonmonotone curve and original failure. Original global publication is not claimed. Completed evaluation candidate calls total5,400; the5,580 curve points permit independent SUN/AUDC recomputation. Earlier snapshots overlap and must not be added as new execution costs.

The previous [01:10 partial B50 snapshot](budget_sweep/20260919T010939_CST/report.md) remains unchanged.

The following table preserves the earlier observations and is not a live progress table.

| Snapshot / cohort | Budget | Completed trajectories | Complete pairs | SUN total, baseline → full | Mean AUDC, baseline → full |
|---|---:|---:|---:|---:|---:|
| Original two systems, Sep17 16:32 CST | 10 | 4/4 | 2/2 | 6 → 6 | 0.34000 → 0.25000 |
| Original five systems, accepted Sep17; read back Sep18 | 10 | 10/10 | 5/5 | 12 → 6 | 0.29600 → 0.10000 |
| Sep18 01:04:38 CST, partial all30 | 10 | 53/60 | 23/30 | 22 → 31 | 0.11652 → 0.14130 |
| Sep18 15:32:27 CST | 10 | **60/60** | **30/30** | **42 → 54** | **0.15467 → 0.18867** |
| Sep18 15:32:27 CST, partial | 30 | **45/60** | **15/30** | **110 → 68** | **0.26578 → 0.16370** |
| Sep18 15:32:27 CST | 50 | 0/60 | 0/30 | unavailable | unavailable |
| Sep18 18:44:26 CST | 10 | **60/60** | **30/30** | **42 → 54** | **0.15467 → 0.18867** |
| Sep18 18:44:26 CST | 30 | **60/60** | **30/30** | **136 → 97** | **0.16081 → 0.12419** |
| Sep18 18:44:26 CST | 50 | 0/60 | 0/30 | unavailable | unavailable |

Arm comparisons use **only complete system/seed pairs**. The15:32 B30 comparison uses15 baseline and15 full trajectories, while all45 completed trajectories remain in its historical CSV. The18:44 B30 comparison uses all30 complete pairs. Both the earlier partial decline and the completed B30 negative result are retained.

The table above is historical. The latest snapshot has all180 complete trajectories, including one explicitly reconciled GaPt result, while the original global publication receipt is not asserted. At the historical18:44 observation B50 was still unfinished; those missing values remain NA in that snapshot. Cross-system sample variance (ddof=1) and SD use seed1 only; a smaller variance is not evidence of better uncertainty calibration or across-seed repeatability.

Separate development evidence: the [frozen NN information audit, Sep19 08:06:02 CST](development/nn_information_audit/20260919T080602_CST/report.md), includes 386 original train/dev proposals, 200 event-level observations and a cached-data [recomputation script](development/nn_information_audit/20260919T080602_CST/analyze.py). Its single dev episode previously participated in model selection and calibration; it is not an additional held-out test result or part of the 180-trajectory test total.

Separately accepted benchmark: [Summit SnAr, Sep19 16:25:48 CST](summit_snar/20260919T162548_CST/report.md). Seven arms each have five B50 evaluation seeds. The independent acceptance reconciles 2,300 distinct calls, including the shared 550-call prior; its CPU recovery made no new scientific calls. The original failed report remains preserved. Full-minus-base mean HV gain is negative; [per-seed data](summit_snar/20260919T162548_CST/per_seed.csv), [sample variances](summit_snar/20260919T162548_CST/statistics.csv), and [recomputation](summit_snar/20260919T162548_CST/recompute.py) are included. SnAr calls and episodes are separate from the 180 MADE trajectories.

The [component-study snapshot at September 19, 17:26:03 CST](made_components/20260919T172603_CST/progress.md) contains 172/1,080 accepted new evaluations. Its [complete four-arm case comparison](made_components/20260919T172603_CST/report.md) uses the same 17 B10/seed 2 systems: full SUN is 27 versus baseline 34, with mean AUDC .174706 versus .202353. All positive, negative and tied cases remain visible; this partial subset is not a repeated-seed variance estimate.

A [source-bound support-method mechanism audit](development/support_mechanism_audit/20260919T174119_CST/REPORT.md) examines the fixed 17-system comparison and all six new full-ES train/dev trajectories. Both ES updates recorded 723 nonzero parameter blocks, but none of those six trajectories used a neural retry, threshold crossing or risk-ranking change. Deduplicated dev AUROC values (.75 and .7083 over ten events each) coexist with Brier scores near .25. All negative cases and contrary MACE examples are retained; these observations do not establish a neural or screening causal benefit. G0 was not evaluated by the original selection rule, and the report records only the missing-control design at its observation time.

## Files

- Latest per-system tables: [B10](budget_sweep/20260919T052138_CST/systems_B10.md), [B30](budget_sweep/20260919T052138_CST/systems_B30.md), [B50](budget_sweep/20260919T052138_CST/systems_B50.md). [Mean/variance/SD data](budget_sweep/20260919T052138_CST/statistics.csv) and [recomputation script](budget_sweep/20260919T052138_CST/recompute.py) cover the same observation.
- [index.json](index.json): historical snapshot index plus the dated statistical-export index.
- [statistics/b10_b30_complete_20260918T184426_CST](statistics/b10_b30_complete_20260918T184426_CST/README.md): complete B10/B30 pairs, mean/variance/SD tables, source hashes and a stdlib-only reproducible `analyze.py`.
- `snapshots/<id>/trajectories.csv`: every completed test trajectory in that snapshot; model/method/system/seed/budget and SUN/AUDC/mSUN.
- `paired_systems.csv`: complete matched pairs, including all negative and tied outcomes.
- `curves.csv`: cached original current official SUN curves (including dynamic-hull decreases); sufficient to independently recompute the exported SUN/AUDC/mSUN.
- `timings_and_costs.csv`: observed rollout, initialization, graph and verification times/counters. Blank means unavailable, **not zero**.
- `summary.json`: snapshot time, expected/completed counts, publication status, wins/losses/ties and matched-pair means.
- `provenance.json`: original source-cache/result/RPC hashes and scope fingerprints.
- [manifest.json](manifest.json): separate hashes of the **derived published files**.
- [validate_exports.py](validate_exports.py): stdlib validation of hashes, curves, metrics, pairing and repeated-trajectory identity.

The snapshots overlap. Original trajectories are reused read-only; **do not add trajectory or cost totals across snapshots**. The15:32 curve snapshot contains105 distinct completed test trajectories (60 B10 +45 B30), accounting for1950 candidate evaluations in completed trajectories. The18:44 statistical export covers120 distinct completed trajectories (60 B10 +60 B30), accounting for2400 candidate evaluations in completed trajectories. This excludes unfinished trajectories and all training/development/diagnostic costs; it is not the end-to-end research cost.

Detailed interpretation, method/source identities and limitations: [docs/results.md](../docs/results.md).
