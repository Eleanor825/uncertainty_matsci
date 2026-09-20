# Experiment results and historical snapshots

The [complete original B50 mechanism audit](../results/budget_sweep/B50_mechanism_analysis/20260920T054512_CST/REPORT.md) covers all **30 systems × baseline/full at seed1**, 60 existing trajectories and3,000 ORB steps. Stable-and-new submission events are **237 versus148**, while dynamic-hull reclassification removes **4 versus1** old discoveries, leaving unchanged official SUN **233 versus147**. Reclassification therefore narrows Full's deficit by3; it does not explain the main loss. Full has417 non-novel candidates versus183, but no within-episode repeated candidate hashes. All positive/negative systems, five temporal bands and a portable CPU convex-hull recomputation are included. This single-seed posthoc audit makes no new scientific calls and does not select training data or hyperparameters.

The [all-complete B10 mechanism audit](../results/made_components/mechanism_analysis/20260920T043843_CST/REPORT.md) covers all **24 systems × three seeds × baseline/full** complete at the fixed429/1,080 snapshot:144 existing trajectories and1,440 ORB steps. Full total SUN is **92 versus118** baseline, with mean AUDC **.151111 versus.175278**. Full had zero scalar-threshold crossings and zero risk-ranking selection changes; all21 type-triggered extra proposals were unused, while44 selected second proposals followed schema repair. Positive and negative cases, graph support, calibrated-risk diagnostics, dynamic-hull offsets and full raw-source hashes are retained. This posthoc analysis makes no new calls and does not establish a neural or ES causal benefit.

A new [CPU-only grouped scalar-risk diagnostic](../results/development/grouped_scalar_risk/20260919T192455_CST/report.md) completed on September 19 at 19:24:55 CST. Training-only episode cross-validation selected 8 updates for internal features and 32 for external action/sampling features, but original-dev AUROC was **0.495935 and 0.430894**, respectively. Both calibrated models had zero probabilities above the fixed 0.6 threshold and did not beat the constant-0.5 Brier score. The release includes all 54 candidate states’ recorded metrics, 166 new prediction rows, 386-row coverage, byte-exact scientific training code and a portable checker. This was a predictive development diagnostic; no controller, G2 checkpoint or primary experiment result changed.

This release preserves the chronology of real MADE test outcomes for **Qwen3.5-4B, evaluation seed1, baseline versus the fixed G2 `esopt_graph_risk` method**. B10, B30 and B50 are independent episodes. The latest audit finished on **2026-09-19 05:21:38 CST (September18 21:21:38 UTC)**. Earlier dated snapshots remain unchanged.

## Complete B10/B30/B50 results

The [complete report](../results/budget_sweep/20260919T052138_CST/report.md) and [180-row system matrix](../results/budget_sweep/20260919T052138_CST/all_system_results.csv) now contain **180/180 completed trajectories and30/30 matched chemical-system pairs at each budget**. This comprises179 originally accepted results and **one independently reconciled Ga-Pt-Tm B50 baseline**; the original runner's failure record remains unchanged.

| Budget | Completed trajectories | Complete pairs | SUN total baseline → full | Mean AUDC baseline → full | SUN wins / ties / losses | AUDC wins / ties / losses |
|---|---:|---:|---:|---:|---:|---:|
| B10 | 60/60 | 30/30 | 42 → 54 | 0.154667 → 0.188667 | 7 / 18 / 5 | 8 / 16 / 6 |
| B30 | 60/60 | 30/30 | 136 → 97 | 0.160815 → 0.124185 | 5 / 17 / 8 | 4 / 16 / 10 |
| B50 | 60/60 | 30/30 | 233 → 147 | 0.158840 → 0.112787 | 9 / 10 / 11 | 10 / 10 / 10 |

The full method has a higher mean at B10 and lower means at B30 andB50 on both reported metrics. All positive, negative and zero outcomes are retained. These are descriptive comparisons for the fixed model/checkpoint and seed1; they do not establish general effectiveness or isolate the causal contribution of uncertainty guidance.

Readable all30-system tables: [B10](../results/budget_sweep/20260919T052138_CST/systems_B10.md), [B30](../results/budget_sweep/20260919T052138_CST/systems_B30.md), [B50](../results/budget_sweep/20260919T052138_CST/systems_B50.md). The [paired CSV](../results/budget_sweep/20260919T052138_CST/paired_systems.csv) includes every system, SUN/mSUN/AUDC and full-minus-baseline differences. The last Al-Hg-K-Mg-W B50 baseline completed50 evaluations andclose with SUN0/AUDC0, matching the already-completed full arm.

Ga-Pt-Tm baseline B50 completed its50 physical evaluations, but the original post-processing gate incorrectly required a nondecreasing discovery count. The official dynamic hull changed SUN from9 at step25 to7 at step26; the final SUN is18 andAUDC0.3664. The [independent reconciliation](../results/budget_sweep/20260919T052138_CST/ga_pt_tm_reconciliation.json) preserves that curve, hashes the original evidence and records explicit adoption for combined reporting. It does not fabricate an original runner success or rerun the experiment.

**Run-to-run variance is unavailable:** every evaluated configuration has only seed1. [Seed coverage](../results/budget_sweep/20260919T052138_CST/seed_coverage.csv) records that absence. The sample variance/SD in [statistics.csv](../results/budget_sweep/20260919T052138_CST/statistics.csv) describe variation across chemical systems at seed1 (ddof=1), not different random-seed repetitions. A lower cross-system variance does not demonstrate more reliable uncertainty estimates.

The export contains [5,580 original-curve points](../results/budget_sweep/20260919T052138_CST/curves.csv), original result/RPC/receipt-or-ledger hashes, and [separate cost counters](../results/budget_sweep/20260919T052138_CST/cost_summary.csv). Completed evaluation candidate calls total5,400:600 atB10,1,800 atB30 and3,000 atB50. Initialization, surrogate calls, training, and offline diagnostics are not silently folded into this count. No scientific call was made to produce the export.

The original global experiment publication receipt is **not asserted**: the old global path still has the unrevised GaPt acceptance failure. This snapshot is the explicitly documented combined result audit, not a claim that the original multi-model/multi-seed/CrystalGym study is complete.

The earlier [01:10:39 CST snapshot](../results/budget_sweep/20260919T010939_CST/report.md) remains intact with154 completed trajectories and only4 B50 pairs. It is not retroactively labelled complete.

## Historical outcomes

The original two-system B10 comparison had unchanged final SUN (6 versus6) and lower mean AUDC under the full method (0.34→0.25). The original five-system comparison was also negative: SUN12→6 and AUDC0.296→0.100. These results remain in the release.

At01:04:38 CST on September18, B10 had **53/60** completed trajectories and only **23/30** complete pairs. Among those pairs, SUN was22→31 and mean AUDC0.11652→0.14130. This was a partial snapshot, not a completed30-system result. B30 then had only one completed baseline trajectory, so no paired B30 effect was available.

At15:32:27 CST, all **60 B10 trajectories /30 pairs** had completed. SUN was42→54; mean AUDC was0.154667→0.188667 (full minus baseline +0.034). For SUN there were7 winning,5 losing and18 tied systems; for AUDC,8 winning,6 losing and16 tied systems.

At that same observation, B30 had **45/60** completed trajectories: all30 baselines and15 full trajectories, yielding15 complete pairs. On those15 pairs, SUN was110→68 and mean AUDC0.265778→0.163704 (−0.102074). SUN wins/losses/ties were2/6/7; AUDC wins/losses/ties were1/8/6. This partial decline is not omitted or averaged into B10. The remaining15 full outcomes were unavailable. No B50 result had completed.

At18:44:26 CST, **B10 and B30 each had60/60 real trajectories and30/30 complete chemical-system pairs**. B10 remains SUN42→54 and mean AUDC0.154667→0.188667. Complete B30 is SUN136→97 and mean AUDC0.160815→0.124185 (full minus baseline −0.036630). Thus the full method has lower mean SUN and AUDC at B30; the negative result is retained. B50 had0 completed trajectories,10 claimed unfinished/running and50 unclaimed. Its missing comparative statistics are unavailable, not zero performance.

The [dated statistics export](../results/statistics/b10_b30_complete_20260918T184426_CST/README.md) provides all paired values, means, sample variances, SDs and population variances, including each system's full-minus-baseline difference. The following sample variances use **ddof=1 across the30 chemical systems at seed1**:

| Budget | Metric | Baseline mean | Baseline variance | Baseline SD | Full mean | Full variance | Full SD |
|---|---|---:|---:|---:|---:|---:|---:|
| B10 | SUN | 1.400000 | 5.351724 | 2.313379 | 1.800000 | 6.510345 | 2.551538 |
| B10 | AUDC | 0.154667 | 0.062929 | 0.250857 | 0.188667 | 0.077557 | 0.278490 |
| B30 | SUN | 4.533333 | 50.602299 | 7.113529 | 3.233333 | 22.116092 | 4.702775 |
| B30 | AUDC | 0.160815 | 0.058769 | 0.242423 | 0.124185 | 0.030844 | 0.175624 |

These describe variation across chemical systems, **not independent random-seed repetitions**. Smaller variance does not establish improved uncertainty calibration, repeatability or uncertainty-guidance efficacy; B30's full-method mean is also lower. The paired-difference variance is calculated after subtracting the two arms within each system.

The earlier dated tables and cached curves are in [results/snapshots](../results/snapshots). Comparisons use equal weighting across complete system/seed pairs; generation steps are not treated as independent samples. Unpaired completed trajectories are still exported and counted. There is one seed, so no general significance or broad uncertainty-improvement claim is made. The full-method arm bundles a fixed ES-updated policy and risk-guided control; this comparison alone does not isolate the causal contribution of uncertainty guidance from the other components.

## Scientific and accounting scope

The evaluated settings retain the same pretrained model, official MADE evaluator, chemical splits and fixed G2 checkpoint/controllers. The official30 test systems exclude collection/training/development systems. Coverage expansions and the budget sweep were registered **after earlier test outcomes were known**, without checkpoint reselection or refitting on those outcomes. These are transparent post-results extensions, not a claim that the complete sequence was preregistered before the initial tests.

The representation corpus used historical B50 train/dev trajectories. G2 selection used the separately registered B10 ES protocol. Evaluations at B30/B50 do not change those training/selection budgets. B10, B30 and B50 are fresh independent trajectories, not continuations of a B10 episode.

The actual-model-state hashes recorded on B10 are:

- Baseline: `6a308686126804830c81a7d2a3495feda3b3a13b074b9958e0e6026759826516`.
- Fixed full/G2: `2e5adce2a0d3785a1e1df3c41888f42eed2a23b9e175e3e0b1a4e1d3ef534fff`.
- Model revision: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

The new-budget audit checked equality with the sealed execution profile. Where a cached projection did not duplicate revision/state-hash fields, the CSV leaves those cells blank and provenance explains the separate profile check; no missing field is fabricated.

Scope fingerprints are preserved in each provenance file:

| Scope | Fingerprint |
|---|---|
| Original FAST core | `725a5febe58990defc7a681dce24b503bbf605016f410d089ee36a128210b154` |
| Five-system extension | `2e1c40d56bbde23fe1c4a42b1afda84dd7fb5ba4d1879723a90a24636219f274` |
| All30 B10 extension | `a33c801bf2cea72eb048a23ac231144ffcdb0cbbb32a7bbc141aad22764f4140` |
| B30/B50 sweep | `407b6cf226a47ce6b1d52bef8e8ed617abeabb9f70d3ee9df3e4d946da0d3f95` |

The B10 all30 runner source SHA is `3573a8e712bb5f7cad99cdc00f764cc4e57e47849f054a011854777f59986f6d`; the frozen sweep runner SHA is `ce38b4614d67cbf50423808bb9f61a87660c5b72efd8023dfd4acc833bce5d66`. Historical scopes used their own frozen sources. The code publication and the per-snapshot source fingerprints must not be conflated.

At the earlier complete-B30 snapshot, B10 and B30 **data computation** were complete, while B50 was unfinished. The latest snapshot above now covers all180 trajectories, including explicit independent GaPt acceptance. This does not assert that the original global publication receipt has been repaired or emitted. The original two-system core and five-system extension have separately preserved acceptance evidence. Completion here is limited to the specified4B/seed1 three-budget data; multi-model, repeated-seed and CrystalGym study completion are not claimed.

## Metrics and time accounting

Let `D(t)` be the current official SUN discovery count (which may decrease as the hull changes) after candidate attempt `t`, with `D(0)=0` and budget `B`:

- `SUN = D(B)`; stable/unique/novel status comes from the official recorded MADE results.
- `mSUN = D(B) / B`.
- `AUDC = 2 × Σ[(t₁−t₀)(D(t₁)+D(t₀))/2] / B²`, using the consecutive recorded curve points.

All budget attempts remain in the denominator. A low SUN count is not evidence that fewer evaluations ran. Curves, complete flags and candidate counts were checked against the cited result/RPC audits. Missing costs are left blank rather than inferred to be zero.

Graph seconds are included in rollout wall time; adding them again would double-count. Initialization is separately recorded. Sum of rollout wall times is not elapsed multi-GPU stage time or measured GPU-hours. The timing CSVs preserve these distinctions; later snapshots repeat historical runs and must not be summed as new costs.

## Provenance and publication boundary

These files are **derived exports**, not byte-identical copies of private result envelopes or RPC logs. `raw_RPC_sha256` identifies original RPC bytes. The legacy `raw_result_sha256` identifies the actual source result envelope; for the explicitly labelled GaPt reconciliation this is the derived envelope, not an invented original result; `results/manifest.json` hashes the different exported CSV/JSON bytes. A `source_relative_path` is relative to the original experiment root and does not imply that raw artifact is present in this repository. Local/remote account names and absolute storage paths were removed by field allowlisting.

The early two-system record was initially verified from a cached observer projection; its report was still running at that snapshot. A subsequent independent transfer verified the report/receipt chain, and later audits reverified the same result hashes. The five-system accepted receipt was observed in the later01:04 cache; its preserved mtime is23:23:21 CST on September17. The export labels this distinction rather than inventing an earlier observation timestamp.

No model, evaluator or materials call was made for publication. Large weights, activations, datasets, full RPC bodies, execution/authentication helpers and private configuration are excluded. The source caches are referenced by SHA but are not published verbatim. This release permits reproducible metric calculations from cached curves; it does not claim that this export process reran the complete remote artifact acceptance.

Validate the published export with:

```sh
python3 results/validate_exports.py
python3 results/statistics/b10_b30_complete_20260918T184426_CST/analyze.py
python3 results/budget_sweep/20260919T010939_CST/recompute.py
```

The statistics script uses only the standard library and regenerates its two output files from the published pairs. Publication checks confirmed byte-identical regeneration, matched all30 B10 pairs to the earlier public export, and recomputed all60 B30 metrics from the audited RPC curves. The B30 audit snapshot SHA256 is `2297d8394f4d1e0662700a42d5da8ed13d7eb87738f8029c9ca4b41c59aa4822`; per-result/receipt/RPC hashes are retained in the statistical export provenance.


## Additive analysis publication, September20

The [paper-ready SnAr table](../results/summit_snar/paper_table/20260919T162548_CST/REPORT.md) consolidates all **35/35 held-out trajectories**, seven arms × five evaluation seeds × B50, with means, sample variances and a [LaTeX table](../results/summit_snar/paper_table/20260919T162548_CST/table.tex). GP-EI has the highest mean incremental HV. Full remains below Qwen baseline; both ES branches selected G0 after their actual development runs, so full/UQ and ES-only/baseline curves coincide. These are existing completed results, not additional experiments.

The [sampling+hidden diagnostic](../results/summit_snar/sampling_hidden_diagnostic/20260919T221018_CST/REPORT.md) adds three recorded CPU NN fits (180 updates;15 cumulative models/926 updates) on the same224 executed actions. Mean AUROC is **.717125** and calibrated Brier **.541182**. Its matched comparison with all285 features is mixed across NN seeds and metrics; no replacement was deployed and no graph-causal benefit is claimed. The eight original JSON artifacts, source/plan,672 predictions and588 independently recomputed metrics are included.

The [complete-core MADE mechanism analysis](../results/made_components/core5/mechanism_analysis/20260919T220626_CST/REPORT.md) joins all60 existing trajectories to1,237 proposals,1,174 executed tools and600 ORB steps. UQ's observed gain coincides with fewer failed selections and schema/controller behavior; its15 runs had **zero neural-triggered extra proposals or risk-ranking changes**. Full produced8 neural extra proposals, all unexecuted; no risk-ranking changes were observed. This does not isolate an NN causal contribution or justify test tuning. The complete scientific projection and Python3.12 byte-exact analysis are portable.

## Capacity planning is separate from results

A separate [24-hour capacity plan](../results/planning/capacity/20260920T010300_CST/REPORT.md) uses the actual362-completion baseline at September20,01:03 CST and measured per-job timings. Under the stated17-worker allocation plus one B/E priority card, it projects roughly320–390 additional4B MADE trajectories, with explicit B50 extrapolation and queue-order bottlenecks. **These are conditional planning numbers, not completed results.** The independently published [01:17:20 result snapshot](../results/made_components/incremental/20260920T011720_CST/report.md) has369 actual accepted evaluations. Forecasts never enter experimental means or replace the30-minute result updates.
