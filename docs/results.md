# Experiment results and historical snapshots

The release preserves the experiment's chronology. It reports real MADE test outcomes for **Qwen3.5-4B, seed1, baseline versus the fixed G2 `esopt_graph_risk` method**. B10, B30 and B50 are independent episodes. The latest data export was observed on **2026-09-18 15:32:27 CST (07:32:27 UTC)**; it is not a live dashboard.

## Outcomes

The original two-system B10 comparison had unchanged final SUN (6 versus6) and lower mean AUDC under the full method (0.34→0.25). The original five-system comparison was also negative: SUN12→6 and AUDC0.296→0.100. These results remain in the release.

At01:04:38 CST on September18, B10 had **53/60** completed trajectories and only **23/30** complete pairs. Among those pairs, SUN was22→31 and mean AUDC0.11652→0.14130. This was a partial snapshot, not a completed30-system result. B30 then had only one completed baseline trajectory, so no paired B30 effect was available.

At15:32:27 CST, all **60 B10 trajectories /30 pairs** had completed. SUN was42→54; mean AUDC was0.154667→0.188667 (full minus baseline +0.034). For SUN there were7 winning,5 losing and18 tied systems; for AUDC,8 winning,6 losing and16 tied systems.

At that same observation, B30 had **45/60** completed trajectories: all30 baselines and15 full trajectories, yielding15 complete pairs. On those15 pairs, SUN was110→68 and mean AUDC0.265778→0.163704 (−0.102074). SUN wins/losses/ties were2/6/7; AUDC wins/losses/ties were1/8/6. This partial decline is not omitted or averaged into B10. The remaining15 full outcomes were unavailable. No B50 result had completed.

The complete tables and original cached curves are in [results/snapshots](../results/snapshots). Comparisons use equal weighting across complete system/seed pairs; generation steps are not treated as independent samples. Unpaired completed trajectories are still exported and counted. There is one seed, so no general significance or broad uncertainty-improvement claim is made. The full-method arm bundles a fixed ES-updated policy and risk-guided control; this comparison alone does not isolate the causal contribution of uncertainty guidance from the other components.

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

B10 **data computation** was complete at the latest snapshot, but its all30 global receipt was not yet published because the administrative guard awaited120 new B30/B50 jobs. The original two-system core and five-system extension have separately preserved acceptance evidence. No full three-budget, multi-model, multi-seed or CrystalGym completion is claimed.

## Metrics and time accounting

Let `D(t)` be the cumulative official SUN discovery count after candidate attempt `t`, with `D(0)=0` and budget `B`:

- `SUN = D(B)`; stable/unique/novel status comes from the official recorded MADE results.
- `mSUN = D(B) / B`.
- `AUDC = 2 × Σ[(t₁−t₀)(D(t₁)+D(t₀))/2] / B²`, using the consecutive recorded curve points.

All budget attempts remain in the denominator. A low SUN count is not evidence that fewer evaluations ran. Curves, complete flags and candidate counts were checked against the cited result/RPC audits. Missing costs are left blank rather than inferred to be zero.

Graph seconds are included in rollout wall time; adding them again would double-count. Initialization is separately recorded. Sum of rollout wall times is not elapsed multi-GPU stage time or measured GPU-hours. The timing CSVs preserve these distinctions; later snapshots repeat historical runs and must not be summed as new costs.

## Provenance and publication boundary

These files are **derived exports**, not byte-identical copies of private result envelopes or RPC logs. `raw_result_sha256` and `raw_RPC_sha256` identify original bytes recorded by the source audits; `results/manifest.json` hashes the different exported CSV/JSON bytes. A `source_relative_path` is relative to the original experiment root and does not imply that raw artifact is present in this repository. Local/remote account names and absolute storage paths were removed by field allowlisting.

The early two-system record was initially verified from a cached observer projection; its report was still running at that snapshot. A subsequent independent transfer verified the report/receipt chain, and later audits reverified the same result hashes. The five-system accepted receipt was observed in the later01:04 cache; its preserved mtime is23:23:21 CST on September17. The export labels this distinction rather than inventing an earlier observation timestamp.

No model, evaluator or materials call was made for publication. Large weights, activations, datasets, full RPC bodies, execution/authentication helpers and private configuration are excluded. The source caches are referenced by SHA but are not published verbatim. This release permits reproducible metric calculations from cached curves; it does not claim that this export process reran the complete remote artifact acceptance.

Validate the published export with:

```sh
python3 results/validate_exports.py
```
