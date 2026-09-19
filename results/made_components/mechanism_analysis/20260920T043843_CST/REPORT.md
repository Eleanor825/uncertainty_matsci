# MADE: mechanisms across all complete three-seed B10 baseline/full systems

**The full method is lower overall in this fixed matched subset. Its scalar NN never crosses the decision threshold, and no supported valid-candidate risk ranking changes the selected action.** There are positive individual systems, but the observed gains cannot be attributed to scalar-risk intervention or demonstrated beneficial NN reranking.

This is a posthoc held-out audit of all 24 chemistries with complete baseline/full pairs for seeds2/3/4 at the fixed 429/1080 snapshot (2026-09-20 04:38:43 CST): 144 trajectories and 1440 candidate ORB evaluations. The six other chemistries were incomplete at that snapshot and remain listed in the cohort. This is not the entire1080 study, the old seed1/180 study, or prospective training evidence. No held-out observation enters the new scalar-NN development scope.

| Quantity | Baseline | Full |
|---|---:|---:|
| Complete trajectories | 72 | 72 |
| Final SUN summed over24systems ×3seeds | 118 | 92 |
| Mean AUDC over matched runs | 0.175277778 | 0.151111111 |
| Per-seed total SUN | 44,40,34 | 33,32,27 |
| Per-seed macro AUDC | 0.1925,0.163333333,0.17 | 0.159583333,0.165833333,0.127916667 |
| Across-seed variance of total SUN (ddof1) | 25.3333333 | 10.3333333 |
| Across-seed variance of macro AUDC (ddof1) | 0.000233565 | 0.000413252 |

By chemistry mean, full has6 SUN wins,9 ties and9 losses; AUDC has8 wins,7 ties and9 losses. Per-seed total SUN differences are−11,−8,−7; macro-AUDC differences are−0.032916667,+0.0025,−0.042083333. The variance above is across the three evaluation seeds of the same24-chemistry aggregate, not across chemistry or independent training repeats.

## What the controller actually did

Full produced1340 proposals:905 had admitted native graphs and435 had unavailable graphs. Of1299 successful generations with recorded scalar NN values, the range was0.498466790–0.501267910. On supported candidates the range was0.499067724–0.501267910. The fixed scalar threshold is0.6: there were **zero scalar-threshold retries or crossings**. Baseline’s separate logging-only scalar is not an intervention and is not the graph NN.

| Full controller event | Count |
|---|---:|
| Observed schema retry requests |45|
| Actual extra proposals after schema requests |44|
| Actual selected second proposals after schema requests |44|
| Predicted-type retry requests (including later candidates) |24|
| Actual extra proposals after predicted-type requests |21|
| Those21 extra proposals selected |0|
| Those21 extra proposals lacking graph support |20|
| Candidate sets with at least2 controller-valid proposals |20|
| Such sets with every candidate supported |1|
| Supported risk ranking changed selection from first valid candidate |0|

All21 extra proposals prompted by predicted type risk were rejected, including the one supported second proposal. They consumed318.090428s of generation and249.986853s of graph processing, with44469 prompt tokens and3232 completion tokens. These are measured extra-computation costs, not additional oracle calls. Type requests predominantly involved `not_new` (21 triggered-head occurrences), with `tool_execution_failure`3 and `unstable`1; a request can include multiple heads. A request or `type_signals_used` flag is not evidence that a different action executed.

Both positive and negative outcome groups have zero scalar-threshold interventions and zero supported risk-selection changes. Schema retry/second-selection behavior occurs in both groups. Their shared occurrence does not independently explain why one group improved and another regressed. The comparison also changes policy weights: baseline is the fixed theta0 state while full is the selectedG2 state. It therefore does not isolate ES, rules, changed stochastic trajectories, or uncertainty as a causal contributor.

## Predictive diagnostics, without recalibration

For full’s **527 supported direct material selections**, scalar predictions average0.500307588 while observed next-evaluation failure is0.842504744; Brier0.249795888, AUROC0.492700532 and ECE10=0.342197156. These are nearly constant risk predictions with poor calibration on this slice. The192 unsupported direct selections have Brier0.249481728 and AUROC0.467032967; their scores are retained only diagnostically, not applied to control. The remaining one material selection is a fixed recovery with no corresponding policy-risk row. No missing risk is replaced with zero.

For all supported executed tool decisions,901 predictions share563 subsequent physical outcomes. The equal-outcome-weighted Brier is0.249810990 and AUROC0.496247478. Intermediate tool labels are associations with the next evaluation, not proof of a causal reasoning error. The package retains per-decision and equal-observed-outcome metrics, exact point values, fixed ten-bin calibration, and all seven typed-head diagnostics. Typed schema components are not assumed to be pure NN outputs. No fit, temperature change, threshold search or test-based repair was performed.

## Different tool and material trajectories

| Executed tool | Baseline | Full |
|---|---:|---:|
| generate_structures |524|318|
| score_buffer |157|45|
| select_for_evaluation |848|873|
| query_structures |7|31|
| get_buffer_stats |0|9|
| list_compositions |2|0|

All1440 physical evaluation steps returned without a recorded step failure. There were no repeated known candidate hashes within any trajectory; this does not imply scientific novelty, which is separately logged. Tool-level failures still occurred:148 baseline versus160 full, including128 versus153 failed selections. Full’s most frequent failure was a hash not belonging to the requested composition buffer (139 occurrences, versus61 baseline); this is a logged tool error, not a hypothetical oracle outcome. Full had fewer missing-composition selection errors (12 versus63).

Baseline requested more generation and screening: raw environment-event counts are8415 MACE candidate evaluations versus2609 full, with720 candidate ORB evaluations and2460 initialization ORB evaluations per arm. The legacy MACE cost field is absent in15 baseline and37 full episodes; its absence was not converted into zero. Event counts come from the complete hash-verified environment logs. Summed episode wall time was34426.852s baseline versus85352.786s full, including57544.973s recorded graph processing for full. These are sums of episode times, not a serial makespan or measured GPU utilization.

## Positive and negative illustrations

Ag-Nd-Pd-Pt-Tb is the requested positive example. SUN changes from[1,5,3] to[4,6,3], mean3 to4.3333333 (sample variances4 and2.3333333). AUDC changes from[0.19,0.43,0.51] to[0.46,0.60,0.51], mean0.3766667 to0.5233333. The full method’s two type-triggered extra candidates were both unused; no scalar threshold or risk rerank changed selection. Material choice differs: in seed2, baseline evaluates Ag1 Pt1 Tb1 eight times and obtains only one stable-new material overall; full evaluates a broader set of formulas and obtains four stable-new materials. In seed4, full evaluates ten distinct hashes of Ag1 Nd1 Pd1 Tb1, all stable but only three new; its final SUN and curve tie baseline. This is an observed trajectory contrast, not proof that diversity or ES caused the gain.

Al-Hg-K-Mg-W is the strongest negative chemistry by mean paired AUDC, selected only after the all-system summary. Baseline SUN[7,0,0] and AUDC[0.77,0,0] become full[0,0,0] for both. Thus the large negative average is driven by seed2; seeds3/4 tie at zero. Baseline seed2 evaluates Al–Mg compositions, yielding10 stable and7 new outcomes; full seed2 evaluates different multielement formulas, with0 stable and9 new outcomes. None of the three full type-triggered extra candidates was selected. No risk selection change, step failure or dynamic-hull offset accounts for this contrast. The observations identify different candidate regions, but do not establish an OOD cause or the quality of any unexecuted proposal.

## Dynamic hull accounting

No trajectory has a negative net SUN step, yet two steps have `stable=true` and `new=true` with SUN delta0: baseline Al-V-Zn seed2 step9, and full Al-V-Zn seed3 step6. Event-time stable-new counts are119/93 while final SUN totals are118/92. Each arm therefore has a net offset of1. The per-step diagnostic `int(stable and new) − SUN_delta` and event-total-minus-final-SUN are explicitly source-inferred accounting differences. They do not prove which earlier material was reclassified or why; official SUN/AUDC remain unchanged.

## Every matched chemistry

| Chemistry | Baseline mean SUN | Full mean SUN | Baseline mean AUDC | Full mean AUDC |
|---|---:|---:|---:|---:|
| Al-Li-V | 1 | 0.66666667 | 0.076666667 | 0.093333333 |
| Al-V-Zn | 5 | 3.6666667 | 0.61666667 | 0.41 |
| Au-K-Tb | 1.3333333 | 1.3333333 | 0.086666667 | 0.14666667 |
| Co-Dy-W | 0 | 0.33333333 | 0 | 0.043333333 |
| Co-Mg-Na | 0 | 0.66666667 | 0 | 0.026666667 |
| Co-Pd-Tl | 0 | 0 | 0 | 0 |
| Ga-Ho-Lu | 4.6666667 | 2.6666667 | 0.43333333 | 0.42666667 |
| Ga-Pt-Tm | 4 | 4.3333333 | 0.48666667 | 0.47666667 |
| Hf-Ni-Zr | 6 | 5 | 0.74666667 | 0.59666667 |
| Mg-Sn-Sr | 4.6666667 | 5 | 0.45333333 | 0.53666667 |
| Au-Cr-Cs-Dy | 0 | 0 | 0 | 0 |
| Au-Tb-V-Y | 1 | 0 | 0.096666667 | 0 |
| Ba-Nd-Ni-W | 0 | 0 | 0 | 0 |
| Ce-Ir-Pt-Sn | 2.3333333 | 0 | 0.24333333 | 0 |
| Co-Dy-Ta-Y | 0.66666667 | 0 | 0.04 | 0 |
| Dy-K-Pd-Sm | 0 | 0 | 0 | 0 |
| Eu-Nb-Sn-Tl | 0 | 0 | 0 | 0 |
| Ag-Nd-Pd-Pt-Tb | 3 | 4.3333333 | 0.37666667 | 0.52333333 |
| Al-Hg-K-Mg-W | 2.3333333 | 0 | 0.25666667 | 0 |
| Al-Lu-Pt-Rb-Sm | 1.3333333 | 1.3333333 | 0.1 | 0.20666667 |
| Ca-Fe-Gd-Pb-Tb | 0 | 0 | 0 | 0 |
| Co-Hg-Mg-Sr-W | 0 | 0 | 0 | 0 |
| Cr-Fe-Lu-Pt-Sc | 1.3333333 | 0 | 0.11333333 | 0 |
| Ho-In-Mg-Pd-Zr | 0.66666667 | 1.3333333 | 0.08 | 0.14 |

All three-seed means, sample variances and SD are in `chemistry_statistics.csv`; exact matched differences, per-seed macros, costs and source-bound mechanism counts are separate tables. `mechanism_by_outcome_sign.csv` is descriptive posthoc grouping, not a fitted predictor.

## Reproduction and limits

The original compressed scientific projection is included byte-for-byte as `observations.json.gz`; its SHA is13f29ed763b8814fc588d5bc005408ed81e991c83a2a34f4958dd6284570c8e7. The decompressed source SHA is0b58bd2507d536c5a6a20d713bfa87da915f4fa55bbd1acf629bb33c269380ec. It contains no raw geometry, weights, credentials, hostnames, GPU identifiers or absolute machine paths. The remote extractor matched accepted receipt/result hashes and exact decision/event/tool/step order for144/144 runs with0 gaps. This package retains those scientific artifact references, not operator logs.

Run `python3 -B recompute.py` for8142 outcome/identity/missingness checks, recalculated three-seed statistics and independent pairwise weighted-AUROC/Brier/ECE calculations from the saved prediction points. It validates all derived files and manifest hashes without a model or oracle. This does not recreate physical experiments or establish causal effects. New-model/graph/oracle calls and new fits for this analysis are all zero.

The current complete subset does not support a general claim that full improves baseline, nor that its scalar NN or risk reranking produced the positive cases. Separate prospective development controls are needed to assess a replacement risk model; this held-out audit does not change their training data, thresholds or evaluation scope.
