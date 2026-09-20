# Why the original Full policy loses B50 discoveries: all 60 trajectories

**The original fixed-G2 Full result is lower mainly because fewer evaluated candidates are both stable and novel, not because it loses more discoveries to convex-hull reclassification.** Across all 30 official systems at the single original seed1, baseline has 237 stable-and-new submission events and Full 148. Baseline later loses four counted entries to an updated hull; Full loses one. Thus the exact final SUN difference is **(148 − 237) − (1 − 4) = −89 + 3 = −86**, giving the unchanged official totals **233 versus 147**. Reclassification slightly narrows Full's deficit.

This is a complete, fixed-cohort posthoc audit, not a new benchmark, new fit, metric replacement or tuning set. All 60 trajectories and 3,000 charged candidate ORB calls are included, including all positive systems and Ga-Pt-Tm's explicitly derived acceptance. The readback completed 2026-09-20 05:45:12 CST / 2026-09-19 21:45:12 UTC; original data SHA is `ff48583be406530716a3168ec1718c52abd98eb8af9622d08af1308a1b0f5396`. Source/result/RPC/acceptance hashes were verified in the readback and all original files remained unchanged. No model, native graph, oracle, optimizer, or training-data update was invoked by this analysis.

## Complete accounting

Each arm has 30 independent task trajectories, each B50, but **only one evaluation seed per system**. These are not 1,500 independent experimental replicates. There is no across-seed variance estimate or causal significance claim.

| Recorded outcome at submission | Baseline | Fixed-G2 Full |
|---|---:|---:|
| Stable and novel | 237 | 148 |
| Stable and not novel | 83 | 114 |
| Unstable and novel | 1,080 | 935 |
| Unstable and not novel | 100 | 303 |
| All stable | 320 / 1,500 (21.33%) | 262 / 1,500 (17.47%) |
| All novel | 1,317 / 1,500 (87.80%) | 1,083 / 1,500 (72.20%) |
| Stable given novel | 237 / 1,317 (18.00%) | 148 / 1,083 (13.67%) |
| Prior stable entries reclassified | 4 | 1 |
| Final official SUN | 233 | 147 |
| Mean official AUDC across systems | 0.158840 | 0.112787 |

No candidate `step` failed in these 3,000 records. This does not establish absence of earlier tool/schema errors, and ORB-return success does not prove DFT validity or geometry-optimizer convergence. It means the current gap is not caused by missing candidate outcomes or an oracle exception being silently scored as zero.

A second exact bookkeeping decomposition is useful: Full has 58 fewer stable submissions overall (262−320), and 31 more stable-but-not-novel submissions (114−83). Consequently it has 89 fewer stable-and-novel events. This arithmetic does not identify independent causal contributions of novelty versus stability: composition, generated structures, previous discoveries and the evolving hull jointly determine both.

## Novelty failures are not literal duplicate-hash calls

Full records 417 non-novel outcomes versus baseline's 183. Nevertheless **each trajectory's 50 recorded candidate hashes is unique** in both arms. The data therefore do not show literal repetition of the same canonical hash within an episode. They show the official StructureMatcher declaring a candidate equivalent to an initial reference or an earlier novel structure. This projection contains no geometry, so those two sources of non-novelty cannot be separated here.

The distinction is source-grounded. `structure_hash` applies primitive/LLL reduction and six-decimal lattice/coordinate formatting; it is not the same equivalence relation as StructureMatcher with ltol 0.2, stol 0.3 and angle tolerance5. Different hashes can still match structurally. Official novelty is checked on the submitted structure; ORB supplies relaxed energy without returning relaxed geometry. We must not call every non-novel outcome an exact physical replay or assume it reached the identical relaxed minimum.

The average number of unique reduced formulas per trajectory is **21.63 baseline versus12.73 Full** (medians22 and12). The mean fraction occupied by the most frequent formula is **17.33% versus32.87%**. These are descriptive concentration measures, not alternative discovery metrics. A repeated formula can represent a useful new polymorph, so it is not itself a failure label.

## Progress through the fixed 50-call budget

Every table cell below aggregates the same 30 systems, 300 candidate calls per ten-step band. Stable+novel is measured when submitted; reclassification is separate.

| Attempts | Baseline stable+novel | Full stable+novel | Baseline non-novel | Full non-novel | Baseline all stable | Full all stable |
|---|---:|---:|---:|---:|---:|---:|
| 1–10 | 56 | 41 | 52 | 67 | 76 | 63 |
| 11–20 | 41 | 36 | 37 | 84 | 61 | 56 |
| 21–30 | 45 | 24 | 35 | 86 | 62 | 50 |
| 31–40 | 49 | 22 | 35 | 95 | 61 | 49 |
| 41–50 | 46 | 25 | 24 | 85 | 60 | 44 |

Full's non-novel counts rise from67/300 in the first band to95/300 at steps31–40 and85/300 at41–50. Baseline falls from52/300 to35/300 and24/300. Full's stable-and-novel yield is41,36,24,22,25 versus56,41,45,49,46. This is consistent with a longer-horizon concentration/novelty problem, but does not establish its cause. The original G2 was trained/selected at B10; horizon mismatch is a hypothesis, not a diagnosis proved by these test traces. No threshold or NN training choice follows from this observation.

## All 30 systems, including gains

Full has **9 SUN wins, 10 ties and11 losses**. Winners contribute+40 discoveries; losing systems−126, net−86. The table retains every system rather than selecting only adverse cases. B/F means baseline/Full. All values are original seed1.

| System | Final SUN B/F | Stable+new events B/F | Non-novel B/F | All stable B/F | Prior entries lost B/F |
|---|---:|---:|---:|---:|---:|
| Ag-Nd-Pd-Pt-Tb | 10/19 | 10/19 | 8/16 | 12/30 | 0/0 |
| Al-Hg-K-Mg-W | 0/0 | 0/0 | 2/0 | 0/0 | 0/0 |
| Al-Li-V | 22/3 | 22/3 | 12/4 | 34/3 | 0/0 |
| Al-Lu-Pt-Rb-Sm | 0/3 | 0/3 | 5/13 | 0/6 | 0/0 |
| Al-V-Zn | 22/17 | 22/18 | 21/13 | 42/30 | 0/1 |
| Au-Cr-Cs-Dy | 14/0 | 14/0 | 2/9 | 14/0 | 0/0 |
| Au-K-Tb | 8/2 | 8/2 | 4/21 | 9/12 | 0/0 |
| Au-Tb-V-Y | 0/0 | 0/0 | 9/1 | 0/0 | 0/0 |
| Ba-Be-Hf-Li | 0/0 | 0/0 | 4/11 | 0/0 | 0/0 |
| Ba-Nd-Ni-W | 0/0 | 0/0 | 2/28 | 0/0 | 0/0 |
| Ca-Fe-Gd-Pb-Tb | 3/0 | 3/0 | 2/11 | 3/0 | 0/0 |
| Ca-Pd-Sn-W | 0/1 | 0/1 | 6/14 | 0/1 | 0/0 |
| Cd-Gd-Mn-Na-Ta | 0/0 | 0/0 | 2/3 | 0/0 | 0/0 |
| Cd-Li-Nd-Ti-W | 2/3 | 2/3 | 7/13 | 6/4 | 0/0 |
| Ce-Er-Pb-Rh | 3/9 | 3/9 | 2/20 | 3/17 | 0/0 |
| Ce-Ir-Pt-Sn | 7/11 | 8/11 | 13/13 | 11/23 | 1/0 |
| Co-Dy-Ta-Y | 0/11 | 0/11 | 1/2 | 0/11 | 0/0 |
| Co-Dy-W | 1/0 | 1/0 | 10/21 | 2/0 | 0/0 |
| Co-Hf-In-Ru-Tm | 0/0 | 0/0 | 0/18 | 0/0 | 0/0 |
| Co-Hg-Mg-Sr-W | 0/4 | 0/4 | 5/22 | 0/5 | 0/0 |
| Co-Mg-Na | 0/0 | 0/0 | 4/14 | 0/0 | 0/0 |
| Co-Pd-Tl | 29/2 | 29/2 | 7/8 | 36/2 | 0/0 |
| Cr-Fe-Lu-Pt-Sc | 0/0 | 0/0 | 6/16 | 0/0 | 0/0 |
| Dy-K-Pd-Sm | 0/0 | 0/0 | 3/8 | 0/0 | 0/0 |
| Eu-Nb-Sn-Tl | 0/0 | 0/0 | 1/25 | 0/0 | 0/0 |
| Ga-Ho-Lu | 18/19 | 18/19 | 9/9 | 24/23 | 0/0 |
| Ga-Pt-Tm | 18/13 | 21/13 | 20/32 | 38/40 | 3/0 |
| Hf-Ni-Zr | 17/16 | 17/16 | 4/8 | 18/19 | 0/0 |
| Ho-In-Mg-Pd-Zr | 18/0 | 18/0 | 6/15 | 21/0 | 0/0 |
| Mg-Sn-Sr | 41/14 | 41/14 | 6/29 | 47/36 | 0/0 |

The largest losses expose different mechanisms, not one universal explanation:

- **Co-Pd-Tl:29→2.** Stable submissions fall36→2 while non-novel counts barely change7→8. Baseline evaluates39 binary and11 ternary candidates; Full evaluates2 binary and48 ternary candidates. The changed composition mix is observed. This does not prove the ternary choice caused the outcome or prescribe a test-derived arity restriction.
- **Mg-Sn-Sr:41→14.** Stable submissions fall47→36, while non-novel outcomes rise6→29. Baseline's50 candidates are binary; Full has21 binary and29 ternary candidates, with7 reduced formulas rather than12. Both novelty and stability differ; there is no hull-reclassification loss in either trace.
- **Al-Li-V:22→3.** Both arms use50 ternary candidates. Full actually tries more reduced formulas (26 versus12), but only3 are stable versus34 baseline. Thus lower formula diversity does not explain this case and increasing diversity is not a universal remedy.
- **Ho-In-Mg-Pd-Zr:18→0.** Full concentrates50 evaluations in four quinary formulas and finds no stable candidate; baseline has21 stable submissions over19 formulas, primarily quaternary compositions. This is another observed composition/structure-selection difference, with no reclassification penalty.

Positive controls prevent overgeneralization:

- **Ag-Nd-Pd-Pt-Tb:10→19.** Full is also more concentrated (10 formulas versus29) and all50 Full candidates contain all five elements, yet stable submissions increase12→30. Non-novel counts also rise8→16, but the stability gain is larger. Concentration and higher arity are not uniformly harmful.
- **Co-Dy-Ta-Y:0→11.** Both arms evaluate50 quaternary candidates; Full tries19 formulas versus33 baseline, with11 stable-and-novel outcomes versus0. The same broad concentration pattern accompanies a gain here.
- **Ga-Pt-Tm:18→13.** Full has *more* stable submissions (40 versus38) but fewer stable-and-novel events (13 versus21), because27 stable Full candidates are non-novel versus17 baseline. Baseline then loses3 previous discoveries to one hull update, narrowing the event gap from8 to the final gap5. This cleanly shows why high stability alone is insufficient for the registered discovery objective.

## Exact dynamic-hull reconstruction

The readback retains the actual initial `phase_diagram_all_entries` projected to composition and total ORB energy, plus every candidate's composition/energy. A CPU linear program computes the lower convex envelope without importing MADE, PyTorch, Qwen or any oracle. All **3,000 official SUN values match**; the maximum candidate `e_above_hull` absolute difference is **1.33×10⁻¹⁴ eV/atom**. This is a secondary consistency diagnostic; the official metric is unchanged. Runtime was approximately2.48 seconds and4,288 small LP solves, with0 new physical calls.

Three update events reclassify five old entries:

| Arm/system | New step | Old counted step(s) | Old e_above_hull before→after (eV/atom) | Net SUN effect |
|---|---:|---|---|---|
| Baseline/Ce-Ir-Pt-Sn |20|17|0.096337→0.116963|new1−old1=0|
| Baseline/Ga-Pt-Tm |26|11,12,25|0.087728→0.103780;0.099870→0.115921;0.099836→0.107021|new1−old3=−2 (9→7)|
| Full/Al-V-Zn |50|27|0.098109→0.106396|new1−old1=0|

All crossings use the unchanged0.1 eV/atom threshold. Full's step50 `Al1 V1 Zn2`, hash `19ce2ea83412395bbdbe6b9fea1492e2fe7ca8b1158cc69386662cd5e6412dbd`, lowers the local hull by0.0248612 eV/atom and removes the step27 `Al3 V2 Zn1` entry. Ga-Pt-Tm's step26 `Ga5 Pt9 Tm3`, hash `39c800e40b090c149eee9cb79b71916f53d6ebfcfb4ea5b29d6894664fc2456b`, lowers its local hull by0.0232445 eV/atom, moving three prior entries beyond tolerance. Exact identities and unrounded values are in `reclassified_entries.csv`.

The scalar target in the original method uses the next event's stable-and-new flag (and tool-error labels), whereas ES fitness uses AUDC of the current surviving count. Thus a successful event can have zero or negative net discovery gain. The mismatch exists, but this cohort quantifies only a small metric effect and one that favors Full comparatively. It cannot explain the bulk of Full's negative result. No label is altered and no new training set is constructed from this held-out audit.

## Limits and reproduction

The two policies do not have matched physical candidate streams, and the changing observed hull is trajectory-specific. These observational traces cannot establish whether ES weights, neural advice, schema handling, generator stochasticity, score-tool usage or memory caused the composition and novelty differences. Tool selection and retry text were intentionally not re-read in this scalar-only extraction. Initial reference datasets/evaluator settings are fixed; energies are separately evaluated, so identical source references are not a guarantee of bit-identical floating-point hulls.

The current audit covers one seed per system; per-system across-seed variance is undefined. Do not treat the30 heterogeneous chemistries or3,000 actions as repeated seeds. Null and positive cases remain in all tables. No alternative energy metric is selected to favor an arm, and raw per-atom energies from different compositions are not ranked as a replacement for SUN/AUDC.

`outputs/steps.csv` contains every attempt; `episodes.csv` every trajectory; `paired_systems.csv` every system; `ten_step_bands.csv` and `per_episode_bands.csv` preserve all temporal strata; `composition_arity.csv` records the descriptive composition mix. `analyze_actual.py` and source hashes permit CPU recomputation. A separate scientific-only portable input/recompute bundle omits host, process and operator records. The original59 per-job accepted traces and one Ga-Pt-Tm derived reconciliation remain distinguished. Original failure, reports, checkpoints and scientific results are untouched.
