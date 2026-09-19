# MADE fixed-five B10 results: three evaluation seeds

Observed **September19,2026 at20:34:17.577945 CST** (source capture began20:34:05 CST). The source directory label `2036_narrow` is not the measurement time. This is a new, fixed-cohort analysis of the ongoing1,080-evaluation component study; it is separate from the original180 seed1 trajectories and SnAr.

**Baseline, full and independent ES-only each have all15 planned core evaluations**: five systems×seeds2/3/4×B10. UQ-only has11/15, with two open claims and two unclaimed runs. The four-arm core ablation is therefore **56/60 complete**, not complete. The whole new study is254/1,080 accepted (all254 currently B10), ten claimed without results, zero failed and816 unclaimed. Core56 is a subset of254, not additional experiments.

## Main result: the aggregate full comparison is negative

Baseline's per-trajectory SUN counts sum to **22**, versus **20** for full. The corresponding means are1.466667 and1.333333 per trajectory. Mean AUDC over the same15 pairs is **.156 versus.144**, a full-minus-baseline difference of **−.012**. These sums are sums of per-run discoveries; they are not a deduplicated union of materials across repeated seeds.

Across the five systems' three-seed means, full has2 positive SUN differences,2 negative and1 tie; for AUDC it has4 positive and1 negative. **Counting favorable systems does not establish an overall improvement.** Al–V–Zn contributes−.62 to the sum of the15 AUDC differences, exceeding the other four systems' combined+.44; the net−.18 divided by15 is−.012.

Every original seed value, including zeros and missing entries, is in [runs.csv](runs.csv), with the original receipt/result SHA. Within-run curves are in [completed_curves.csv](completed_curves.csv). Seed arrays below are ordered[2,3,4].

| System | Baseline SUN | Full SUN | Mean SUN delta | Baseline AUDC | Full AUDC | Mean AUDC delta |
|---|---|---|---:|---|---|---:|
|Al-Li-V|[0, 2, 1]|[0, 0, 2]|-0.333333|[0.0, 0.08, 0.15]|[0.0, 0.0, 0.28]|0.0166667|
|Al-V-Zn|[4, 5, 6]|[4, 3, 4]|-1.33333|[0.5, 0.69, 0.66]|[0.42, 0.47, 0.34]|-0.206667|
|Au-K-Tb|[4, 0, 0]|[0, 3, 1]|0|[0.26, 0.0, 0.0]|[0.0, 0.37, 0.07]|0.06|
|Co-Dy-W|[0, 0, 0]|[0, 1, 0]|0.333333|[0.0, 0.0, 0.0]|[0.0, 0.13, 0.0]|0.0433333|
|Co-Mg-Na|[0, 0, 0]|[2, 0, 0]|0.666667|[0.0, 0.0, 0.0]|[0.08, 0.0, 0.0]|0.0266667|

The15 individual full-minus-baseline pairs have SUN5wins/6ties/4losses and AUDC5wins/5ties/5losses. These differ from counts of favorable **system means**, and neither count substitutes for the magnitude-weighted aggregate above.

- **Al–Li–V:** Full loses one endpoint SUN across the three seeds (3→2), yet AUDC increases in mean (+.016667). Endpoint yield and discovery timing measure different behavior.
- **Al–V–Zn:** Full's total SUN drops15→11. Its AUDC is lower on **all three seeds** (.50→.42, .69→.47, .66→.34), making this the largest aggregate loss.
- **Au–K–Tb:** Total SUN ties4→4, while mean AUDC rises+.06. The paired SUN differences[−4,+3,+1] have sample variance13; the tie in the mean hides considerable seed variation.
- **Co–Dy–W:** The positive mean comes from one discovery on seed3 only; the other two full seeds and all baseline seeds are zero.
- **Co–Mg–Na:** The positive mean comes from two discoveries on seed2 only; the other two full seeds and all baseline seeds are zero. These two favorable systems do not show improvement on every seed.

Independent ES-only also has complete coverage: its SUN total is17 and mean AUDC.131333 on the same15 settings. These descriptive values do not establish which component caused the full result. The complete planned four-arm comparison must wait for the missing UQ-only runs.

## Fixed-system evaluation-seed mean, sample variance and SD

Each row holds the chemical system, budget and arm fixed, then uses its observed evaluation seeds. Sample variance is `sum((x-mean)^2)/(n-1)` (**ddof=1**); SD is its square root. These are evaluation repeats with the same learned weights/controller and evaluator configuration. They do not repeat training, checkpoint selection, or the whole research pipeline. They are not variances across different systems.

UQ-only rows with n=2 are explicitly partial and lack seed4. Their partial means are **not compared with complete three-seed arm aggregates**. Zero variance in a repeatedly zero-discovery setting is not evidence of useful discovery performance.

| System | Arm | n /3 | Missing seed | SUN mean | SUN variance | SUN SD | AUDC mean | AUDC variance | AUDC SD |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
|Al-Li-V|Baseline|3|none|1|1|1|0.0766667|0.00563333|0.0750555|
|Al-Li-V|UQ-only|3|none|0.333333|0.333333|0.57735|0.03|0.0027|0.0519615|
|Al-Li-V|ES-only|3|none|0.333333|0.333333|0.57735|0.05|0.0075|0.0866025|
|Al-Li-V|Full|3|none|0.666667|1.33333|1.1547|0.0933333|0.0261333|0.161658|
|Al-V-Zn|Baseline|3|none|5|1|1|0.616667|0.0104333|0.102144|
|Al-V-Zn|UQ-only|2|4|6|2|1.41421|0.72|0.0002|0.0141421|
|Al-V-Zn|ES-only|3|none|4|1|1|0.48|0.0199|0.141067|
|Al-V-Zn|Full|3|none|3.66667|0.333333|0.57735|0.41|0.0043|0.0655744|
|Au-K-Tb|Baseline|3|none|1.33333|5.33333|2.3094|0.0866667|0.0225333|0.150111|
|Au-K-Tb|UQ-only|2|4|3|0|0|0.31|0.0072|0.0848528|
|Au-K-Tb|ES-only|3|none|1|0|0|0.123333|0.00973333|0.0986577|
|Au-K-Tb|Full|3|none|1.33333|2.33333|1.52753|0.146667|0.0386333|0.196554|
|Co-Dy-W|Baseline|3|none|0|0|0|0|0|0|
|Co-Dy-W|UQ-only|2|4|1|2|1.41421|0.08|0.0128|0.113137|
|Co-Dy-W|ES-only|3|none|0|0|0|0|0|0|
|Co-Dy-W|Full|3|none|0.333333|0.333333|0.57735|0.0433333|0.00563333|0.0750555|
|Co-Mg-Na|Baseline|3|none|0|0|0|0|0|0|
|Co-Mg-Na|UQ-only|2|4|0|0|0|0|0|0|
|Co-Mg-Na|ES-only|3|none|0.333333|0.333333|0.57735|0.00333333|3.33333e-05|0.0057735|
|Co-Mg-Na|Full|3|none|0.666667|1.33333|1.1547|0.0266667|0.00213333|0.046188|

Exact values and mSUN statistics: [per_system_seed_statistics.csv](per_system_seed_statistics.csv). Paired differences use matched system/budget/seed rows, and their variance is computed **from the differences**, not by subtracting two variances: [paired_run_differences.csv](paired_run_differences.csv), [paired_system_seed_statistics.csv](paired_system_seed_statistics.csv).

## Five-system aggregate first, variance across three seeds second

For each seed, sum SUN over the same five systems and average AUDC over those same five. Only after constructing these three complete, equally sized seed aggregates do we calculate their mean, sample variance and SD. This avoids mixing a varying set of available systems into a claimed seed repetition.

| Seed | Baseline total SUN | Full total SUN | SUN total delta | Baseline macro AUDC | Full macro AUDC | AUDC delta |
|---|---:|---:|---:|---:|---:|---:|
|2|8|6|-2|0.152|0.1|-0.052|
|3|7|7|0|0.154|0.194|0.04|
|4|7|7|0|0.162|0.138|-0.024|

| Arm | Complete seed aggregates | Mean total SUN per seed | SUN-total variance | SUN-total SD | Mean macro AUDC | Macro-AUDC variance | Macro-AUDC SD |
|---|---:|---:|---:|---:|---:|---:|---:|
|Baseline|3|7.33333|0.333333|0.57735|0.156|2.8e-05|0.0052915|
|ES-only|3|5.66667|0.333333|0.57735|0.131333|0.00246533|0.0496521|
|Full|3|6.66667|0.333333|0.57735|0.144|0.002236|0.0472864|

Full-minus-baseline total-SUN differences across the three seed aggregates are[−2,0,0], with mean-0.666667, sample variance1.33333, and SD1.1547. Macro-AUDC differences are[−.052,+.040,−.024], with mean-0.012, sample variance0.002224, and SD0.0471593. Three repeats do not support a broad statistical efficacy claim.

All seed aggregates, including explicit missing UQ cells: [five_system_aggregates_by_seed.csv](five_system_aggregates_by_seed.csv). Across-seed means/variances/SDs: [aggregate_seed_statistics.csv](aggregate_seed_statistics.csv). Matched aggregate differences and their variances: [paired_aggregates_by_seed.csv](paired_aggregates_by_seed.csv), [paired_aggregate_seed_statistics.csv](paired_aggregate_seed_statistics.csv).

UQ's full15-run SUN total, overall AUDC mean and three-seed aggregate variance are **NA until all15 exist**. We do not compare its11 observed runs with another arm's15. Its seed4 missing runs are Al–V–Zn and Au–K–Tb (claimed without completed results), plus Co–Dy–W and Co–Mg–Na (no claim in this snapshot). Open claims are not proof of current GPU occupancy. No incomplete value is replaced with zero.

## Scope, cost and provenance

The fixed working cohort is Al–Li–V, Al–V–Zn, Au–K–Tb, Co–Dy–W and Co–Mg–Na. It predates these new seed2/3/4 outcomes. Historically the first two older-system results were already observed before the original expansion to five, so this is not claimed as an entirely outcome-unseen prospective subset. It was not chosen from the favorable cases in the present snapshot.

The new study still requires all30 systems×three separately registered budget settings10/30/50×three evaluation seeds×four arms=1,080 evaluations. Finishing the baseline/full core B10 subset is not finishing that benchmark matrix, all budgets, all models, or the four-arm ablation. The original180 seed1 runs have a different study/protocol scope and are not pooled into these variances. SnAr is separate as well.

The56 completed core runs account for560 completed candidate evaluations, a subset of the new study's completed calls. Initialization, screening, model calls and graph/wall time remain separate in [timings_and_costs.csv](timings_and_costs.csv). Those560 calls are not end-to-end research cost. The616 cached curve points include each run's initial point. AUDC is recomputed from the official SUN curve as `2 * trapezoidal_area / B^2`; possible official dynamic-hull decreases are preserved rather than replaced with a cumulative maximum.

The scientific input projection retains exact observed metrics, curves, costs and original receipt/result hashes. It excludes authentication, process and resource-control records. The source snapshot hash is pinned in [provenance.json](provenance.json), and the same two scientific registration identities as the prior18:50 publication remain bound. We checked the local projection and recomputed metrics/variance; we did not rerun the full remote scientific acceptance or any model/oracle.

```sh
python3 -B analyze.py
```

[validation.json](validation.json) records56 curve checks and independent algebraic variance checks. [manifest.json](manifest.json) hashes this standalone scientific export. The fixed observation is retained; existing public results are unchanged. [publication_provenance.json](publication_provenance.json) distinguishes the original analysis hashes from these exported files.
