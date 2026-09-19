# MADE component study: observation at September 19, 17:26:03 CST

This is **09:26:03 UTC**, a dated partial snapshot rather than a live dashboard. It is separate from the completed original 180 seed 1 trajectories and from the accepted SnAr study.

| New arm | Accepted / planned | Open jobs | Failed |
|---|---:|---:|---:|
| Baseline reference | 58/270 | 1 | 0 |
| UQ-only support-aware | 38/270 | 2 | 0 |
| Independent ES-only | 59/270 | 1 | 0 |
| Full support-aware | 17/270 | 1 | 0 |

Total: **172/1080 accepted evaluations**, with 5 open jobs. No GPU occupancy is inferred from this table. The registered new matrix is 30 systems × 3 budgets × 3 evaluation seeds × 4 arms; this snapshot does not declare that matrix complete.

The same 17 complete B10/seed 2 systems are used for the direct four-arm comparison:

| Arm | SUN total | Mean AUDC |
|---|---:|---:|
| Baseline | 34 | .202353 |
| UQ-only | 40 | .252941 |
| ES-only | 24 | .209412 |
| Full | 27 | .174706 |

Full-minus-baseline has SUN W/T/L 1/11/5 and AUDC W/T/L 3/7/7. **Full currently trails baseline on this subset.** UQ's positive partial result must not be substituted for the full-method result. The [17-case audit](report.md) preserves every case, its curve and cost information. Zero SUN means no stable/unique/novel discovery under the metric; it is not a claim that the oracle or trajectory failed.

Separate complete pair sets cover UQ 38 pairs, ES-only 58 pairs and full 17 pairs. Their respective SUN totals baseline→arm are 63→69,90→58,34→27; mean AUDC values are .176579→.197632,.163103→.126552,.202353→.174706. These unequal-denominator means should not be directly compared between arms. [All pair values](progress_pairs.csv).

The five core systems are Al–Li–V, Al–V–Zn, Au–K–Tb, Co–Dy–W and Co–Mg–Na. New seeds 2/3/4 only:

| B10 completion | Baseline | UQ-only | ES-only | Full |
|---|---:|---:|---:|---:|
| seed 2 | 5/5 | 5/5 | 5/5 | 5/5 |
| seed 3 | 5/5 | 5/5 | 5/5 | 0/5 |
| seed 4 | 0/5 | 0/5 | 0/5 | 0/5 |

B30/B50 core5 completion is 0 throughout. This is 35/60 B10 trajectories, or 35/180 across the three budgets. Only seed 2 has a completed core5 four-arm comparison; **the multi-seed core5 experiment is not complete**. [Matrix](core5_matrix.csv), [seed coverage](core5_seed_coverage.csv).

The case report is descriptive analysis of one held-out evaluation seed across 17 systems. It does not estimate across-seed variance and must not select thresholds, checkpoints or favorable cases to rerun. Missing optional surrogate/cost fields remain blank, with explicit missing-row counts; equal candidate ORB budgets do not imply equal computation.

The source snapshot is pinned by SHA256 in [provenance.json](provenance.json), but its operational/process information is not published. Three case CSVs, the case report and summary are byte-preserved from the reviewed analysis. The original analysis script depended on that unpublished snapshot; the public checker works directly from the exported tables:

```bash
python3 recompute.py
```

No experiment, model, source code or existing snapshot was changed to produce this export.
