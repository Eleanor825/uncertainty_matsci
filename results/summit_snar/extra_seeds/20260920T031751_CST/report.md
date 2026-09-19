# SnAr: additional fixed-policy evaluation seeds

Observed **2026-09-20T03:17:51.971199+08:00**. **5/12** additional B50 trajectories have passed the original oracle, trajectory and graph audit. Only these complete trajectories enter statistics.

The new scope comprises four arms × seeds5201/5202/5203 ×50 calls (600 planned). These seeds were checked against prior train/dev/evaluation/mutation seeds before execution. This precision extension was requested after the original35 results were known; it is separate from the original five-seed study, not a retrospective claim of a prespecified eight-seed study. No original trajectory, fitting or checkpoint selection is replayed.

All arms start from the same frozen550-query adaptation archive (HV0.8810117795428745). Model, NN, controller, evaluator and budget remain fixed. Both ES branches previously selected G0: full uses the same weights as UQ-only, and ES-only uses the same weights as baseline. Extra seeds cannot establish an ES improvement or isolate the NN contribution.

| Arm | Seed | Final HV gain | Mean querywise HV gain | Selected-action Brier |
|---|---:|---:|---:|---:|
| uq_esopt | 5201 | 0 | 0 | 0.731628809501768 |
| uq_only | 5201 | 0 | 0 | 0.731628809501768 |
| qwen_base | 5201 | 7.87991523543e-06 | 5.04314575068e-06 | None |
| es_only | 5201 | 7.87991523543e-06 | 5.04314575068e-06 | None |
| uq_esopt | 5202 | 0 | 0 | 0.7396632008255477 |

Metrics use two-objective HV, not MADE AUDC. Coordinate transforms are `sty/13000` and `(1000-e_factor)/1000`. Gain subtracts the shared prior HV. Brier uses observed selected proposals only and `no_hvi=1` for non-improvement; candidates never executed have no invented outcomes.

Per-arm means/sample variance/SD use accepted evaluation seeds only; incomplete sets are labelled and n<2 variance is blank. Paired differences use exact matching new seeds only. Pending, failed and unknown rows stay metric-free. Early uneven arm coverage is not an efficacy comparison.

Original journal closure, physical query identity, graph gates and actual executed G0 identity were revalidated using the frozen acceptance functions. The portable input contains executed objectives/risks and the shared prior, with original evidence SHA references. `python3 -B recompute.py` recomputes every HV point, label, gain, Brier and statistic without a model or oracle. Snapshots overlap; do not add their counts.
