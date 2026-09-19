# Full-method versus baseline: fixed interim evidence

Current snapshot: **2026-09-20T02:47:20.330199+08:00**, **395/1080** completed evaluations. Baseline and full both have all three B10 seeds for **24/30 systems**. This is an interim completion-defined subset, not a preselected all-system estimate.

| Metric | Positive system means | Tied system means | Negative system means |
|---|---:|---:|---:|
| SUN | 6 | 9 | 9 |
| AUDC | 8 | 7 | 9 |

The unit above is a system mean over evaluation seeds2/3/4. `system_statistics.csv` includes every eligible positive, zero and negative system; `paired_seeds.csv` retains each seed. Sample variance/SD are within-system evaluation-seed statistics, not across-system or training-run variance.

Ag–Nd–Pd–Pt–Tb is a positive descriptive example: SUN [1,5,3] → [4,6,3], mean3 →4.333333, sample variance4 →2.333333; AUDC [.19,.43,.51] → [.46,.60,.51], mean.3766667 →.5233333, sample variance.0277333 →.0050333. Both metrics have two winning seeds and one tie. This example was identified from completed results and does not demonstrate general efficacy or causal NN benefit.

The **separate original seed1 B50 study** is complete for30 systems/60 trajectories: baseline totalSUN233 versus original fixed-G2 method147; meanAUDC.15884 versus.1127866667. That older method/seed scope must not be pooled with the new support-aware four-arm study. Its across-system dispersion is not evaluation-seed variance.

At this current snapshot there are **0 complete new B50 baseline/full task-seed pairs**. The old B50 negative aggregate cannot be presented as a current-study B50 result, and newly running B50 jobs have no publishable paired effect yet.

Paper framing supported by these records: performance is heterogeneous, with positive examples across evaluation seeds and substantial negative cases. The evidence does not justify a generic improvement claim; component/controller and NN contributions must remain separate. No new fitting, model call or oracle invocation was used for this recheck.

Run `python3 -B recompute.py`. Inputs retain source snapshot/result hashes and complete discovery curves.
