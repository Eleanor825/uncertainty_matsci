# MADE mean and variance across chemical systems

Observed through 2026-09-18 18:44:26 CST. Model: Qwen3.5-4B. Both methods use seed1. Baseline is the original baseline; full is fixed G2 ESOpt plus graph-risk policy. B10 and B30 each have60 completed trajectories and30 matched chemical systems. The complete three-budget study/global acceptance remains pending because B50 is unfinished.

Each row below uses the same30 official chemical systems. Variance is the sample variance, with denominator n−1. This describes variation **across chemical systems**, not variation across independent random-seed repetitions. Standard deviation (SD), population variance (denominator n), and per-system paired differences are also provided in `statistics.csv`. The paired difference is full minus baseline, computed before summarizing across systems.

| Budget | Metric | Baseline mean | Baseline sample variance | Full mean | Full sample variance |
|---|---|---:|---:|---:|---:|
| B10 | SUN | 1.400000 | 5.351724 | 1.800000 | 6.510345 |
| B10 | AUDC | 0.154667 | 0.062929 | 0.188667 | 0.077557 |
| B30 | SUN | 4.533333 | 50.602299 | 3.233333 | 22.116092 |
| B30 | AUDC | 0.160815 | 0.058769 | 0.124185 | 0.030844 |

B10 SUN totals are42→54; B30 totals are136→97. The direction of the average improvement therefore differs by budget. A smaller cross-system variance is not evidence of improved uncertainty calibration or repeatability; B30 also has a lower mean for the full method. There are no multi-seed variance estimates in this study. B50 has0 completed trajectories,10 running and50 unclaimed as of this snapshot, so its comparative statistics are missing (NA), not zero.

## Recompute

Run `python3 analyze.py` in this directory. It reads `pairs_B10.csv` and `pairs_B30.csv`, verifies30 distinct matched systems and seed/budget identities, and regenerates `statistics.csv` and `summary.json`. It uses only the Python standard library and makes no model, oracle, or network calls. Source CSV digests are retained in the summary. The n−1 variance was checked against a direct squared-deviation calculation; the independent B30 audit agrees with all nine mean/variance records.

The B10 source is the audited15:32:27 snapshot. The full B30 source is the audited18:44:26 snapshot with SHA256 `2297d8394f4d1e0662700a42d5da8ed13d7eb87738f8029c9ca4b41c59aa4822`. Its60 result/receipt identities, fixed execution profiles, and original RPC curves were verified. Individual result SHA256 values are retained in both paired-system CSVs.
