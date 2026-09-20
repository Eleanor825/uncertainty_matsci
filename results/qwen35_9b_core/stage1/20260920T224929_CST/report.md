# Qwen3.5-9B stage1: accepted baseline results

Observed **2026-09-20T22:49:29.048318+08:00**. **5/15** B10 test trajectories are accepted. The seven B50 train/dev collections are separate and are excluded from test statistics.

This model is Qwen/Qwen3.5-9B at the pinned revision. These are unchanged theta0 baseline evaluations: no neural uncertainty controller or ES update is applied. They are separate from the 4B 1080-run experiment and the earlier seed1 180-run study.

| Chemistry | Seed | SUN | AUDC | Candidate calls | Initialization calls | Episode wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| Al-Li-V | 2 | 1 | 0.01 | 10 | 16 | 1164.209 |
| Al-Li-V | 3 | 3 | 0.37 | 10 | 16 | 1086.888 |
| Al-Li-V | 4 | 1 | 0.05 | 10 | 16 | 1590.000 |
| Al-V-Zn | 2 | 7 | 0.69 | 10 | 12 | 619.829 |
| Al-V-Zn | 3 | 1 | 0.03 | 10 | 12 | 3029.554 |

Pending/failed rows have blank metrics in `all_jobs.csv`. Sample variance and SD are computed only across completed evaluation seeds of the same chemistry; n<2 is blank, and n<3 is explicitly partial. Five-system macro statistics are emitted only for fully completed five-system seed blocks. An early result is not a reliable model-scale comparison or evidence of an uncertainty/ES benefit.

AUDC is the normalized trapezoidal discovery-curve area, `sum((x1-x0)*(SUN1+SUN0))/B²`. Candidate calls, initialization calls and tool attempts are reported separately. The final SUN is official end-state SUN; dynamic hull reclassification may lower intermediate values.

Acceptance re-executed the original source-bound per-job CPU audit, including raw file inventory, actual RPC/physics labels, generation prefixes and unchanged policy identity. It hashes checkpoint files but does not instantiate a model, call the oracle or replay an experiment. Exported episodes retain original result/profile/episode SHA references; they are scientific projections rather than copies of private runtime metadata.

Run `python3 -B recompute.py` to verify this portable projection and all derived statistics. This does not replace the original raw-evidence audit or reproduce a model run. Snapshots overlap; do not add their counts.
