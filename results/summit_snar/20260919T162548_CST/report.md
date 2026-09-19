# Summit SnAr: independently accepted seven-arm results

**The independent CPU acceptance passed on September 19, 2026 at 16:25:48 CST (08:25:48 UTC).** It reconciled 60 episodes and **2,300 unique physical oracle calls**: 550 adaptation calls and 1,750 held-out calls across seven arms, five evaluation seeds per arm, 50 calls per episode. There were no failed or unknown physical attempts. The recovery performed **zero new oracle, model or training calls**.

The original report job remains failed with `ModuleNotFoundError: No module named 'study'`. An independently registered wrapper bound the original pure `study` module and ran the unchanged acceptance/reporting code against a separate view of the original evidence. It wrote the accepted report only in that new derived namespace. This export does not overwrite the old failure or claim that the original failed job exited successfully. [Acceptance evidence](acceptance_evidence.json), [preserved failure](original_report_failure.json), [provenance](source_provenance.json).

This is the deterministic [Summit SnAr benchmark](https://github.com/sustainable-processes/summit/tree/1de682d05e97adcfb96cd8376e876cef2d6160d3), with Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The complete settings are in [protocol.json](protocol.json). Both objectives enter the registered hypervolume metric; the five seeds are evaluation repeats on **one known reaction function**, with one fixed adaptation/training schedule.

Every arm shares the same **550-call adaptation prior**, whose HV is **0.881011779543**. Thus absolute final HV is largely inherited from the common prior. The table reports incremental HV gains; the prior's cost is not free.

| Arm | Final HV gain mean | Sample variance | SD | Mean querywise HV gain | SD of querywise gain |
|---|---:|---:|---:|---:|---:|
| Random | 0.0000264017 | 2.471925e-9 | 0.0000497185 | 0.0000180475 | 0.0000372640 |
| GP-EI, fixed scalarization | **0.0003670644** | 1.078543e-8 | 0.0001038529 | **0.0002867876** | 0.0001042270 |
| Qwen base | 0.0000531005 | 3.441641e-9 | 0.0000586655 | 0.0000162716 | 0.0000167630 |
| UQ-only | 0.0000226923 | 1.705823e-9 | 0.0000413016 | 0.0000149235 | 0.0000289232 |
| Independent ES-only | 0.0000531005 | 3.441641e-9 | 0.0000586655 | 0.0000162716 | 0.0000167630 |
| Random controller | 0.0000293139 | 3.959034e-9 | 0.0000629209 | 0.0000114629 | 0.0000247892 |
| Full, UQ + ES | **0.0000226923** | 1.705823e-9 | 0.0000413016 | **0.0000149235** | 0.0000289232 |

Each row has **n=5**. Variance is sample variance with **ddof=1 across evaluation seeds 5101–5105**, conditional on the common prior and fixed adaptation. It is not cross-material variance or a measure of variation across independently retrained models. Absolute-HV means, all variances/SDs, invalid-proposal rates and the five raw values are retained in [statistics.csv](statistics.csv) and [per_seed.csv](per_seed.csv).

**Full did not improve over Qwen base in mean gain.** The paired full-minus-base final-HV-gain mean is **−0.000030408221**, sample variance **5.176106086e-9**; mean-querywise-gain delta is **−0.000001348153**, sample variance **9.979133526e-10**. Both comparisons have **2 wins, 1 tie and 2 losses** across the five paired seeds. GP-EI has the largest observed mean gain. These small-sample results do not support a general improvement claim. [Paired values](paired_full_minus_base.csv), [component comparisons](ablation_deltas.csv).

The full ES branch made two actual weight updates but its registered development rule selected **G0**, because fitness decreased from 0.65502166 to 0.64734298 and 0.60512441 at G1/G2. Full and UQ-only have identical per-seed HV curves and final weight hashes; ES-only and Qwen base do too. There is no demonstrated ES benefit in this study. [Prior and selection evidence](prior_and_selection.json).

Additional limits: GP-EI uses fixed product scalarization, not EHVI or Summit TSEMO. Full/UQ can propose a second candidate, so policy-compute and wall-time costs differ. B10/B30 values in the original report are **prefixes of these same B50 episodes**, not independent budget runs. These results concern one deterministic function, not unseen-function generalization. The graph is the registered native cut-Jacobian implementation, not a claim of reproducing the original CRV replacement model.

The cached [1,750-point curves](curves.csv) and [35 source-referenced summaries](episode_summaries.json) permit independent recomputation:

```bash
python3 recompute.py
```

The checker uses only these local exports. All per-seed values match the actual accepted report fields. A few variance/SD values differ by a last floating-point bit between Python runtimes; both accepted values and the comparison are preserved in [accepted_report_fields.json](accepted_report_fields.json) and [aggregate_rounding_check.json](aggregate_rounding_check.json). Original remote artifact hashes are distinguished from derived export hashes. Publication did not download large weights or raw RPC logs, reread the full remote acceptance file, or perform any new scientific call.
