# Eight-fit risk data refresh — 22 September 2026, 21:29:53 CST

**All eight CPU fits completed; online improvement is not established.** The run recorded 500 optimizer updates in 67.124 seconds. Both base84 models exactly reproduce their earlier CV2 initial/final tensors, preprocessing, selected hyperparameters and calibration. The earlier v1 preparation failure is retained in the previous dated report; v2 corrects the zero-counter consumer interface in a separate namespace.

The fixed 2×2 data variants contain 84 base rows, 96 with ES rows, 87 with milestone rows and 99 combined rows. All 20 raw executed ES rows remain auditable; 12 satisfy the unchanged eligibility rule and 8 are excluded from fitting. Combined is **99, not the possible maximum 107**. The three milestone rows are original world0 training actions, not new environment calls.

| Data variant | Features | Fit rows | AUROC | Brier ↓ | NLL ↓ | FNR at 0.5 ↓ |
|---|---|---:|---:|---:|---:|---:|
| base84 | Internal | 84 | 0.960000 | 0.068870 | 0.220769 | 0.300 |
| base84 | OutputAction | 84 | 0.910000 | 0.110689 | 0.355796 | 0.200 |
| with_ES | Internal | 96 | 0.950000 | 0.091745 | 0.279636 | 0.200 |
| with_ES | OutputAction | 96 | 0.943333 | 0.103779 | 0.346184 | 0.200 |
| with_milestones | Internal | 87 | 0.960000 | 0.068525 | 0.221672 | 0.300 |
| with_milestones | OutputAction | 87 | 0.953333 | 0.089175 | 0.276790 | 0.200 |
| combined | Internal | 99 | 0.953333 | 0.084463 | 0.258935 | 0.300 |
| combined | OutputAction | 99 | 0.960000 | 0.082908 | 0.227814 | 0.200 |

The prespecified combined Internal model is worse than Internal-base84 in Brier (0.068870 → 0.084463; difference +0.015593) and NLL (+0.038166), with unchanged FNR 0.30. Combined OutputAction improves Brier (0.110689 → 0.082908; difference −0.027782), NLL (−0.127982) and AUROC (+0.05), with unchanged FNR 0.20. OutputAction includes sampling/output and action features; it is not an internal-graph model or a pure output-probability comparator. All eight outcomes are reported. No development winner is selected and no production predictor is replaced.

Each feature family reuses its earlier base84 training-only CV choice: Internal 25 updates / decay 0.0001, OutputAction 100 / 0.01. All variants use the same seed, architecture, row-uniform fitting loss and unchanged world1 60-row temperature calibration. This run performs four times (25 + 100) = 500 updates; it does not repeat the earlier 2,525-update CV search. Eight model/temperature seals precede the development predictions. The labels concern next-action failure, not long-horizon scientific progress.

Evaluation reuses the same world2 40 first-proposal rows from two episodes: 10 failures and 30 nonfailures, prevalence 0.25. Episode metrics are exported separately. Rows are correlated; there is one fit seed per variant, no independent training-replicate variance, and no statistical-significance or fresh-held-out claim. The two added ES trajectories come from nearby perturbed policies sharing one world/policy seed, not two independent worlds or the final deployed policy. Milestone selection is informed by training outcomes. The factorial arithmetic describes this fixed experiment and cannot establish general causal improvement or performance on revised proposals.

No new policy generation, environment call, graph extraction, LLM parameter update or held-out test read occurs in the eight-fit refresh. The existing ES/graph collection costs belong to their earlier recorded stages. B30 online comparisons remain prospective at this snapshot; these CPU results are not an online Native-versus-Full result.

[All eight exact metric rows](metrics.csv), [16 per-episode metric rows](episode_metrics.csv), [models and calibration](models_and_calibration.csv), [source projection](summary.json), [raw receipt hashes](provenance.json). Run `python3 validate.py` for arithmetic/inventory checks. Raw logits and private RPC logs are not republished; their hashes identify different bytes from these derived exports. Publication adds zero scientific calls.
