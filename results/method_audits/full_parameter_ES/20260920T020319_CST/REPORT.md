# Actual completed ES parameter audit — 2026-09-20

All four recorded runs used full-parameter policy ES: **723 unique parameter tensors, 4,539,265,536 scalar parameters**. The independently summed actual manifests are identical (canonical manifest SHA256 `477a30e2c510f587cc43a91892bb2cea33190ea064f865021eefd76f2d745501`). Tied parameter aliases are counted once. This scope excludes the auxiliary risk NN, transcoder bank and physical evaluator.

| Actual run | G0 → G1 nonzero tensor deltas | G1 → G2 nonzero tensor deltas | State changed at G1 / G2 | Selected |
|---|---:|---:|---|---|
| MADE support-aware full | 723 / 723 | 723 / 723 | yes / yes | G2 |
| MADE independent ES-only | 723 / 723 | 723 / 723 | yes / yes | G1 |
| SnAr full | 723 / 723 | 723 / 723 | yes / yes | G0 |
| SnAr ES-only | **0 / 723** | 723 / 723 | **no** / yes | G0 |

Thus “each ES generation applies the full-parameter update rule” is accurate; “every generation necessarily changes every parameter” is not. SnAr ES-only’s first population rewards were both `0.41085579650323145`; population z-scores were `[0,0]`, all actual tensor deltas were zero, and the state hash stayed identical. A nonzero tensor L2 norm means at least one scalar in that tensor changed, not that every scalar changed. Both SnAr heldout branches selected G0 and consequently used original policy weights, despite conducting and recording the two-generation ES training.

These are two **training-generation** updates, not per-material-step or per-test-seed training. Within a generation each population perturbation is evaluated and restored exactly before the committed update. Evaluation trajectories reuse the selected frozen checkpoint. MADE’s registered selection compares G1/G2; SnAr compares G0/G1/G2 with earliest tie-breaking.

Actual read-only observation: **2026-09-20 02:03:19 CST / 2026-09-19 18:03:19 UTC**. All four projections verified; all 33 read source/JSON inputs retained identical SHA/mtime. The observer read no model tensor, loaded no model, performed no fit/oracle call and wrote no remote artifact. Exact source/input refs, rewards, state hashes, and marker consistency are retained in `snapshot.json`; compact per-generation values are in `actual_updates.csv`. This is a metadata consistency audit of prior physical/clean-reload evidence, not a rerun of those physical validations.

Snapshot SHA256: `114357c0259d4743b46e83e892e61f82867ce2be1e1d58bd6c1baef48415f57c`. Observer SHA256: `cf18eaee1d86ed763c763aa105fd2bda27c4510ef2de294c86b944eb07078079`. Initial `REPORT_local.md`, its partial local evidence and original manifest remain unchanged.
