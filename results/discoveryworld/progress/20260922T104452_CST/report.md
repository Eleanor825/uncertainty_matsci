# DiscoveryWorld: one Baseline result; Full comparison incomplete

Observed **2026-09-22T10:44:52.819018+08:00**. Model: **Qwen3.5-4B**; scenario: **Proteomics Normal**.

**The fixed Baseline trajectory is complete. Full has no accepted held-out result, so there is no completed pair or demonstrated method improvement.**

| Stage | Accepted completion | Interpretation |
|---|---:|---|
| Held-out Baseline | 1/1 trajectory | World seed 3, policy seed 401, B30 |
| Held-out Full | 0/1 trajectory | No result available |
| Baseline–Full matched pairs | 0/1 pair | No paired effect can be estimated |
| DW-domain transcoder bank | 32/32 layers pass development fidelity | Does not establish full attribution qualification |
| Formally qualified graph rows | 0/160 | Required graph dataset unavailable |
| Fitted and calibrated risk NN | 0 | No accepted fit/calibration completion |
| Full parameter-updating ES | 0/7 episodes | No accepted Full ES trajectory |

## Fixed held-out Baseline

| Attempts | Normalized score | Task success | Failed attempts | Failure fraction | Fitness F = J |
|---:|---:|---|---:|---:|---:|
| 30 | 0.0 | False | 29/30 | 0.966667 | -0.096667 |

The original completion record is accepted, but the environment task was not solved. Failed action attempts are an episode metric, not 29 separate evaluation runs or a technical failure of the exported result. This single Baseline trajectory is insufficient to establish a Full-method effect.

## Bank fidelity and finite-difference diagnosis

The new DW-domain bank was trained on world 0 preparatory sources and checked for development fidelity on world 1. Its completion receipt records 32/32 layers passing the unchanged development FVU limit of 0.5. The test world was not used. The original eight-prefix attribution qualification remains required.

The original attribution check at epsilon 0.001 failed on the first fixed TRAIN prefix. A separate diagnostic then measured the same four selected edges at four fixed step sizes. All diagnostic outcomes are retained below; this does not turn the diagnostic into a formal qualification.

| Finite-difference epsilon | Passed checks | Scope |
|---:|---:|---|
| 0.03 | 4/4 | Same single TRAIN prefix |
| 0.01 | 4/4 | Same single TRAIN prefix |
| 0.003 | 4/4 | Same single TRAIN prefix |
| 0.001 | 3/4 | Same single TRAIN prefix |

At epsilon 0.01, all four numerical checks pass the original rtol=0.05 and atol=0.0001. At epsilon 0.001, the token-to-action-score edge still fails (analytic 0.01641509; finite difference 0.01943111). The diagnostic reports `qualified=false` and preserves the original failed qualification. Agreement at larger steps is consistent with a small-step numerical issue; its exact cause has not been established.

The diagnostic used 36 policy forwards (2 capture and 34 intervention/control forwards), zero new policy generations, zero environment calls and zero training updates. It used no held-out test information. Formal adoption of a changed step requires a separately registered prospective protocol, all eight fixed prefixes and qualification for each policy state used by the Full method. No later V3 execution is included in this snapshot.

## Retained earlier results

The MADE-trained bank previously failed cross-domain transfer at layer 0: FVU 0.843559980392456 exceeded 0.5. That negative result remains valid and is not replaced by the newly trained DW bank.

The [05:24 historical report](../20260922T052401_CST/report.md) records eight completed preparatory Native B100 trajectories (800 attempts) and six of seven completed episodes in the earlier, separate ES-only B100 branch. Those counts retain their original observation time here. They are training/development evidence, not additional held-out Baseline or Full tests.

The planned Full method still requires graph features, a fitted/calibrated risk NN, bounded proposal revisions and full-parameter Agentic ESOpt. The seven planned B10 ES episodes comprise four candidate trajectories and three development trajectories including G0. The fixed held-out Baseline and Full tests use the same world seed 3, policy seed 401 and B30. No effectiveness claim is made before this chain completes.

## Export validation

`snapshot.json` is an explicit scientific-field projection with source hashes and project-relative artifact references. Operational records and raw prompts are excluded. Original result acceptance and bank receipts are read as recorded; raw simulator acceptance and raw bank tensors were not rerun for this export. Source observations were collected sequentially and retain their own timestamps.

Run `python3 -B validate.py` to check the whitelist, identities, metric arithmetic, all 16 diagnostic checks, report rendering and bundle hashes. Export generation makes zero model, simulator, graph, NN or ES calls.
