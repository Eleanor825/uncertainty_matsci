# Qwen3.5-9B: three independently accepted training collections

Updated **2026-09-21 17:07:16 CST**. The existing Au-Li-Pd, seed1 B50 collection passed the unmodified original CPU auditor once. The actual audit exited0 with empty stderr; original evidence hashes were unchanged. Together with the [previous two accepted Al-Au-Hf collections](../20260920T231045_CST/report.md), this gives **3/7 accepted collections**. This cumulative table overlaps that prior report; only one collection is newly accepted here.

| Training chemistry | Seed | Existing candidate ORB calls | Initialization ORB calls | Proposals | Layers | Rows/layer | FP32 tensor bytes | SUN | AUDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Al-Au-Hf | 1 | 50 | 28 | 180 | 32 | 2880 | 3019898880 | 14 | 0.2888 |
| Al-Au-Hf | 2 | 50 | 28 | 156 | 32 | 2496 | 2617245696 | 14 | 0.368 |
| Au-Li-Pd | 1 | 50 | 24 | 164 | 32 | 2624 | 2751463424 | 43 | 0.842 |

**These are training outcomes, not held-out method efficacy.** The separate test count remains5/15 at the latest accepted test snapshot. No new model, graph, oracle or fitting calls were made for these audits. A high training SUN/AUDC is not evidence of cross-chemistry generalization or a complete9B risk/controller/ES method. The seven-collection scope is not complete.

Across these three existing collections:150 candidate evaluations and80 initialization evaluations;500 proposals,8000 captured positions per layer and8,388,608,000 tensor bytes. Future-failure labels are 220 non-failure, 229 failure and 51 unknown. Repeated proposals associated with one physical step are not independent physical observations. Unknown/unexecuted outcomes are preserved.

The new acceptance checked the fixed registration/checkpoint/runtime, raw file inventory, actual generated token prefixes and sampling seeds,50-step physical accounting/journal closure, all32 finite FP32 activation shards, and original label derivation. Permanent audit claims and absence of conflicting prior claims were checked before dispatch; a failed or unknown audit is never automatically replayed. The prior two accepted records were referenced byte-for-byte and not reaudited.

All original costs remain in each accepted record, including auxiliary generator/surrogate/tool attempts, token counts and elapsed times; they are not replaced by the candidate-query budget. The newly accepted Au-Li-Pd record contains316 surrogate attempts,151 tool attempts and5758.667695965618 seconds of original rollout time. These are existing scientific costs, not costs of the CPU acceptance.

Run `python3 -B verify.py` to verify the three portable receipt hashes and recompute identities, totals and AUDC. This does not repeat the original raw-tensor audit, which requires external server artifacts. Scientific accepted JSON is retained byte-for-byte; credentials, process/operator material, tensors and model weights are excluded.
