# Seeded world4 confirmation — local snapshot

Observed 2026-09-23T12:28:47.114256+08:00. Publication-ready: **False**. This file does not publish anything. Until all ten arms and both resource closures are verified, treat it as progress; unavailable final metrics are not zero outcomes.

| Policy seed | Condition | Status | Actions | Final score | Failed actions | Task success |
|---|---|---|---:|---:|---:|---|
| 336 | Native1 | complete | 30 | 0.000000 | 29 | False |
| 336 | NoGraphRisk | complete | 30 | 0.000000 | 29 | False |
| 336 | ExplicitRepeatRisk | complete | 30 | 0.000000 | 29 | False |
| 336 | ExplicitRepeatCommon2 | complete | 30 | 0.125000 | 9 | False |
| 336 | ExplicitRepeatInternal2 | partial | 6 recorded | — | — | — |
| 337 | Native1 | complete | 30 | 0.000000 | 2 | False |
| 337 | NoGraphRisk | complete | 30 | 0.000000 | 2 | False |
| 337 | ExplicitRepeatRisk | complete | 30 | 0.000000 | 2 | False |
| 337 | ExplicitRepeatCommon2 | complete | 30 | 0.000000 | 21 | False |
| 337 | ExplicitRepeatInternal2 | complete | 30 | 0.000000 | 4 | False |

## Same-seed, same-executed-prefix audit

Every comparison is within its policy seed. After the first executed-action difference, later states are not required to match and are not tested as the same prefix. Different sampling seeds are not required to produce the same actions. Empty mismatch lists mean no mismatch among the available aligned snapshots, not complete private-state equivalence. When an adapter rejects input without a tick, the prior public UI is explicitly reused rather than treated as a new observation.

| Policy seed | Pair | Matching executed prefix length | Public UI mismatch after attempt | Recorded global RNG mismatch after attempt |
|---|---|---:|---|---|
| 336 | Native1 / NoGraphRisk | 30 | [] | [] |
| 336 | NoGraphRisk / ExplicitRepeatRisk | 30 | [] | [] |
| 336 | Native1 / ExplicitRepeatRisk | 30 | [] | [] |
| 336 | ExplicitRepeatCommon2 / ExplicitRepeatInternal2 | 6 | [] | [] |
| 337 | Native1 / NoGraphRisk | 30 | [] | [] |
| 337 | NoGraphRisk / ExplicitRepeatRisk | 30 | [] | [] |
| 337 | Native1 / ExplicitRepeatRisk | 30 | [] | [] |
| 337 | ExplicitRepeatCommon2 / ExplicitRepeatInternal2 | 3 | [] | [] |

The actual sidecar records global Python RNG fingerprints and object-construction count/order fingerprints. It does **not** record current per-object, world or UUID RNG state, so this report cannot repeat the separate fixed-13 diagnostic's five-gate equality claim. Raw prompts, token IDs, private answers/task maps, model weights and activation arrays are not exported.

Closed metrics are checked against original public action-return events and official summary scalars; candidate0 prompt/token/seed identities are recorded only as hashes or seed metadata. Unsupported or inconsistent reads retain an explicit unavailable status. Positive transitions reflect action plus world tick, without labels for unexecuted candidates.

Both policy seeds use one world. Any complete two-seed mean, sample variance and SD are descriptive sampling-seed summaries; they do not establish cross-world generalization. Historical unseeded p335 and the original Full follow-up are separate. No controller or threshold is selected or changed by this export.

All exporter scientific-call counts are zero. The accompanying snapshot, source receipt hash and local manifest preserve the evidence for a later complete-results publication.
