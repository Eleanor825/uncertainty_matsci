# Qwen3.5-9B: two independently accepted training collections

Completed CPU acceptance: **2026-09-20 23:10:45 CST**. This bounded audit accepts the existing Al-Au-Hf B50 training collections for seeds1 and2: **2 of7 registered collection jobs**. It adds **zero** model, graph, oracle or fitting calls. Both CPU audits passed the unmodified original9B `audit_job`; the actual supervisor exited0, and original evidence hashes remained unchanged.

| Training chemistry | Seed | Existing candidate ORB calls | Initialization ORB calls | Proposals | Captured layers | Rows per layer | FP32 activation tensor bytes | SUN | AUDC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Al-Au-Hf |1|50|28|180|32|2880|3019898880|14|0.2888|
| Al-Au-Hf |2|50|28|156|32|2496|2617245696|14|0.368|

These are **training-data outcomes, not held-out efficacy results**. Their acceptance does not increment the separately reported5/15 baseline tests, complete the7-collection scope, establish a9B method benefit, or fit any TC/NN/ES model. The table characterizes existing data; no statistical comparison with the4B test matrix is made.

The original audit rechecked immutable raw inventory, full token prefixes, actual sampling seeds and G0 policy identity, official50-step RPC/ORB accounting and closed journal, failure labels and all32 finite FP32 input/output activation shards. There are336 proposals and5376 captured positions per layer across the two collections (5637144576 tensor bytes). Labels attach repeated tool decisions to their subsequent physical outcomes:301 proposals have known future-failure labels and35 unexecuted proposals retain unknown labels; these are not336 independent physical experiments.

Before dispatch, no prior periodic collection-audit claim, representation-stage preparation claim or independent per-job audit claim was present. New permanent CPU claims ensure failed/unknown audits cannot be replayed. Original result/claim/profile/manifest/episode SHA references are preserved in the two byte-exact accepted records. Operator logs, process metadata, credentials, tensors and model weights are not published.

The original periodic test-only exporter continues to show raw collection results as requiring a separate audit; this dedicated acceptance record is that separate evidence. It does not modify the frozen test exporter or count training data as tests. Other collection jobs are outside this bounded acceptance record; pending or failed jobs are not converted to completed data.

Run `python3 -B verify.py` to check portable receipt hashes, identities and counts. This portable check does not repeat the original raw-tensor audit, which requires the external server artifacts. The actual original audit was executed once per listed job with CUDA hidden and no model/oracle construction.
