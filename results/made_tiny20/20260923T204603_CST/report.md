# MADE Tiny20: first three accepted trajectories (3/20)

This interim snapshot contains **3 accepted trajectories out of 20 canonical jobs**, all on **Mg–Sn–Sr, B5, seed1**. The three-result snapshot was read on **23 September 2026 at 20:46:03 CST (12:46:03 UTC)**. Native and Hidden145Fixed2 were already present in the first-pair readback at 20:32:06 CST. The two comparisons below share the same Native trajectory and are one system/seed block, not independent replications. They do not establish an overall treatment effect.

| Arm | Final SUN | mSUN | Normalized AUDC | Candidate ORB attempts | LLM calls |
|---|---:|---:|---:|---:|---:|
| Native | 5 | 1.0 | 1.0 | 5 | 16 |
| Hidden145Fixed2 | 4 | 0.8 | 0.8 | 5 | 36 |
| G2Full | 4 | 0.8 | 0.96 | 5 | 23 |

Within this block, Hidden145Fixed2 minus Native is −1 SUN, −0.2 mSUN and −0.2 AUDC, with 20 additional LLM calls. G2Full minus the same Native is −1 SUN, −0.2 mSUN and −0.04 AUDC, with 7 additional LLM calls. Native's recorded curve is 0, 1, 2, 3, 4, 5. Hidden145Fixed2's is 0, 1, 2, 2, 3, 4. G2Full's is 0, 1, 2, 3, 4, 4. No smoothing, cumulative-maximum correction, baseline replacement or outcome selection was applied in this export.

SUN is the final stable, unique and novel discovery count. mSUN = SUN / B. Normalized AUDC = 2 × trapezoidal area under the recorded SUN curve / B². Candidate ORB attempts are the budgeted calls; they do not include all environment costs. Each trajectory also records 96 initialization oracle attempts, and Native / Hidden145Fixed2 record 8 / 9 surrogate oracle attempts. G2Full records 418.58 graph seconds and 804.95 wall seconds. The CSV retains recorded cost fields. Missing fields, including G2Full surrogate-oracle and score-sentinel counts and failure rate in this source, remain blank rather than being filled with zero.

## Scope and interpretation

The registered matrix is Mg–Sn–Sr and Au–K–Tb × B5 and B15 × Native, G0GraphRisk, G2ControllerOff, G2Full and Hidden145Fixed2, all at seed1: 20 fresh trajectories and 200 planned candidate ORB attempts. The model is Qwen/Qwen3.5-4B, with model and tokenizer revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. These are two selected official systems. The B5/B15 scope was chosen after previous results were known. This snapshot supports neither full-benchmark coverage nor repeated-seed inference.

Native uses the unchanged original G0 baseline. Hidden145Fixed2 is the second-version combination of G0, the frozen original MADE hidden145 risk head, exactly two proposals and explicit public feedback. It is not the original G2Full method and it contains no new ES update. G2Full uses the original selected generation-2 full graph-risk/controller method. The batch has no Common2 arm, so the Hidden145Fixed2 difference combines hidden-risk ranking, extra proposals and feedback; an isolated NN contribution cannot be inferred. G2ControllerOff uses the original Full-trained selected G2 with its controller disabled and is not independently trained ES-only. No new ES or risk-head training occurs in this batch.

## Preserved infrastructure failures

A separate read-only inspection at **20:37:01 CST (12:37:01 UTC)** identified **8 original attempts that failed during environment initialization because the required ORB checkpoint was missing from the cache**. Each recorded request trace contains only initialization, no decision file and zero candidate oracle calls. These are infrastructure failures with unavailable scores, not zero-score scientific outcomes. The eight reruns had not executed at that snapshot, and no retry result enters this report. The other 9 canonical jobs have no accepted result in this three-row export; their current execution state is not inferred here.

## Evidence

- [3rows.csv](3rows.csv): the three accepted records, including recorded costs, generation identities and raw-result/completion hashes.
- [curves.csv](curves.csv): all six recorded points per trajectory, including the origin.
- [evidence.json](evidence.json): sanitized snapshot, scope and the eight original initialization failures.
- [scientific_sha256.json](scientific_sha256.json): SHA-256 references for the source snapshots, scientific protocol/lineage and original result/failure/RPC files.
- [manifest.json](manifest.json) and [validate.py](validate.py): public bundle hashes and numerical consistency checks; run `python3 validate.py` from this directory.

The accepted-result reader verified original raw-result hashes, completed status and canonical registration identity before this projection. Its source SHA matches the readback's bound source SHA. This publication rechecked local snapshot and frozen-source hashes, matching result/verification receipts, exact candidate counts, SUN endpoints, trapezoidal AUDC and CSV consistency. Original raw server files are retained in the research archive rather than uploaded here. This export performs no new scientific, experiment-server or GPU calls, and changes no model, baseline or scientific outcome.
