# One previously used MADE development seed: G0 control

Read [report.md](report.md) and [comparison.json](comparison.json). Run `python3 -B recompute.py` for local arithmetic/identity validation; no model or oracle is called. G0/G1/G2 AUDC is .68/.62/.64 on Al-Pd-Sm B10 with one previously used development seed. Only the separate G0 diagnostic added10 candidate calls; G1/G2 are existing references. Initialization and surrogate-call costs are separately recorded.

This is not a held-out comparison, training replication or part of the180/1,080 test matrices. Missing G1/G2 SUN remains null. The original G2 selection and experiment results are unchanged. [publication_provenance.json](publication_provenance.json) separates original scientific hashes from the export manifest.
