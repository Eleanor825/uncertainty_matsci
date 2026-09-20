# Original seed1 B50 all-60 posthoc hull audit

Scientific data only: 60 trajectory identities, 3,000 existing ORB scalar outcomes, exact initial composition/energy hulls, and original evidence hashes. No geometry, LLM text, tensor, credential, host, operator or process records are included.

Run `python -B recompute.py` with NumPy and SciPy to rebuild all event, novelty, time-band and convex-hull tables in a fresh `recomputed/` directory. No PyTorch, model, MADE environment, oracle, network or fitting is used. The original execution used Python3.12, NumPy/SciPy versions listed in outputs/summary.json. Existing output directories are not overwritten. Official SUN/AUDC stay unchanged; this is single-seed held-out posthoc analysis, not a training selection dataset.

The portable source differs from the internal analyze_actual.py only in its input path/gzip decoding, pinned projection checksum and output directory. All numerical/statistical logic is byte-preserved. The source input retains the original readback SHA and per-job raw evidence SHA; the original raw envelopes and their acceptance distinction remain authoritative. The new projection is not a substitute original scientific receipt.
