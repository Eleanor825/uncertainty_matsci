# Fixed144-run MADE mechanism audit

Read [REPORT.md](REPORT.md). This posthoc dataset is the full set of complete three-seed baseline/full B10 chemistry pairs at the fixed429/1080 snapshot, not a new experiment or the complete1080 matrix.

Run `python3 -B recompute.py` with the standard library. It reads the included exact compressed scientific observations, verifies original hashes and all derived tables, and performs no fitting or oracle work. Sample variance uses ddof=1 across matching evaluation seeds. Missing risks/costs stay missing.

Original-source and export hashes are distinguished in [provenance.json](provenance.json). The input retains every executed and unexecuted compact risk row; unexecuted physical labels remain null. Original large raw logs, geometry, model weights, runtime operator material and credentials are excluded.
