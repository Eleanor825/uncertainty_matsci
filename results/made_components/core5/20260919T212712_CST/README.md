# Complete fixed-core MADE B10 ablation: five systems × three seeds × four arms

Read [report.md](report.md) and [summary.json](summary.json). All60 core cells are complete, while the larger registered study is279/1,080. UQ/controller has a positive core mean difference; full remains below baseline. UQ seed4 declines. The two ES policies were trained separately, and the NN contribution is not isolated.

Run `python3 -B analyze.py` to recompute and byte-verify the local cached tables and manifest. No model, graph, oracle or network is invoked. `--write` recreates only derived tables; `python3 -B render_report.py` renders the report from them. All variances use ddof=1 over evaluation seeds within fixed systems, or over complete five-system seed aggregates. They are not between-system variance or repeated-training variance.

The prior56 completed core results are unchanged; the four formerly missing UQ rows are now present. This package remains separate from the older180 seed1 study, SnAr and development diagnostics.
