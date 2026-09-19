# Exact SnAr executed-candidate diagnosis

Read [report.md](report.md). This reuses 15 of the existing35 B50 episodes (Qwen base/full/GP-EI, five seeds each); it adds no experiments. Full and UQ selected the same G0 in the accepted study, so these results do not demonstrate an ES improvement.

Run `python3 -B analyze.py` with the Python standard library to verify the included hash-pinned [gzip scientific projection](observations.json.gz), all15 derived tables/summary, and package manifest. `--write` regenerates only derived outputs. Original extraction SHA and path-relocated export SHA are separate in [provenance.json](provenance.json). The analysis functions are unchanged; only portable input loading and source-reference prefixes differ.

The projection contains750 executed queries,248 full candidates and125 train/dev feature rows. All numerical values, arrays and nulls are preserved, including24 unexecuted candidate outcomes left blank. No causal retry effect, test recalibration or new parameter selection is claimed. Original scientific-source checks are historical evidence; portable recomputation does not reconstruct the full550-query prior or rerun model/graph/oracle computation.
