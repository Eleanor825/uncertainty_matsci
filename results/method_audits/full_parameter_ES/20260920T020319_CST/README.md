# Scope of the parameter-update evidence

This is an audit of recorded full-parameter ES histories, tensor-manifest counts, actual delta summaries and selected-state hashes, not a fresh model reload. The723 unique tensors contain4,539,265,536 scalar policy parameters. Auxiliary NNs, transcoders and evaluators are outside that scope. A nonzero tensor norm only proves at least one scalar changed in that tensor. Tied aliases are counted once.

The original readback SHA and all scientific source references are retained; snapshot export only removes host identity and relocates machine-specific reference paths. No update arrays, rewards, norms, model/state hashes or selection values are changed. Both SnAr selected policies remainG0, so performing ES training is not itself evidence of an ES test benefit. Run `python3 -B verify.py` for portable metadata/table consistency and hashes.
