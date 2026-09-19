# Portable SnAr feature-family diagnostic archive v2

Read `REPORT.md` and `CORRECTION.json` first. The original full and independent ES-only selections both use **G0/θ0**, not evolved weights. The full/UQ physical curves coincide, as do ES-only/base. This archive does not claim additional independent mechanism successes from those duplicate curves.

The 17 original downloaded diagnostic JSON artifacts total 1,111,076 bytes. Their deterministic gzip copies retain byte-exact originals when decompressed. Additional scientific sources provide the 125 original fit-feature rows, 224 executed-action associations, frozen fit settings, original NN code, and G0/prior selection evidence. `source_manifest.json` gives compressed and uncompressed hashes and actual byte sizes. `MANIFEST.json` covers the full review bundle. No `.pt`, model weights, browser/authentication data, process observer code, or credentials are included.

Run with **Python 3.9+ and its standard library only**, from any directory:

```sh
python /path/to/bundle/recompute.py
```

Optionally write new outputs (existing paths are refused):

```sh
python /path/to/bundle/recompute.py --output verification-new.json --calibration-csv bins-new.csv
```

It verifies all gzip source bytes, original JSON seals, the 12-model/224-unique-action matrix, unchanged action identities/labels, all 2,688 scores, test/development AUROC with ties, average precision, Brier/NLL/ECE and risk coverage, five policy-seed strata, constants, and means/sample variances over the three NN seeds. It also checks recorded epoch selection/early stopping and 746 actual CPU optimizer steps. It does not load or fit any neural model. The copied original fitting runner remains provenance with original absolute resource paths; it is **not** the portable verifier and should not be invoked by archive users.

The fixed ten-bin calibration CSV contains all 12 models, raw and calibrated scores, pooled and five policy-seed strata; empty-bin means remain missing. Its observed fractions are descriptions of dependent selected actions, not independent-action confidence intervals. The verifier checks a 1e−12 cross-implementation floating arithmetic bound; the observed maximum discrepancy is 8.8818e−16. This does not alter the separate original **byte-exact** saved-model report reconstruction acceptance. Original audit rc=1 is preserved, with a separate technical derived acceptance.

The archive is explicitly posthoc. There are only six HVI improvements; all actions were selected by the original full controller. No threshold/configuration/checkpoint/model-family winner is chosen. No policy is deployed and no new LLM, graph, physical query or fitting is performed by this archive.

Published as a dated posthoc diagnostic archive. Original scientific artifact bytes and the original failed audit remain unchanged; publication_provenance.json records original/export manifest identities.
