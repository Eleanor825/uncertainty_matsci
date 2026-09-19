# Actual archive-label addendum, 2026-09-20 02:27:03 CST

The fixed read-only extractor completed and retained every input SHA/mtime. Original cold-start labels were recomputed from the original physical outcomes and agreed. No model, graph, fit or oracle was called.

| Existing rows | Original no-HVI | Alternate archive no-HVI | Label flips |
|---|---:|---:|---:|
| 75 train eligible; other two train episodes as 60-row prior | 70/75 (93.33%) | 73/75 (97.33%) | 3 |
| 50 dev eligible; all three train episodes as 90-row prior | 45/50 (90%) | 48/50 (96%) | 3 |

Only collection_train_1102 and collection_dev_2101 change (three labels each). This directly demonstrates archive-dependence of the target on already observed outcomes, without changing any actual query. The original first-candidate risk mean moved from development0.867484 to selected-policy test-first0.190787: the label-prior shift is toward *more* failures, whereas the predicted risk moved sharply in the opposite direction. The small label-count shifts cannot alone explain that probability movement. Input/prompt/feature support changes and selection effects remain distinct mechanisms, not isolated causes.

These alternate labels are **not** valid warm-prompt training examples: their original prompts and activations saw the cold-start context. They remain descriptive counterfactual rescoring only. In particular the 550-row prior contains the old fitting outcomes, so it must not be used to relabel those old rows for fitting.

The next proposed collection has now been clarified by root: five *new* G0 trajectories will actually receive the fixed550-row prior, use new disjoint train/dev seeds, and capture their actual new prefixes/graphs. Original test records are untouched; prior is fixed before all these new observations. This supersedes the report's earlier90-train-row-prior deployment candidate only; the read-only60/90-row diagnostic remains exactly as executed.

Actual new full seed5201 is now independently read: 45 scored actions, all45 no-HVI, mean risk0.152521584, range[0.0371407494,0.7580465674], one p≥0.5, Brier0.7316288095, prior550 and final HV gain0. This is additional descriptive evidence of the same problem, not data for choosing a threshold or model.

Snapshot SHA256: `1f1a67d21a7b45d981c6f0ea75b3ea2208495aab3a207320eceebcda202c7526`. Original `REPORT.md`/analysis/source bundle remain unchanged.
