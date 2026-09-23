# Frozen graph increment: CPU paired diagnostic

Each stratum is one policy-conditioned episode in world4/p336; rows are serially correlated. No row-level CI, p-value, causal policy gain or broad generalization claim.
Loss/AUROC use common-supported executed actions only; all unsupported actions stay in denominators and CSV. NoGraph capture success on graph-unsupported candidates is unobserved.
Candidate ranking differences do not identify outcomes of unexecuted actions. Old controller generated and selected the observed candidates.

| Cohort | Stratum | Official labels | Common support | Family | Brier | Logloss | AUROC |
|---|---|---:|---:|---|---:|---:|---:|
| G0_UQ | G0 | 30 | 30 | Internal | 0.04614951838185843 | 0.15149631292181248 | 0.9807692307692307 |
| G0_UQ | G0 | 30 | 30 | NoGraph | 0.09223894831820778 | 0.28276595806375077 | 0.9423076923076923 |
| G0_UQ | G0 | 30 | 30 | GraphOnly | 0.051433644379249924 | 0.18485310757935475 | 0.9711538461538461 |

G0_UQ paired differences (negative Brier/logloss favours the first family):
{"GraphOnly_minus_NoGraph": {"auroc": 0.028846153846153855, "brier": -0.04080530393895786, "logloss": -0.09791285048439602}, "Internal_minus_GraphOnly": {"auroc": 0.009615384615384581, "brier": -0.005284125997391496, "logloss": -0.033356794657542266}, "Internal_minus_NoGraph": {"auroc": 0.038461538461538436, "brier": -0.04608942993634935, "logloss": -0.13126964514193828}}
Same generated-candidate decisions: {"GraphOnly_NoGraph_disagreements": 3, "Internal_NoGraph_disagreements": 2, "all_attempts": 30, "common_supported_two_candidate_attempts": 6, "recorded_vs_recomputed_Internal_disagreements": 0}

| Full | selected_G1 | 30 | 30 | Internal | 0.14491567750631024 | 0.5229616149018989 | 0.9330357142857143 |
| Full | selected_G1 | 30 | 30 | NoGraph | 0.1842228869328549 | 0.6653473018406101 | 0.8125 |
| Full | selected_G1 | 30 | 30 | GraphOnly | 0.0786682217557241 | 0.2489139141986189 | 0.9732142857142857 |

Full paired differences (negative Brier/logloss favours the first family):
{"GraphOnly_minus_NoGraph": {"auroc": 0.1607142857142857, "brier": -0.1055546651771308, "logloss": -0.41643338764199117}, "Internal_minus_GraphOnly": {"auroc": -0.0401785714285714, "brier": 0.06624745575058613, "logloss": 0.27404770070327994}, "Internal_minus_NoGraph": {"auroc": 0.1205357142857143, "brier": -0.03930720942654467, "logloss": -0.14238568693871123}}
Same generated-candidate decisions: {"GraphOnly_NoGraph_disagreements": 4, "Internal_NoGraph_disagreements": 3, "all_attempts": 30, "common_supported_two_candidate_attempts": 14, "recorded_vs_recomputed_Internal_disagreements": 0}

| selected_theta_NN_off | selected_G1 | 30 | 0 | Internal | None | None | None |
| selected_theta_NN_off | selected_G1 | 30 | 0 | NoGraph | None | None | None |
| selected_theta_NN_off | selected_G1 | 30 | 0 | GraphOnly | None | None | None |

selected_theta_NN_off paired differences (negative Brier/logloss favours the first family):
{"GraphOnly_minus_NoGraph": {"auroc": null, "brier": null, "logloss": null}, "Internal_minus_GraphOnly": {"auroc": null, "brier": null, "logloss": null}, "Internal_minus_NoGraph": {"auroc": null, "brier": null, "logloss": null}}
Same generated-candidate decisions: {"GraphOnly_NoGraph_disagreements": 0, "Internal_NoGraph_disagreements": 0, "all_attempts": 30, "common_supported_two_candidate_attempts": 0, "recorded_vs_recomputed_Internal_disagreements": 0}

Not analyzed at protocol snapshot: []
No model fitting, new graph extraction, LLM or environment calls. No scientific configuration was changed.
