"""Recompute this diagnostic from exported recorded evidence only."""
from pathlib import Path
from collections import Counter
import ast, hashlib, json
P = Path(__file__).resolve().parent
m = json.loads((P / "manifest.json").read_text())
for name, ref in m["files"].items():
    b = (P / name).read_bytes()
    assert len(b) == ref["bytes"] and hashlib.sha256(b).hexdigest() == ref["sha256"], name
assert hashlib.sha256((P / "audit_source.py").read_bytes()).hexdigest() == m["source_receipt_exporter_sha256"]
ast.parse((P / "audit_source.py").read_text())
d = json.loads((P / "evidence.json").read_text())
assert d["schema"] == "dw_p336_two_arm_revision_failure_readonly_audit_v1"
assert d["read_only"] and all(v == 0 for v in d["scientific_calls"].values())
assert d["risks_are_only_previously_recorded_values"] and not d["unexecuted_outcome_labels_created"]
for a in d["arms"]:
    assert a["closed_B30_validated"] and a["policy_seed"] == 336 and a["world_seed"] == 4
    steps = a["steps"]
    assert [r["attempt"] for r in steps] == list(range(1, 31))
    assert dict(Counter(r["selection_path"] for r in steps)) == a["selection_paths"]
    assert all(r["selection"]["selected_index"] == 0 and not r["selected_action_changed_from_first"] for r in steps)
    alt = [r for r in steps if r["alternative"] is not None]
    assert len(alt) == 28
    assert all(r["alternative"]["generation_success"] and isinstance(r["alternative"]["parsed_action"], dict) for r in alt)
    unsupported = [r for r in alt if not r["alternative"]["NoGraph_supported"]]
    compared = [r for r in alt if r["selection"]["supported_comparison"]]
    layers = Counter(r["alternative"]["capture_failure"]["layer"] for r in unsupported)
    for r in unsupported:
        f = r["alternative"]["capture_failure"]
        assert f["category"] == "transcoder_fidelity_gate" and f["fvu_defined"] and f["output_fvu"] > 0.5
        assert r["selection_path"] == "incomplete_support_fallback"
        assert r["selection"]["risks"] is None and r["alternative"]["recorded_risk"] is None
    for r in compared:
        risks = r["selection"]["risks"]
        assert risks[1] > risks[0]
        assert r["selection_path"] == "alternative_higher_risk_keeps_first"
    if a["condition"] == "NoGraphRisk":
        assert len(unsupported) == 6 and len(compared) == 22
        assert layers == {"model.language_model.layers.13.mlp": 1, "model.language_model.layers.16.mlp": 5}
    else:
        assert a["condition"] == "ExplicitRepeatRisk"
        assert len(unsupported) == 28 and len(compared) == 0
        assert layers == {"model.language_model.layers.12.mlp": 28}
print(json.dumps({"closed_arms": 2, "decisions": 60, "parsed_alternatives": 56, "unsupported_alternatives": 34, "higher_risk_alternatives_retained_first": 22, "new_scientific_calls": 0}))
