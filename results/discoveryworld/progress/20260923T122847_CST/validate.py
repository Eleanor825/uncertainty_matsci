"""Validate only this public partial snapshot; no experiment or large artifact reads."""
from pathlib import Path
import hashlib, importlib.util, json
P = Path(__file__).resolve().parent
manifest = json.loads((P / "manifest.json").read_text())
for name, ref in manifest["files"].items():
    raw = (P / name).read_bytes()
    assert len(raw) == ref["bytes"]
    assert hashlib.sha256(raw).hexdigest() == ref["sha256"], name
d = json.loads((P / "snapshot.json").read_text())
assert d["publication_ready"] is False and d["all_ten_complete"] is False
assert manifest["publication_kind"] == "partial_progress_only"
assert all(x == 0 for x in d["scientific_calls"].values())
assert d["old_unseeded_p335_pooling"] is False
assert d["original_Full_followup_included"] is False
assert len(d["arms"]) == 10
closed = [a for a in d["arms"] if a["status"] == "complete"]
partial = [a for a in d["arms"] if a["status"] == "partial"]
assert len(closed) == 9 and len(partial) == 1
assert (partial[0]["policy_seed"], partial[0]["condition"]) == (336, "ExplicitRepeatInternal2")
assert partial[0]["metrics"] is None
assert partial[0]["progress"]["completed_action_returns"] == 6
for a in closed:
    m = a["metrics"]
    assert m["agent_attempts"] == len(a["action_rows"]) == 30
    assert m["failure_count"] == sum(r["official_success"] is False for r in a["action_rows"])
    assert m["completedSuccessfully"] is False
    assert a["RPC_closure"]["closed_response"] and a["RPC_closure"]["returncode"] == 0
    assert not a["audit_errors"]
for seed in (336, 337):
    arms = {a["condition"]: a for a in d["arms"] if a["policy_seed"] == seed}
    native = arms["Native1"]
    for name in ("NoGraphRisk", "ExplicitRepeatRisk"):
        assert [r["executed_action_sha256"] for r in native["action_rows"]] == [r["executed_action_sha256"] for r in arms[name]["action_rows"]]
        assert native["metrics"] == arms[name]["metrics"]
assert sum(p["complete_pair"] for p in d["paired_contrasts"]) == 7
for p in d["same_seed_prefix_audits"]:
    assert not p["public_mismatch_after_same_prefix"] and not p["global_rng_mismatch_after_same_prefix"]
    assert p["complete_hidden_state_equivalence_claimed"] is False
    for s in p["aligned_same_action_prefix_snapshots"]:
        for key in ("global_rng_equal", "public_UI_equal", "object_construction_summary_equal"):
            assert s[key] is True
        if s["next_proposal0_available"]:
            assert all(s[k] is True for k in ("next_prompt_messages_equal", "next_prompt_tokens_equal", "next_sampling_seed_equal"))
spec = importlib.util.spec_from_file_location("snapshot_renderer", P / "render.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
report, stats = renderer.render(d)
assert (P / "report.md").read_text() == report
assert json.loads((P / "complete_two_seed_statistics.json").read_text()) == stats == []
assert hashlib.sha256((P / "render.py").read_bytes()).hexdigest() == manifest["renderer_source_sha256"]
print(json.dumps({"closed_episodes": 9, "partial_episodes": 1, "complete_pairs": 7, "complete_results_ready": False, "two_seed_statistics": 0, "new_scientific_calls": 0}))
