"""Pure scheduling/integrity fixtures; no models or scientific tools execute."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from matdiscovery.accounting import FULL_COUNTS, file_sha256, fingerprint

PROJECT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / (name + ".py"))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_unsealed_queue_is_safe_for_already_running_old_supervisor(tmp_path, monkeypatch):
    script = module("build_stage_queue")
    (tmp_path / "scripts").mkdir()
    for name in ("run_representation_condition.py", "train_es_condition.py", "run_final_evaluation.py", "report_made_phase.py", "report_full_study.py", "finalize_stage_queue.py"):
        (tmp_path / "scripts" / name).write_text("# fixture only\n")
    original = [{"id": f"collect-{model}-{bench}", "command": ["synthetic-no-execution", "--model-key", model, "--benchmark", bench]}
                for model in ("qwen35_4b", "qwen35_9b") for bench in ("made", "crystalgym")]
    write(tmp_path / "configs/stage_queue.json", {"stages": original, "sealed": False})
    write(tmp_path / "logs/pipeline_status.json", {"stages": {original[0]["id"]: {"status": "started"}}})
    monkeypatch.setattr(sys, "argv", ["build_stage_queue", "--project", str(tmp_path)])
    script.main()
    unsealed = json.loads((tmp_path / "configs/stage_queue.json").read_text())
    assert unsealed["stages"] == [original[0], original[2]]
    assert len(unsealed["declared_next_stages"]) == 29 and len(unsealed["deferred_stages"]) == 72
    final = [s for s in unsealed["declared_next_stages"] + unsealed["deferred_stages"] if s["id"].startswith("final-")]
    assert len(final) == 4 and "--resume" not in final[0]["command"]
    assert all("--resume" in s["command"] for s in final[1:])
    monkeypatch.setattr(sys, "argv", ["build_stage_queue", "--project", str(tmp_path), "--seal"])
    script.main()
    sealed = json.loads((tmp_path / "configs/stage_queue.json").read_text())
    assert sealed["stages"][:2] == [original[0], original[2]] and len(sealed["stages"]) == 31
    assert sealed["declared_next_stages"] == [] and all(s["requires_sealed"] for s in sealed["stages"][2:])
    assert sealed["stages"][-1]["id"] == "report-made-phase"
    assert all("crystalgym" not in s["id"] for s in sealed["stages"])
    assert sealed["completion_scope"] == "full_current_two_model_MADE_phase"
    assert not sealed["final_acceptance_verified"] and not sealed["latest_3_to_4_model_extension_complete"]
    assert len(sealed["deferred_stages"]) == 72
    assert len({s["id"] for s in sealed["stages"] + sealed["deferred_stages"]}) == 103

    # Later-benchmark activation is explicit; its commands retain full budgets.
    monkeypatch.setattr(sys, "argv", ["build_stage_queue", "--project", str(tmp_path), "--seal", "--phase", "all"])
    script.main()
    all_phases = json.loads((tmp_path / "configs/stage_queue.json").read_text())
    assert len(all_phases["stages"]) == 103 and all_phases["deferred_stages"] == []
    assert all_phases["stages"][:31] == sealed["stages"]
    assert all_phases["stages"][31:] == sealed["deferred_stages"]
    assert all_phases["counts"]["final_jobs"] == 2160
    assert all_phases["counts"]["MADE_final_jobs"] == 1800

    # A started later benchmark cannot be hidden by switching back to MADE.
    write(tmp_path / "logs/pipeline_status.json", {"stages": {original[1]["id"]: {"status": "started"}}})
    monkeypatch.setattr(sys, "argv", ["build_stage_queue", "--project", str(tmp_path), "--seal"])
    with pytest.raises(RuntimeError, match="silently deferred"):
        script.main()


def test_locked_snapshot_plan_entry_and_project_override_guard(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "configs").mkdir()
    entry = tmp_path / "scripts/fixture_entry.py"
    entry.write_text("import argparse,json\np=argparse.ArgumentParser();p.add_argument('--project');p.add_argument('--plan-only',action='store_true');a=p.parse_args();print(json.dumps(vars(a)))\n")
    command = [sys.executable, str(PROJECT / "scripts/run_locked_stage.py"), "--project", str(tmp_path), "--entry", entry.name, "--", "--plan-only"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[-1]["plan_only"] is True and Path(rows[-1]["project"]).parent == tmp_path / "frozen_sources"
    assert (Path(rows[-1]["project"]) / "source_manifest.json").is_file()
    rejected = subprocess.run(command + ["--project=/not-the-frozen-project"], text=True, capture_output=True)
    assert rejected.returncode != 0 and "wrapper supplies" in rejected.stderr
    source = Path(rows[-1]["project"]) / "source_manifest.json"
    payload = json.loads(source.read_text())
    payload["original_project"] = "/different-project"
    write(source, payload)
    rejected = subprocess.run(command, text=True, capture_output=True)
    assert rejected.returncode != 0 and "provenance" in rejected.stderr


def test_requested_method_gap_prevents_sealing_downstream_experiments(tmp_path, monkeypatch):
    script = module("build_stage_queue")
    write(tmp_path / "configs/method_alignment_gate.json", {
        "implementation_matches_requested_method": False,
        "full_baseline_collection_may_continue": True,
    })
    monkeypatch.setattr(sys, "argv", ["build_stage_queue", "--project", str(tmp_path), "--seal"])
    with pytest.raises(RuntimeError, match="method alignment is incomplete"):
        script.main()
    write(tmp_path / "configs/method_alignment_gate.json", {"implementation_matches_requested_method": True})
    with pytest.raises(RuntimeError, match="actual validation evidence"):
        script.main()


def test_representation_gate_checks_raw_chain_and_duplicate_progress(tmp_path, monkeypatch):
    from collection_cost_fixtures import write_cost_collection_fixtures
    from matdiscovery.collection_provenance import CollectionProvenanceError
    from matdiscovery.experiment_plan import collection_jobs
    script = module("run_representation_condition")
    project = Path(__file__).resolve().parents[1]
    jobs = [j for j in collection_jobs(project) if j["model_key"] == "qwen35_4b" and j["benchmark"] == "made"]
    directories = write_cost_collection_fixtures(tmp_path, jobs)
    monkeypatch.setattr(script, "collection_jobs", lambda root: jobs)
    manifests = [directory / "collection_manifest.json" for directory in directories]
    assert script.completed_collections(tmp_path, "qwen35_4b", "made") == manifests
    base = directories[0].parent
    progress = json.loads((base / "progress.json").read_text())
    progress["collection_manifests"].append(str(manifests[0]))
    write(base / "progress.json", progress)
    with pytest.raises(RuntimeError, match="duplicated"):
        script.completed_collections(tmp_path, "qwen35_4b", "made")
    progress["collection_manifests"].pop()
    write(base / "progress.json", progress)
    (directories[0] / "decisions.jsonl").write_text("changed raw evidence\n")
    with pytest.raises(CollectionProvenanceError, match="changed"):
        script.completed_collections(tmp_path, "qwen35_4b", "made")


def test_report_graph_missingness_and_prior_unknown_cost_not_zero(tmp_path):
    script = module("report_full_study")
    path = tmp_path / "decisions.jsonl"
    row = {"episode_index": 0, "disposition": "executed", "label_future_failure": 1,
           "predicted_failure_probability": .5, "graph_status": "unavailable"}
    path.write_text(json.dumps(row) + "\n")
    envelope = {"artifacts": [{"path": str(path), "sha256": file_sha256(path)}], "episodes": [{"episode_id": "0", "metrics": {}}]}
    _, counts, _ = script.reliability_for_job(envelope, tmp_path)
    assert counts["graph_proposed"] == counts["graph_unavailable"] == 1
    events = [{"job_id": "final-made-fixture", "event": "claimed", "details": {"attempt_id": "a"}},
              {"job_id": "final-made-fixture", "event": "failed", "details": {"attempt_id": "a", "costs": None}},
              {"job_id": "final-made-fixture", "event": "claimed", "details": {"attempt_id": "b"}}]
    states = {"final-made-fixture": {"attempt_number": 2}}
    report = script.audit_attempt_history(events, states)
    assert report["prior_attempts_with_unknown_physical_cost"] == 1 and report["all_attempt_costs_accounted"] is False
    events[1]["details"]["costs"] = {"candidate_oracle_attempts": 3}
    report = script.audit_attempt_history(events, states)
    assert report["all_attempt_costs_accounted"] and report["known_prior_attempt_costs"]["candidate_oracle_attempts"] == 3


def test_finalizer_rejects_unknown_costs_before_claiming_study_complete(tmp_path, monkeypatch):
    script = module("finalize_stage_queue")
    acceptance = {"all_final_experiments_complete": True, "expected_counts": FULL_COUNTS,
                  "all_attempt_costs_accounted": False, "all_es_training_conditions_accounted": True}
    acceptance["fingerprint"] = fingerprint(acceptance)
    path = tmp_path / "acceptance.json"
    write(path, acceptance)
    monkeypatch.setattr(sys, "argv", ["finalize", "--project", str(tmp_path), "--acceptance", str(path)])
    with pytest.raises(RuntimeError, match="Unknown prior"):
        script.main()
