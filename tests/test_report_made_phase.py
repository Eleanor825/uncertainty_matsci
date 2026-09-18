"""Complete-count CPU fixtures only: never invoke a policy, ORB, QE or a GPU."""
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from matdiscovery.accounting import AccountingError, FULL_COUNTS, build_final_manifest, file_sha256, fingerprint
from matdiscovery.experiment_plan import collection_jobs
from matdiscovery.execution_contract import declared_mace_workers
from matdiscovery.metrics import IncompleteResultsError, summarize_results
from matdiscovery import study_costs
from collection_cost_fixtures import write_cost_collection_fixtures
from matdiscovery.collection_provenance import CollectionProvenanceError

PROJECT = Path(__file__).resolve().parents[1]


def load_report():
    spec = importlib.util.spec_from_file_location("report_made_phase_test", PROJECT / "scripts/report_made_phase.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture(scope="module")
def report():
    return load_report()


@pytest.fixture(scope="module")
def manifest():
    return build_final_manifest(PROJECT)


@pytest.fixture(scope="module")
def made_rows(manifest):
    return [{**{key: job[key] for key in ("benchmark", "model_key", "method", "task_id", "seed")},
             "episode_id": "0", "environment_seed": job["seed"], "complete": True, "status": "succeeded",
             "costs": {"candidate_oracle_attempts": 50, "dft_episode_attempts": 0, "wall_seconds": 1.0},
             "metrics": {"AUDC": .1 + (.2 if job["method"] == "esopt_graph_risk" else 0), "mSUN": 1.0,
                         "decision_brier": .2, "decision_overconfident_error_rate": .1}}
            for job in manifest["jobs"] if job["benchmark"] == "made"]


def es_conditions():
    return {f"synthetic-{model}-{method}-{seed}": {"model_key": model, "benchmark": "made", "method": method,
        "seed": seed, "property": None, "training_directory": "synthetic-no-execution",
        "costs": {"candidate_oracle_attempts": 8800, "dft_episode_attempts": 0, "wall_seconds": 1.0}}
        for model in ("qwen35_4b", "qwen35_9b") for method in ("esopt", "esopt_graph_risk") for seed in range(1, 6)}


def cost_fixture():
    return {"benchmark_scope": "made", "experimental_cost_accounting_complete": True,
        "shared_collection": {"complete": True, "expected_jobs": 420, "completed_jobs": 420,
            "costs": {"completed_episodes": 420, "candidate_oracle_attempts": 21000, "dft_episode_attempts": 0}},
        "offline_and_other_stage_times": {"complete": True, "expected_stages": 30, "representation_expected_stages": 6},
        "experimental_end_to_end": {"cost_categories": {
            "independent_es_training": {"candidate_oracle_attempts": 176000},
            "matched_final_evaluation": {"candidate_oracle_attempts": 90000}}}}


def test_made_complete_does_not_reduce_or_satisfy_full_study_gate(report, manifest, made_rows):
    before = fingerprint(manifest)
    result = report.summarize_made_phase(made_rows, manifest)
    assert result["accounting"]["expected_counts"] == report.FINAL_COUNTS
    assert result["accounting"]["failure_denominator_expected"] == 1800
    assert result["accounting"]["complete"]
    assert result["global_study_complete"] is False
    assert all(row["benchmark"] == "made" for row in result["tasks"])
    assert fingerprint(manifest) == before and manifest["expected_counts"] == FULL_COUNTS
    with pytest.raises(IncompleteResultsError) as error:
        summarize_results(made_rows, manifest, final=True)
    assert error.value.report["missing_episodes"] == 1800  # Deferred CG only.
    reduced = copy.deepcopy(manifest)
    reduced["jobs"] = report.made_jobs(manifest)
    reduced["manifest_fingerprint"] = fingerprint({key: value for key, value in reduced.items() if key != "manifest_fingerprint"})
    with pytest.raises(AccountingError):
        report.made_jobs(reduced)


@pytest.mark.parametrize("change", ["missing", "duplicate", "short_budget", "incomplete"])
def test_phase_rejects_any_missing_duplicate_or_underbudget_made_episode(report, manifest, made_rows, change):
    rows = copy.deepcopy(made_rows)
    if change == "missing": rows.pop()
    elif change == "duplicate": rows.append(rows[0])
    elif change == "short_budget": rows[0]["costs"]["candidate_oracle_attempts"] = 49
    else: rows[0]["complete"] = False
    with pytest.raises((RuntimeError, AccountingError)):
        report.summarize_made_phase(rows, manifest)


def test_failed_and_uncalibrated_episodes_are_not_dropped(report, manifest, made_rows):
    rows = copy.deepcopy(made_rows)
    rows[0]["status"] = "failed"
    rows[0]["metrics"]["AUDC"] = None
    rows[0]["metrics"]["decision_brier"] = None
    summary = report.summarize_made_phase(rows, manifest)
    assert summary["accounting"]["failed_episode_count"] == 1
    assert summary["accounting"]["failure_denominator_expected"] == 1800
    protocol = json.loads((PROJECT / "configs/main_protocol.json").read_text())
    protocol["statistics"]["bootstrap_replicates"] = 2  # CPU fixture only, no file modified.
    estimates = report.phase_estimates(rows, manifest, protocol)
    assert len(estimates) == 64
    assert any(row["status"] == "undefined_scientific_metric_no_failed_episode_dropping" for row in estimates)
    assert any(row["status"] == "undefined_incomplete_confidence_coverage" for row in estimates)
    known = next(row for row in estimates if row["model_key"] == "qwen35_9b" and row["method"] == "esopt_graph_risk" and row["reference"] == "baseline" and row["metric"] == "AUDC")
    assert known["delta"] == pytest.approx(.2) and known["n_tasks"] == 30 and known["n_seed_pairs"] == 150
    assert known["n_episodes_per_method"] == 150


@pytest.mark.parametrize("change", ["missing_seed", "short_budget", "duplicate_condition", "foreign_benchmark"])
def test_exact_twenty_es_conditions_and_176000_attempts(report, change):
    conditions = es_conditions()
    assert report.validate_es_conditions(conditions)["candidate_oracle_attempts"] == 176000
    entries = list(conditions)
    if change == "missing_seed": conditions.pop(entries[0])
    elif change == "short_budget": conditions[entries[0]]["costs"]["candidate_oracle_attempts"] -= 1
    elif change == "duplicate_condition": conditions[entries[0]] = copy.deepcopy(conditions[entries[1]])
    else: conditions[entries[0]]["benchmark"] = "crystalgym"
    with pytest.raises(RuntimeError):
        report.validate_es_conditions(conditions)


def stage_plan_fixture(root):
    runnable, deferred, states = [], [], {}
    ordinal = 0
    for benchmark in ("made", "crystalgym"):
        for kind, count in (("collect", 2), ("transcoders", 2), ("graphs", 2), ("risk", 2), ("es", 20 if benchmark == "made" else 60), ("final", 2)):
            for i in range(count):
                stage = {"id": f"{kind}-synthetic-{benchmark}-{i}", "command": ["never-run-test-fixture", "--benchmark", benchmark]}
                (runnable if benchmark == "made" else deferred).append(stage)
                if benchmark == "made":
                    start = 100 + ordinal * 5
                    states[stage["id"]] = {"status": "succeeded", "returncode": 0, "started_at": start, "finished_at": start + 2,
                        "command": stage["command"], "command_fingerprint": hashlib.sha256(json.dumps(stage, sort_keys=True).encode()).hexdigest()}
                ordinal += 1
    queue = {"stages": runnable, "declared_next_stages": [], "deferred_stages": deferred}
    write(root / "configs/stage_queue.json", queue)
    write(root / "logs/pipeline_status.json", {"stages": states})
    return queue, states


def test_deferred_cg_stage_times_do_not_fail_made_or_satisfy_global(tmp_path):
    queue, states = stage_plan_fixture(tmp_path)
    phase = study_costs.audit_stage_times(tmp_path, benchmark="made")
    assert phase["complete"] and phase["expected_stages"] == 30
    assert phase["representation_expected_stages"] == 6 and phase["representation_wall_seconds"] == 12
    global_report = study_costs.audit_stage_times(tmp_path)
    assert global_report["expected_stages"] == 100 and not global_report["complete"]
    assert len(global_report["unknown"]) == 70
    queue["deferred_stages"].append(queue["stages"][0])
    write(tmp_path / "configs/stage_queue.json", queue)
    with pytest.raises(RuntimeError, match="Duplicate"):
        study_costs.audit_stage_times(tmp_path, benchmark="made")


def test_only_made_collection_artifacts_needed_but_all_420_are_verified(tmp_path, monkeypatch):
    jobs = collection_jobs(PROJECT)
    monkeypatch.setattr(study_costs, "collection_jobs", lambda root: jobs)
    directories = write_cost_collection_fixtures(tmp_path, jobs, benchmark="made")
    phase = study_costs.audit_collection_costs(tmp_path, benchmark="made")
    assert phase["completed_jobs"] == 420 and phase["costs"]["completed_episodes"] == 420
    assert phase["costs"]["candidate_oracle_attempts"] == 21000
    assert phase["costs"]["initialization_oracle_attempts"] == 840
    assert phase["costs"]["surrogate_oracle_attempts"] == 420
    assert all(row["source_provenance"]["oracle_attempt_journal"]["valid"] for row in phase["jobs"])
    with pytest.raises(FileNotFoundError):
        study_costs.audit_collection_costs(tmp_path)
    (directories[-1] / "decisions.jsonl").write_text("tampered original receipt evidence")
    with pytest.raises(CollectionProvenanceError, match="changed"):
        study_costs.audit_collection_costs(tmp_path, benchmark="made")


def test_acceptance_is_phase_only_and_unknown_or_changed_costs_block(report, manifest, made_rows, tmp_path):
    summary = report.summarize_made_phase(made_rows, manifest)
    costs = cost_fixture()
    attempts = {"all_attempt_costs_accounted": True, "ledger_events_fingerprint": "synthetic"}
    artifacts = []
    for name in sorted(report.REPORT_ARTIFACTS):
        path = tmp_path / name
        write(path, {"unit_test_fixture_not_results": True})
        artifacts.append({"path": str(path), "sha256": file_sha256(path)})
    evidence_hash = next(item["sha256"] for item in artifacts if Path(item["path"]).name == "verified_made_final_evidence.json")
    def acceptance():
        return report.make_phase_acceptance(manifest, summary, es_conditions(), attempts, costs,
            evidence_hash=evidence_hash, artifacts=artifacts, primary=[])
    value = acceptance()
    assert value["made_phase_complete"] and value["scope"] == "full_current_two_model_MADE_phase"
    assert value["completed_models"] == ["qwen35_4b", "qwen35_9b"]
    assert value["global_study_complete"] is value["all_final_experiments_complete"] is False
    assert value["latest_3_to_4_model_extension_complete"] is False
    assert value["full_study_expected_counts_unchanged"] == FULL_COUNTS
    attempts["all_attempt_costs_accounted"] = False
    with pytest.raises(RuntimeError, match="unknown"):
        acceptance()
    attempts["all_attempt_costs_accounted"] = True
    costs["shared_collection"]["completed_jobs"] = 419
    with pytest.raises(RuntimeError, match="420"):
        acceptance()
    costs["shared_collection"]["completed_jobs"] = 420
    Path(artifacts[0]["path"]).write_text("changed")
    with pytest.raises(RuntimeError, match="artifact changed"):
        acceptance()


def test_failed_cli_audit_replaces_a_stale_success_gate(report, tmp_path, monkeypatch):
    write(tmp_path / "made_phase_acceptance.json", {"made_phase_complete": True, "stale_fixture": True})
    def fail(*args): raise RuntimeError("synthetic missing MADE receipt")
    monkeypatch.setattr(report, "report_made_phase", fail)
    monkeypatch.setattr(sys, "argv", ["report_made_phase", "--project", str(tmp_path), "--output", str(tmp_path)])
    with pytest.raises(RuntimeError, match="synthetic missing"):
        report.main()
    receipt = json.loads((tmp_path / "made_phase_acceptance.json").read_text())
    assert receipt["status"] == "audit_failed" and not receipt["made_phase_complete"]
    assert receipt["global_study_complete"] is False


def test_phase_end_to_end_costs_keep_exact_budgets_and_failed_extra_calls(tmp_path, monkeypatch, made_rows):
    stage_plan_fixture(tmp_path)
    collection = {**cost_fixture()["shared_collection"], "evidence_files": []}
    def collection_audit(root, *, benchmark=None):
        assert benchmark == "made"
        return collection
    monkeypatch.setattr(study_costs, "audit_collection_costs", collection_audit)
    attempts = {"known_prior_attempt_costs": {"candidate_oracle_attempts": 3}, "all_attempt_costs_accounted": True}
    value = study_costs.build_study_cost_report(tmp_path, made_rows, es_conditions(), attempts, benchmark="made")
    assert value["experimental_cost_accounting_complete"]
    assert value["matched_final_budget"]["costs"]["candidate_oracle_attempts"] == 90000
    assert value["experimental_end_to_end"]["known_physical_call_counts"]["candidate_oracle_attempts"] == 90000 + 21000 + 176000 + 3
    assert value["global_study_complete"] is False
    attempts["all_attempt_costs_accounted"] = False
    unknown = study_costs.build_study_cost_report(tmp_path, made_rows, es_conditions(), attempts, benchmark="made")
    assert not unknown["experimental_cost_accounting_complete"]


def test_complete_report_checks_all_1800_receipts_with_cg_deferred_and_raw_tamper_rejected(report, manifest, tmp_path, monkeypatch):
    """Synthetic scientific verifier only; filesystem/ledger/calibration gates run."""
    project = tmp_path / "fixture_project"
    output = tmp_path / "fixture_reports"
    for name in ("benchmark_tasks", "main_protocol", "made_splits", "model_manifest"):
        data = json.loads((PROJECT / "configs" / (name + ".json")).read_text())
        if name == "main_protocol":
            data["statistics"]["bootstrap_replicates"] = 2  # This temporary CPU fixture only.
            # This legacy receipt fixture intentionally has no typed evidence.
            # The production enabled protocol requires the separate seven-head audit.
            data.pop("failure_control", None)
        write(project / "configs" / (name + ".json"), data)
    final = project / "experiments/final_evaluation"
    write(final / "manifest.json", manifest)
    database = sqlite3.connect(final / "ledger.sqlite3")
    database.execute("CREATE TABLE jobs (job_id TEXT PRIMARY KEY,state TEXT,attempt_number INTEGER,result_path TEXT,result_sha256 TEXT)")
    database.execute("CREATE TABLE events (event_id INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT,event TEXT,details TEXT)")
    first_raw = None
    for job in manifest["jobs"]:
        if job["benchmark"] != "made":
            database.execute("INSERT INTO jobs VALUES (?, 'pending', 0, NULL, NULL)", (job["job_id"],))
            continue
        directory = final / "synthetic_receipts" / job["job_id"]
        raw = directory / "decisions.jsonl"
        raw.parent.mkdir(parents=True)
        raw.write_text(json.dumps({"unit_test_fixture_not_physics": True, "episode_index": 0, "disposition": "executed",
            "label_future_failure": 1, "predicted_failure_probability": .2,
            "graph_status": "unavailable" if "graph_risk" in job["method"] else None}) + "\n")
        if first_raw is None: first_raw = raw
        selected = None
        if job["method"] in {"esopt", "esopt_graph_risk"}:
            selected = {"training_run_fingerprint": fingerprint([job["model_key"], job["method"], job["seed"]]),
                "training_directory": "synthetic-only-no-training-run", "training_costs": {"candidate_oracle_attempts": 8800, "dft_episode_attempts": 0}}
        episode = {**{key: job[key] for key in ("benchmark", "model_key", "method", "task_id", "seed")},
                   "episode_id": "0", "environment_seed": job["seed"], "complete": True, "status": "succeeded",
                   "costs": {"candidate_oracle_attempts": 50, "dft_episode_attempts": 0}, "metrics": {"AUDC": .2, "mSUN": 1.0}}
        path = directory / "result.json"
        write(path, {"unit_test_fixture_not_scientific_evidence": True, "execution_profile": {"selected_es": selected},
                     "episodes": [episode], "artifacts": [{"path": str(raw), "sha256": file_sha256(raw)}]})
        database.execute("INSERT INTO jobs VALUES (?, 'succeeded', 1, ?, ?)", (job["job_id"], str(path), file_sha256(path)))
        database.execute("INSERT INTO events(job_id,event,details) VALUES (?, 'claimed', ?)", (job["job_id"], json.dumps({"attempt_id": "fixture-" + job["job_id"]})))
    database.commit()
    database.close()
    seen = []
    expected_workers = declared_mace_workers(json.loads((project / "configs/main_protocol.json").read_text()))
    def scientific_verifier(path, job, manifest_fingerprint, *, tasks, expected_mace_num_workers):
        # Never execute the real oracle. Tests of its reconstruction contract live
        # in the final_evaluation suite; this test verifies correct gate composition.
        assert json.loads(path.read_text())["unit_test_fixture_not_scientific_evidence"]
        assert manifest_fingerprint == manifest["manifest_fingerprint"]
        assert job["benchmark"] == "made"
        assert expected_mace_num_workers == expected_workers
        seen.append(job["job_id"])
    monkeypatch.setattr(report, "verify_final_envelope", scientific_verifier)
    monkeypatch.setattr(report, "build_study_cost_report", lambda *args, benchmark: cost_fixture())
    receipt = report.report_made_phase(project, output)
    assert len(seen) == len(set(seen)) == 1800
    assert receipt["made_phase_complete"] and not receipt["global_study_complete"]
    calibration = json.loads((output / "made_calibration.json").read_text())
    assert len(calibration) == 12
    assert sum(row["coverage_counts"]["executed"] for row in calibration) == 1800
    typed = json.loads((output / "made_failure_type_calibration.json").read_text())
    assert typed["job_status_counts"] == {"not_recorded": 1800}
    assert "made_failure_type_calibration.json" in {Path(a["path"]).name for a in receipt["artifacts"]}
    assert len(json.loads((output / "verified_made_final_evidence.json").read_text())) == 1800
    first_raw.write_text("tampered decision evidence\n")
    monkeypatch.setattr(sys, "argv", ["report_made_phase", "--project", str(project), "--output", str(output)])
    with pytest.raises(RuntimeError, match="missing or changed"):
        report.main()
    failed = json.loads((output / "made_phase_acceptance.json").read_text())
    assert not failed["made_phase_complete"] and failed["status"] == "audit_failed"
