"""Independent all30 workers: tiny CPU policy, real synthetic RPC/journal gate.

These contract fixtures are not MADE experiments and never construct an oracle.
Only legacy source-layout/provider setup is isolated; physical evidence verifiers
remain real. New all30 runtime source validation is exercised without a stub.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from matdiscovery import made_all30_runner as mod
from matdiscovery.accounting import AccountingError, file_sha256, fingerprint, write_json_atomic
from matdiscovery.made_all30_extension import build_all30, planned_jobs
from test_made_all30_extension import all30_prior_fixture
from test_fast_core_accounting import fast_core
from test_training_jobs import project


def snapshot(root):
    return {str(p): (file_sha256(p), p.stat().st_mtime_ns) for p in Path(root).rglob("*") if p.is_file()}


def balanced_shards(jobs, workspace):
    tasks = sorted({j["task_id"] for j in jobs})
    assert len(tasks) == 25
    return [{"worker_id": f"worker_{index}", "execution_site": f"logical_gpu_slot_{index}",
        "output_workspace": str(workspace / "workers" / f"worker_{index}"),
        "job_ids": [j["job_id"] for j in jobs if j["task_id"] in tasks[index * 5:(index + 1) * 5]]}
        for index in range(5)]


@pytest.fixture
def setup_all30(all30_prior_fixture, monkeypatch):
    f = all30_prior_fixture
    jobs = planned_jobs(f["core"], f["prior"])
    registration = build_all30(f["core"], f["prior"], f["workspace"], execution_shards=balanced_shards(jobs, f["workspace"]))
    monkeypatch.setattr(mod, "CoreFinalRunner", lambda *args, **kwargs: f["runner"])
    f.update(registration=registration, new_calls=[],
        parent_before=snapshot(f["core"]["workspace"]), prior_before=snapshot(f["prior_workspace"]))
    return f


def runner(f, worker_id="worker_0", *, fail_after_closed=False):
    from test_fast_es_budget import b10_trajectory
    class UnitRollout:
        def __init__(self, project, policy, *, settings, risk_model, attributor):
            self.project = project
            assert settings.failure_aware_control is True and settings.max_generation_retries == 2
        def run(self, job, output, *, collection):
            assert collection is False and job["budget"] == 10
            assert job["group_id"] == job["task_id"]
            assert any(all(job.get(k) == v for k, v in registered.items()) for registered in f["registration"]["new_jobs"])
            f["new_calls"].append(job["job_id"])
            b10_trajectory(self.project, job, output, discovered=job["method"] == "esopt_graph_risk")
            if fail_after_closed:
                raise RuntimeError("synthetic closed all30 failure")
    return mod.MadeAll30Runner(f["workspace"], worker_id=worker_id, rollout_factory=UnitRollout)


def test_five_workers_accept_only_50_new_and_aggregate_60_without_charging_training(setup_all30):
    f = setup_all30
    workers = [runner(f, f"worker_{i}") for i in range(5)]
    assert all([j["method"] for j in worker.manifest["jobs"]] == ["baseline"] * 5 + ["esopt_graph_risk"] * 5 for worker in workers)
    one = workers[0].run()
    assert one["complete"] and one["worker_complete"] and not one["all30_complete"]
    pending = mod.aggregate_all30(f["workspace"])
    assert not pending["complete"] and len(pending["missing_workers"]) == 4
    for worker in workers[1:]: worker.run()
    result = mod.aggregate_all30(f["workspace"])
    assert result["complete"] and result["all30_complete"] and not result["global_study_complete"]
    assert result["expected_new_counts"]["jobs"] == 50 and result["expected_cumulative_counts"]["candidate_oracle_attempts"] == 600
    assert result["original_five_registered_before_expansion"]["paired_systems"] == 5
    assert result["new_twenty_five_after_expansion"]["paired_systems"] == 25
    assert result["cumulative_thirty_descriptive"]["paired_systems"] == 30
    assert result["cumulative_thirty_descriptive"]["inferential_p_value"] is None
    for method in mod.METHODS:
        assert result["costs"]["new_fifty_by_method"][method]["candidate_oracle_attempts"] == 250
        assert result["costs"]["original_ten_by_method"][method]["candidate_oracle_attempts"] == 50
        assert result["costs"]["cumulative_sixty_by_method"][method]["candidate_oracle_attempts"] == 300
    assert result["costs"]["additional_training_or_ES_cost"] == 0
    assert len(f["new_calls"]) == len(set(f["new_calls"])) == 50
    assert len(f["calls"]) == 4 and len(f["old_calls"]) == 6
    assert snapshot(f["core"]["workspace"]) == f["parent_before"]
    assert snapshot(f["prior_workspace"]) == f["prior_before"]


def test_profile_once_per_method_and_no_parent_deep_recursion_per_job(setup_all30, monkeypatch):
    from matdiscovery import core_protocol
    f = setup_all30
    worker = runner(f)
    original_launch, profile = mod.validate_launch, f["runner"].profile
    launches, profiles = [], []
    def launch(registration):
        launches.append(registration["fingerprint"])
        proof = original_launch(registration)
        monkeypatch.setattr(core_protocol, "read_core", lambda *a, **k: pytest.fail("per-job parent deep verification"))
        return proof
    def load_profile(*args):
        profiles.append(args[1]["method"])
        return profile(*args)
    monkeypatch.setattr(mod, "validate_launch", launch)
    monkeypatch.setattr(f["runner"], "profile", load_profile)
    worker.run()
    assert len(launches) == 1 and profiles == ["baseline", "esopt_graph_risk"]
    assert len(f["new_calls"]) == 10


def test_complete_reuse_does_not_load_policy_or_rewrite_worker_receipts(setup_all30):
    f = setup_all30
    worker = runner(f)
    worker.run()
    paths = [worker.output / p for p in ("ledger.json", "completion.json", "stage_receipt.json")]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    f["runner"].policy_factory = lambda _: pytest.fail("completed worker must not load another policy")
    worker.run()
    assert len(f["new_calls"]) == 10
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items())


def test_unknown_partial_and_closed_failure_never_replay(setup_all30):
    f = setup_all30
    worker = runner(f, fail_after_closed=True)
    with pytest.raises(RuntimeError, match="synthetic closed"): worker.run()
    assert len(f["new_calls"]) == 1
    f["runner"].policy_factory = lambda _: pytest.fail("failure check must precede model load")
    with pytest.raises(ValueError, match="reconciliation"): runner(f).run()
    assert len(f["new_calls"]) == 1


def test_orphaned_complete_envelope_reconciles_without_second_physical_attempt(setup_all30, monkeypatch):
    f = setup_all30
    worker = runner(f)
    finish = mod.MadeAll30Ledger.finish
    monkeypatch.setattr(mod.MadeAll30Ledger, "finish", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt): worker.run()
    assert len(f["new_calls"]) == 1
    monkeypatch.setattr(mod.MadeAll30Ledger, "finish", finish)
    runner(f).run()
    assert len(f["new_calls"]) == len(set(f["new_calls"])) == 10
    ledger = json.loads((worker.output / "ledger.json").read_text())
    assert any(e["event"] == "reconciled_complete" for e in ledger["events"])


def test_unknown_running_directory_refused_before_policy_load(setup_all30):
    f = setup_all30
    worker = runner(f)
    ledger = mod.MadeAll30Ledger(worker.output / "ledger.json", worker.manifest, registration=worker.registration)
    job = worker.manifest["jobs"][0]
    attempt = ledger.claim(job["job_id"])
    directory = worker.output / "jobs" / job["job_id"] / attempt
    directory.mkdir(parents=True)
    (directory / "unknown.txt").write_text("unit unresolved call")
    f["runner"].policy_factory = lambda _: pytest.fail("unresolved work must block policy load")
    with pytest.raises(ValueError, match="partial.*reconciliation"): worker.run()
    assert not f["new_calls"] and (directory / "unknown.txt").is_file()


def test_worker_lock_and_persisted_invocation_precede_model_load(setup_all30, monkeypatch):
    f = setup_all30
    worker, other = runner(f), runner(f, "worker_1")
    with mod.exclusive_lock(worker.output / ".runner.lock"):
        with pytest.raises(ValueError, match="Another worker"): worker.run()
        with mod.exclusive_lock(other.output / ".runner.lock"): pass
    loader = f["runner"].load_policy
    def load():
        assert len(list((worker.output / "launch_checks").glob("invocation-*.json"))) == 1
        with pytest.raises(ValueError, match="Another worker"):
            with mod.exclusive_lock(worker.output / ".runner.lock"): pass
        return loader()
    monkeypatch.setattr(f["runner"], "load_policy", load)
    worker.run()


def test_worker_manifest_cannot_add_old_job_or_other_worker(setup_all30):
    f = setup_all30
    worker = runner(f)
    bad = deepcopy(worker.manifest)
    bad["jobs"][0] = deepcopy(runner(f, "worker_1").manifest["jobs"][0])
    bad["fingerprint"] = fingerprint({k: v for k, v in bad.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="ownership"): mod.validate_worker_manifest(bad, worker.registration)
    with pytest.raises(ValueError, match="sealed ownership"): runner(f, "unknown")
    with pytest.raises(ValueError, match="Output override"):
        mod.MadeAll30Runner(f["workspace"], worker_id="worker_0", output=worker.output / "other")


def test_source_drift_and_changed_profile_refuse_before_physics(setup_all30):
    f = setup_all30
    worker = runner(f)
    path = f["prior_workspace"] / "src/matdiscovery/rpc.py"
    path.write_text(path.read_text() + "\n# unit tampering\n")
    with pytest.raises(ValueError, match="frozen runtime/control"): worker.run()
    assert not f["new_calls"]


def test_loaded_profile_cannot_change_selected_weights(setup_all30, monkeypatch):
    f = setup_all30
    worker = runner(f)
    original = f["runner"].profile
    def changed(*args):
        profile, risk, graph = original(*args)
        return dict(profile, actual_model_state_hash="unit wrong weights"), risk, graph
    monkeypatch.setattr(f["runner"], "profile", changed)
    with pytest.raises(ValueError, match="profile differs"): worker.run()
    assert not f["new_calls"]


@pytest.mark.parametrize("field", ["source", "selected_generation"])
def test_resealed_source_or_generation_metadata_is_rejected(setup_all30, field):
    f = setup_all30
    worker = runner(f)
    worker.run()
    job = worker.manifest["jobs"][0]
    path = next((worker.output / "jobs" / job["job_id"]).glob("*/result.json"))
    value = json.loads(path.read_text())
    value["execution_job"][field] = "wrong" if field == "source" else 999
    write_json_atomic(path, value)
    with pytest.raises(ValueError, match="source changed|generation changed"):
        mod.verify_all30_envelope(path, job, worker.manifest, worker.registration, tasks=worker.tasks, core=worker.core)


def test_old_strict_gate_rejects_new_all30_job_without_explicit_admission(setup_all30):
    from matdiscovery.final_evaluation import verify_final_envelope
    f = setup_all30
    worker = runner(f)
    worker.run()
    job = worker.manifest["jobs"][0]
    path = next((worker.output / "jobs" / job["job_id"]).glob("*/result.json"))
    with pytest.raises(AccountingError, match="sealed FAST final matrix"):
        verify_final_envelope(path, job, worker.manifest["fingerprint"], tasks=worker.tasks,
            expected_profile=worker.registration["execution_profiles"][job["method"]],
            expected_made_budget=10, core_protocol=worker.core)
    raw = path.parent / "episodes.json"
    raw.write_text("[]")
    f["runner"].policy_factory = lambda _: pytest.fail("tamper must be caught without model")
    with pytest.raises(ValueError): worker.run()
