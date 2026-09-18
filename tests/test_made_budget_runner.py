"""Dynamic evaluation contracts with tiny CPU policy and real synthetic RPC gates.

No pretrained policy, GPU, or scientific oracle is used. Empty reservations in
bounded tests represent work owned elsewhere, never completed scientific data.
"""
from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import socket

import pytest

from matdiscovery import made_budget_runner as mod
from matdiscovery.accounting import AccountingError, file_sha256, fingerprint, write_json_atomic
from test_made_budget_sweep import budget_prior_fixture, budget_trajectory, frozen_prior_modules
from test_made_all30_extension import make_prior_fixture
from test_fast_core_accounting import fast_core
from test_training_jobs import project


def snapshot(root):
    return {str(p): (file_sha256(p), p.stat().st_mtime_ns) for p in Path(root).rglob("*") if p.is_file()}


@pytest.fixture
def all30_prior_fixture(fast_core, tmp_path, monkeypatch):
    # Match the real CoreFinalRunner execution schema before any synthetic
    # parent result/profile is sealed; never rewrite a completed ancestor.
    import test_core_final_integration as integration
    original = integration.runner_fixture
    def provider(*args, **kwargs):
        result = original(*args, **kwargs)
        result.execution["runtime"] = deepcopy(result.runtime)
        return result
    monkeypatch.setattr(integration, "runner_fixture", provider)
    return make_prior_fixture(fast_core, tmp_path, monkeypatch, complete_prior=True)


@pytest.fixture
def setup_budget(budget_prior_fixture, monkeypatch):
    f = budget_prior_fixture
    monkeypatch.setattr(mod, "CoreFinalRunner", lambda *a, **kw: f["runner"])
    f["new_calls"] = []
    return f


def actor(f, actor_id="actor_00", *, fail=False, partial=False, gpu_uuid=None, before_rollout=None):
    class UnitRollout:
        def __init__(self, project, policy, *, settings, risk_model, attributor):
            self.project = project
            assert settings.failure_aware_control and settings.max_generation_retries == 2
        def run(self, job, output, *, collection):
            assert not collection and job["budget"] in (30, 50)
            assert job["group_id"] == job["task_id"] and job["selected_generation"] == (2 if job["method"] == "esopt_graph_risk" else 0)
            assert any(all(job.get(k) == v for k, v in expected.items()) for expected in f["registration"]["new_jobs"])
            if before_rollout is not None:
                before_rollout(job)
            f["new_calls"].append(job["job_id"])
            if partial:
                output.mkdir(); (output / "unknown-physical-attempt.txt").write_text("unit unresolved call")
                raise RuntimeError("unit unknown physical outcome")
            budget_trajectory(self.project, job, output, discovered=job["method"] == "esopt_graph_risk")
            if fail: raise RuntimeError("unit closed physical transcript before envelope")
    return mod.MadeBudgetActor(f["workspace"], actor_id=actor_id, rollout_factory=UnitRollout,
        gpu_provider=lambda: {"index": 0, "uuid": gpu_uuid or "unit-GPU-" + actor_id, "name": "CPU fixture only"})


def reserve_except(registration, selected):
    root = Path(registration["workspace"]) / "jobs"
    root.mkdir(exist_ok=True)
    for job in registration["new_jobs"]:
        if job["job_id"] not in selected:
            (root / job["job_id"]).mkdir()


def make_invocation(registration, actor_id, execution, proof):
    spec = mod.actor_spec(registration, actor_id)
    directory = Path(registration["workspace"]) / "actors" / actor_id
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(directory / "launch_validation.json", proof)
    gpu = {"index": 0, "uuid": "unit-GPU-" + actor_id, "name": "CPU fixture only"}
    mod.create_once(directory / "invocation.json", {"schema": "made_budget_actor_invocation_v1", "scope": mod.SCOPE,
        "sweep_fingerprint": registration["fingerprint"], "actor": spec, "gpu": gpu,
        "hostname": socket.gethostname(), "pid": os.getpid(), "registered_runtime": execution["runtime"],
        "parent_execution": execution, "launch_validation": mod.artifact(directory / "launch_validation.json"),
        "automatic_physical_replay": False})
    return spec, mod.artifact(directory / "invocation.json"), gpu


def _claim_process(registration, actor_id, execution, proof, queue):
    """Filesystem administration only; fork receives no scientific authorization."""
    try:
        spec, invocation, gpu = make_invocation(registration, actor_id, execution, proof)
        result = []
        while True:
            job, claim = mod.claim_next_job(registration, spec, invocation, gpu, spec["initial_method"])
            if job is None: break
            if job["budget"] == 50:
                assert all((Path(registration["workspace"]) / "jobs" / j["job_id"]).exists()
                           for j in registration["new_jobs"] if j["budget"] == 30)
            result.append(job["job_id"])
        queue.put((actor_id, result, None))
    except BaseException as error:
        queue.put((actor_id, [], repr(error)))
        raise


def test_eight_processes_claim_all120_once_and_B30_reservations_precede_B50(setup_budget):
    f = setup_budget; reg = f["registration"]
    proof = mod.validate_launch(reg)
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    children = [context.Process(target=_claim_process, args=(reg, f"actor_{index:02d}", f["runner"].execution, proof, queue)) for index in range(8)]
    for child in children: child.start()
    messages = [queue.get(timeout=30) for _ in children]
    for child in children:
        child.join(timeout=30)
        assert child.exitcode == 0
    assert not any(message[2] for message in messages)
    ids = [job for _, jobs, _ in messages for job in jobs]
    assert len(ids) == len(set(ids)) == 120 and set(ids) == {j["job_id"] for j in reg["new_jobs"]}
    state = mod.queue_state(reg)
    assert state == {"30": {"claimed_not_complete": 60}, "50": {"claimed_not_complete": 60}}
    assert mod.global_receipt(reg) is None and not f["new_calls"]


def test_B30_priority_over_preferred_B50_and_unknown_reservation_never_reclaimed(setup_budget):
    f = setup_budget; reg = f["registration"]
    spec, invocation, gpu = make_invocation(reg, "actor_02", f["runner"].execution, mod.validate_launch(reg))
    first, _ = mod.claim_next_job(reg, spec, invocation, gpu, "esopt_graph_risk")
    assert first["budget"] == 30 and first["method"] == "esopt_graph_risk"
    root = Path(reg["workspace"]) / "jobs"
    blocked = next(j for j in reg["new_jobs"] if j["budget"] == 30 and not (root / j["job_id"]).exists())
    (root / blocked["job_id"]).mkdir()
    ids = []
    while True:
        job, _ = mod.claim_next_job(reg, spec, invocation, gpu, "esopt_graph_risk")
        if job is None: break
        ids.append(job["job_id"])
    assert blocked["job_id"] not in ids and len(ids) == 118
    assert not (root / blocked["job_id"] / "claim.json").exists()
    assert mod.queue_state(reg)["30"]["unknown_reserved"] == 1


def test_resident_actor_loads_once_and_never_repeats_parent_deep_audit_per_job(setup_budget, monkeypatch):
    from matdiscovery import core_protocol, evaluation_extension
    f = setup_budget; reg = f["registration"]
    selected = [next(j for j in reg["new_jobs"] if j["budget"] == b and j["method"] == m)
                for b, m in [(30, "baseline"), (30, "esopt_graph_risk"), (50, "esopt_graph_risk")]]
    reserve_except(reg, {j["job_id"] for j in selected})
    worker = actor(f)
    original_launch, original_load, original_profile = mod.validate_launch, f["runner"].load_policy, f["runner"].profile
    launches, loads, profiles = [], [], []
    def launch(value):
        launches.append(True); proof = original_launch(value)
        monkeypatch.setattr(core_protocol, "read_core", lambda *a, **kw: pytest.fail("Repeated parent deep validation"))
        monkeypatch.setattr(evaluation_extension, "read_extension", lambda *a, **kw: pytest.fail("Repeated prior deep validation"))
        return proof
    def load():
        loads.append(True)
        assert (worker.directory / "invocation.json").is_file()
        return original_load()
    def profile(*args):
        profiles.append(args[1]["method"])
        return original_profile(*args)
    monkeypatch.setattr(mod, "validate_launch", launch)
    monkeypatch.setattr(f["runner"], "load_policy", load)
    monkeypatch.setattr(f["runner"], "profile", profile)
    result = worker.run(wait_for_global=False)
    assert not result["complete"] and result["all_jobs_claimed_is_not_completion"]
    assert len(launches) == len(loads) == 1 and profiles == ["baseline", "esopt_graph_risk"]
    assert f["new_calls"] == [j["job_id"] for j in selected]
    for job in selected:
        receipt = mod.read(f["workspace"] / "jobs" / job["job_id"] / "receipt.json")
        assert receipt["complete"] and receipt["expected_counts"]["candidate_oracle_attempts"] == job["budget"]


def test_all120_new_and_original60_publish_separate_budgets_without_retraining(setup_budget, monkeypatch):
    from test_fast_es_budget import b10_trajectory
    f = setup_budget; reg = f["registration"]
    modules = frozen_prior_modules(f, monkeypatch); old_calls = []
    class B10UnitRollout:
        def __init__(self, project, policy, **kwargs): self.project = project
        def run(self, job, output, *, collection):
            assert not collection and job["budget"] == 10
            old_calls.append(job["job_id"]); b10_trajectory(self.project, job, output, discovered=False)
    modules.runner.MadeAll30Runner(f["all30_workspace"], worker_id="primary", rollout_factory=B10UnitRollout).run()
    modules.runner.aggregate_all30(f["all30_workspace"])
    before = {name: snapshot(f[name]) for name in ("prior_workspace", "all30_workspace")}
    parent_before = snapshot(f["core"]["workspace"])
    early_B30 = {}
    def publish_first_budget(job):
        if job["budget"] == 50 and not early_B30:
            pending = mod.aggregate_sweep(f["workspace"])
            assert not pending["complete"] and set(pending["completed_budget_reports"]) == {"30"}
            path = Path(pending["completed_budget_reports"]["30"]["path"])
            early_B30.update(path=path, content=path.read_bytes(), mtime_ns=path.stat().st_mtime_ns)
    result = actor(f, before_rollout=publish_first_budget).run(wait_for_global=False)
    assert not result["complete"] and len(f["new_calls"]) == 120
    assert len(set(f["new_calls"])) == 120
    report = mod.aggregate_sweep(f["workspace"])
    assert report["complete"] and report["sweep_complete"] and not report["global_study_complete"]
    assert report["expected_new_counts"]["candidate_oracle_attempts"] == 4800
    assert report["expected_cumulative_counts"]["candidate_oracle_attempts"] == 5400
    assert report["new_training_or_ES_cost"] == 0 and not report["cross_budget_pooled_metric_claimed"]
    assert early_B30 and early_B30["path"].read_bytes() == early_B30["content"]
    assert early_B30["path"].stat().st_mtime_ns == early_B30["mtime_ns"]
    assert set(report["budget_reports"]) == {"10", "30", "50"}
    for budget, count in [(10, 600), (30, 1800), (50, 3000)]:
        single = mod.read(report["budget_reports"][str(budget)]["path"])
        assert single["expected_counts"]["candidate_oracle_attempts"] == count
        assert single["paired_results"]["paired_systems"] == 30
        assert sum(cost["candidate_oracle_attempts"] for cost in single["costs_by_method"].values()) == count
    assert len(old_calls) == 50 and len(f["calls"]) == 4 and len(f["old_calls"]) == 6
    assert all(snapshot(f[name]) == value for name, value in before.items())
    assert snapshot(f["core"]["workspace"]) == parent_before
    assert len(list((f["workspace"] / "actors").glob("*/invocation.json"))) == 1  # nine optional actors never started
    f["runner"].policy_factory = lambda _: pytest.fail("accepted sweep must not reload a policy")
    immutable = snapshot(f["workspace"] / "experiments")
    assert actor(f, "actor_09").run()["complete"]
    assert mod.aggregate_sweep(f["workspace"])["complete"]
    assert snapshot(f["workspace"] / "experiments") == immutable


def test_B30_complete_report_does_not_claim_B50_or_B10_complete(setup_budget):
    f = setup_budget; reg = f["registration"]
    reserve_except(reg, {j["job_id"] for j in reg["new_jobs"] if j["budget"] == 30})
    actor(f).run(wait_for_global=False)
    result = mod.aggregate_sweep(f["workspace"])
    assert not result["complete"] and len(result["missing_new_jobs"]) == 60
    assert set(result["completed_budget_reports"]) == {"30"}
    assert not (f["workspace"] / "experiments/made_budget_stage_receipts/final.json").exists()


@pytest.mark.parametrize("partial", [False, True])
def test_failed_or_unknown_claim_stays_reserved_and_actor_cannot_restart(setup_budget, partial):
    f = setup_budget; reg = f["registration"]
    job = reg["new_jobs"][0]; reserve_except(reg, {job["job_id"]})
    worker = actor(f, fail=not partial, partial=partial)
    with pytest.raises(RuntimeError, match="unit"): worker.run(wait_for_global=False)
    directory = f["workspace"] / "jobs" / job["job_id"]
    assert (directory / "failure.json").is_file() and not (directory / "receipt.json").exists()
    with pytest.raises(ValueError, match="invocation already exists"): actor(f).run(wait_for_global=False)
    other = actor(f, "actor_01").run(wait_for_global=False)
    assert not other["complete"] and len(f["new_calls"]) == 1
    result = mod.aggregate_sweep(f["workspace"])
    assert not result["complete"] and len(result["missing_new_jobs"]) == 120


def test_complete_orphan_is_reconciled_under_free_lock_without_physical_replay(setup_budget, monkeypatch):
    f = setup_budget; reg = f["registration"]
    job = reg["new_jobs"][0]; reserve_except(reg, {job["job_id"]})
    finish = mod.finish_job
    monkeypatch.setattr(mod, "finish_job", lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt): actor(f).run(wait_for_global=False)
    directory = f["workspace"] / "jobs" / job["job_id"]
    assert len(list(directory.glob("*/result.json"))) == 1 and not (directory / "receipt.json").exists()
    monkeypatch.setattr(mod, "finish_job", finish)
    with mod.locked(directory / ".execution.lock"):
        assert not mod.aggregate_sweep(f["workspace"])["complete"]
        assert not (directory / "receipt.json").exists()
    result = mod.aggregate_sweep(f["workspace"])
    assert not result["complete"] and (directory / "receipt.json").is_file()
    assert len(f["new_calls"]) == 1


def test_actor_gpu_locks_and_invocation_are_before_policy_load(setup_budget):
    f = setup_budget
    worker = actor(f, gpu_uuid="unit-shared-GPU")
    with mod.locked(worker.directory / ".actor.lock"):
        with pytest.raises(BlockingIOError): worker.run(wait_for_global=False)
    with mod.locked(f["workspace"] / "locks" / ("gpu-" + fingerprint("unit-shared-GPU") + ".lock")):
        with pytest.raises(BlockingIOError): actor(f, "actor_01", gpu_uuid="unit-shared-GPU").run(wait_for_global=False)
    assert not list((f["workspace"] / "actors").glob("*/invocation.json"))
    assert not f["new_calls"]


@pytest.mark.parametrize("field", ["source", "selected_generation", "budget", "raw", "claim"])
def test_tampered_result_claim_or_raw_budget_is_rejected(setup_budget, field):
    f = setup_budget; reg = f["registration"]
    job = next(j for j in reg["new_jobs"] if j["budget"] == 30 and j["method"] == "esopt_graph_risk")
    reserve_except(reg, {job["job_id"]}); worker = actor(f); worker.run(wait_for_global=False)
    directory = f["workspace"] / "jobs" / job["job_id"]
    path = next(directory.glob("*/result.json"))
    if field == "raw": (path.parent / "episodes.json").write_text("[]")
    elif field == "claim":
        claim = mod.read(directory / "claim.json"); claim["attempt_number"] = 2
        write_json_atomic(directory / "claim.json", claim)
    else:
        value = mod.read(path)
        value["execution_job"][field] = "wrong_source" if field == "source" else 99
        write_json_atomic(path, value)
    with pytest.raises(ValueError): mod.verify_budget_envelope(path, job, reg, tasks=worker.tasks, core=worker.core)


def test_B30_still_rejected_without_new_scope_and_source_drift_stops_execution(setup_budget):
    from matdiscovery.accounting import expected_episodes
    f = setup_budget; job = f["registration"]["new_jobs"][0]
    with pytest.raises(AccountingError): expected_episodes({"jobs": [job]}, expected_made_budget=30, core_protocol=f["core"])
    worker = actor(f)
    path = f["all30_workspace"] / "src/matdiscovery/native_attribution.py"
    path.write_text(path.read_text() + "\n# unit source tamper\n")
    with pytest.raises(ValueError, match="frozen runtime/control|source artifact changed"): worker.run(wait_for_global=False)
    assert not f["new_calls"]
