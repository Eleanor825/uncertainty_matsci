"""All30 admission/data-range tests; tiny CPU/scalar fixtures, never science."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from matdiscovery import made_all30_extension as mod
from matdiscovery.accounting import AccountingError, expected_episodes, file_sha256, fingerprint, write_json_atomic
from test_fast_core_accounting import fast_core, fast_manifest, zero_selected
from test_training_jobs import project


def make_prior_fixture(core, tmp_path, monkeypatch, *, complete_prior):
    """Build real-shaped parent+prior receipts with tiny G2 null-update fixtures."""
    from matdiscovery import evaluation_extension as prior_module
    from matdiscovery import evaluation_extension_runner as prior_runner
    from matdiscovery.esopt import tensor_state_hash
    from test_core_final_integration import runner_fixture
    from test_fast_es_budget import b10_trajectory
    root = Path(core["workspace"])
    monkeypatch.setattr(prior_module, "PARENT_CORE_FINGERPRINT", core["fingerprint"])
    monkeypatch.setattr(mod, "PARENT_CORE_FINGERPRINT", core["fingerprint"])
    write_json_atomic(root / "source_manifest.json", {"classification": "unit source only, not an experiment"})
    calls = []; provider = runner_fixture(root, tmp_path, calls)
    provider.output = root / "experiments/core_final"
    provider.core, provider.manifest = core, fast_manifest(core)
    def selected(policy, directory, **kwargs):
        base = tensor_state_hash(dict(policy.model.state_dict()))
        value = zero_selected(base); value["selected_generation"] = 2
        value["zero_update_proof"]["selected_generation"] = 2; value["artifacts"] = {}
        policy.mark_state("reload", generation=2)
        return value
    provider.selected_loader = selected
    class ParentRollout:
        def __init__(self, project, policy, **kwargs): self.project = project
        def run(self, job, output, *, collection):
            assert not collection
            calls.append(job["job_id"]); b10_trajectory(self.project, job, output, discovered=False)
    provider.rollout_factory = ParentRollout
    provider.run(); assert len(calls) == 4
    prior_workspace = tmp_path / "prior_five"
    prior = prior_module.build_extension(core, prior_workspace)
    monkeypatch.setattr(mod, "PRIOR_EXTENSION_FINGERPRINT", prior["fingerprint"])
    # Only the old fixture's missing source layout/provider is isolated. New
    # all30 source checks and all raw result/RPC/journal verifiers stay active.
    monkeypatch.setattr(prior_runner, "verify_execution_sources", lambda value: None)
    monkeypatch.setattr(prior_runner, "CoreFinalRunner", lambda *a, **kw: provider)
    old_calls = []
    class PriorRollout:
        def __init__(self, project, policy, **kwargs): self.project = project
        def run(self, job, output, *, collection):
            assert not collection
            old_calls.append(job["job_id"]); b10_trajectory(self.project, job, output, discovered=False)
    if complete_prior:
        prior_runner.EvaluationExtensionRunner(prior_workspace, rollout_factory=PriorRollout).run()
        assert len(old_calls) == 6
    else:
        manifest = prior_runner.build_shard_manifest(prior, "primary")
        output = Path(prior["execution_shards"][0]["output_workspace"])
        write_json_atomic(output / "manifest.json", manifest)
        prior_runner.EvaluationExtensionLedger(output / "ledger.json", manifest, extension=prior)
    source = Path(mod.__file__).parent
    for path in source.rglob("*.py"):
        relative = path.relative_to(source)
        if str(relative) in mod.NEW_MODULES: continue
        target = prior_workspace / "src/matdiscovery" / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    return {"core": core, "runner": provider, "calls": calls, "old_calls": old_calls,
        "prior_workspace": prior_workspace, "prior": prior, "workspace": tmp_path / "all30"}


@pytest.fixture
def all30_prior_fixture(fast_core, tmp_path, monkeypatch):
    return make_prior_fixture(fast_core, tmp_path, monkeypatch, complete_prior=True)


@pytest.fixture
def pending_prior_fixture(fast_core, tmp_path, monkeypatch):
    return make_prior_fixture(fast_core, tmp_path, monkeypatch, complete_prior=False)


@pytest.fixture
def registered(pending_prior_fixture):
    f = pending_prior_fixture
    f["registration"] = mod.build_all30(f["core"], f["prior"], f["workspace"])
    return f


def reseal(value):
    value["fingerprint"] = value["all30_fingerprint"] = fingerprint(
        {k: v for k, v in value.items() if k not in {"fingerprint", "all30_fingerprint"}})
    write_json_atomic(Path(value["workspace"]) / "configs/made_all30_extension.json", value)
    return value


def test_exact50_and_deferred_prior10_without_fake_result_hashes(registered):
    f = registered; reg = f["registration"]
    assert len(reg["new_jobs"]) == 50 and len(reg["new_tasks"]) == 25 and len(reg["official_tasks"]) == 30
    assert reg["expected_new_counts"]["candidate_oracle_attempts"] == 500
    assert reg["expected_cumulative_counts"]["candidate_oracle_attempts"] == 600
    assert {j["task_id"] for j in reg["new_jobs"]}.isdisjoint(mod.OLD_TASKS)
    assert [len(t["elements"]) for t in reg["new_tasks"]] == [3]*5+[4]*10+[5]*10
    assert all(j["budget"] == 10 and j["seed"] == 1 and j["environment_seeds"] == [1] for j in reg["new_jobs"])
    assert len({j["job_id"] for j in reg["new_jobs"]}) == 50
    assert reg["training_jobs"] == reg["es_jobs"] == []
    assert len(reg["imported_dependencies"]) == 10
    assert sum(d["result"] is None for d in reg["imported_dependencies"]) == 6
    assert f["old_calls"] == []
    before = (Path(reg["workspace"]) / "configs/made_all30_extension.json").read_bytes()
    assert mod.build_all30(f["core"], f["prior"], f["workspace"]) == reg
    assert (Path(reg["workspace"]) / "configs/made_all30_extension.json").read_bytes() == before


@pytest.mark.parametrize("change", ["budget", "seed", "task", "omit", "duplicate", "denominator", "checkpoint", "disclosure", "train"])
def test_rehashed_mutations_cannot_change_protocol(registered, change):
    bad = deepcopy(registered["registration"])
    if change == "budget": bad["new_jobs"][0]["budget"] = 50
    elif change == "seed": bad["new_jobs"][0]["seed"] = 2
    elif change == "task": bad["new_jobs"][0]["task_id"] = "Al-Au-Hf"
    elif change == "omit": bad["new_jobs"].pop()
    elif change == "duplicate": bad["new_jobs"][0] = deepcopy(bad["new_jobs"][1])
    elif change == "denominator": bad["expected_cumulative_counts"]["candidate_oracle_attempts"] = 500
    elif change == "checkpoint": bad["execution_profiles"]["esopt_graph_risk"]["actual_model_state_hash"] = "another-model"
    elif change == "disclosure": bad["registration"]["registered_after_partial_prior5_results_known"] = False
    else: bad["training_jobs"] = [{"new": "training"}]
    reseal(bad)
    with pytest.raises(ValueError): mod.validate_all30(bad)


def test_unvalidated_json_and_new_process_cannot_use_fast_budget_gate(registered, monkeypatch):
    reg = registered["registration"]; job = reg["new_jobs"][0]
    kwargs = {"expected_made_budget": 10, "core_protocol": registered["core"], "made_all30_extension": reg}
    with pytest.raises(AccountingError, match="validate_launch"):
        expected_episodes({"jobs": [job]}, **kwargs)
    fake = deepcopy(reg); fake["validated"] = True
    with pytest.raises(AccountingError): expected_episodes({"jobs": [job]}, **{**kwargs,"made_all30_extension":fake})
    proof = mod.validate_launch(reg); assert proof["complete"]
    assert len(expected_episodes({"jobs": [job]}, **kwargs)) == 1
    real_pid = mod.os.getpid()
    monkeypatch.setattr(mod.os, "getpid", lambda: real_pid+1)
    with pytest.raises(AccountingError, match="validate_launch"):
        expected_episodes({"jobs": [job]}, **kwargs)


def test_per_job_fast_admission_does_not_recurse_into_parent_verifiers(registered, monkeypatch):
    from matdiscovery import core_protocol, evaluation_extension
    reg = registered["registration"]; proof = mod.validate_launch(reg)
    def forbidden(*a, **kw): raise AssertionError("Deep ancestor audit repeated per job")
    monkeypatch.setattr(core_protocol, "read_core", forbidden)
    monkeypatch.setattr(evaluation_extension, "read_extension", forbidden)
    for job in reg["new_jobs"][:2]:
        assert len(expected_episodes({"jobs":[job]}, expected_made_budget=10,
            core_protocol=registered["core"], made_all30_extension=reg)) == 1
    assert mod.validate_launch(reg) == proof
    with pytest.raises(AccountingError):
        expected_episodes({"jobs":[reg["new_jobs"][0]]}, expected_made_budget=50, made_all30_extension=reg)
    wrong = deepcopy(reg["new_jobs"][0]); wrong["environment_seeds"]=[2]
    with pytest.raises(AccountingError):
        expected_episodes({"jobs":[wrong]}, expected_made_budget=10, made_all30_extension=reg)


def test_mutated_source_after_launch_is_not_hidden_by_process_capability(registered):
    reg=registered["registration"];mod.validate_launch(reg)
    path=Path(reg["parent_core"]["path"]);path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError, match="artifact changed"):
        mod.validate_job_scope(reg, reg["new_jobs"][0])


def test_pending_prior_results_block_final_aggregation_but_not_new_launch(registered):
    reg=registered["registration"];mod.validate_launch(reg)
    pending=mod.audit_imported_results(reg,require_complete=False)
    assert not pending["complete"] and len(pending["results"]) == 4 and len(pending["missing_jobs"]) == 6
    with pytest.raises(ValueError, match="original ten"):
        mod.audit_imported_results(reg)


def test_all_original_ten_results_are_verified_without_replay(all30_prior_fixture):
    f=all30_prior_fixture;reg=mod.build_all30(f["core"],f["prior"],f["workspace"])
    mod.validate_launch(reg)
    proof=mod.audit_imported_results(reg)
    assert proof["complete"] and len(proof["results"]) == 10 and proof["missing_jobs"] == []
    assert len(f["calls"]) == 4 and len(f["old_calls"]) == 6
    victim=Path(proof["results"][-1]["result"]["path"])
    victim.write_text(victim.read_text()+"\n")
    with pytest.raises(ValueError): mod.audit_imported_results(reg)


def test_fixed25_selection_rejects_test_train_overlap(registered):
    root=Path(registered["core"]["workspace"])
    path=root/"configs/made_splits.json";value=json.loads(path.read_text())
    value["splits"]["train"].append(value["splits"]["test"][-1])
    write_json_atomic(path,value)
    with pytest.raises(ValueError,match="artifact changed|train/development"):
        mod.validate_all30(registered["registration"])


def test_static_workers_cover_each_new_job_once(registered):
    f=registered;new=f["workspace"].parent/"sharded_all30";jobs=mod.planned_jobs(f["core"],f["prior"])
    tasks=[t["id"] for t in f["registration"]["new_tasks"]]
    shards=[{"worker_id":f"worker_{i}","execution_site":f"slot_{i}","output_workspace":str(new/"workers"/f"worker_{i}"),
        "job_ids":[j["job_id"] for j in jobs if j["task_id"] in tasks[i*5:(i+1)*5]]} for i in range(5)]
    reg=mod.build_all30(f["core"],f["prior"],new,execution_shards=shards)
    assert [len(mod.all30_jobs(reg,worker_id=f"worker_{i}")) for i in range(5)] == [10]*5
    bad=deepcopy(reg);bad["execution_shards"][1]["job_ids"][0]=bad["execution_shards"][0]["job_ids"][0];reseal(bad)
    with pytest.raises(ValueError):mod.validate_all30(bad)
