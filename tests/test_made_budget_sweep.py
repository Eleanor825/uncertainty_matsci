"""Independent-budget contract fixtures only; no actual model or material oracle."""
from copy import deepcopy
from contextlib import contextmanager
import json
import importlib
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from matdiscovery import made_budget_sweep as mod
from matdiscovery.accounting import AccountingError, expected_episodes, file_sha256, fingerprint, write_json_atomic
from test_made_all30_extension import all30_prior_fixture
from test_fast_core_accounting import fast_core
from test_training_jobs import project


@pytest.fixture
def budget_prior_fixture(all30_prior_fixture, monkeypatch):
    from matdiscovery import made_all30_extension as old, made_all30_runner as runner
    f = all30_prior_fixture
    package = Path(mod.__file__).parent
    for name in mod.NEW_MODULES:
        # This legacy fixture copies today's source into the old-five layout;
        # remove only the two not-yet-existing sweep modules from that toy copy.
        (f["prior_workspace"] / "src/matdiscovery" / name).unlink(missing_ok=True)
    # Frozen ancestor layout: the two sweep modules did not exist in B10.
    sources = []
    for path in package.rglob("*.py"):
        relative = path.relative_to(package)
        if str(relative) in mod.NEW_MODULES: continue
        target = f["workspace"] / "src/matdiscovery" / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target); sources.append(target)
    # Existing tiny ancestor/provider fixtures execute through the local module;
    # bind its five files too. Production only uses its own frozen namespace.
    sources += [package / name for name in (*old.SHARED_ADMISSION_MODULES, *old.NEW_MODULES)]
    prior = old.build_all30(f["core"], f["prior"], f["workspace"], source_files=sources)
    monkeypatch.setattr(mod, "PARENT_CORE_FINGERPRINT", f["core"]["fingerprint"])
    monkeypatch.setattr(mod, "PRIOR_ALL30_FINGERPRINT", prior["fingerprint"])
    monkeypatch.setattr(runner, "CoreFinalRunner", lambda *a, **kw: f["runner"])
    f.update(prior_all30=prior, all30_workspace=f["workspace"], workspace=f["workspace"].parent / "budget_sweep")
    f["registration"] = mod.build_sweep(f["core"], prior, f["workspace"])
    return f


def frozen_prior_modules(f, monkeypatch):
    """Actual frozen verifier namespace; only synthetic parent pins are isolated."""
    from matdiscovery import core_protocol
    modules = mod._b10_modules(f["prior_all30"])
    namespace = modules.protocol.__package__
    frozen_core = importlib.import_module(namespace + ".core_protocol")
    frozen_prior = importlib.import_module(namespace + ".evaluation_extension")
    monkeypatch.setattr(frozen_core, "read_core", core_protocol.read_core)
    monkeypatch.setattr(frozen_prior, "PARENT_CORE_FINGERPRINT", f["core"]["fingerprint"])
    monkeypatch.setattr(modules.protocol, "PARENT_CORE_FINGERPRINT", f["core"]["fingerprint"])
    monkeypatch.setattr(modules.protocol, "PRIOR_EXTENSION_FINGERPRINT", f["prior"]["fingerprint"])
    monkeypatch.setattr(modules.runner, "CoreFinalRunner", lambda *a, **kw: f["runner"])
    return modules


def budget_trajectory(project, job, output, *, discovered=False, failed_last=False):
    """Full B30/B50-shaped synthetic RPC and the production attempt recorder."""
    from matdiscovery.mace_parallel import mace_execution_metadata, AuditedOracleExecutor
    from matdiscovery.benchmark_adapters import BaseAdapter
    from matdiscovery.native_attribution import CONTRACT
    budget = job["budget"]; assert budget in (30, 50)
    output.mkdir(parents=True)
    task = json.loads((project / "configs/benchmark_tasks.json").read_text())["made"]
    metadata = {"vendor": {"commit": task["official_commit"]}, "oracle": task["oracle"],
        "stability_tolerance": .1, "mace_num_workers": 4, "orb_num_workers": 1, "mace_execution": mace_execution_metadata(4)}
    write_json_atomic(output / "job.json", job); write_json_atomic(output / "environment_metadata.json", metadata)
    events = []; counts = {"initialization_oracle_attempts": 3}
    def observation(): return {"episode_index": 0, "episode_seed": job["environment_seeds"][0], "counts": dict(counts)}
    def rpc(op, args=None, result=None, error=None):
        index = len(events)//2
        events.append({"direction": "request", "payload": {"id": index, "op": op, "args": args or {}}})
        events.append({"direction": "response", "elapsed_seconds": .01,
            "payload": {"id": index, "ok": error is None, **({"result": result} if error is None else {"error": error})}})
    rpc("init", {"benchmark": "made", "seed": job["environment_seeds"][0], "budget": budget,
        "elements": job["task"]["elements"], "mace_num_workers": 4}, {"metadata": metadata, "observation": observation()})
    curve = [[0, 0]]
    for index in range(1, budget+1):
        counts.update(candidate_oracle_attempts=index, step_attempts=index)
        count = int(discovered and index >= 5)
        if failed_last and index == budget:
            rpc("step", error={"code": "oracle_or_environment_exception", "details": observation()})
        else: rpc("step", result={"observation": observation(), "official_metrics": {"num_newly_discovered_stable": count}})
        curve.append([index, count])
    rpc("close", result={"closed": True})
    (output / "rpc").mkdir(); (output / "rpc/rpc.jsonl").write_text("".join(json.dumps(row)+"\n" for row in events))
    token = output / "tokens/episode.pt"; token.parent.mkdir(); torch.save({"input_ids": torch.tensor([[1,2,3]])}, token)
    row = {"decision_id": job["job_id"]+":0", "episode_index": 0,
        **{k: job[k] for k in ("split", "model_key", "task_id", "group_id")},
        "policy_stamp": {"state_id": job["policy_state_id"], "checkpoint_hash": job["initial_checkpoint_manifest_hash"]},
        "input_ids_file": str(token), "generation": {"completion_count": 3, "prompt_token_count": 5},
        "prefix_hash": "unit-prefix", "graph_status": "succeeded", "graph_contract": CONTRACT,
        "graph_metadata": {"unit_test_only": True, "full_prefix_hash": "unit-prefix",
            "policy": {"state_id": job["policy_state_id"], "generation": job["selected_generation"]}}}
    (output / "decisions.jsonl").write_text(json.dumps(row)+"\n")
    area = sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(curve,curve[1:]))
    summary = {**{k:job[k] for k in ("benchmark","model_key","method","task_id","seed")},
        "complete": True, "status": "succeeded", "episode_id": "0", "environment_seed": job["environment_seeds"][0],
        "costs": {**counts,"llm_calls":1,"completion_tokens":3,"prompt_tokens":5,"graph_seconds":0,"wall_seconds":.8},
        "metrics": {"AUDC":2*area/budget**2,"mSUN":curve[-1][1]/budget,"failure_rate":1-curve[-1][1]/budget},"discovery_curve":curve}
    write_json_atomic(output / "episodes.json", [summary])
    recorder = BaseAdapter({"vendor_root":str(project/"src"),"work_dir":str(output/"environment"),"seed":1,"budget":budget})
    recorder.work_dir.mkdir(); recorder.event_file=recorder.work_dir/"env_events.jsonl"; recorder.oracle_attempt_file=recorder.work_dir/"oracle_attempts.jsonl"
    class ScalarFixture:
        num_workers=1
        def evaluate(self,value): return {"unit_test_only":True,"energy":float(value)}
        def batch_evaluate(self,values): return [self.evaluate(value) for value in values]
    scalar=ScalarFixture()
    AuditedOracleExecutor(scalar,role="orb",candidate_hash=lambda value:"synthetic-"+str(value),invoke=recorder.invoke_oracle,record=recorder.oracle_audit)
    scalar.batch_evaluate(range(3)); recorder._phase="candidate"; scalar.batch_evaluate(range(budget))


def reseal(value):
    value["fingerprint"]=value["sweep_fingerprint"]=fingerprint({k:v for k,v in value.items() if k not in {"fingerprint","sweep_fingerprint"}})
    write_json_atomic(Path(value["workspace"])/"configs/made_budget_sweep.json",value)


def test_exact_independent120_and_optional10actors_without_mutable_B10_seals(budget_prior_fixture):
    f=budget_prior_fixture; reg=f["registration"]
    assert len(reg["new_jobs"])==len({j["job_id"] for j in reg["new_jobs"]})==120
    assert reg["expected_new_counts"]==mod.NEW_COUNTS and reg["expected_cumulative_counts"]==mod.CUMULATIVE_COUNTS
    assert [len(mod.sweep_jobs(reg,budget=b)) for b in (30,50)]==[60,60]
    assert reg["historical_collection_budget"]==50 and reg["selected_ES_execution_budget"]==10
    assert f["core"]["budget"]==50 and f["core"]["execution_budget"]==10 and f["prior_all30"]["execution_budget"]==10
    assert len(reg["actors"])==10 and reg["actors_are_optional"]
    assert [a["initial_method"] for a in reg["actors"]]==["baseline"]*2+["esopt_graph_risk"]*8
    assert len(reg["imported_dependencies"])==60 and sum(d["result"] is None for d in reg["imported_dependencies"])==56
    assert all('ledger' not in a["path"] for a in reg["source_files"])
    assert reg["training_jobs"]==reg["es_jobs"]==[]
    before=Path(reg["workspace"])/"configs/made_budget_sweep.json"; data=before.read_bytes()
    assert mod.build_sweep(f["core"],f["prior_all30"],f["workspace"])==reg and before.read_bytes()==data


@pytest.mark.parametrize("mutation",["budget","seed","omit","duplicate","train","checkpoint","denominator","actor","disclosure","corpus"])
def test_rehashed_scope_mutations_are_rejected(budget_prior_fixture,mutation):
    value=deepcopy(budget_prior_fixture["registration"])
    if mutation=="budget": value["new_jobs"][0]["budget"]=10
    elif mutation=="seed": value["new_jobs"][0]["environment_seeds"]=[2]
    elif mutation=="omit": value["new_jobs"].pop()
    elif mutation=="duplicate": value["new_jobs"][1]=deepcopy(value["new_jobs"][0])
    elif mutation=="train": value["new_jobs"][0]["task"]={"id":"Al-Au-Hf","elements":["Al","Au","Hf"]}
    elif mutation=="checkpoint": value["execution_profiles"]["esopt_graph_risk"]["selected_es"]["selected_generation"]=1
    elif mutation=="denominator": value["expected_new_counts"]["candidate_oracle_attempts"]=1200
    elif mutation=="actor": value["actors"].append(deepcopy(value["actors"][0]))
    elif mutation=="disclosure": value["registration"]["registered_after_B10_results_known"]=False
    else:value["historical_collection_budget"]=30
    reseal(value)
    with pytest.raises(ValueError):mod.validate_sweep(value)


def test_budget_optin_requires_real_PID_launch_and_no_scope_mix(budget_prior_fixture,monkeypatch):
    f=budget_prior_fixture; reg=f["registration"];job=reg["new_jobs"][0]
    with pytest.raises(AccountingError,match="budget"):expected_episodes({"jobs":[job]},expected_made_budget=30)
    kw={"expected_made_budget":30,"core_protocol":f["core"],"made_budget_sweep":reg}
    with pytest.raises(AccountingError,match="validate_launch"):expected_episodes({"jobs":[job]},**kw)
    fake=deepcopy(reg);fake["validated"]=True
    with pytest.raises(AccountingError):expected_episodes({"jobs":[job]},**{**kw,"made_budget_sweep":fake})
    proof=mod.validate_launch(reg);assert proof["complete"]
    assert list(expected_episodes({"jobs":[job]},**kw).values())[0]["candidate_oracle_attempts"]==30
    for other in ({"evaluation_extension":f["prior"]},{"made_all30_extension":f["prior_all30"]}):
        with pytest.raises(AccountingError,match="mix"):expected_episodes({"jobs":[job]},**kw,**other)
    with pytest.raises(AccountingError):expected_episodes({"jobs":[job]},**{**kw,"expected_made_budget":50})
    real_pid=mod.os.getpid();monkeypatch.setattr(mod.os,"getpid",lambda:real_pid+1)
    with pytest.raises(AccountingError,match="validate_launch"):expected_episodes({"jobs":[job]},**kw)


def test_fast_scope_never_repeats_deep_parent_and_rejects_source_mutation(budget_prior_fixture,monkeypatch):
    from matdiscovery import core_protocol,evaluation_extension
    f=budget_prior_fixture;reg=f["registration"];proof=mod.validate_launch(reg)
    def forbidden(*a,**kw):raise AssertionError("Repeated deep parent audit")
    monkeypatch.setattr(core_protocol,"read_core",forbidden);monkeypatch.setattr(evaluation_extension,"read_extension",forbidden)
    for budget in (30,50):
        job=next(j for j in reg["new_jobs"] if j["budget"]==budget)
        assert len(expected_episodes({"jobs":[job]},expected_made_budget=budget,made_budget_sweep=reg))==1
    assert mod.validate_launch(reg)==proof
    target=Path(f["prior_all30"]["workspace"])/"src/matdiscovery/policy.py";target.write_text(target.read_text()+"\n")
    with pytest.raises(ValueError):mod.validate_job_scope(reg,reg["new_jobs"][0],expected_made_budget=30)


@pytest.mark.parametrize("budget",[30,50])
def test_real_shared_scientific_verifier_uses_exact_budget_and_failure_denominator(budget_prior_fixture,tmp_path,budget):
    from matdiscovery.training_jobs import reconstruct_scientific_evidence,TrainingContractError,TrainingJobCallbacks
    from matdiscovery.final_evaluation import verify_final_envelope
    f=budget_prior_fixture;reg=f["registration"];mod.validate_launch(reg)
    job=next(j for j in reg["new_jobs"] if j["budget"]==budget and j["method"]=="esopt_graph_risk")
    profile=reg["execution_profiles"][job["method"]]
    actual={**job,"policy_state_id":"unit-state","initial_checkpoint_manifest_hash":"unit-checkpoint",
        "actual_model_state_hash":profile["actual_model_state_hash"],"selected_generation":2,"execution_profile_fingerprint":fingerprint(profile)}
    output=tmp_path/"synthetic_budget_trajectory";budget_trajectory(Path(f["core"]["workspace"]),actual,output,discovered=True,failed_last=True)
    tasks=json.loads((Path(f["core"]["workspace"])/"configs/benchmark_tasks.json").read_text())
    kw={"tasks":tasks,"partition":"final_test","expected_made_budget":budget,"core_protocol":f["core"],"made_budget_sweep":reg,"expected_mace_num_workers":4}
    proof=reconstruct_scientific_evidence(output,actual,**kw)
    assert proof["costs"]["candidate_oracle_attempts"]==budget and proof["costs"]["initialization_oracle_attempts"]==3
    with pytest.raises(TrainingContractError):reconstruct_scientific_evidence(output,actual,**{**kw,"partition":"training"})
    value={"job_id":job["job_id"],"model_revision":job["model_revision"],"manifest_fingerprint":"unit-manifest","complete":True,
        "episodes":json.loads((output/"episodes.json").read_text()),"artifacts":TrainingJobCallbacks._inventory(output),
        "execution_job":actual,"execution_profile":profile,"execution_profile_fingerprint":fingerprint(profile),
        "evidence_schema":"final_official_rpc_v1","scientific_evidence":proof}
    path=output/"result.json";write_json_atomic(path,value)
    result=verify_final_envelope(path,job,"unit-manifest",tasks=tasks,expected_profile=profile,expected_mace_num_workers=4,
        expected_made_budget=budget,core_protocol=f["core"],made_budget_sweep=reg)
    assert result["candidate_oracle_attempts"]==budget
    bad=deepcopy(value);bad["episodes"][0]["costs"]["candidate_oracle_attempts"]-=1;write_json_atomic(path,bad)
    with pytest.raises(AccountingError):verify_final_envelope(path,job,"unit-manifest",tasks=tasks,expected_made_budget=budget,made_budget_sweep=reg)


def test_original_validator_namespace_is_frozen_and_missing_B10_does_not_fake_complete(budget_prior_fixture,monkeypatch):
    f=budget_prior_fixture;reg=f["registration"];mod.validate_launch(reg)
    frozen=frozen_prior_modules(f,monkeypatch)
    assert Path(frozen.protocol.__file__).is_relative_to(f["all30_workspace"])
    audit=mod.audit_B10_results(reg,require_complete=False)
    assert not audit["complete"] and len(audit["results"])==10 and len(audit["missing_jobs"])==50
    with pytest.raises(ValueError,match="B10 sixty"):mod.audit_B10_results(reg)


def test_frozen_namespace_cache_is_active_for_real_hashes_and_exceptions_propagate(budget_prior_fixture,monkeypatch):
    from matdiscovery.immutable_hash_cache import immutable_hash_cache
    f=budget_prior_fixture;reg=f["registration"];mod.validate_launch(reg)
    modules=frozen_prior_modules(f,monkeypatch)
    cache=importlib.import_module(modules.protocol.__package__+".immutable_hash_cache")
    probe=Path(f["prior_all30"]["workspace"])/"configs/made_all30_extension.json"
    original_hash=cache.maybe_cached_file_sha256; original_context=cache.immutable_hash_cache
    hashes=[]; contexts=[]
    def checked_hash(path):
        value=original_hash(path)
        assert value is not None, "Frozen accounting is hashing outside its own cache context"
        hashes.append(str(path))
        return value
    @contextmanager
    def recorded_context():
        with original_context() as counters:
            contexts.append(counters)
            yield counters
    monkeypatch.setattr(cache,"maybe_cached_file_sha256",checked_hash)
    monkeypatch.setattr(cache,"immutable_hash_cache",recorded_context)
    # Main-namespace activation alone does not activate the frozen ContextVar.
    with immutable_hash_cache():
        assert original_hash(probe) is None
        audit=mod.audit_B10_results(reg,require_complete=False)
        assert len(audit["results"])==10 and hashes and contexts[0]["files_hashed"]>0
        assert original_hash(probe) is None
        def failed_original_audit(*args,**kwargs):
            assert original_hash(probe)==file_sha256(probe)
            raise ValueError("synthetic original B10 verification failure")
        monkeypatch.setattr(modules.protocol,"audit_imported_results",failed_original_audit)
        with pytest.raises(ValueError,match="synthetic original B10 verification failure"):
            mod.audit_B10_results(reg,require_complete=False)
        assert original_hash(probe) is None
    assert len(contexts)==2


def test_all60_B10_dependencies_are_reverified_readonly_in_frozen_namespace(budget_prior_fixture,monkeypatch):
    from test_fast_es_budget import b10_trajectory
    f=budget_prior_fixture;reg=f["registration"];mod.validate_launch(reg)
    modules=frozen_prior_modules(f,monkeypatch);calls=[]
    class UnitRollout:
        def __init__(self,project,policy,**kw):self.project=project
        def run(self,job,output,*,collection):
            assert not collection and job["budget"]==10
            calls.append(job["job_id"]);b10_trajectory(self.project,job,output,discovered=False)
    modules.runner.MadeAll30Runner(f["all30_workspace"],worker_id="primary",rollout_factory=UnitRollout).run()
    pending=mod.audit_B10_results(reg,require_complete=False)
    assert len(pending["results"])==60 and not pending["complete"] and len(pending["missing_acceptance"])==1
    modules.runner.aggregate_all30(f["all30_workspace"])
    before={str(p):(file_sha256(p),p.stat().st_mtime_ns) for p in f["all30_workspace"].rglob("*") if p.is_file()}
    cache=importlib.import_module(modules.protocol.__package__+".immutable_hash_cache")
    original_verify=modules.runner.verify_all30_envelope; audited_jobs=[]
    def cached_original_verify(path,job,*args,**kwargs):
        assert cache.maybe_cached_file_sha256(path)==file_sha256(path)
        audited_jobs.append(job["job_id"])
        return original_verify(path,job,*args,**kwargs)
    monkeypatch.setattr(modules.runner,"verify_all30_envelope",cached_original_verify)
    verified=mod.audit_B10_results(reg)
    assert verified["complete"] and len(verified["results"])==60 and verified["missing_jobs"]==[]
    assert len(audited_jobs)==len(set(audited_jobs))==50
    assert len(calls)==50 and len(f["calls"])==4 and len(f["old_calls"])==6
    assert before=={str(p):(file_sha256(p),p.stat().st_mtime_ns) for p in f["all30_workspace"].rglob("*") if p.is_file()}
    path=Path(verified["results"][-1]["result"]["path"]);path.write_text(path.read_text()+"\n")
    with pytest.raises(ValueError):mod.audit_B10_results(reg)
