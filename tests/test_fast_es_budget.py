"""B10 whole-trajectory and tiny actual ES tests; all physics is synthetic."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from matdiscovery import core_protocol
from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.core_es import (CoreTrainingCondition, CoreTrainingJobCallbacks, _descriptor_path,
    _fast_evolution_evidence, audit_core_es_costs, finalize_core_training, load_core_selected_es, run_core_es)
from matdiscovery.es_training import ESTrainingDriver, TrainingContractError, verify_clean_reload_checkpoint
from matdiscovery.rollouts import RolloutSettings
from matdiscovery.training_jobs import reconstruct_scientific_evidence
from test_core_es import setup_core, full_project_fixture
from test_es_training import TinyAdapter, TinyHF
from torch_runtime_fixture import restore_torch_runtime


@pytest.fixture
def fast_core(setup_core):
    core = setup_core
    # The separate protocol tests verify real source ancestry. This CPU fixture
    # keeps the existing isolated read_core boundary and actual ES arithmetic.
    dev = core_protocol.collection_jobs(core, include_imported=False)[0]
    core.update(registration=copy.deepcopy(core_protocol.FAST_CAUSAL_GRAPH_REGISTRATION),
        execution_budget=10, historical_collection_budget=50, fit_recipe=core_protocol.NORMALIZED_RECIPE,
        imported_development={"job": dev}, passed_bank_reuse_contract={"unit_fixture": True}, causal_graph_amendment={})
    core["study_id"] = core["registration"]["study_id"]
    root = Path(core["workspace"])
    path = root / "configs/main_protocol.json"
    main = json.loads(path.read_text()); main["transcoder"]["epochs"] = 64; main["execution_budget"] = 10
    main["esopt"]["full_training_and_development_candidate_oracle_attempts"] = 60
    write_json_atomic(path, main)
    core["transcoder"], core["esopt"] = copy.deepcopy(main["transcoder"]), copy.deepcopy(main["esopt"])
    core["derived_main_protocol"] = {"path": str(path), "sha256": file_sha256(path)}
    return core


def b10_trajectory(project, job, output, *, discovered, failed_last=False):
    """Produce ten closed fake responses plus the real audit-recorder shape."""
    output.mkdir(parents=True)
    assert job["budget"] == 10
    task = json.loads((project / "configs/benchmark_tasks.json").read_text())["made"]
    from matdiscovery.mace_parallel import mace_execution_metadata, AuditedOracleExecutor
    from matdiscovery.benchmark_adapters import BaseAdapter
    metadata = {"vendor": {"commit": task["official_commit"]}, "oracle": task["oracle"],
        "stability_tolerance": .1, "mace_num_workers": 4, "orb_num_workers": 1, "mace_execution": mace_execution_metadata(4)}
    write_json_atomic(output / "job.json", job); write_json_atomic(output / "environment_metadata.json", metadata)
    rows = []
    counts = {"initialization_oracle_attempts": 3}
    def obs(): return {"episode_index": 0, "episode_seed": job["environment_seeds"][0], "counts": dict(counts)}
    def rpc(op, args=None, result=None, error=None):
        number = len(rows) // 2
        rows.append({"direction": "request", "payload": {"id": number, "op": op, "args": args or {}}})
        rows.append({"direction": "response", "elapsed_seconds": .01,
            "payload": {"id": number, "ok": error is None, **({"result": result} if error is None else {"error": error})}})
    rpc("init", {"benchmark": "made", "seed": job["environment_seeds"][0], "budget": 10,
        "elements": job["task"]["elements"], "mace_num_workers": 4}, {"metadata": metadata, "observation": obs()})
    curve = [[0, 0]]
    for index in range(1, 11):
        counts.update(candidate_oracle_attempts=index, step_attempts=index)
        count = int(discovered and index >= 5)
        if failed_last and index == 10:
            rpc("step", error={"code": "oracle_or_environment_exception", "details": obs()})
        else:
            rpc("step", result={"observation": obs(), "official_metrics": {"num_newly_discovered_stable": count}})
        curve.append([index, count])
    rpc("close", result={"closed": True})
    (output / "rpc").mkdir(); (output / "rpc/rpc.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    token = output / "tokens/episode.pt"; token.parent.mkdir(); torch.save({"input_ids": torch.tensor([[1, 2, 3]])}, token)
    request = job.get("training_request", {"phase": "final", "generation": job.get("selected_generation", 0)})
    from matdiscovery.native_attribution import CONTRACT
    row = {"decision_id": job["job_id"] + ":0", "episode_index": 0,
        **{key: job[key] for key in ("split", "model_key", "task_id", "group_id")},
        "policy_stamp": {"state_id": job["policy_state_id"], "checkpoint_hash": job["initial_checkpoint_manifest_hash"]},
        "input_ids_file": str(token), "generation": {"completion_count": 3, "prompt_token_count": 5},
        "prefix_hash": "unit-prefix", "graph_status": "succeeded", "graph_contract": CONTRACT,
        "graph_metadata": {"unit_test_only": True, "full_prefix_hash": "unit-prefix",
            "policy": {"state_id": job["policy_state_id"], "generation": request["generation"] - int(request["phase"] == "train")}}}
    (output / "decisions.jsonl").write_text(json.dumps(row) + "\n")
    area = sum((b[0]-a[0])*(a[1]+b[1])/2 for a, b in zip(curve, curve[1:]))
    summary = {**{key: job[key] for key in ("benchmark", "model_key", "method", "task_id", "seed")},
        "complete": True, "status": "succeeded", "episode_id": "0", "environment_seed": job["environment_seeds"][0],
        "costs": {**counts, "llm_calls": 1, "completion_tokens": 3, "prompt_tokens": 5, "graph_seconds": 0, "wall_seconds": .8},
        "metrics": {"AUDC": 2*area/100, "mSUN": curve[-1][1]/10, "failure_rate": 1-curve[-1][1]/10}, "discovery_curve": curve}
    write_json_atomic(output / "episodes.json", [summary])
    recorder = BaseAdapter({"vendor_root": str(project / "src"), "work_dir": str(output / "environment"),
                            "seed": job["environment_seeds"][0], "budget": 10})
    recorder.work_dir.mkdir(); recorder.event_file = recorder.work_dir / "env_events.jsonl"
    recorder.oracle_attempt_file = recorder.work_dir / "oracle_attempts.jsonl"
    class ScalarFixture:
        num_workers = 1
        def evaluate(self, value): return {"unit_test_only": True, "energy": float(value)}
        def batch_evaluate(self, values): return [self.evaluate(value) for value in values]
    scalar = ScalarFixture()
    AuditedOracleExecutor(scalar, role="orb", candidate_hash=lambda value: "synthetic-" + str(value),
        invoke=recorder.invoke_oracle, record=recorder.oracle_audit)
    scalar.batch_evaluate(range(3)); recorder._phase = "candidate"; scalar.batch_evaluate(range(10))


def train_fast(core, *, tied=False, tied_first=False, failed_last=False):
    condition = CoreTrainingCondition(Path(core["workspace"]))
    torch.manual_seed(821); base = TinyHF(); policy = TinyAdapter(copy.deepcopy(base))
    requests = []
    class RolloutFixture:
        def __init__(self, project, live_policy, **kwargs): self.project = project
        def run(self, job, output, collection=False):
            requests.append(job["training_request"])
            req = job["training_request"]
            discovered = False if tied or (tied_first and req["phase"] == "train" and req["generation"] == 1) else (req["candidate_index"] == 1 if req["phase"] == "train" else req["generation"] == 1)
            b10_trajectory(self.project, job, output, discovered=discovered, failed_last=failed_last)
    contract = core_protocol.corpus_contract(core, core_protocol.collection_manifest_paths(core))
    callbacks = CoreTrainingJobCallbacks(condition, policy, condition.output_directory(), corpus_contract=contract,
        settings=RolloutSettings(), risk_model=object(), attributor=object(), rollout_factory=RolloutFixture)
    write_json_atomic(_descriptor_path(callbacks.output), {"core_fingerprint": core["fingerprint"],
        "callback_identity": callbacks.callback_identity, "callback_fingerprint": callbacks.fingerprint})
    def reload(policy, checkpoint, expected):
        return verify_clean_reload_checkpoint(policy, checkpoint, expected,
            fresh_policy_factory=lambda: copy.deepcopy(base), input_ids=torch.tensor([[1, 2, 3]]))
    train, dev = condition.cases()
    driver = ESTrainingDriver(policy, condition.config(), train_cases=train, dev_cases=dev,
        job_factory=callbacks.job_factory, evaluate=callbacks.evaluate, result_verifier=callbacks.result_verifier,
        recover_result=callbacks.recover_result, clean_reload_validator=reload,
        output_dir=callbacks.output, callback_fingerprint=callbacks.fingerprint)
    return driver, callbacks, policy, requests, reload


@pytest.mark.parametrize("tied", [False, True])
def test_fast_g2p2_real_parameter_updates_or_honest_zero_signal(fast_core, tied):
    driver, callbacks, policy, requests, reload = train_fast(fast_core, tied=tied)
    summary = driver.run()
    assert len(requests) == 6 and summary["actual_evaluator_costs"]["candidate_oracle_attempts"] == 60
    profile = finalize_core_training(policy, callbacks.output, core_protocol=fast_core, clean_reload_validator=reload)
    assert profile["evolution_signal_observed"] is not tied
    assert (profile["actual_model_state_hash"] == profile["initial_actual_model_state_hash"]) is tied
    assert profile["nonzero_text_parameter_update_verified"] is not tied
    assert profile["initial_dev_evaluated"] is False
    assert [r["generation"] for r in profile["generation_curve"]] == [1, 2]
    assert profile["generation_curve"][0]["dev_environment_seeds"] == profile["generation_curve"][1]["dev_environment_seeds"]
    if tied:
        assert profile["evolution_status"] == "no_evolution_signal"
        assert profile["zero_update_proof"]["selected_matches_initial"]
        assert all(r["population_fitness_tied"] and r["all_parameter_deltas_zero"] for r in profile["generation_curve"])
    else:
        assert profile["zero_update_proof"] is None
        assert profile["generation_curve"][1]["dev_mean"] < profile["generation_curve"][0]["dev_mean"]
    costs = audit_core_es_costs(callbacks.output, fast_core)
    assert costs["training"]["costs"]["candidate_oracle_attempts"] == 40
    assert costs["development"]["costs"]["candidate_oracle_attempts"] == 20
    assert load_core_selected_es(policy, callbacks.output, core_protocol=fast_core) == profile
    assert len(requests) == 6


def test_fast_science_requires_explicit_scope_and_failed_attempt_uses_b10_denominator(fast_core):
    driver, callbacks, _, _, _ = train_fast(fast_core, failed_last=True)
    driver.run()
    output = next((callbacks.output / "rollouts").iterdir()); job = json.loads((output / "job.json").read_text())
    tasks = json.loads((Path(fast_core["workspace"]) / "configs/benchmark_tasks.json").read_text())
    with pytest.raises(TrainingContractError, match="expected execution budget"):
        reconstruct_scientific_evidence(output, job, tasks=tasks)
    with pytest.raises(TrainingContractError, match="sealed core"):
        reconstruct_scientific_evidence(output, job, tasks=tasks, expected_made_budget=10)
    proof = reconstruct_scientific_evidence(output, job, tasks=tasks, expected_made_budget=10,
        core_protocol=fast_core, expected_mace_num_workers=4)
    assert proof["costs"]["candidate_oracle_attempts"] == 10 and proof["costs"]["initialization_oracle_attempts"] == 3
    rows = json.loads((output / "episodes.json").read_text()); rows[0]["metrics"]["AUDC"] = .999
    write_json_atomic(output / "episodes.json", rows)
    with pytest.raises(TrainingContractError, match="AUDC"):
        reconstruct_scientific_evidence(output, job, tasks=tasks, expected_made_budget=10, core_protocol=fast_core)


def test_fast_plan_and_zero_proof_tampering_fail_closed(fast_core):
    assert run_core_es(fast_core["workspace"], plan_only=True)["candidate_oracle_attempts"] == 60
    driver, callbacks, policy, _, reload = train_fast(fast_core, tied=True)
    driver.run(); finalize_core_training(policy, callbacks.output, core_protocol=fast_core, clean_reload_validator=reload)
    path = callbacks.output / "core_training_receipt.json"
    receipt = json.loads(path.read_text()); receipt["evolution_signal_observed"] = True
    receipt["fingerprint"] = fingerprint({k: v for k, v in receipt.items() if k != "fingerprint"})
    write_json_atomic(path, receipt)
    with pytest.raises(TrainingContractError, match="evolution/zero-update"):
        load_core_selected_es(policy, callbacks.output, core_protocol=fast_core)


def test_selected_zero_generation_and_later_negative_dev_update_remain_distinct(fast_core):
    driver, callbacks, policy, _, reload = train_fast(fast_core, tied_first=True)
    driver.run()
    profile = finalize_core_training(policy, callbacks.output, core_protocol=fast_core, clean_reload_validator=reload)
    assert profile["selected_generation"] == 1 and profile["evolution_signal_observed"] is False
    assert profile["zero_update_proof"]["any_generation_updated"] is True
    assert profile["generation_curve"][0]["population_fitness_tied"] is True
    assert profile["generation_curve"][1]["parameter_update_observed"] is True
    assert profile["generation_curve"][1]["dev_mean"] < profile["generation_curve"][0]["dev_mean"]
    state = json.loads((driver.output_dir / "checkpoints/generation_0002/driver_state.json").read_text())
    optimizer = json.loads((driver.output_dir / "checkpoints/generation_0002/es_history.json").read_text())["history"]
    optimizer[1]["normalized_rewards"] = [0., 0.]
    with pytest.raises(TrainingContractError, match="z-score"):
        _fast_evolution_evidence(state["generations"], optimizer,
            initial=profile["initial_actual_model_state_hash"], selected_generation=1, config=driver.config)
