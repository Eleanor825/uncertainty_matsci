"""Tiny CPU callback tests, not MADE/CrystalGym scientific experiments."""
import copy
from dataclasses import replace
import json
import random
from types import SimpleNamespace

import pytest
import torch

from matdiscovery.esopt import AgenticESOpt, tensor_state_hash
from matdiscovery.es_training import (
    ESTrainingConfig, ESTrainingDriver, EvaluationResult, TrainingCase,
    TrainingContractError, TrainingHalted, verify_clean_reload_checkpoint,
)


class TinyHF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny_causal_lm")
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(6, 3)
        self.model.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3), torch.nn.Linear(3, 3)])
        self.model.visual = torch.nn.Linear(2, 2, bias=False)
        self.model.visual.weight.requires_grad_(False)
        self.lm_head = torch.nn.Linear(3, 6, bias=False)
        self.eval()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, use_cache=False, **kwargs):
        assert use_cache is False
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = torch.tanh(layer(hidden))
        return SimpleNamespace(logits=self.lm_head(hidden))


class TinyAdapter:
    model_id = "fixture/tiny-hf"
    revision = "fixture-revision"
    checkpoint_hash = "INITIAL-DOWNLOAD-MANIFEST-HASH-NOT-UPDATED-TENSOR-HASH"
    configuration_fingerprint = "fixed-tiny-decoder-settings"

    def __init__(self, model):
        self.model = model
        self.transitions = []
        self.counter = 0
        self.model_stamp = SimpleNamespace(state_id="initial", generation=0,
                                           perturbation_seed=None, perturbation_sigma=None,
                                           checkpoint_hash=self.checkpoint_hash)

    def get_state_id(self):
        return self.model_stamp.state_id

    def mark_state(self, reason, *, generation=None, perturbation_seed=None, perturbation_sigma=None):
        self.counter += 1
        self.model_stamp = SimpleNamespace(
            state_id=f"state-{id(self)}-{self.counter}",
            generation=self.model_stamp.generation if generation is None else generation,
            perturbation_seed=perturbation_seed, perturbation_sigma=perturbation_sigma,
            checkpoint_hash=self.checkpoint_hash,
        )
        self.transitions.append({"reason": reason, "state_id": self.get_state_id(),
                                 "hash": tensor_state_hash(dict(self.model.state_dict()))})
        return self.model_stamp

    def parameter_report(self):
        return {"runtime_total_parameters": sum(p.numel() for p in self.model.parameters()),
                "unused_visual_parameters": self.model.model.visual.weight.numel(),
                "gradient_trainable_visual_parameters": 0, "es_scope_must_be_declared_separately": True}


class MockOfficialCallbacks:
    """Synthetic official-shaped evidence. Nothing here invokes a real oracle."""
    def __init__(self, root, base):
        self.root, self.base = root, copy.deepcopy(base)
        self.executed = []
        self.factory_calls = []
        self.recovered = []
        self.fail_candidate_once = None
        self.bad_reward_source = False
        self.nested = None
        self.driver = None

    def job_factory(self, request):
        # Durability must precede a factory that could dispatch physical work.
        record = json.loads(self.driver._job_record_path(request.request_id).read_text())
        assert record["status"] == "started"
        self.factory_calls.append(request.request_id)
        return {"request_id": request.request_id}

    def evaluate(self, policy, job, request):
        self.executed.append(request)
        assert job["request_id"] == request.request_id
        assert request.case.split == request.phase and request.phase in {"train", "dev"}
        assert policy.get_state_id() == request.policy_state_id
        if request.phase == "train":
            assert self.driver.optimizer.active_perturbation == {
                "seed": request.perturbation_seed, "sigma": request.perturbation_sigma}
        else:
            assert self.driver.optimizer.active_perturbation is None
        # Simulate multiple trajectory turns under one unchanged perturbation.
        for _ in range(3):
            assert tensor_state_hash(dict(policy.model.state_dict())) == request.actual_model_state_hash
            assert policy.get_state_id() == request.policy_state_id
        if self.nested:
            with pytest.raises(TrainingHalted, match="same policy"):
                self.nested.run()
        metric = float(torch.sigmoid(policy.model.lm_head.weight[0, 0]).detach())
        if request.phase == "dev":
            metric = 1 / (1 + abs(request.generation - 4))
        costs = {"wall_seconds": 0.5, "llm_calls": 3,
                 "candidate_oracle_attempts": 2 if request.case.benchmark == "made" else 0,
                 "dft_episode_attempts": 1 if request.case.benchmark == "crystalgym" else 0}
        source = "llm_self_score" if self.bad_reward_source else request.reward_source
        self.root.mkdir(parents=True, exist_ok=True)
        evidence = self.root / f"{request.request_id}.json"
        evidence.write_text(json.dumps({"request": request.semantic_identity(), "metric_name": request.metric_name,
                                        "metric_value": metric, "costs": costs, "reward_source": source,
                                        "complete": True, "unit_test_only": True}))
        result = EvaluationResult(request.request_id, request.metric_name, metric, source, costs, [str(evidence)])
        if self.fail_candidate_once == (request.generation, request.candidate_index, request.phase):
            self.fail_candidate_once = None
            raise RuntimeError("Synthetic transport failure after physical-shaped result was persisted")
        return result

    def verify(self, request, result):
        evidence = json.loads(open(result.result_paths[0]).read())
        assert evidence["request"] == request.semantic_identity()
        assert evidence["complete"] and evidence["reward_source"] == request.reward_source
        return {"verified": True, "evidence": "CPU unit-test evidence only; no real material oracle",
                "metric_name": evidence["metric_name"], "metric_value": evidence["metric_value"], "costs": evidence["costs"]}

    def recover(self, request, record):
        self.recovered.append(request.request_id)
        path = self.root / f"{request.request_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return EvaluationResult(request.request_id, data["metric_name"], data["metric_value"],
                                data["reward_source"], data["costs"], [str(path)])

    def reload(self, policy, path, expected_hash):
        return verify_clean_reload_checkpoint(policy, path, expected_hash,
                    fresh_policy_factory=lambda: copy.deepcopy(self.base), input_ids=torch.tensor([[1, 2, 3]]))


def cases(benchmark="made", property_name=None):
    train = [TrainingCase(f"train-{i}", "train", benchmark, property_name, {"budget": 2, "unit_test_only": True}) for i in range(3)]
    dev = [TrainingCase(f"dev-{i}", "dev", benchmark, property_name, {"budget": 2, "unit_test_only": True}) for i in range(2)]
    return train, dev


def build(tmp_path, *, config=None, base=None, callbacks=None, resume_recovery=False, directory="run"):
    if base is None:
        torch.manual_seed(123)
        base = TinyHF()
    config = config or ESTrainingConfig(generations=2, population=3, dev_every=1, noise_chunk_size=5)
    policy = TinyAdapter(copy.deepcopy(base))
    callback = callbacks or MockOfficialCallbacks(tmp_path / "evidence", base)
    train, dev = cases(config.benchmark, config.property_name)
    driver = ESTrainingDriver(policy, config, train_cases=train, dev_cases=dev,
                              job_factory=callback.job_factory, evaluate=callback.evaluate,
                              result_verifier=callback.verify, clean_reload_validator=callback.reload,
                              output_dir=tmp_path / directory, callback_fingerprint="synthetic-callback-v1",
                              recover_result=callback.recover if resume_recovery else None)
    callback.driver = driver
    return driver, callback, base


def test_complete_default_16_generations_population_pairing_and_dev_selection(tmp_path):
    config = ESTrainingConfig(noise_chunk_size=5)
    driver, callback, _ = build(tmp_path, config=config)
    before_visual = driver.policy.model.model.visual.weight.detach().clone()
    summary = driver.run()
    assert summary["completed_generations"] == summary["required_generations"] == 16
    assert summary["completed_evaluations"] == 16 * 8 + 4 * 2
    assert len(callback.executed) == summary["completed_evaluations"]
    assert summary["actual_evaluator_costs"]["wall_seconds"] == 0.5 * summary["completed_evaluations"]
    assert summary["best_generation"] == 4 and "generation_0016" in summary["final_checkpoint"]
    assert summary["clean_reload"]["verified"] and summary["clean_reload"]["max_abs_logit_difference"] == 0
    assert summary["initial_checkpoint_manifest_hash"] == driver.policy.checkpoint_hash
    assert summary["final_actual_model_state_hash"] != driver.policy.checkpoint_hash
    assert not torch.equal(before_visual, driver.policy.model.model.visual.weight)
    assert driver.policy.model.model.visual.weight.requires_grad is False
    assert len(driver.optimizer.history) == 16
    assert len(list((tmp_path / "run/checkpoints").glob("generation_*/complete.json"))) == 17
    for plan in driver.plan:
        train_requests = [r for r in callback.executed if r.generation == plan["generation"] and r.phase == "train"]
        assert len({r.perturbation_seed for r in train_requests}) == 8
        assert len({(r.case.case_id, r.environment_seed) for r in train_requests}) == 1
        assert len({r.policy_state_id for r in train_requests}) == 8
        assert all(r.perturbation_sigma == plan["sigma"] for r in train_requests)
    assert driver.plan[0]["sigma"] == pytest.approx(0.001)
    assert driver.plan[-1]["sigma"] == pytest.approx(0.0002)
    dev_requests = [r for r in callback.executed if r.phase == "dev"]
    assert {r.generation for r in dev_requests} == {4, 8, 12, 16}
    assert len({r.environment_seed for r in dev_requests}) == len(driver.dev_cases)
    changes = driver._completed[-1]["actual_layer_deltas"]
    assert changes["unused_visual"]["actual_delta_l2"] > 0
    assert changes["model.layers.0"]["changed_parameter_tensors"] > 0
    reasons = [event["reason"] for event in driver.policy.transitions]
    assert reasons.count("perturb") == reasons.count("restore") == 16 * 8
    assert reasons.count("update") == 16


def test_seed_schedule_is_independent_of_global_rng_and_identical_for_paired_methods(tmp_path):
    state = random.getstate()
    a, _, base = build(tmp_path / "a")
    assert random.getstate() == state
    random.seed(90867)
    b, _, _ = build(tmp_path / "b", base=base, config=replace(a.config, method="es_plus_risk"))
    assert a.plan == b.plan and a.run_fingerprint != b.run_fingerprint
    assert len({seed for plan in a.plan for seed in plan["population_seeds"]}) == a.config.generations * a.config.population


def test_unknown_physical_outcome_halts_restore_then_read_only_recovery_without_reexecution(tmp_path):
    config = ESTrainingConfig(generations=3, population=3, dev_every=2, noise_chunk_size=5)
    driver, callback, base = build(tmp_path, config=config)
    callback.fail_candidate_once = (1, 1, "train")
    initial_hash = tensor_state_hash(dict(driver.policy.model.state_dict()))
    with pytest.raises(TrainingHalted, match="without retry"):
        driver.run()
    assert len(callback.executed) == 2
    assert driver.optimizer.active_perturbation is None
    assert tensor_state_hash(dict(driver.policy.model.state_dict())) == initial_hash
    assert driver.policy.transitions[-1]["reason"] == "restore"
    pending = [json.loads(p.read_text()) for p in (tmp_path / "run/jobs").glob("*.json") if json.loads(p.read_text())["status"] == "outcome_unknown"]
    assert len(pending) == 1 and pending[0]["actual_costs"] is None
    blocked, _, _ = build(tmp_path, config=config, base=base, callbacks=callback)
    with pytest.raises(TrainingHalted, match="no automatic retry"):
        blocked.run(resume=True)
    assert len(callback.executed) == 2
    resumed, _, _ = build(tmp_path, config=config, base=base, callbacks=callback, resume_recovery=True)
    summary = resumed.run(resume=True)
    ids = [r.request_id for r in callback.executed]
    assert len(ids) == len(set(ids)) == summary["expected_evaluations"]
    assert len(callback.recovered) == 1
    assert summary["current_session"]["results_reused"] >= 2
    assert summary["actual_evaluator_costs"]["llm_calls"] == 3 * summary["expected_evaluations"]


def test_interrupted_checkpoint_commit_reuses_results_but_never_loads_partial_checkpoint(tmp_path, monkeypatch):
    driver, callback, base = build(tmp_path)
    original = AgenticESOpt.save_checkpoint
    failed = [False]
    def interrupt_after_save(self, path):
        result = original(self, path)
        if "generation_0001" in str(path) and not failed[0]:
            failed[0] = True
            raise OSError("synthetic commit interruption")
        return result
    monkeypatch.setattr(AgenticESOpt, "save_checkpoint", interrupt_after_save)
    with pytest.raises(OSError, match="commit interruption"):
        driver.run()
    assert (tmp_path / "run/checkpoints/generation_0001/policy.pt").exists()
    assert not (tmp_path / "run/checkpoints/generation_0001/complete.json").exists()
    original_ids = [r.request_id for r in callback.executed]
    monkeypatch.setattr(AgenticESOpt, "save_checkpoint", original)
    resumed, _, _ = build(tmp_path, config=driver.config, base=base, callbacks=callback)
    summary = resumed.run(resume=True)
    assert all(sum(r.request_id == request_id for r in callback.executed) == 1 for request_id in original_ids)
    assert summary["completed_generations"] == 2 and summary["current_session"]["results_reused"] == len(original_ids)


def test_changed_replay_weights_halt_in_original_job_slot_without_dispatch(tmp_path, monkeypatch):
    driver, callback, base = build(tmp_path)
    callback.fail_candidate_once = (1, 1, "train")
    with pytest.raises(TrainingHalted):
        driver.run()
    original = AgenticESOpt.begin_perturbation
    def changed_replay(self, seed, sigma):
        original(self, seed, sigma)
        with torch.no_grad():
            next(self.model.parameters()).add_(0.01)
    monkeypatch.setattr(AgenticESOpt, "begin_perturbation", changed_replay)
    resumed, _, _ = build(tmp_path, config=driver.config, base=base, callbacks=callback, resume_recovery=True)
    before = len(callback.executed)
    with pytest.raises(TrainingContractError, match="different weights"):
        resumed.run(resume=True)
    assert len(callback.executed) == before
    assert resumed.optimizer.active_perturbation is None


def test_corrupt_complete_checkpoint_and_changed_artifacts_are_not_silently_replayed(tmp_path):
    config = ESTrainingConfig(generations=1, population=2, dev_every=1, noise_chunk_size=5)
    driver, callback, base = build(tmp_path, config=config)
    driver.run()
    checkpoint = tmp_path / "run/checkpoints/generation_0001/policy.pt"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original + b"tampered")
    resumed, _, _ = build(tmp_path, config=config, base=base, callbacks=callback)
    count = len(callback.executed)
    with pytest.raises(TrainingContractError, match="corrupt"):
        resumed.run(resume=True)
    assert len(callback.executed) == count
    checkpoint.write_bytes(original)
    artifact = next((tmp_path / "evidence").glob("*.json"))
    artifact.write_text(artifact.read_text() + "\n")
    resumed, _, _ = build(tmp_path, config=config, base=base, callbacks=callback)
    with pytest.raises(TrainingContractError, match="changed result evidence"):
        resumed.run(resume=True)
    assert len(callback.executed) == count


@pytest.mark.parametrize("mutation", ["test", "mixed_benchmark", "mixed_property"])
def test_test_cases_and_mixed_units_are_rejected_before_jobs(tmp_path, mutation):
    config = ESTrainingConfig(benchmark="crystalgym", property_name="bm", generations=1, population=2)
    train, dev = cases("crystalgym", "bm")
    if mutation == "test":
        train[0] = replace(train[0], split="test")
    elif mutation == "mixed_benchmark":
        train[0] = replace(train[0], benchmark="made")
    else:
        train[0] = replace(train[0], property_name="density")
    base = TinyHF()
    callback = MockOfficialCallbacks(tmp_path / "evidence", base)
    with pytest.raises(TrainingContractError):
        ESTrainingDriver(TinyAdapter(base), config, train_cases=train, dev_cases=dev,
                         job_factory=callback.job_factory, evaluate=callback.evaluate, result_verifier=callback.verify,
                         clean_reload_validator=callback.reload, output_dir=tmp_path / "run", callback_fingerprint="fixture")
    assert not callback.executed and not (tmp_path / "run").exists()


def test_self_scores_rejected_and_full_model_cannot_contain_risk_network(tmp_path):
    driver, callback, _ = build(tmp_path / "score")
    callback.bad_reward_source = True
    with pytest.raises(TrainingHalted):
        driver.run()
    assert len(driver.optimizer.history) == 0 and driver.optimizer.active_perturbation is None
    other, callback, _ = build(tmp_path / "network")
    other.policy.model.risk_network = torch.nn.Linear(3, 1)
    with pytest.raises(TrainingContractError, match="outside"):
        other.run()
    assert not callback.executed


def test_same_policy_cannot_run_concurrent_task_and_method_resume_is_separate(tmp_path):
    config = ESTrainingConfig(generations=1, population=2, dev_every=1, noise_chunk_size=5)
    driver, callback, base = build(tmp_path, config=config)
    nested = ESTrainingDriver(driver.policy, replace(config, method="es_plus_risk"),
                             train_cases=driver.train_cases, dev_cases=driver.dev_cases,
                             job_factory=callback.job_factory, evaluate=callback.evaluate, result_verifier=callback.verify,
                             clean_reload_validator=callback.reload, output_dir=tmp_path / "other", callback_fingerprint="fixture")
    callback.nested = nested
    driver.run()
    assert not (tmp_path / "other").exists()
    mismatch, _, _ = build(tmp_path, config=replace(config, method="es_plus_risk"), base=base, callbacks=callback)
    with pytest.raises(TrainingContractError, match="identical method"):
        mismatch.run(resume=True)


def test_clean_reload_is_mandatory_and_can_finish_later_without_oracle_reexecution(tmp_path):
    config = ESTrainingConfig(generations=1, population=2, dev_every=1, noise_chunk_size=5)
    driver, callback, base = build(tmp_path, config=config)
    driver.clean_reload_validator = lambda *_: {"verified": True}
    with pytest.raises(TrainingContractError, match="Fresh-model"):
        driver.run()
    assert len(driver.optimizer.history) == 1
    assert not (tmp_path / "run/training_summary.json").exists()
    completed_calls = len(callback.executed)
    resumed, _, _ = build(tmp_path, config=config, base=base, callbacks=callback)
    summary = resumed.run(resume=True)
    assert len(callback.executed) == completed_calls and summary["status"] == "complete"
    with pytest.raises(TrainingContractError, match="different object"):
        verify_clean_reload_checkpoint(resumed.policy, summary["final_checkpoint"], summary["final_actual_model_state_hash"],
                                       fresh_policy_factory=lambda: resumed.policy.model, input_ids=torch.tensor([[1, 2]]))


def test_crystalgym_uses_official_reward_and_fixed_property(tmp_path):
    config = ESTrainingConfig(benchmark="crystalgym", property_name="density", generations=1, population=2,
                              dev_every=4, noise_chunk_size=5)
    driver, callback, _ = build(tmp_path, config=config)
    summary = driver.run()
    assert all(r.metric_name == "reward" and r.case.property_name == "density" for r in callback.executed)
    assert summary["actual_evaluator_costs"]["dft_episode_attempts"] == summary["expected_evaluations"]
    assert summary["best_generation"] == 1  # the final generation is dev-evaluated even before dev_every
