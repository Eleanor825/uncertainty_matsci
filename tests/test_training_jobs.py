"""Synthetic contracts only: no policy weights, ORB or Quantum ESPRESSO runs."""
import copy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from matdiscovery.es_training import EvaluationRequest, TrainingCase, TrainingContractError, TrainingHalted
from matdiscovery.esopt import AgenticESOpt, tensor_state_hash
from matdiscovery.rollouts import RolloutSettings
from matdiscovery.training_jobs import (
    TrainingCondition, TrainingJobCallbacks, cpu_clean_reload_validator,
    make_training_job, reconstruct_scientific_evidence,
)


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "new_project"
    (root / "configs").mkdir(parents=True)
    for name in ("main_protocol.json", "benchmark_tasks.json", "made_splits.json", "model_manifest.json"):
        shutil.copy2(PROJECT / "configs" / name, root / "configs" / name)
    # CPU mocks exercise orchestration, not a deployment backend. Keep their
    # numerical runtime independent of the real protocol's GPU acceptance.
    protocol = json.loads((root / "configs/main_protocol.json").read_text())
    protocol.pop("policy_runtime", None)
    (root / "configs/main_protocol.json").write_text(json.dumps(protocol))
    for name in ("configs/assets.runtime.json", "configs/qe.runtime.json", "data/raw/materials_project/index.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (root / "src/matdiscovery").mkdir(parents=True)
    (root / "src/matdiscovery/mock.py").write_text("# synthetic callback fixture\n")
    (root / "scripts").mkdir()
    (root / "scripts/train_es_condition.py").write_text("# synthetic callback fixture\n")
    return root


def setup_request(project, benchmark="made", phase="train", *, method="esopt", candidate=0, envseed=42):
    condition = TrainingCondition(project, "qwen35_4b", benchmark, method, 1, "density" if benchmark == "crystalgym" else None)
    entry = json.loads((project / "configs/model_manifest.json").read_text())["models"][0]
    policy = SimpleNamespace(model_id=entry["model_id"], revision=entry["revision"], configuration_fingerprint="decoder",
                             checkpoint_hash="base-hash")
    train, dev = condition.cases()
    case = (train if phase == "train" else dev)[0]
    request = EvaluationRequest("a" * 64, "run", phase, 1, candidate if phase == "train" else None,
                                case, method, envseed, 13 if phase == "train" else None,
                                .001 if phase == "train" else None, "actual-weights", "base-hash", "state")
    job = make_training_job(condition, request, model_entry=entry, policy_configuration_fingerprint=policy.configuration_fingerprint)
    return condition, policy, request, job


def dump(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))


def write_mock_scientific_trajectory(project, job, output, *, failed_last_made=False):
    """Official-shaped synthetic protocol evidence, never a material result."""
    output.mkdir(parents=True)
    tasks = json.loads((project / "configs/benchmark_tasks.json").read_text())
    benchmark = job["benchmark"]
    args = {"benchmark": benchmark, "seed": job["environment_seeds"][0], "budget": job["budget"]}
    metadata = {"vendor": {"commit": tasks[benchmark]["official_commit"]}}
    if benchmark == "made":
        args["elements"] = job["task"]["elements"]
        metadata.update(oracle=tasks[benchmark]["oracle"], stability_tolerance=.1)
        from matdiscovery.execution_contract import declared_mace_workers
        from matdiscovery.mace_parallel import mace_execution_metadata
        workers = declared_mace_workers(json.loads((project / "configs/main_protocol.json").read_text()))
        args["mace_num_workers"] = workers
        metadata.update(mace_num_workers=workers, orb_num_workers=1, mace_execution=mace_execution_metadata(workers))
    else:
        metadata.update(prototype_index=job["task"]["prototype"]["index"], split="heldout" if job["stage"] == "final_eval" else "train", property=job["task"]["property"]["id"], target=job["task"]["property"]["target"],
                        qe={"calculation": "vc-relax", "occupations": "smearing", "ecutwfc": 50, "ecutrho": 400})
    dump(output / "job.json", job)
    dump(output / "environment_metadata.json", metadata)
    rows, summaries, decisions = [], [], []
    counts = {"initialization_oracle_attempts": 3 if benchmark == "made" else 0}
    def observation(episode):
        return {"episode_index": episode, "episode_seed": job["environment_seeds"][episode], "counts": dict(counts)}
    def rpc(op, args=None, result=None, error=None):
        number = len(rows) // 2
        rows.append({"direction": "request", "payload": {"id": number, "op": op, "args": args or {}}})
        rows.append({"direction": "response", "elapsed_seconds": .01,
                     "payload": {"id": number, "ok": error is None, **({"result": result} if error is None else {"error": error})}})
    rpc("init", args, {"metadata": metadata, "observation": observation(0)})
    for episode in range(job["expected_counts"]["episodes"]):
        before = dict(counts)
        if episode:
            rpc("reset", result=observation(episode))
        if benchmark == "made":
            curve, discovered = [[0, 0]], 0
            for attempt in range(1, 51):
                counts.update(candidate_oracle_attempts=attempt, step_attempts=attempt)
                if failed_last_made and attempt == 50:
                    rpc("step", error={"code": "oracle_or_environment_exception", "details": observation(episode)})
                else:
                    discovered += int(attempt % 10 == 0)
                    rpc("step", result={"observation": observation(episode), "official_metrics": {"num_newly_discovered_stable": discovered}})
                curve.append([attempt, discovered])
            area = sum((x1-x0)*(y1+y0)/2 for (x0,y0),(x1,y1) in zip(curve, curve[1:]))
            metrics = {"AUDC": 2*area/2500}
        else:
            counts["dft_episode_attempts"] = episode + 1
            counts["atomic_action_attempts"] = 4 * (episode + 1)
            metrics = {"reward": -1. if episode == 0 else .3, "property_value": None if episode == 0 else 4.8}
            rpc("step", result={"episode_done": True, "observation": observation(episode),
                "scientific_result": {"terminal": True, "property": "density", **metrics}})
        token = output / "tokens" / f"episode_{episode}.pt"
        token.parent.mkdir(exist_ok=True)
        torch.save({"input_ids": torch.tensor([[1, 2, 3]])}, token)
        decisions.append({"decision_id": f"{job['job_id']}:{episode}", "episode_index": episode,
            "split": job["split"], "model_key": job["model_key"], "task_id": job["task_id"], "group_id": job["group_id"],
            "policy_stamp": {"state_id": job["policy_state_id"], "checkpoint_hash": job["initial_checkpoint_manifest_hash"]},
            "input_ids_file": str(token), "generation": {"completion_count": 3, "prompt_token_count": 5}})
        costs = {key: value - before.get(key, 0) for key, value in counts.items()}
        costs.update(llm_calls=1, completion_tokens=3, prompt_tokens=5, graph_seconds=0, wall_seconds=.8)
        if episode == 0:
            costs["initialization_oracle_attempts"] = 3 if benchmark == "made" else 0
        summary = {key: job[key] for key in ("benchmark", "model_key", "method", "task_id", "seed")}
        summary.update(complete=True, status="succeeded" if benchmark == "made" or episode else "failed", episode_id=str(episode), environment_seed=job["environment_seeds"][episode], costs=costs, metrics=metrics)
        if benchmark == "made":
            summary["discovery_curve"] = curve
        summaries.append(summary)
    rpc("close", result={"closed": True})
    (output / "rpc").mkdir()
    (output / "rpc/rpc.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in decisions))
    dump(output / "episodes.json", summaries)
    if benchmark == "made":
        # Exercise the real recorder with a scalar mock only. These are synthetic
        # unit events, never ORB/MACE calls or admissible study results.
        from matdiscovery.benchmark_adapters import BaseAdapter
        from matdiscovery.mace_parallel import AuditedOracleExecutor
        recorder = BaseAdapter({"vendor_root": str(project / "src"), "work_dir": str(output / "environment"),
                                "seed": job["environment_seeds"][0], "budget": 50})
        recorder.work_dir.mkdir()
        recorder.event_file = recorder.work_dir / "env_events.jsonl"
        recorder.oracle_attempt_file = recorder.work_dir / "oracle_attempts.jsonl"
        class ScalarMock:
            num_workers = 1
            def evaluate(self, value):
                return {"unit_test_only": True, "energy": float(value)}
            def batch_evaluate(self, values):
                return [self.evaluate(v) for v in values]
        scalar = ScalarMock()
        AuditedOracleExecutor(scalar, role="orb", candidate_hash=lambda value: "synthetic-" + str(value),
                              invoke=recorder.invoke_oracle, record=recorder.oracle_audit)
        scalar.batch_evaluate(list(range(3)))
        recorder._phase = "candidate"
        scalar.batch_evaluate(list(range(50)))
    return summaries


def test_complete_case_catalog_and_budget(project):
    for benchmark, sizes in (("made", (30, 12)), ("crystalgym", (5, 5))):
        condition, _, request, job = setup_request(project, benchmark)
        train, dev = condition.cases()
        assert tuple(map(len, (train, dev))) == sizes
        assert not {c.case_id for c in train} & {c.case_id for c in dev}
        assert condition.config().generations == 16
        assert job["budget"] == (50 if benchmark == "made" else 1)
        if benchmark == "crystalgym":
            assert {c.payload["task"]["prototype"]["id"] for c in train + dev} == {"C2", "C3", "C4", "C5", "C6"}


def test_seed_and_case_do_not_depend_on_candidate_or_perturbation(project):
    condition, policy, request, job = setup_request(project, "crystalgym")
    entry = {"model_id": policy.model_id, "revision": policy.revision}
    other = make_training_job(condition, replace(request, candidate_index=7, perturbation_seed=999, request_id="b"*64), model_entry=entry, policy_configuration_fingerprint="decoder")
    assert job["environment_seeds"] == other["environment_seeds"]
    assert job["seed"] == other["seed"] and job["task"] == other["task"]
    _, _, _, dev = setup_request(project, "crystalgym", "dev")
    assert dev["budget"] == 2 and dev["environment_seeds"][1] == dev["environment_seeds"][0] + 1
    assert not set(dev["environment_seeds"]) & set(job["environment_seeds"])


def test_forbid_heldout_and_changed_partitions(project):
    condition, policy, request, _ = setup_request(project, "crystalgym")
    payload = copy.deepcopy(request.case.payload)
    payload["task"]["prototype"].update(id="C1", index=3403, split="heldout")
    invalid = replace(request, case=replace(request.case, payload=payload))
    with pytest.raises(TrainingContractError, match="catalog"):
        make_training_job(condition, invalid, model_entry={"model_id": policy.model_id, "revision": policy.revision}, policy_configuration_fingerprint="decoder")
    with pytest.raises(TrainingContractError, match="test"):
        make_training_job(condition, replace(request, phase="test"), model_entry={}, policy_configuration_fingerprint="decoder")
    path = project / "configs/made_splits.json"
    content = json.loads(path.read_text())
    content["splits"]["train"][0] = content["splits"]["test"][0]
    dump(path, content)
    with pytest.raises(TrainingContractError, match="Held-out"):
        TrainingCondition(project, "qwen35_4b", "made", "esopt", 1).cases()


@pytest.mark.parametrize("failed_last", [False, True])
def test_made_official_attempt_curve_not_model_self_score(project, tmp_path, failed_last):
    _, _, _, job = setup_request(project)
    output = tmp_path / "rollout"
    write_mock_scientific_trajectory(project, job, output, failed_last_made=failed_last)
    tasks = json.loads((project / "configs/benchmark_tasks.json").read_text())
    proof = reconstruct_scientific_evidence(output, job, tasks=tasks)
    assert proof["metric_name"] == "AUDC" and proof["costs"]["candidate_oracle_attempts"] == 50
    assert proof["costs"]["initialization_oracle_attempts"] == 3
    summaries = json.loads((output / "episodes.json").read_text())
    summaries[0]["metrics"]["AUDC"] = .999
    dump(output / "episodes.json", summaries)
    with pytest.raises(TrainingContractError, match="AUDC"):
        reconstruct_scientific_evidence(output, job, tasks=tasks)


def test_crystal_dev_two_true_rewards_including_failure(project, tmp_path):
    _, _, _, job = setup_request(project, "crystalgym", "dev")
    output = tmp_path / "rollout"
    write_mock_scientific_trajectory(project, job, output)
    tasks = json.loads((project / "configs/benchmark_tasks.json").read_text())
    evidence = reconstruct_scientific_evidence(output, job, tasks=tasks)
    assert evidence["metric_value"] == (-1 + .3) / 2
    assert evidence["costs"]["dft_episode_attempts"] == 2
    assert evidence["environment_seeds"] == job["environment_seeds"]
    summaries = json.loads((output / "episodes.json").read_text())
    summaries[1]["environment_seed"] = summaries[0]["environment_seed"]
    dump(output / "episodes.json", summaries)
    with pytest.raises(TrainingContractError, match="seed"):
        reconstruct_scientific_evidence(output, job, tasks=tasks)


def test_callback_receipts_recovery_hashes_and_no_physical_retry(project, tmp_path):
    condition, policy, request, job = setup_request(project)
    calls = []
    class MockRollout:
        def __init__(self, project, policy, **kwargs):
            self.project = project
        def run(self, job, output, *, collection):
            assert collection is False
            calls.append(job["job_id"])
            return write_mock_scientific_trajectory(self.project, job, output)
    callbacks = TrainingJobCallbacks(condition, policy, tmp_path / "driver", settings=RolloutSettings(), rollout_factory=MockRollout)
    durable = {"request": request.to_dict()}
    assert callbacks.recover_result(request, durable) is None
    result = callbacks.evaluate(policy, job, request)
    assert callbacks.result_verifier(request, result)["verified"]
    # JSON checkpoint round-trip changes tuple paths to list; it remains equivalent.
    assert callbacks.result_verifier(request, replace(result, result_paths=list(result.result_paths)))["verified"]
    reused = callbacks.recover_result(replace(request, policy_state_id="new-replay-state"), durable)
    assert reused == result and len(calls) == 1
    with pytest.raises(TrainingHalted, match="reconciliation"):
        callbacks.evaluate(policy, job, request)
    output = callbacks._directory(request)
    (output / "rpc/rpc.jsonl").write_text("corrupt")
    with pytest.raises(TrainingContractError, match="changed"):
        callbacks.recover_result(request, durable)
    assert len(calls) == 1


def test_incomplete_rpc_and_wrong_oracle_rejected(project, tmp_path):
    _, _, _, job = setup_request(project)
    output = tmp_path / "rollout"
    write_mock_scientific_trajectory(project, job, output)
    tasks = json.loads((project / "configs/benchmark_tasks.json").read_text())
    path = output / "rpc/rpc.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(TrainingContractError, match="complete"):
        reconstruct_scientific_evidence(output, job, tasks=tasks)
    path.write_text("\n".join(lines) + "\n")
    metadata = json.loads((output / "environment_metadata.json").read_text())
    metadata["oracle"]["model"] = "mock_surrogate"
    dump(output / "environment_metadata.json", metadata)
    with pytest.raises(TrainingContractError, match="metadata"):
        reconstruct_scientific_evidence(output, job, tasks=tasks)


def test_main_cli_plan_has_full_budget_and_requires_graph_artifacts(project, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("train_es_condition_cli", PROJECT / "scripts/train_es_condition.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    arguments = ["--project", str(project), "--model-key", "qwen35_4b", "--benchmark", "made", "--method", "esopt", "--seed", "1", "--output", str(tmp_path / "runs"), "--plan-only"]
    assert module.main(arguments) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["training_evaluations"] == 128 and plan["development_evaluations"] == 48
    assert plan["expected_physical_attempts"] == 8800 and plan["train_case_count"] == 30 and plan["dev_case_count"] == 12
    assert not (tmp_path / "runs").exists()
    arguments[arguments.index("esopt")] = "esopt_graph_risk"
    with pytest.raises(SystemExit):
        module.main(arguments)


def test_cpu_reload_does_not_move_or_change_live_state(tmp_path):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="tiny_causal_lm")
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(6, 3)
            self.model.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3)])
            self.lm_head = torch.nn.Linear(3, 6)
            self.eval()
        def get_input_embeddings(self):
            return self.model.embed_tokens
        def get_output_embeddings(self):
            return self.lm_head
        def forward(self, input_ids, **kwargs):
            assert kwargs["use_cache"] is False
            return SimpleNamespace(logits=self.lm_head(self.model.layers[0](self.model.embed_tokens(input_ids))))
    model = Tiny()
    optimizer = AgenticESOpt(model, policy_model_id="fixture/tiny", parameter_scope="full")
    checkpoint = tmp_path / "policy.pt"
    optimizer.save_checkpoint(checkpoint)
    policy = SimpleNamespace(model=model, get_state_id=lambda: "unchanged")
    before = tensor_state_hash(dict(model.state_dict()))
    proof = cpu_clean_reload_validator(policy, checkpoint, before, fresh_model_factory=Tiny)
    assert proof["verified"] and proof["max_abs_logit_difference"] == 0
    assert policy.get_state_id() == "unchanged" and optimizer.model_state_hash() == before


def test_real_risk_serialization_loader_pins_benchmark_and_checksum(tmp_path):
    import numpy as np
    from matdiscovery.accounting import file_sha256
    from matdiscovery.training_jobs import load_frozen_controllers
    from matdiscovery.uncertainty import CalibratedRiskModel, RiskTrainingConfig
    features = ["sampling.entropy_mean", "sampling.mean_logprob"]
    model = CalibratedRiskModel().fit(np.array([[0., 0.], [1., 1.], [2., 0.], [3., 1.]]), np.array([0, 1, 0, 1]),
        np.array([[.5, 0.], [2.5, 1.]]), np.array([0, 1]), feature_names=features,
        train_groups=["a", "b", "c", "d"], dev_groups=["e", "f"], train_episode_ids=["a", "b", "c", "d"],
        config=RiskTrainingConfig(label_kind="future_failure", epochs=1, batch_size=2, patience=1))
    schema = {"feature_names": features}
    model.provenance.update(method="entropy_risk", feature_schema=schema,
        collection_provenance={"model_key": "qwen35_4b", "checkpoint_hash": "base", "benchmarks": ["made"],
                               "test_used_for_fit": False, "test_used_for_threshold_selection": False, "label_kind": "future_failure",
                               "policy_runtime": {"fixture": "cpu"}, "policy_configuration_fingerprint": "fixture"})
    checkpoint = tmp_path / "entropy_risk.pt"
    model.save(checkpoint)
    fit = {"status": "succeeded", "method": "entropy_risk", "checkpoint_sha256": file_sha256(checkpoint)}
    dump(checkpoint.with_suffix(".fit.json"), fit)
    dump(checkpoint.with_suffix(".schema.json"), schema)
    arguments = dict(model_key="qwen35_4b", benchmark="made", method="esopt", risk_checkpoint=checkpoint,
                     transcoder_manifest=None, graph_config={}, device="cpu")
    policy = SimpleNamespace(checkpoint_hash="base", configuration_fingerprint="fixture", runtime_precision_record=lambda: {"fixture": "cpu"})
    loaded, graph, files = load_frozen_controllers(policy, **arguments)
    assert graph is None and len(files) == 3 and all(not p.requires_grad for p in loaded.model.parameters())
    with pytest.raises(TrainingContractError, match="Risk fit"):
        load_frozen_controllers(policy, **{**arguments, "benchmark": "crystalgym"})
    fit["checkpoint_sha256"] = "0" * 64
    dump(checkpoint.with_suffix(".fit.json"), fit)
    with pytest.raises(TrainingContractError, match="SHA256"):
        load_frozen_controllers(policy, **arguments)
