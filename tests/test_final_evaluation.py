"""Final-matrix integration contracts using synthetic evidence, never physics."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from matdiscovery.accounting import RunLedger, build_final_manifest, file_sha256, fingerprint, write_json_atomic
from matdiscovery.es_training import TrainingContractError, TrainingHalted
from matdiscovery.final_evaluation import FinalEvaluationRunner, execution_job, select_final_jobs, verify_final_envelope
from matdiscovery.training_jobs import TrainingJobCallbacks, reconstruct_scientific_evidence
from test_training_jobs import PROJECT, project, write_mock_scientific_trajectory


class TinyPolicy:
    def __init__(self, entry):
        self.model = torch.nn.Linear(2, 2)
        torch.nn.init.constant_(self.model.weight, .1)
        torch.nn.init.constant_(self.model.bias, 0)
        self.model.eval()
        self.model_id, self.revision = entry["model_id"], entry["revision"]
        self.checkpoint_hash, self.configuration_fingerprint = "base-hash", "decoder"
        self.counter = 0
        self.model_stamp = SimpleNamespace(generation=0)
    def get_state_id(self):
        return f"state-{id(self)}-{self.counter}"
    def mark_state(self, reason, *, generation):
        self.counter += 1
        self.model_stamp = SimpleNamespace(generation=generation)
        return self.model_stamp


def make_runner(project, tmp_path, *, calls, fail_after_complete=False):
    (project / "scripts/run_final_evaluation.py").write_text("# synthetic final fixture\n")
    entry = json.loads((project / "configs/model_manifest.json").read_text())["models"][0]
    def controller(policy, **kwargs):
        assert kwargs["risk_checkpoint"] == tmp_path / "risk" / kwargs["model_key"] / kwargs["benchmark"] / "entropy_risk.pt"
        assert kwargs["transcoder_manifest"] is None
        return None, None, []
    class MockRollout:
        def __init__(self, root, policy, **kwargs):
            self.project = root
        def run(self, job, output, *, collection):
            calls.append(job["job_id"])
            summaries = write_mock_scientific_trajectory(self.project, job, output)
            if fail_after_complete:
                raise RuntimeError("synthetic uncertain transport after closed transcript")
            return summaries
    return FinalEvaluationRunner(project, tmp_path / "final", risk_root=tmp_path / "risk", transcoder_root=tmp_path / "tc", es_root=tmp_path / "es",
        device="cpu", policy_factory=lambda key: TinyPolicy(entry), rollout_factory=MockRollout, controller_loader=controller)


def test_filters_never_reduce_full_manifest(project):
    manifest = build_final_manifest(project)
    assert len(manifest["jobs"]) == 2160
    assert len(select_final_jobs(manifest, model_key="qwen35_4b", benchmark="made")) == 900
    assert len(select_final_jobs(manifest, model_key="qwen35_4b", benchmark="crystalgym")) == 180
    jobs = select_final_jobs(manifest, model_key="qwen35_4b", benchmark="crystalgym", property_name="density", method="baseline", seed=1)
    assert len(jobs) == 2 and sum(j["expected_counts"]["episodes"] for j in jobs) == 10
    assert {j["task"]["prototype"]["id"] for j in jobs} == {"C1", "C7"}
    assert manifest["expected_counts"]["jobs"] == 2160
    with pytest.raises(TrainingContractError):
        select_final_jobs(manifest, model_key="qwen35_4b", benchmark="made", property_name="density")


def test_complete_made_batch_and_verified_resume(project, tmp_path):
    calls = []
    runner = make_runner(project, tmp_path, calls=calls)
    filters = dict(model_key="qwen35_4b", benchmark="made", method="baseline", seed=1)
    summary = runner.run(**filters)
    assert len(calls) == 30 and summary["batch_complete"] is True and summary["complete"] is False
    assert summary["expected_counts"]["jobs"] == 2160
    assert summary["verified_counts"]["candidate_oracle_attempts"] == 1500
    resumed = runner.run(**filters, resume=True)
    assert len(calls) == 30 and resumed["batch_completed_jobs"] == 30
    path = next((tmp_path / "final/jobs").glob("*/*/episodes.json"))
    path.write_text("[]")
    with pytest.raises((TrainingContractError, ValueError), match="hash"):
        runner.run(**filters, resume=True)
    assert len(calls) == 30


def test_crystal_final_requires_both_heldout_and_five_real_attempts_each(project, tmp_path):
    calls = []
    runner = make_runner(project, tmp_path, calls=calls)
    summary = runner.run(model_key="qwen35_4b", benchmark="crystalgym", property_name="density", method="baseline", seed=1)
    assert len(calls) == 2 and summary["verified_counts"]["dft_episode_attempts"] == 10
    assert summary["failed_scientific_episodes"] == 2
    assert summary["complete"] is False


def test_failed_physical_batch_is_not_automatically_replayed(project, tmp_path):
    calls = []
    runner = make_runner(project, tmp_path, calls=calls, fail_after_complete=True)
    filters = dict(model_key="qwen35_4b", benchmark="made", method="baseline", seed=1)
    with pytest.raises(RuntimeError, match="synthetic"):
        runner.run(**filters)
    assert len(calls) == 1
    with pytest.raises(TrainingHalted, match="reconciliation"):
        runner.run(**filters, resume=True)
    assert len(calls) == 1
    with RunLedger(tmp_path / "final/ledger.sqlite3", runner.manifest) as ledger:
        assert ledger.get(calls[0])["state"] == "failed"


def test_finished_envelope_survives_crash_before_ledger_success(project, tmp_path, monkeypatch):
    calls = []
    runner = make_runner(project, tmp_path, calls=calls)
    filters = dict(model_key="qwen35_4b", benchmark="made", method="baseline", seed=1)
    original = RunLedger.succeed
    def interrupted(self, *args):
        raise KeyboardInterrupt("mock kill before ledger commit")
    monkeypatch.setattr(RunLedger, "succeed", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.run(**filters)
    # The complete envelope is reconciled read-only; only the remaining 29
    # pending cases execute. The completed physical case is never repeated.
    monkeypatch.setattr(RunLedger, "succeed", original)
    summary = runner.run(**filters, resume=True)
    assert summary["batch_complete"] and len(calls) == len(set(calls)) == 30


def test_final_cli_plan_counts_without_loading_models(project, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("final_cli", PROJECT / "scripts/run_final_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--project", str(project), "--model-key", "qwen35_4b", "--benchmark", "made", "--output", str(tmp_path / "final"),
                       "--risk-root", str(tmp_path / "risk"), "--transcoder-root", str(tmp_path / "tc"), "--es-root", str(tmp_path / "es"), "--plan-only"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["whole_study_expected_counts"]["jobs"] == 2160 and plan["batch_expected_jobs"] == 900
    assert plan["batch_expected_counts"]["candidate_oracle_attempts"] == 45000
    assert not (tmp_path / "final").exists()


def test_es_selected_loader_requires_all_generations_and_uses_actual_best_hash(project, tmp_path):
    import copy
    from matdiscovery.checkpoint_retention import retire_completed_es_checkpoints
    from matdiscovery.es_training import ESTrainingDriver
    from matdiscovery.final_evaluation import selected_es_artifact
    from matdiscovery.training_jobs import TrainingCondition
    from test_es_training import TinyHF, TinyAdapter, MockOfficialCallbacks
    condition = TrainingCondition(project, "qwen35_4b", "made", "esopt", 1)
    train, dev = condition.cases()
    base = TinyHF()
    policy = TinyAdapter(copy.deepcopy(base))
    callbacks = MockOfficialCallbacks(tmp_path / "synthetic_evidence", base)
    root = tmp_path / "es"
    directory = condition.output_directory(root)
    driver = ESTrainingDriver(policy, condition.config(), train_cases=train, dev_cases=dev,
        job_factory=callbacks.job_factory, evaluate=callbacks.evaluate, result_verifier=callbacks.verify,
        clean_reload_validator=callbacks.reload, callback_fingerprint="synthetic-full-schedule-only", output_dir=directory)
    callbacks.driver = driver
    summary = driver.run()
    retire_completed_es_checkpoints(directory)
    loaded = TinyAdapter(copy.deepcopy(base))
    profile, files = selected_es_artifact(loaded, condition, root)
    assert profile["selected_generation"] == 4 and loaded.model_stamp.generation == 4
    assert profile["actual_model_state_hash"] == summary["best_actual_model_state_hash"]
    assert profile["actual_model_state_hash"] != loaded.checkpoint_hash
    assert (directory / "checkpoint_retirement.json") in files
    marker = directory / "checkpoints/generation_0015/complete.json"
    marker.unlink()
    with pytest.raises(FileNotFoundError):
        selected_es_artifact(TinyAdapter(copy.deepcopy(base)), condition, root)


def test_final_graph_must_match_current_policy_state_and_complete_prefix(project, tmp_path):
    from matdiscovery.native_attribution import CONTRACT
    manifest = build_final_manifest(project)
    job = select_final_jobs(manifest, model_key="qwen35_4b", benchmark="made", method="graph_risk", seed=1)[0]
    entry = json.loads((project / "configs/model_manifest.json").read_text())["models"][0]
    policy = TinyPolicy(entry)
    from matdiscovery.execution_contract import declared_mace_workers
    profile = {"actual_model_state_hash": "test-only-current-weight-hash", "artifacts": {},
               "mace_num_workers": declared_mace_workers(json.loads((project / "configs/main_protocol.json").read_text()))}
    current = execution_job(job, policy, profile)
    output = tmp_path / "graph_trajectory"
    write_mock_scientific_trajectory(project, current, output)
    path = output / "decisions.jsonl"
    row = json.loads(path.read_text())
    row.update(prefix_hash="complete-prefix", graph_status="succeeded", graph_contract=CONTRACT,
               graph_metadata={"policy": {"state_id": policy.get_state_id(), "generation": 0}, "full_prefix_hash": "complete-prefix"})
    path.write_text(json.dumps(row) + "\n")
    tasks = json.loads((project / "configs/benchmark_tasks.json").read_text())
    evidence = reconstruct_scientific_evidence(output, current, tasks=tasks, partition="final_test")
    envelope = {"job_id": job["job_id"], "manifest_fingerprint": manifest["manifest_fingerprint"], "model_revision": job["model_revision"],
        "complete": True, "episodes": json.loads((output / "episodes.json").read_text()), "artifacts": TrainingJobCallbacks._inventory(output),
        "evidence_schema": "final_official_rpc_v1", "scientific_evidence": evidence, "execution_job": current,
        "execution_profile": profile, "execution_profile_fingerprint": fingerprint(profile)}
    result = output / "result.json"
    write_json_atomic(result, envelope)
    verify_final_envelope(result, job, manifest["manifest_fingerprint"], tasks=tasks)
    row["graph_metadata"]["policy"]["state_id"] = "stale-state"
    path.write_text(json.dumps(row) + "\n")
    envelope["artifacts"] = [artifact for artifact in TrainingJobCallbacks._inventory(output) if artifact["path"] != str(result)]
    write_json_atomic(result, envelope)
    with pytest.raises(TrainingContractError, match="attributed"):
        verify_final_envelope(result, job, manifest["manifest_fingerprint"], tasks=tasks)
