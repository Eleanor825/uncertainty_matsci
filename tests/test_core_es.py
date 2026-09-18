"""Tiny CPU contracts only; synthetic RPC evidence is not a material experiment."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery import core_protocol
from matdiscovery.core_es import (
    CoreTrainingCondition, CoreTrainingJobCallbacks, _descriptor_path,
    audit_core_es_costs, finalize_core_training, load_core_selected_es, run_core_es,
)
from matdiscovery.es_training import ESTrainingDriver, TrainingContractError, verify_clean_reload_checkpoint
from matdiscovery.esopt import tensor_state_hash
from matdiscovery.rollouts import RolloutSettings
from matdiscovery.training_jobs import TrainingCondition
from test_es_training import TinyHF, TinyAdapter
from test_training_jobs import project as full_project_fixture, write_mock_scientific_trajectory


@pytest.fixture
def setup_core(full_project_fixture, monkeypatch):
    root = full_project_fixture
    main_path = root / "configs/main_protocol.json"
    main = json.loads(main_path.read_text())
    main.update(models=["qwen35_4b"], training_seeds=[1])
    main["esopt"].update(generations=2, population=2, cases_per_generation=1, validation_every_generations=1)
    main["graph"].update(max_feature_nodes=32, max_backward_targets=42)
    main["made_execution"] = {"mace_num_workers": 4, "orb_num_workers": 1}
    write_json_atomic(main_path, main)
    entry = {"key": "qwen35_4b", "model_id": TinyAdapter.model_id, "revision": TinyAdapter.revision}
    write_json_atomic(root / "configs/model_manifest.json", {"models": [entry]})
    tasks = lambda *values: [{"id": "-".join(elements), "elements": list(elements)} for elements in values]
    core = {"workspace": str(root), "parent_project": str(root.parent), "fingerprint": "fixture-core-fingerprint",
        "study_id": core_protocol.REGISTRATION["study_id"], "registration": copy.deepcopy(core_protocol.REGISTRATION),
        "transcoder": copy.deepcopy(main["transcoder"]), "model_key": "qwen35_4b", "model": entry, "training_seed": 1,
        "budget": 50, "methods": ["baseline", "esopt_graph_risk"],
        "train_tasks": tasks(("Al", "Au", "Hf")), "dev_tasks": tasks(("Al", "Pd", "Sm")),
        "test_tasks": tasks(("Al", "Li", "V"), ("Al", "V", "Zn")), "esopt": main["esopt"],
        "derived_main_protocol": {"path": str(main_path), "sha256": file_sha256(main_path)}, "imported_train": []}
    # The protocol module's own tests cover signed workspace validation. Here its
    # read boundary provides a tiny catalog so ES arithmetic/RPC/reload contracts
    # can run with no actual collection, asset download or physical oracle.
    monkeypatch.setattr(core_protocol, "read_core", lambda project: copy.deepcopy(core))
    for seed in (1, 2, 3):
        path = root / f"imported/seed{seed}/collection_manifest.json"
        job = {"job_id": f"collection-made-unit-train-{seed}", "split": "train", "seed": seed}
        write_json_atomic(path, {"complete": True, "job_id": job["job_id"]})
        core["imported_train"].append({"job": job, "manifest": {"path": str(path), "sha256": file_sha256(path)}})
    for path, job in zip(core_protocol.collection_manifest_paths(core), core_protocol.collection_jobs(core), strict=True):
        write_json_atomic(path, {"complete": True, "job_id": job["job_id"]})
    write_json_atomic(root / "configs/deadline_core_protocol.json", {"unit_fixture_only": True})
    return core


def train_fixture(core, *, zero_update=False, missing_graphs=False):
    condition = CoreTrainingCondition(Path(core["workspace"]))
    torch.manual_seed(821)
    base = TinyHF()
    policy = TinyAdapter(copy.deepcopy(base))
    directory = condition.output_directory()
    requests = []
    class SyntheticRollout:
        def __init__(self, project, live_policy, **kwargs):
            self.project = project
        def run(self, job, output, collection=False):
            assert collection is False
            request = job["training_request"]
            requests.append(request)
            failed = not zero_update and ((request["phase"] == "train" and request["candidate_index"] == 0)
                                         or (request["phase"] == "dev" and request["generation"] == 2))
            write_mock_scientific_trajectory(self.project, job, output, failed_last_made=failed)
            path = output / "decisions.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            from matdiscovery.native_attribution import CONTRACT
            for row in rows:
                row["prefix_hash"] = "synthetic-prefix"
                row["graph_status"] = "unavailable" if missing_graphs else "succeeded"
                row["graph_error"] = "unit missing graph" if missing_graphs else None
                row["graph_contract"] = CONTRACT
                row["graph_metadata"] = {"unit_test_only": True, "full_prefix_hash": row["prefix_hash"],
                    "policy": {"state_id": request["policy_state_id"],
                               "generation": request["generation"] - int(request["phase"] == "train")}}
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    contract = core_protocol.corpus_contract(core, core_protocol.collection_manifest_paths(core))
    callbacks = CoreTrainingJobCallbacks(condition, policy, directory, corpus_contract=contract,
        settings=RolloutSettings(), risk_model=object(), attributor=object(), rollout_factory=SyntheticRollout)
    descriptor = {"core_fingerprint": core["fingerprint"], "callback_identity": callbacks.callback_identity,
                  "callback_fingerprint": callbacks.fingerprint}
    write_json_atomic(_descriptor_path(directory), descriptor)
    def clean_reload(policy, checkpoint, expected):
        return verify_clean_reload_checkpoint(policy, checkpoint, expected,
            fresh_policy_factory=lambda: copy.deepcopy(base), input_ids=torch.tensor([[1, 2, 3]]))
    train, dev = condition.cases()
    driver = ESTrainingDriver(policy, condition.config(), train_cases=train, dev_cases=dev,
        job_factory=callbacks.job_factory, evaluate=callbacks.evaluate, result_verifier=callbacks.result_verifier,
        recover_result=callbacks.recover_result, clean_reload_validator=clean_reload,
        output_dir=directory, callback_fingerprint=callbacks.fingerprint)
    return driver, callbacks, policy, base, requests, clean_reload


def test_core_catalog_plan_has_two_complete_generations_and_original_gate_stays_strict(setup_core):
    core = setup_core
    plan = run_core_es(core["workspace"], plan_only=True)
    assert plan["training_evaluations"] == 4 and plan["development_evaluations"] == 2
    assert plan["candidate_oracle_attempts"] == 300
    assert plan["config"]["generations"] == plan["config"]["population"] == 2
    assert not Path(plan["output_directory"]).exists()
    with pytest.raises(TrainingContractError, match="complete model/seed matrix"):
        TrainingCondition(Path(core["workspace"]), "qwen35_4b", "made", "esopt_graph_risk", 1).protocol()
    core["dev_tasks"] = core["train_tasks"]
    with pytest.raises(TrainingContractError, match="different dev"):
        CoreTrainingCondition(Path(core["workspace"])).cases()


def test_full_parameter_g2_p2_science_reconstruction_selection_and_clean_reload(setup_core):
    driver, callbacks, policy, base, requests, reload = train_fixture(setup_core)
    summary = driver.run()
    assert len(requests) == 6 and summary["best_generation"] == 1
    assert summary["actual_evaluator_costs"]["candidate_oracle_attempts"] == 300
    assert not torch.equal(base.model.visual.weight, policy.model.model.visual.weight)
    profile = finalize_core_training(policy, driver.output_dir, core_protocol=setup_core, clean_reload_validator=reload)
    assert profile["actual_model_state_hash"] != profile["initial_actual_model_state_hash"]
    assert profile["selected_generation"] == 1 and profile["selected_clean_reload"]["max_abs_logit_difference"] == 0
    costs = audit_core_es_costs(driver.output_dir, setup_core)
    assert costs["training"]["job_count"] == 4 and costs["training"]["costs"]["candidate_oracle_attempts"] == 200
    assert costs["development"]["job_count"] == 2 and costs["development"]["costs"]["candidate_oracle_attempts"] == 100
    assert (Path(setup_core["workspace"]) / "experiments/core_stage_receipts/esopt.json").is_file()
    fresh = TinyAdapter(copy.deepcopy(base))
    before = len(requests)
    loaded = load_core_selected_es(fresh, driver.output_dir, core_protocol=setup_core)
    assert loaded == profile and len(requests) == before
    assert tensor_state_hash(dict(fresh.model.state_dict())) == profile["actual_model_state_hash"]
    for generation in (1, 2):
        cohort = [r for r in requests if r["phase"] == "train" and r["generation"] == generation]
        assert len({r["environment_seed"] for r in cohort}) == 1
        assert len({r["actual_model_state_hash"] for r in cohort}) == 2


def test_zero_reward_difference_cannot_claim_nonzero_selected_policy(setup_core):
    driver, _, policy, _, _, reload = train_fixture(setup_core, zero_update=True)
    driver.run()
    with pytest.raises(TrainingContractError, match="nonzero"):
        finalize_core_training(policy, driver.output_dir, core_protocol=setup_core, clean_reload_validator=reload)
    assert not (driver.output_dir / "core_training_receipt.json").exists()


def test_core_rejects_all_missing_graph_method_instead_of_claiming_closed_loop(setup_core):
    driver, *_ = train_fixture(setup_core, missing_graphs=True)
    from matdiscovery.es_training import TrainingHalted
    with pytest.raises(TrainingHalted, match="without retry"):
        driver.run()
    assert not (driver.output_dir / "training_summary.json").exists()


def test_loader_rejects_tampered_science_and_other_scope_without_replay(setup_core):
    driver, _, policy, _, requests, reload = train_fixture(setup_core)
    driver.run()
    finalize_core_training(policy, driver.output_dir, core_protocol=setup_core, clean_reload_validator=reload)
    before = len(requests)
    with pytest.raises(TrainingContractError, match="another model/method/seed"):
        load_core_selected_es(policy, driver.output_dir, core_protocol=setup_core, seed=2)
    evidence = next((driver.output_dir / "rollouts").glob("*/episodes.json"))
    data = json.loads(evidence.read_text())
    data[0]["costs"]["candidate_oracle_attempts"] = 49
    evidence.write_text(json.dumps(data))
    with pytest.raises(TrainingContractError):
        load_core_selected_es(policy, driver.output_dir, core_protocol=setup_core)
    assert len(requests) == before


def test_committed_receipt_recovers_missing_stage_marker_without_trajectory_replay(setup_core):
    driver, _, policy, _, requests, reload = train_fixture(setup_core)
    driver.run()
    expected = finalize_core_training(policy, driver.output_dir, core_protocol=setup_core, clean_reload_validator=reload)
    stage = Path(setup_core["workspace"]) / "experiments/core_stage_receipts/esopt.json"
    old_bytes = stage.read_bytes()
    stage.unlink()
    got = finalize_core_training(policy, driver.output_dir, core_protocol=setup_core,
                                 clean_reload_validator=lambda *a: pytest.fail("must not rerun a committed proof"))
    assert got == expected and stage.read_bytes() == old_bytes and len(requests) == 6


def test_corrupt_checkpoint_and_loosened_reload_proof_never_load(setup_core):
    driver, _, policy, _, _, reload = train_fixture(setup_core)
    driver.run()
    profile = finalize_core_training(policy, driver.output_dir, core_protocol=setup_core, clean_reload_validator=reload)
    receipt_path = driver.output_dir / "core_training_receipt.json"
    original = receipt_path.read_bytes()
    receipt = json.loads(original)
    receipt["selected_clean_reload"]["rtol"] = .5
    receipt["fingerprint"] = fingerprint({k: v for k, v in receipt.items() if k != "fingerprint"})
    write_json_atomic(receipt_path, receipt)
    with pytest.raises(TrainingContractError, match="fixed-tolerance"):
        load_core_selected_es(policy, driver.output_dir, core_protocol=setup_core)
    receipt_path.write_bytes(original)
    with Path(profile["checkpoint_path"]).open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(TrainingContractError, match="missing or changed"):
        load_core_selected_es(policy, driver.output_dir, core_protocol=setup_core)


def test_full_16_generation_selected_loader_cannot_accept_core_receipt(setup_core):
    from matdiscovery.final_evaluation import selected_es_artifact
    driver, _, policy, _, _, _ = train_fixture(setup_core)
    driver.run()
    condition = CoreTrainingCondition(Path(setup_core["workspace"]))
    with pytest.raises(TrainingContractError, match="16-generation"):
        selected_es_artifact(policy, condition, driver.output_dir)


def test_core_controller_exact_corpus_and_typed_inventory_never_satisfy_default210(setup_core):
    import numpy as np
    from matdiscovery.failure_risk import HEADS, FailureAwareRiskModel, FailureRiskTrainingConfig
    from matdiscovery.training_jobs import load_frozen_controllers
    from matdiscovery.uncertainty import CalibratedRiskModel, RiskTrainingConfig
    root = Path(setup_core["workspace"])
    contract = core_protocol.corpus_contract(setup_core, core_protocol.collection_manifest_paths(setup_core))
    output = root / "risk_fixture"
    output.mkdir()
    raw_source = output / "raw_features.json"
    write_json_atomic(raw_source, {"unit_only": True})
    runtime = {"unit_only": "cpu-float32"}
    configuration = {"policy_runtime": runtime, "enable_thinking": False}
    cfg_hash = fingerprint(configuration)
    names = ["sampling.entropy_mean", "sampling.mean_logprob"]
    x = np.array([[0., 0.], [1., 1.], [2., 0.], [3., 1.]])
    dx, labels = np.array([[.5, 0.], [2.5, 1.]]), np.array([0, 1, 0, 1])
    source = {"model_key": "qwen35_4b", "checkpoint_hash": "unit-base", "benchmarks": ["made"],
        "test_used_for_fit": False, "test_used_for_threshold_selection": False, "label_kind": "future_failure",
        "policy_runtime": runtime, "policy_configuration": configuration, "policy_configuration_fingerprint": cfg_hash,
        "source_files": [{"path": str(raw_source), "sha256": file_sha256(raw_source)}],
        "collection_manifests": [{"path": item["path"], "content": json.loads(Path(item["path"]).read_text())}
                                 for item in contract["manifests"]]}
    source["dataset_fingerprint"] = fingerprint(source)
    primary = CalibratedRiskModel().fit(x, labels, dx, labels[:2], feature_names=names,
        train_groups=["t1", "t2", "t3", "t4"], dev_groups=["d1", "d2"], train_episode_ids=["t1", "t2", "t3", "t4"],
        config=RiskTrainingConfig(label_kind="future_failure", epochs=1, batch_size=2, patience=1))
    primary.provenance.update(method="entropy_risk", feature_schema={"feature_names": names}, collection_provenance=source)
    checkpoint = output / "entropy_risk.pt"
    primary.save(checkpoint)
    write_json_atomic(checkpoint.with_suffix(".fit.json"), {"status": "succeeded", "method": "entropy_risk", "checkpoint_sha256": file_sha256(checkpoint)})
    write_json_atomic(checkpoint.with_suffix(".schema.json"), {"feature_names": names})
    sidecars = []
    for i, entry in enumerate(contract["manifests"]):
        path = output / f"labels_{i}.json"
        write_json_atomic(path, {"unit_only_label_sidecar": True, "source_manifest": entry})
        sidecars.append({"path": str(path), "sha256": file_sha256(path),
                         "source_manifest": {"path": entry["path"], "sha256": entry["sha256"]}})
    inventory = {"model_key": "qwen35_4b", "heads": list(HEADS), "dataset_fingerprint": source["dataset_fingerprint"],
                 "sources": sidecars, "expected_collection_jobs": 4, "corpus_contract": contract}
    inventory["fingerprint"] = fingerprint(inventory)
    inventory_path = output / "failure_label_inventory.json"
    write_json_atomic(inventory_path, inventory)
    typed = FailureAwareRiskModel.fit(primary, x, np.tile(labels[:, None], (1, len(HEADS))),
        dx, np.tile(labels[:2, None], (1, len(HEADS))), feature_names=names,
        train_groups=["t1", "t2", "t3", "t4"], dev_groups=["d1", "d2"],
        train_episode_ids=["t1", "t2", "t3", "t4"], dev_episode_ids=["d1", "d2"],
        primary_checkpoint=checkpoint, expected_primary_sha256=file_sha256(checkpoint),
        expected_policy_runtime=runtime, provenance=source,
        config=FailureRiskTrainingConfig(epochs=1, batch_size=2, patience=1))
    typed.failure_provenance["label_inventory"] = inventory
    saved = typed.save(output / "entropy_risk.failure_heads.pt")
    write_json_atomic(output / "entropy_risk.failure_heads.fit.json", {**saved,
        "status": "succeeded", "method": "entropy_risk", "heads": list(HEADS), "source_provenance": source,
        "primary_checkpoint_sha256": file_sha256(checkpoint),
        "label_inventory": {"path": str(inventory_path), "sha256": file_sha256(inventory_path)}})
    policy = SimpleNamespace(checkpoint_hash="unit-base", configuration_fingerprint=cfg_hash,
                             runtime_precision_record=lambda: runtime)
    arguments = dict(model_key="qwen35_4b", benchmark="made", method="baseline", risk_checkpoint=checkpoint,
        transcoder_manifest=None, graph_config={}, device="cpu", failure_control={"enabled_benchmarks": ["made"]})
    risk, graph, _ = load_frozen_controllers(policy, **arguments, corpus_contract=contract)
    assert isinstance(risk, FailureAwareRiskModel) and graph is None
    with pytest.raises(TrainingContractError, match="sidecars are incomplete"):
        load_frozen_controllers(policy, **arguments)
    with pytest.raises(ValueError, match="modified"):
        load_frozen_controllers(policy, **arguments, corpus_contract={**contract, "fingerprint": "bad"})
    raw_source.write_text('{"unit_only": "tampered"}')
    with pytest.raises(TrainingContractError, match="source data changed"):
        load_frozen_controllers(policy, **arguments, corpus_contract=contract)
