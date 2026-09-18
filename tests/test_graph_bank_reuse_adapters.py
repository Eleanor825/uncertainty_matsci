"""Consumer contracts for an already-verified reuse result; no policy/oracle.

The recovery helper is replaced at its public boundary here. Its scientific
corpus-equivalence and real bank integrity tests belong to graph recovery.
"""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from matdiscovery.accounting import file_sha256, write_json_atomic
from matdiscovery.native_attribution import PolicyStamp
from matdiscovery.representation_training import GraphStageConfig, QWEN_MLP_PATHS, RepresentationError
from matdiscovery.training_jobs import TrainingContractError
from matdiscovery.transcoders import TopKTranscoder, TranscoderConfig
from torch_runtime_fixture import restore_torch_runtime

RULE = "qwen_final_mlp_causal_activation_v1"


def artifact(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    from matdiscovery import representation_training as offline, native_attribution as native
    from matdiscovery import policy as policy_module, transcoders, uncertainty, core_protocol
    old_manifest, new_manifest = tmp_path / "old_collection.json", tmp_path / "new_collection.json"
    for path, version in ((old_manifest, "old"), (new_manifest, "new")):
        write_json_atomic(path, {"unit_only": True, "derived_manifest_version": version})
    core_path = tmp_path / "new_core.json"; core_path.write_text("{}")
    contract_path = tmp_path / "reuse_contract.json"; contract_path.write_text("unit immutable reuse contract")
    source_evidence = tmp_path / "source_receipt.json"; source_evidence.write_text("unit original bank receipt")
    contract = {"core_fingerprint": "target-v4", "benchmark": "made", "model_key": "qwen35_4b",
                "protocol_path": str(core_path), "expected_collection_jobs": 1,
                "manifests": [artifact(new_manifest)]}
    runtime = {"torch_cpu_threads": 1, "unit_only": True}
    collection = {"model_key": "qwen35_4b", "model_id": "unit-model", "checkpoint_hash": "base",
        "policy_runtime": runtime, "policy_configuration_fingerprint": "cfg", "policy_configuration": {"unit": True},
        "collections": [{**artifact(new_manifest), "job_id": "collection-made-unit"}],
        "source_policy_stamps": [], "records": [{"decision_id": "unit-new-source"}]}
    tc = TopKTranscoder(TranscoderConfig(2, 2, 2, 2))
    layers, metadata_by_path = [], {}
    for index, layer_path in enumerate(QWEN_MLP_PATHS):
        checkpoint = tmp_path / f"layer_{index:02d}.pt"
        checkpoint.write_text(f"unit loader boundary {index}; not real trained weights")
        metadata = {"unit_only": True, "provenance": {"layer_path": layer_path}}
        metadata_by_path[str(checkpoint)] = metadata
        layers.append({"checkpoint": str(checkpoint), "layer_path": layer_path, "metadata": metadata,
                       "transcoder_hash": tc.checkpoint_hash()})
    bank = {key: copy.deepcopy(collection[key]) for key in
            ("model_key", "model_id", "checkpoint_hash", "policy_runtime", "policy_configuration_fingerprint")}
    bank.update(bank_fingerprint="original-v3-bank", layers=layers, test_used_for_training_or_fidelity=False,
                collections=[{**artifact(old_manifest), "job_id": "collection-made-unit"}])
    bank_path = tmp_path / "original_bank.json"; write_json_atomic(bank_path, bank)
    proof = {"source_bank_manifest": artifact(bank_path), "source_core_fingerprint": "source-v3",
        "target_core_fingerprint": "target-v4", "target_source_selection_rule": RULE,
        "contract": artifact(contract_path), "source_evidence_files": [artifact(source_evidence), artifact(old_manifest)]}
    state = SimpleNamespace(bank=bank, collection=collection, proof=proof, contract=contract,
        bank_path=bank_path, new_manifest=new_manifest, old_manifest=old_manifest,
        root=tmp_path, calls=[], reuse=True, validated=[], replayed=[], attributor_options=[])
    helper = ModuleType("matdiscovery.graph_recovery")
    def bank_for_corpus(corpus, *, manifest_paths=None, model_key, transcoder_manifest):
        assert corpus == contract and model_key == "qwen35_4b" and Path(transcoder_manifest).resolve() == bank_path
        assert [Path(path).resolve() for path in manifest_paths] == [new_manifest]
        state.calls.append("reuse_helper")
        return (state.bank, state.collection, state.proof) if state.reuse else None
    helper.bank_for_corpus = bank_for_corpus
    monkeypatch.setitem(sys.modules, "matdiscovery.graph_recovery", helper)
    def validate_corpus(corpus, *, manifest_paths=None, model_key=None):
        assert corpus == contract
        if manifest_paths is not None:
            assert [Path(path).resolve() for path in manifest_paths] == [new_manifest]
        return contract
    monkeypatch.setattr(core_protocol, "validate_corpus_contract", validate_corpus)
    def standard_bank(*args, **kwargs):
        state.validated.append((args, kwargs)); return bank
    monkeypatch.setattr(offline, "validate_transcoder_bank", standard_bank)
    monkeypatch.setattr(offline, "read_collections", lambda *a, **kw: collection)
    monkeypatch.setattr(offline, "_load_resume", lambda *a, **kw: {})
    def replay(records, *args, **kwargs):
        state.replayed.append((records, kwargs)); return {"complete": True, "unit_only": True}
    monkeypatch.setattr(offline, "_replay_records", replay)
    def attributor(*args, **kwargs):
        state.attributor_options.append(kwargs)
        return SimpleNamespace(runtime_metadata=lambda: {"unit_only": True})
    monkeypatch.setattr(native, "NativeAttributor", attributor)
    stamp = PolicyStamp("unit", "base", "unit-model", 0)
    policy = SimpleNamespace(model=torch.nn.Linear(2, 2), model_stamp=stamp, checkpoint_hash="base", model_id="unit-model",
        mlp_paths=QWEN_MLP_PATHS, architecture_review={"unit_only": True}, configuration_fingerprint="cfg",
        get_state_id=lambda: "unit", runtime_precision_record=lambda: runtime)
    state.policy = policy
    monkeypatch.setattr(policy_module.QwenPolicyAdapter, "from_verified_checkpoint", lambda *a, **kw: policy)
    monkeypatch.setattr(transcoders, "load_transcoder", lambda path, **kw: (tc, metadata_by_path[str(path)]))
    risk_path = tmp_path / "graph_risk.pt"; risk_path.write_text("unit fitted loader boundary")
    schema = {"feature_names": ["unit-feature"]}
    write_json_atomic(risk_path.with_suffix(".fit.json"), {"status": "succeeded", "method": "graph_risk", "checkpoint_sha256": file_sha256(risk_path)})
    write_json_atomic(risk_path.with_suffix(".schema.json"), schema)
    risk_collection = {"model_key": "qwen35_4b", "benchmarks": ["made"], "policy_runtime": runtime,
        "policy_configuration_fingerprint": "cfg", "checkpoint_hash": "base", "test_used_for_fit": False,
        "test_used_for_threshold_selection": False, "label_kind": "future_failure",
        "collection_manifests": [{"path": str(new_manifest), "content": json.loads(new_manifest.read_text())}],
        "source_files": [artifact(new_manifest)]}
    risk = SimpleNamespace(provenance={"method": "graph_risk", "collection_provenance": risk_collection,
        "test_used_for_fit": False, "feature_schema": schema}, features=SimpleNamespace(names=["unit-feature"]), model=torch.nn.Linear(1, 1))
    state.risk, state.risk_path = risk, risk_path
    monkeypatch.setattr(uncertainty.CalibratedRiskModel, "load", lambda *a, **kw: risk)
    return state


def run_offline(case, *, rule=RULE):
    from matdiscovery.representation_training import generate_graph_features
    return generate_graph_features([case.new_manifest], model_key="qwen35_4b", model_manifest=case.root / "models.json",
        checkpoint_dir=case.root, transcoder_manifest=case.bank_path, report_path=case.root / "report.json",
        config=GraphStageConfig(device="cpu", source_selection_rule=rule), corpus_contract=case.contract)


def run_online(case, *, rule=RULE):
    from matdiscovery.training_jobs import load_frozen_controllers
    return load_frozen_controllers(case.policy, model_key="qwen35_4b", benchmark="made", method="esopt_graph_risk",
        risk_checkpoint=case.risk_path, transcoder_manifest=case.bank_path, device="cpu",
        graph_config=asdict(GraphStageConfig(device="cpu", source_selection_rule=rule)), corpus_contract=case.contract)


def test_offline_reuses_original_bank_binds_proof_and_replays_only_new_corpus(boundary):
    before = (file_sha256(boundary.bank_path), boundary.bank_path.stat().st_mtime_ns)
    assert run_offline(boundary)["complete"]
    assert not boundary.validated and boundary.calls == ["reuse_helper"]
    saved = json.loads((boundary.root / "report.provenance.json").read_text())
    assert saved["pipeline"]["passed_bank_reuse"] == boundary.proof
    assert saved["pipeline"]["transcoder_bank_fingerprint"] == "original-v3-bank"
    assert saved["collections"] == boundary.collection["collections"] != boundary.bank["collections"]
    assert boundary.replayed[0][0] == boundary.collection["records"]
    assert before == (file_sha256(boundary.bank_path), boundary.bank_path.stat().st_mtime_ns)


def test_online_reuse_preserves_risk_gates_and_includes_both_source_lineages(boundary):
    risk, attributor, files = run_online(boundary)
    assert risk is boundary.risk and not boundary.validated
    needed = [boundary.new_manifest, boundary.old_manifest, boundary.bank_path,
              Path(boundary.proof["contract"]["path"]), Path(boundary.proof["source_evidence_files"][0]["path"])]
    assert set(needed) <= set(files)
    assert boundary.attributor_options[-1]["source_selection_rule"] == RULE
    boundary.risk.provenance["collection_provenance"]["test_used_for_fit"] = True
    with pytest.raises(TrainingContractError, match="never test"):
        run_online(boundary)


@pytest.mark.parametrize("consumer", [run_offline, run_online])
def test_reuse_rejects_wrong_rule_or_changed_proof_evidence_before_attributor(boundary, consumer):
    with pytest.raises(RepresentationError, match="source-selection rule"):
        consumer(boundary, rule="global_activation")
    assert not boundary.attributor_options
    Path(boundary.proof["contract"]["path"]).write_text("changed")
    with pytest.raises(RepresentationError, match="evidence changed"):
        consumer(boundary)
    assert not boundary.attributor_options


def test_reuse_cannot_replace_policy_or_target_corpus_identity(boundary):
    boundary.collection["checkpoint_hash"] = "other-base"
    with pytest.raises(RepresentationError, match="checkpoint/runtime/configuration"):
        run_offline(boundary)
    boundary.collection["checkpoint_hash"] = "base"
    boundary.proof["target_core_fingerprint"] = "other-core"
    with pytest.raises(RepresentationError, match="target core"):
        run_online(boundary)


def test_ordinary_scope_still_uses_original_validation_and_corpus_match(boundary):
    boundary.reuse = False
    assert run_offline(boundary)["complete"] and len(boundary.validated) == 1
    saved = json.loads((boundary.root / "report.provenance.json").read_text())
    assert "passed_bank_reuse" not in saved["pipeline"]
    # A non-reuse scope must still reject old bank manifest paths against the
    # new corpus; it cannot inherit the new registered mapping exception.
    with pytest.raises(AssertionError):
        run_online(boundary)
    assert len(boundary.validated) == 2
