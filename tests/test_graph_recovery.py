"""Synthetic registration/FD receipts and real tiny CPU bank reuse contracts.

No fixture here is evidence of a real policy FD pass or material evaluation.
Ancestral physical/protocol admission is isolated; CORE1 path transforms,
full token/shard checks, and the unchanged 32x64 tiny bank verifier execute.
"""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pytest
import torch

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.core_protocol import (REGISTRATION, TC64_REGISTRATION, NORMALIZED_REGISTRATION,
    NORMALIZED_RECIPE, CAUSAL_GRAPH_REGISTRATION, CAUSAL_GRAPH_RULE, CoreProtocolError,
    _registered_epochs, _derived_main, collection_manifest_paths)
from matdiscovery import graph_recovery as recovery, core_collection as collections, core_representation as representation
from matdiscovery.parallel_normalized_bank import inventory
from matdiscovery.representation_training import GraphStageConfig
from test_passed_bank_reuse import source_case, target_case, artifact
from torch_runtime_fixture import restore_torch_runtime


@pytest.fixture(scope="module", autouse=True)
def preserve_fixture_runtime():
    # The imported source_case fits its tiny bank in a module-scoped fixture.
    # Capture before that fixture, not only before individual test functions.
    threads, precision = torch.get_num_threads(), torch.get_float32_matmul_precision()
    cuda, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = cuda
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn


def seal_probe(value):
    value["fingerprint"] = fingerprint({k: v for k, v in value.items() if k != "fingerprint"})
    return value


def synthetic_probe(source_project, source_core, bank_reference):
    """Schema fixture only: all numerical measurements below are synthetic."""
    prefix, target, backend = "a" * 64, "b" * 64, "d" * 64
    rule = CAUSAL_GRAPH_RULE
    checks = [{"source": {"kind": kind, "position": 595, "layer": -1 if kind == "token" else 31},
               "target": {"kind": "score", "target_spec_hash": target},
               "analytical": .25 + i, "finite_difference": .25 + i, "passed": True}
              for i, kind in enumerate(("token", "feature", "feature", "feature"))]
    runtime = {"classification": "synthetic_CPU_fixture_not_policy_runtime"}
    source_proof = {"classification": "synthetic_architecture_proof_not_runtime_acceptance"}
    validation = {"passed": True, "checks": checks, "epsilon": .001, "rtol": .05, "atol": .0001,
        "required_direct_action_sink_source_kinds": ["token", "feature"], "prefix_hash": prefix,
        "target_spec_hash": target, "source_selection_rule": rule, "backend_signature": backend,
        "forward_parity_max_abs": 0., "runtime": runtime, "source_selection_proof": source_proof}
    metadata = {"selected_features": 32, "backward_targets": 33, "backend_validation": validation,
        "source_selection_rule": rule, "source_selection_proof": source_proof, "backend_signature": backend,
        "full_prefix_hash": prefix, "full_prefix_tokens": 643, "target_spec_hash": target,
        "forward_parity_max_abs": 0., "future_source_gradient_max_abs": 0., "runtime": runtime,
        "sink_count": 1, "causal_horizon": 604,
        "fidelity": {f"layer{i}": {"output_fvu": .25, "fvu_undefined": 0.} for i in range(32)}}
    return seal_probe({"schema": "causal_native_single_prefix_validation_v1", "complete": True, "passed": True,
        "classification": "synthetic_fixture_not_real_GPU_FD_acceptance", "source_selection_rule": rule,
        "source_core_fingerprint": source_core["fingerprint"], "source_bank_manifest": bank_reference,
        "new_policy_or_oracle_calls": 0, "new_policy_generation_calls": 0, "new_material_oracle_calls": 0,
        "source_sha256": {"native_attribution.py": file_sha256(Path(source_project) / "src/matdiscovery/native_attribution.py")},
        "source_prefix_hash": prefix, "source_decision_id": "synthetic-fixed-643-prefix",
        "validator_kwargs": {"epsilon": .001, "rtol": .05, "atol": .0001, "max_edges": 4},
        "validation": validation, "attribution_metadata": metadata,
        "nonzero_feature_to_score_count": 32, "reachable_feature_count": 32,
        "graph_config": asdict(GraphStageConfig(max_feature_nodes=32, max_backward_targets=42)),
        "intentional_configuration_delta": {"source_selection_rule": {"old": "global_activation", "new": rule}},
        "targets": {"source_length": 643, "source_prefix_hash": prefix, "target_spec_hash": target,
                    "prediction_positions": [595, 596, 604], "causal_horizon": 604},
        "actual_attribution_runtime": runtime, "actual_source_selection_proof": source_proof,
        "completed_validate_backend_calls": 1, "completed_attribute_calls": 1})


@pytest.fixture
def probe_case(tmp_path):
    source = tmp_path / "source"; native = source / "src/matdiscovery/native_attribution.py"
    native.parent.mkdir(parents=True); native.write_text("# synthetic source identity, never executed\n")
    bank = tmp_path / "bank.json"; bank.write_text("synthetic bank identity")
    core = {"fingerprint": "c" * 64}
    value = synthetic_probe(source, core, artifact(bank))
    path = tmp_path / "synthetic_probe.json"; write_json_atomic(path, value)
    return path, value, dict(source_project=source, source_core=core, bank_reference=artifact(bank))


def test_probe_accepts_explicit_synthetic_schema_fixture_only(probe_case):
    path, value, kwargs = probe_case
    assert recovery._probe(path, **kwargs) == value
    assert value["classification"].startswith("synthetic_fixture")


@pytest.mark.parametrize("mutation", ["source", "bank", "core", "rule", "fingerprint", "epsilon", "failed",
    "no_token", "feature_not_score", "wrong_numeric", "nonnumeric", "zero_required", "prefix", "target",
    "selected_budget", "backward_zero", "backward_over", "future_gradient"])
def test_probe_rejects_source_numeric_identity_and_budget_changes(probe_case, mutation):
    path, value, kwargs = probe_case
    bad = copy.deepcopy(value)
    if mutation == "source": bad["source_sha256"]["native_attribution.py"] = "0" * 64
    elif mutation == "bank": bad["source_bank_manifest"]["sha256"] = "0" * 64
    elif mutation == "core": bad["source_core_fingerprint"] = "wrong"
    elif mutation == "rule": bad["source_selection_rule"] = "global_activation"
    elif mutation == "fingerprint": bad["fingerprint"] = "0" * 64
    elif mutation == "epsilon": bad["validator_kwargs"]["epsilon"] = .01
    elif mutation == "failed": bad["validation"]["checks"][0]["passed"] = False
    elif mutation == "no_token": bad["validation"]["checks"][0]["source"]["kind"] = "feature"
    elif mutation == "feature_not_score":
        for row in bad["validation"]["checks"][1:]: row["target"]["kind"] = "feature"
    elif mutation == "wrong_numeric": bad["validation"]["checks"][0]["finite_difference"] = 999.
    elif mutation == "nonnumeric": bad["validation"]["checks"][0]["analytical"] = "NaN"
    elif mutation == "zero_required":
        row = bad["validation"]["checks"][0]; row["analytical"] = row["finite_difference"] = 0.
    elif mutation == "prefix": bad["attribution_metadata"]["full_prefix_hash"] = "0" * 64
    elif mutation == "target": bad["validation"]["target_spec_hash"] = "0" * 64
    elif mutation == "selected_budget": bad["attribution_metadata"]["selected_features"] = 31
    elif mutation == "backward_zero": bad["attribution_metadata"]["backward_targets"] = 0
    elif mutation == "backward_over": bad["attribution_metadata"]["backward_targets"] = 43
    else: bad["attribution_metadata"]["future_source_gradient_max_abs"] = .01
    if mutation != "fingerprint": seal_probe(bad)
    write_json_atomic(path, bad)
    with pytest.raises(CoreProtocolError): recovery._probe(path, **kwargs)


def test_registration_changes_only_declared_graph_rule_and_old_versions_reject_flags():
    project = Path(__file__).parents[1]
    parent = json.loads((project / "configs/main_protocol.json").read_text())
    old = {"registration": NORMALIZED_REGISTRATION, "study_id": "old", "scope": "unit",
           "model_key": "qwen35_4b", "methods": ["baseline", "esopt_graph_risk"], "training_seed": 1}
    new = {**old, "registration": CAUSAL_GRAPH_REGISTRATION, "study_id": "new"}
    before, after = _derived_main(parent, old), _derived_main(parent, new)
    assert after["graph"] == {**before["graph"], "source_selection_rule": CAUSAL_GRAPH_RULE}
    assert {k: v for k, v in before.items() if k not in {"study", "graph"}} == {k: v for k, v in after.items() if k not in {"study", "graph"}}
    for registration in (REGISTRATION, TC64_REGISTRATION, NORMALIZED_REGISTRATION):
        for key in ("passed_bank_reuse_contract", "causal_graph_amendment"):
            with pytest.raises(CoreProtocolError, match="prior registration"):
                _registered_epochs({"registration": registration, "fit_recipe": NORMALIZED_RECIPE, key: {}})
    with pytest.raises(CoreProtocolError, match="explicit original-bank"):
        _registered_epochs({"registration": CAUSAL_GRAPH_REGISTRATION, "fit_recipe": NORMALIZED_RECIPE})


@pytest.fixture
def imported_case(target_case, monkeypatch):
    case = target_case
    # Only ancestral admission is synthetic. The actual current import dispatch,
    # path-only transformations, receipt checks and immutable-bank verifier run.
    ancestor = {"fingerprint": "unit-CORE1-anchor", "imported_train": case.core["imported_train"]}
    monkeypatch.setattr(recovery, "validate_causal_graph_registration", lambda core: (case.core, ancestor))
    original_import, original_dev = collections.verify_import, collections.verify_development
    def ancestral_proof(): return {"evidence_files": [], "classification": "synthetic_ancestral_physical_admission"}
    monkeypatch.setattr(collections, "verify_import", lambda core, item: ancestral_proof() if core is ancestor else original_import(core, item))
    monkeypatch.setattr(collections, "verify_development", lambda core: ancestral_proof() if core is ancestor else original_dev(core))
    def sealed_fixture_read(path):
        value = json.loads((Path(path) / "configs/deadline_core_protocol.json").read_text())
        assert value == case.target_core
        assert value["fingerprint"] == fingerprint({k: v for k, v in value.items() if k not in {"fingerprint", "core_fingerprint"}})
        return value
    monkeypatch.setattr(collections, "read_core", sealed_fixture_read)
    shutil.rmtree(case.target / "experiments/collection")
    collections.import_training(case.target_core)
    collections.collect_development(case.target_core, policy_factory=lambda *_: pytest.fail("No policy/recollection"))
    return case


def test_actual_four_job_import_dispatch_and_unchanged_real_tiny_bank(imported_case):
    case = imported_case
    paths = collections.completed_core_collections(case.target_core)
    assert len(paths) == 4 and all(case.target in path.parents for path in paths)
    assert all("causal_graph_corpus_reuse" in json.loads(path.read_text()) for path in paths)
    assert collections.verify_stage_receipt(case.target_core, "import")["new_physical_calls"] == 0
    assert collections.verify_stage_receipt(case.target_core, "collect")["candidate_oracle_attempts_are_historical"] is True
    before = inventory(case.bankdir)
    bank, collection, proof = recovery.reused_bank_for_core(case.target_core, paths)
    assert bank == case.bank and collection["total_decisions"] == 4
    assert proof["new_transcoder_training_calls"] == proof["new_policy_or_material_oracle_calls"] == 0
    assert representation.paths_for(case.target_core)["bank"] == case.bankdir
    assert inventory(case.bankdir) == before


@pytest.mark.parametrize("mutation", ["unknown_file", "token"])
def test_imported_target_unknown_or_changed_science_is_rejected(imported_case, mutation):
    case = imported_case
    directory = collection_manifest_paths(case.target_core)[0].parent
    path = directory / ("unknown.partial" if mutation == "unknown_file" else "tokens/complete.pt")
    path.write_bytes(b"changed synthetic corpus")
    with pytest.raises(CoreProtocolError): collections.completed_core_collections(case.target_core)


def test_tc_stage_dispatch_reuses_verified_bank_without_training_or_cuda(imported_case, monkeypatch):
    case = imported_case
    paths = collections.completed_core_collections(case.target_core)
    verified = recovery.reused_bank_for_core(case.target_core, paths)
    # Dispatcher-only context supplies runtime settings absent from the tiny
    # ancestral protocol fixture; the returned bank was verified just above.
    dispatch_core = {**case.target_core, "policy_runtime": {"torch_cpu_threads": 2}}
    monkeypatch.setattr(representation, "verify_stage_receipt", lambda *a: {"complete": True})
    monkeypatch.setattr(representation, "completed_core_collections", lambda core: paths)
    monkeypatch.setattr(representation, "corpus_contract", lambda *a: {"unit_only": "dispatcher_contract"})
    monkeypatch.setattr(recovery, "reused_bank_for_core", lambda core, manifests: verified)
    def forbid(*a, **kw): pytest.fail("Bank adoption cannot train, resume training, or touch CUDA")
    monkeypatch.setattr(representation, "train_transcoders_from_collections", forbid)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", forbid)
    captured = {}
    def receipt(core, stage, files, **details):
        captured.update(stage=stage, files=files, **details); return captured
    monkeypatch.setattr(representation, "publish_stage_receipt", receipt)
    before = inventory(case.bankdir)
    result = representation.run_stage(dispatch_core, "transcoders")
    assert result["reused_complete_bank"] and result["new_transcoder_training_calls"] == result["new_physical_calls"] == 0
    assert case.bankdir / "transcoder_manifest.json" in result["files"]
    assert inventory(case.bankdir) == before


def test_prepare_refuses_unknown_existing_destination_without_overwrite(source_case, tmp_path, monkeypatch):
    case = source_case
    if case.target.exists(): shutil.rmtree(case.target)
    case.target.mkdir(); marker = case.target / "unknown.partial"; marker.write_text("preserve")
    source_project = Path(__file__).parents[1]
    probe = tmp_path / "synthetic_probe.json"
    write_json_atomic(probe, synthetic_probe(source_project, case.core, case.contract["source_bank_manifest"]))
    history = tmp_path / "synthetic_history.json"; history.write_text('{"synthetic":true}')
    original_read = recovery.read_core
    monkeypatch.setattr(recovery, "read_core", lambda path: case.core if Path(path).resolve() == case.workspace else original_read(path))
    with pytest.raises((CoreProtocolError, FileNotFoundError)):
        recovery.prepare_causal_workspace(source_project, case.target, predecessor_workspace=case.workspace,
            pause_closure_path=case.closure, validated_probe_path=probe, diagnostic_history_paths=[history])
    assert marker.read_text() == "preserve" and set(case.target.iterdir()) == {marker}
