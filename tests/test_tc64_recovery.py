"""CPU synthetic failed-fit and closed-receipt amendment tests; no real oracle.

Only the TC device adapter is replaced by CPU for this unit fixture. Its actual
32x16 optimizer runs, dev scores, raw receipts and all provenance checks execute.
"""
import copy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.core_protocol import (TC64_REGISTRATION, CoreProtocolError, _core_hash,
    collection_jobs, collection_manifest_paths, read_core, registered_transcoder_epochs)
from matdiscovery.core_collection import (import_training, collect_development, audit_core_corpus_costs,
    completed_core_collections)
from matdiscovery.tc64_recovery import prepare_tc64_workspace, validate_tc64_registration
from matdiscovery.deferred_dev_transcoders import precompute_layer
from matdiscovery.deferred_transcoder_publisher import _provider
from matdiscovery.representation_training import QWEN_MLP_PATHS, train_transcoders_from_collections, read_collections
from matdiscovery.transcoders import TranscoderConfig
from core_corpus_fixture import make_core_fixture, complete_core_dev
from test_collection_provenance import artifact


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    import matdiscovery.core_representation as representation
    original_config = representation.transcoder_config
    monkeypatch.setattr(representation, "transcoder_config", lambda core: replace(original_config(core), device="cpu"))
    torch.set_num_threads(2)
    f = make_core_fixture(tmp_path, representation=True)
    old = f["core"]; import_training(old); complete_core_dev(f); collect_development(old)
    manifests = collection_manifest_paths(old)
    collection = read_collections(manifests, model_key="qwen35_4b")
    cfg = representation.transcoder_config(old)
    pre = tmp_path / "old_precompute/layers"
    receipts = []
    for index, layer in enumerate(QWEN_MLP_PATHS):
        paths = [x["path"] for x in collection["activation_shards"] if x["layer_path"] == layer and x["split"] == "train"]
        precompute_layer(TranscoderConfig(64, 64, 128, 64), paths, pre / f"layer_{index:02d}",
            policy_fingerprint=collection["checkpoint_hash"], layer_path=layer, seed=1729 + index,
            epochs=16, batch_size=cfg.batch_size, learning_rate=cfg.learning_rate, max_dev_fvu=.5, device="cpu")
        receipts.append(artifact(pre / f"layer_{index:02d}/complete.json"))
    bank_dir = tmp_path / "failed_bank"
    with _provider(pre):
        bank = train_transcoders_from_collections(manifests, bank_dir, model_key="qwen35_4b", config=cfg)
    assert bank["complete"] and not bank["ready_for_graphs"]
    result = tmp_path / "old_precompute/result.json"
    write_json_atomic(result, {"complete": True, "core_fingerprint": old["fingerprint"], "layers": 32,
        "epochs_per_layer": 16, "layer_receipts": receipts, "classification": "CPU_unit_not_real_model_oracle"})
    failure = tmp_path / "failure.json"
    write_json_atomic(failure, {"status": "failed_requires_reconciliation", "publisher_exited": True,
        "supervisor_halt_requested": True, "cleanup_errors": [], "classification": "simulated_process_lifecycle_no_signals"})
    kwargs = dict(predecessor_workspace=old["workspace"], failed_bank_path=bank_dir / "transcoder_manifest.json",
        failure_reconciliation_path=failure, prior_precompute_result=result, tc64_output_root=tmp_path / "tc64_epochs")
    return f, kwargs


def test_tc64_only_epoch_amendment_reuses_all_four_closed_jobs_without_oracle(recovery, tmp_path):
    f, kwargs = recovery; old = f["core"]
    old_files = {p: (file_sha256(p), p.stat().st_mtime_ns) for p in Path(old["workspace"]).rglob("*") if p.is_file() and not p.is_symlink()}
    core = prepare_tc64_workspace(f["parent"], tmp_path / "tc64", **kwargs)
    assert core["registration"] == TC64_REGISTRATION
    assert registered_transcoder_epochs(core) == 64 and registered_transcoder_epochs(old) == 16
    assert core["transcoder"] == {**old["transcoder"], "epochs": 64}
    assert collection_jobs(core) == collection_jobs(old)  # Physical identities remain original.
    assert core["fingerprint"] != old["fingerprint"]
    import_training(core)
    collect_development(core, policy_factory=lambda *args: pytest.fail("MUST NOT load policy"),
                        rollout_factory=lambda *args, **kw: pytest.fail("MUST NOT dispatch oracle"))
    paths = completed_core_collections(core)
    assert len(paths) == 4 and all(Path(core["workspace"]) in p.parents for p in paths)
    report = audit_core_corpus_costs(core)
    assert report["total"]["costs"]["candidate_oracle_attempts"] == 200
    assert report["reused_development"]["costs"]["candidate_oracle_attempts"] == 50
    assert report["new_development"]["jobs"] == 0 and report["new_development"]["costs"]["candidate_oracle_attempts"] == 0
    assert not any(report["incremental_physical_costs"].values())
    assert old_files == {p: (file_sha256(p), p.stat().st_mtime_ns) for p in old_files}
    again = prepare_tc64_workspace(f["parent"], tmp_path / "tc64", **kwargs)
    assert again == core
    with pytest.raises(CoreProtocolError, match="another source"):
        prepare_tc64_workspace(f["parent"], tmp_path / "tc64", **{**kwargs, "tc64_output_root": tmp_path / "other"})


def test_tc64_requires_pretest_closed_failure_and_exact_fidelity(recovery, tmp_path):
    f, kwargs = recovery
    old = f["core"]
    marker = Path(old["workspace"]) / "experiments/core_final/attempt.json"
    write_json_atomic(marker, {"unit_only": True})
    with pytest.raises(CoreProtocolError, match="pre-test"):
        prepare_tc64_workspace(f["parent"], tmp_path / "blocked", **kwargs)
    marker.unlink(); marker.parent.rmdir()
    failure = kwargs["failure_reconciliation_path"]
    write_json_atomic(failure, {"status": "failed_requires_reconciliation", "publisher_exited": False,
        "supervisor_halt_requested": True, "cleanup_errors": []})
    with pytest.raises(CoreProtocolError, match="not closed"):
        prepare_tc64_workspace(f["parent"], tmp_path / "unclosed", **kwargs)


def test_tc64_rehashed_scope_change_and_derived_data_tamper_rejected(recovery, tmp_path):
    f, kwargs = recovery
    core = prepare_tc64_workspace(f["parent"], tmp_path / "tc64", **kwargs)
    import_training(core); collect_development(core)
    changed = copy.deepcopy(core); changed["transcoder"]["max_development_output_fvu"] = .7
    changed["fingerprint"] = changed["core_fingerprint"] = _core_hash(changed)
    with pytest.raises(CoreProtocolError): validate_tc64_registration(changed)
    path = collection_manifest_paths(core)[-1].parent / "decisions.jsonl"
    path.write_text('{"fake": true}\n')
    with pytest.raises(CoreProtocolError, match="changed"):
        completed_core_collections(core)
