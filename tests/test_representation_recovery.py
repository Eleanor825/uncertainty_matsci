"""CPU recovery fixtures. Fake completed layer receipts are NOT learned models."""
import copy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from matdiscovery.accounting import file_sha256, write_json_atomic
from matdiscovery import representation_training as rep
from matdiscovery import transcoders as tc
from matdiscovery.native_attribution import PolicyStamp
from test_representation_training import collection_files, graph_record


def training_fixture(tmp_path, monkeypatch):
    items = collection_files(tmp_path)
    for item in items:
        split = item["record"]["split"]
        for index, layer in enumerate(rep.QWEN_MLP_PATHS):
            x = torch.arange(16).reshape(8, 2).float() / 10
            path = item["manifest"].parent / f"synthetic_layer_{index:02d}.pt"
            meta = tc.write_activation_shard(path, x, x + .3, group_ids=[split + "-group"] * 8,
                prefix_hashes=[item["record"]["prefix_hash"]] * 8, split=split,
                policy_fingerprint="a" * 64, layer_path=layer)
            item["manifest_data"]["activation_shards"].append({"path": str(path), "layer_path": layer,
                "split": split, "tensor_hash": meta["tensor_hash"]})
        write_json_atomic(item["manifest"], item["manifest_data"])
    calls = []
    def fake_completed_layer(config, train, dev, **kwargs):
        index = rep.QWEN_MLP_PATHS.index(kwargs["layer_path"])
        calls.append(index)
        model = tc.TopKTranscoder(config, seed=kwargs["seed"])
        provenance = tc.validate_shard_splits(train, dev, policy_fingerprint=kwargs["policy_fingerprint"],
            layer_path=kwargs["layer_path"], config=config)
        measured = {"rows": float(provenance["rows"]["dev"]), "output_mse": .1, "output_fvu": .1,
                    "fvu_undefined": 0.0, "mean_active_features": 2.0}
        metadata = {"schema_version": 1, "classification": "synthetic_fake_completed_layer_not_training",
            "method": "per_mlp_topk_input_to_output_v1", "config": asdict(config),
            **{key: kwargs[key] for key in ("seed", "epochs", "learning_rate", "batch_size", "max_dev_fvu")},
            "selected_epoch": 0, "history": [{"epoch": epoch, "train_rows": provenance["rows"]["train"],
                "train_output_mse": .2, "dev": dict(measured)} for epoch in range(kwargs["epochs"])],
            "dev": measured, "provenance": provenance, "fidelity_gate_passed": True, "transcoder_hash": model.checkpoint_hash()}
        torch.save({"state_dict": model.state_dict(), "metadata": metadata}, kwargs["output_path"])
        Path(str(kwargs["output_path"]) + ".json").write_text(json.dumps(metadata))
        return model, metadata
    monkeypatch.setattr(tc, "train_layer_transcoder", fake_completed_layer)
    config = rep.TranscoderStageConfig(top_k=2, epochs=16, batch_size=4, device="cpu")
    return {"items": items, "manifests": [i["manifest"] for i in items], "output": tmp_path / "bank",
            "config": config, "calls": calls, "trainer": fake_completed_layer}


def train(data, **options):
    return rep.train_transcoders_from_collections(data["manifests"], data["output"], model_key="qwen35_4b",
        config=options.pop("config", data["config"]), **options)


def interrupt_between_layers(data, monkeypatch, completed=2):
    def publish_then_interrupt(path, payload):
        write_json_atomic(path, payload)
        if payload.get("completed_layers") == completed and payload.get("in_progress_layer") is None:
            raise KeyboardInterrupt("synthetic process stop after durable layer publication")
    with monkeypatch.context() as patch:
        patch.setattr(rep, "write_json_atomic", publish_then_interrupt)
        with pytest.raises(KeyboardInterrupt):
            train(data)
    return json.loads((data["output"] / "transcoder_manifest.json").read_text())


@pytest.mark.parametrize("completed", [2, 32])
def test_clean_layer_boundary_resume_skips_only_verified_complete_layers(tmp_path, monkeypatch, completed):
    data = training_fixture(tmp_path, monkeypatch)
    partial = interrupt_between_layers(data, monkeypatch, completed=completed)
    assert partial["in_progress_layer"] is None and partial["completed_layers"] == completed
    assert partial["complete"] is False  # Includes layer 32 published before final integrity/complete write.
    before = {p.name: file_sha256(p) for p in data["output"].glob("layer_*")}
    result = train(data, resume=True)
    assert data["calls"] == list(range(32))
    assert result["complete"] and result["ready_for_graphs"] and result["passed_layers"] == 32
    assert all(file_sha256(data["output"] / name) == digest for name, digest in before.items())
    assert all(len(row["metadata"]["history"]) == 16 for row in result["layers"])
    snapshot = {str(p): (file_sha256(p), p.stat().st_mtime_ns) for p in data["output"].rglob("*") if p.is_file()}
    assert train(data, resume=True) == result
    assert snapshot == {str(p): (file_sha256(p), p.stat().st_mtime_ns) for p in data["output"].rglob("*") if p.is_file()}
    assert data["calls"] == list(range(32))  # Complete bank did not retrain or rewrite.


@pytest.mark.parametrize("mutation", ["checkpoint", "metadata_sidecar", "missing_file", "orphan", "inflight",
    "failed_fidelity", "configuration", "collection_source", "layer_provenance", "short_history", "modified_training_source"])
def test_partial_bank_conflicts_fail_before_any_new_training(tmp_path, monkeypatch, mutation):
    data = training_fixture(tmp_path, monkeypatch)
    bank = interrupt_between_layers(data, monkeypatch)
    path = data["output"] / "transcoder_manifest.json"
    options = {}
    if mutation == "checkpoint":
        (data["output"] / "layer_00.pt").write_bytes(b"modified weights")
    elif mutation == "metadata_sidecar":
        (data["output"] / "layer_00.pt.json").write_text("{}")
    elif mutation == "missing_file":
        (data["output"] / "layer_00.pt.json").unlink()
    elif mutation == "orphan":
        (data["output"] / "layer_02.pt").write_bytes(b"unpublished checkpoint")
    elif mutation == "inflight":
        bank["in_progress_layer"] = {"layer_index": 2, "layer_path": rep.QWEN_MLP_PATHS[2]}
    elif mutation == "failed_fidelity":
        bank["layers"][0]["status"] = "failed_fidelity"
    elif mutation == "configuration":
        options["config"] = replace(data["config"], seed=999)
    elif mutation == "collection_source":
        item = data["items"][0]
        item["manifest_data"]["extra_fixture_change"] = "must be rejected"
        write_json_atomic(item["manifest"], item["manifest_data"])
    elif mutation == "modified_training_source":
        bank["training_source_sha256"]["transcoders.py"] = "another trainer"
    else:
        # Rehash every stored container to verify semantic provenance/epoch checks,
        # not merely the outer checkpoint SHA comparison.
        checkpoint = data["output"] / "layer_00.pt"
        payload = torch.load(checkpoint, weights_only=True)
        metadata = payload["metadata"]
        if mutation == "layer_provenance":
            metadata["provenance"]["rows"]["train"] = 7
        else:
            metadata["history"].pop()
        torch.save(payload, checkpoint)
        sidecar = Path(str(checkpoint) + ".json")
        sidecar.write_text(json.dumps(metadata))
        bank["layers"][0].update(metadata=metadata, checkpoint_sha256=file_sha256(checkpoint), metadata_sha256=file_sha256(sidecar))
    bank["bank_fingerprint"] = rep._bank_fingerprint(bank)
    write_json_atomic(path, bank)
    before = {str(p): file_sha256(p) for p in data["output"].rglob("*") if p.is_file()}
    with pytest.raises((rep.RepresentationError, ValueError), match="reconciliation|source"):
        train(data, resume=True, **options)
    assert data["calls"] == [0, 1]
    assert before == {str(p): file_sha256(p) for p in data["output"].rglob("*") if p.is_file()}


@pytest.mark.parametrize("checkpoint_written", [False, True])
def test_mid_layer_interrupt_never_adopts_or_retrains_unpublished_work(tmp_path, monkeypatch, checkpoint_written):
    data = training_fixture(tmp_path, monkeypatch)
    def interrupted(*args, **kwargs):
        if checkpoint_written:
            data["trainer"](*args, **kwargs)
        raise KeyboardInterrupt("synthetic in-flight interruption")
    monkeypatch.setattr(tc, "train_layer_transcoder", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train(data)
    bank = json.loads((data["output"] / "transcoder_manifest.json").read_text())
    assert bank["layers"] == [] and bank["in_progress_layer"]["layer_index"] == 0
    monkeypatch.setattr(tc, "train_layer_transcoder", lambda *args, **kwargs: pytest.fail("Must not retry in-flight layer"))
    with pytest.raises(rep.RepresentationError, match="in-progress"):
        train(data, resume=True)


def test_orphan_checkpoint_without_manifest_is_not_adopted(tmp_path):
    output = tmp_path / "orphan"
    output.mkdir()
    (output / "layer_00.pt").write_bytes(b"not a published trained layer")
    with pytest.raises(rep.RepresentationError, match="published manifest"):
        rep.train_transcoders_from_collections([], output, model_key="qwen35_4b", resume=True)


def test_filtered_dev_resume_is_read_only_and_detects_modified_rows(tmp_path):
    original, derived = tmp_path / "source.pt", tmp_path / "filtered.pt"
    x = torch.arange(8).reshape(4, 2).float()
    tc.write_activation_shard(original, x, x + 1, group_ids=["dev"] * 4, prefix_hashes=["shared", "a", "shared", "b"],
        split="dev", policy_fingerprint="a" * 64, layer_path=rep.QWEN_MLP_PATHS[0])
    shard = tc._load_shard(original)
    first = rep.filter_development_shard(shard, source_path=original, output_path=derived, train_prefixes={"shared"})
    before = (file_sha256(derived), derived.stat().st_mtime_ns)
    assert rep.filter_development_shard(shard, source_path=original, output_path=derived,
        train_prefixes={"shared"}, verify_existing=True) == first
    assert before == (file_sha256(derived), derived.stat().st_mtime_ns)
    wrong = copy.deepcopy(shard)
    wrong["inputs"][1, 0] += 1
    with pytest.raises(rep.RepresentationError, match="changed filtered"):
        rep.filter_development_shard(wrong, source_path=original, output_path=derived,
                                    train_prefixes={"shared"}, verify_existing=True)
    assert before == (file_sha256(derived), derived.stat().st_mtime_ns)


@pytest.mark.parametrize("tail", ["", "{", "valid_json_without_newline"])
def test_graph_resume_rejects_empty_or_truncated_journal(tmp_path, tail):
    items = collection_files(tmp_path)
    sources = rep.read_collections([i["manifest"] for i in items], model_key="qwen35_4b")["records"]
    journal = Path(sources[0]["decision_file"]).parent / "graph_features.jsonl"
    journal.write_text(json.dumps(graph_record(sources[0])) if tail == "valid_json_without_newline" else tail)
    with pytest.raises(rep.RepresentationError, match="reconciliation"):
        rep._load_resume(sources, "synthetic-bank", "synthetic-pipeline", True)


def test_graph_resume_reuses_success_and_unavailable_without_any_forward(tmp_path):
    items = collection_files(tmp_path)
    sources = rep.read_collections([i["manifest"] for i in items], model_key="qwen35_4b")["records"]
    for index, source in enumerate(sources):
        journal = Path(source["decision_file"]).parent / "graph_features.jsonl"
        journal.write_text(json.dumps(graph_record(source, succeeded=index == 0)) + "\n")
    existing = rep._load_resume(sources, "synthetic-bank", "synthetic-pipeline", True)
    class NoReplayPolicy:
        model_stamp = PolicyStamp("replay", "a" * 64, "Qwen/Qwen3.5-4B", 0)
        def __getattr__(self, name):
            raise AssertionError("Completed graph rows must not invoke policy or oracle")
    result = rep._replay_records(sources, NoReplayPolicy(), object(), config=rep.GraphStageConfig(),
        bank_fingerprint="synthetic-bank", pipeline_fingerprint="synthetic-pipeline", existing=existing,
        report_path=tmp_path / "report.json")
    assert result["complete"] and result["successful_graphs"] == result["unavailable_graphs"] == 1
    assert result["resumed_verified_records"] == 2


@pytest.mark.parametrize("stage", ["graphs", "transcoders"])
def test_production_stage_entry_enables_strict_resume(tmp_path, monkeypatch, stage):
    script = Path(__file__).resolve().parents[1] / "scripts/run_representation_condition.py"
    spec = importlib.util.spec_from_file_location("unit_representation_stage", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = Path(__file__).resolve().parents[1] / "configs/main_protocol.json"
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/main_protocol.json").write_bytes(source.read_bytes())
    manifest = tmp_path / "source/collection_manifest.json"
    write_json_atomic(manifest, {"unit_fixture_not_results": True})
    (manifest.parent / "graph_features.jsonl").write_text("unit fixture\n")
    monkeypatch.setattr(module, "completed_collections", lambda *args: [manifest])
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *args: None)
    seen = []
    def fake_stage(*args, **kwargs):
        seen.append(kwargs)
        if stage == "transcoders":
            return {"status": "succeeded", "ready_for_graphs": True}
        write_json_atomic(kwargs["report_path"], {"unit_fixture_not_results": True})
        return {"status": "succeeded", "complete": True, "successful_graphs": 1, "unavailable_graphs": 0,
                "transcoder_bank_fingerprint": "unit-bank", "graph_pipeline_fingerprint": "unit-pipeline",
                "expected_decisions": 1, "completed_decisions": 1}
    monkeypatch.setattr(module, "generate_graph_features" if stage == "graphs" else "train_transcoders_from_collections", fake_stage)
    monkeypatch.setattr(sys, "argv", [str(script), "--project", str(tmp_path), "--model-key", "qwen35_4b", "--benchmark", "made", "--stage", stage])
    before = torch.get_num_threads()
    try:
        module.main()
    finally:
        torch.set_num_threads(before)
    assert len(seen) == 1 and seen[0]["resume"] is True
    if stage == "transcoders":
        assert seen[0]["config"].epochs == 16 and seen[0]["config"].device == "cuda:0"
