"""32 tiny CPU layers test publication/filter/resume, not Qwen or materials fits.

The separate deferred trainer test proves all 16 epochs numerically. Here two
epochs keep a full-32-layer schema/integration test inexpensive; production
publish_core_precomputed_bank reads and enforces the actual core 16/.5 config.
"""
from dataclasses import asdict
from pathlib import Path
import json

import pytest
import torch

from matdiscovery.accounting import file_sha256, write_json_atomic
from matdiscovery.deferred_dev_transcoders import precompute_parallel
from matdiscovery.deferred_transcoder_publisher import build_external_bank, install_verified_bank, _inventory
from matdiscovery.representation_training import QWEN_MLP_PATHS, TranscoderStageConfig, train_transcoders_from_collections
from matdiscovery.transcoders import TranscoderConfig, write_activation_shard
from test_representation_training import collection_files


def collections(root):
    root.mkdir()
    items = collection_files(root)
    for item in items:
        split = item["record"]["split"]
        if split == "dev":
            duplicate = {**item["record"], "decision_id": "synthetic-duplicate-dev", "prefix_hash": items[0]["record"]["prefix_hash"]}
            item["decisions"].write_text(json.dumps(item["record"]) + "\n" + json.dumps(duplicate) + "\n")
            item["manifest_data"]["decision_files"][0]["sha256"] = file_sha256(item["decisions"])
        shards = []
        for layer in QWEN_MLP_PATHS:
            generator = torch.Generator().manual_seed(42 if split == "train" else 43)
            x = torch.rand(12, 3, generator=generator); y = x[:, :2] * .6 + .1
            path = item["manifest"].parent / (layer.replace(".", "_") + ".pt")
            prefixes = [item["record"]["prefix_hash"]] * 12
            if split == "dev": prefixes[:4] = [items[0]["record"]["prefix_hash"]] * 4
            meta = write_activation_shard(path, x, y, group_ids=[item["record"]["group_id"]] * 12,
                prefix_hashes=prefixes, split=split, policy_fingerprint=item["stamp"]["checkpoint_hash"], layer_path=layer)
            shards.append({"path": str(path), "layer_path": layer, "split": split, "tensor_hash": meta["tensor_hash"]})
        item["manifest_data"]["activation_shards"] = shards
        write_json_atomic(item["manifest"], item["manifest_data"])
    return items


def test_all_32_layers_original_filter_provider_and_installed_original_resume(tmp_path):
    torch.set_num_threads(2)
    items = collections(tmp_path / "collections")
    manifests = [item["manifest"] for item in items]
    cfg = TranscoderStageConfig(feature_dim=8, top_k=4, epochs=2, batch_size=8,
        learning_rate=.03, max_dev_fvu=100., device="cpu")
    tasks = []
    for index, layer in enumerate(QWEN_MLP_PATHS):
        tasks.append({"config": asdict(TranscoderConfig(3, 2, 8, 4)),
            "train_paths": [items[0]["manifest_data"]["activation_shards"][index]["path"]],
            "output_dir": str(tmp_path / "pre" / f"layer_{index:02d}"), "kwargs": {
                "policy_fingerprint": items[0]["stamp"]["checkpoint_hash"], "layer_path": layer,
                "seed": cfg.seed + index, "epochs": cfg.epochs, "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate, "device": cfg.device, "max_dev_fvu": cfg.max_dev_fvu}})
    precompute_parallel(tasks, max_workers=2)
    expected = train_transcoders_from_collections(manifests, tmp_path / "serial", model_key="qwen35_4b", config=cfg)
    actual = build_external_bank(manifests, tmp_path / "pre", tmp_path / "deferred", model_key="qwen35_4b", config=cfg)
    assert actual["complete"] and actual["passed_layers"] == 32
    assert actual["development_exact_prefix_filter"]["duplicate_dev_activation_rows"] == 4 * 32
    for left, right in zip(expected["layers"], actual["layers"]):
        metadata = dict(right["metadata"]); provenance = metadata.pop("execution_provenance")
        assert metadata == left["metadata"]
        assert right["transcoder_hash"] == left["transcoder_hash"]
        assert provenance["original_serial_execution_claimed"] is False
        assert provenance["epochs_scored"] == 2
    before = _inventory(tmp_path / "deferred")
    again = build_external_bank(manifests, tmp_path / "pre", tmp_path / "deferred", model_key="qwen35_4b", config=cfg)
    assert again == actual and _inventory(tmp_path / "deferred") == before
    receipt = install_verified_bank(manifests, tmp_path / "deferred", tmp_path / "installed", model_key="qwen35_4b", config=cfg)
    assert receipt["complete"] and receipt["original_resume_verified"]
    installed = _inventory(tmp_path / "installed")
    train_transcoders_from_collections(manifests, tmp_path / "installed", model_key="qwen35_4b", config=cfg, resume=True)
    assert installed == _inventory(tmp_path / "installed")
    with pytest.raises(ValueError, match="already exists"):
        install_verified_bank(manifests, tmp_path / "deferred", tmp_path / "installed", model_key="qwen35_4b", config=cfg)
    (tmp_path / "pre/layer_00/epoch_000.pt").write_bytes(b"changed precomputation")
    with pytest.raises(ValueError, match="Epoch bytes"):
        build_external_bank(manifests, tmp_path / "pre", tmp_path / "deferred", model_key="qwen35_4b", config=cfg)


def test_provider_binding_restored_on_rejected_training_contract(tmp_path):
    from matdiscovery import transcoders
    from matdiscovery.deferred_transcoder_publisher import _provider
    original = transcoders.train_layer_transcoder
    with pytest.raises(FileNotFoundError):
        with _provider(tmp_path):
            assert transcoders.train_layer_transcoder is not original
            transcoders.train_layer_transcoder(TranscoderConfig(3, 2, 8, 4), [], [],
                policy_fingerprint="unit", layer_path=QWEN_MLP_PATHS[0], output_path=tmp_path / "bad.pt")
    assert transcoders.train_layer_transcoder is original
