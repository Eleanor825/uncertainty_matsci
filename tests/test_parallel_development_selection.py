"""32 tiny CPU layers and 64 real epochs; no policy or scientific oracle."""
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from matdiscovery.deferred_dev_transcoders import precompute_parallel
from matdiscovery.deferred_transcoder_publisher import (
    build_external_bank, install_verified_bank, _inventory, _verify_execution_provenance,
)
from matdiscovery.representation_training import QWEN_MLP_PATHS, TranscoderStageConfig, train_transcoders_from_collections
from matdiscovery.transcoders import TranscoderConfig
from test_deferred_transcoder_publisher import collections


def test_32_layers_64_epochs_four_spawn_exact_selection_and_strict_resume(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    items = collections(tmp_path / "collections")
    manifests = [item["manifest"] for item in items]
    cfg = TranscoderStageConfig(feature_dim=8, top_k=4, epochs=64, batch_size=8,
                               learning_rate=.03, max_dev_fvu=100., device="cpu")
    tasks = [{"config": asdict(TranscoderConfig(3, 2, 8, 4)),
        "train_paths": [items[0]["manifest_data"]["activation_shards"][index]["path"]],
        "output_dir": str(tmp_path / "pre" / f"layer_{index:02d}"), "kwargs": {
            "policy_fingerprint": items[0]["stamp"]["checkpoint_hash"], "layer_path": layer,
            "seed": cfg.seed + index, "epochs": cfg.epochs, "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate, "device": cfg.device, "max_dev_fvu": cfg.max_dev_fvu}}
        for index, layer in enumerate(QWEN_MLP_PATHS)]
    precompute_parallel(tasks, max_workers=4)
    original = train_transcoders_from_collections(manifests, tmp_path / "serial", model_key="qwen35_4b", config=cfg)
    actual = build_external_bank(manifests, tmp_path / "pre", tmp_path / "parallel", model_key="qwen35_4b", config=cfg,
                                 development_workers=4)
    assert actual["complete"] and actual["passed_layers"] == 32
    assert actual["development_exact_prefix_filter"]["duplicate_dev_activation_rows"] == 4 * 32
    for expected, selected in zip(original["layers"], actual["layers"]):
        metadata = dict(selected["metadata"])
        evidence = metadata.pop("execution_provenance")
        assert metadata == expected["metadata"]  # All 64 histories, FVU, first minimum and tensor hash.
        assert len(metadata["history"]) == 64
        assert evidence["parallel_selection"]["workers"] == 4
        left = torch.load(expected["checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
        right = torch.load(selected["checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
        assert all(torch.equal(left[key], right[key]) for key in left)
    cache = tmp_path / "parallel.selection_cache"
    before, cache_before = _inventory(tmp_path / "parallel"), _inventory(cache)
    # Resume/installation must not invoke a trainer or selector at all.
    from matdiscovery import transcoders
    monkeypatch.setattr(transcoders, "train_layer_transcoder", lambda *a, **kw: pytest.fail("no optimizer/selector replay on resume"))
    again = build_external_bank(manifests, tmp_path / "pre", tmp_path / "parallel", model_key="qwen35_4b", config=cfg,
                                development_workers=4)
    assert actual == again and before == _inventory(tmp_path / "parallel") and cache_before == _inventory(cache)
    proof = install_verified_bank(manifests, tmp_path / "parallel", tmp_path / "installed", model_key="qwen35_4b", config=cfg)
    assert proof["complete"] and proof["original_resume_verified"]
    assert cache_before == _inventory(cache)
    # Even an unused old epoch is rehashed during explicit reuse.
    (tmp_path / "pre/layer_00/epoch_000.pt").write_bytes(b"changed precomputed bytes")
    with pytest.raises(ValueError, match="Epoch bytes"):
        _verify_execution_provenance(actual, tmp_path / "pre", None)


def test_parallel_failure_does_not_retry_cache_or_invoke_numeric_selector(tmp_path, monkeypatch):
    from matdiscovery import deferred_transcoder_publisher as publisher
    from matdiscovery import transcoders
    calls = []
    def failed(*args):
        calls.append(args)
        raise ValueError("synthetic partial selection")
    monkeypatch.setattr(publisher, "_parallel_selection", failed)
    kwargs = dict(policy_fingerprint="unit", layer_path=QWEN_MLP_PATHS[0], output_path=tmp_path / "layer_00.pt")
    with publisher._provider(tmp_path / "pre", development_workers=4):
        with pytest.raises(ValueError, match="partial selection"):
            transcoders.train_layer_transcoder(TranscoderConfig(3, 2, 8, 4), [], [], **kwargs)
        with pytest.raises(ValueError, match="already failed"):
            transcoders.train_layer_transcoder(TranscoderConfig(3, 2, 8, 4), [], [], **kwargs)
    assert len(calls) == 1


@pytest.mark.parametrize("workers", [0, 5, True])
def test_worker_limits_reject_before_any_coordinator(tmp_path, workers):
    with pytest.raises(ValueError, match="one to four"):
        build_external_bank([], tmp_path / "pre", tmp_path / "bank", model_key="qwen35_4b",
                            config=TranscoderStageConfig(), development_workers=workers)
