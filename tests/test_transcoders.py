"""Streaming transcoder checks on synthetic tensors, not material experiments."""
import copy

import pytest
import torch

from matdiscovery.transcoders import (TopKTranscoder, TranscoderConfig, iter_activation_batches,
    load_transcoder, reconstruction_metrics, train_layer_transcoder, validate_shard_splits, write_activation_shard)


def make_shards(tmp_path, *, overlap=False):
    generator = torch.Generator().manual_seed(9)
    paths = {}
    for split, rows in (("train", 96), ("dev", 32)):
        x = torch.rand(rows, 2, generator=generator) + 0.2
        y = torch.stack((2 * x[:, 0], -x[:, 1], x[:, 0] + x[:, 1]), dim=1)
        path = tmp_path / (split + ".pt")
        group = "same-group" if overlap else split + "-groups"
        write_activation_shard(path, x, y, group_ids=[group] * rows,
                               prefix_hashes=[f"{split}-prefix-{i}" for i in range(rows)], split=split,
                               policy_fingerprint="teacher@hash", layer_path="layers.0.mlp")
        paths[split] = path
    return paths


def test_transcoder_predicts_mlp_outputs_not_its_input():
    config = TranscoderConfig(2, 3, 12, 4)
    transcoder = TopKTranscoder(config, seed=4)
    x = torch.randn(10, 2)
    assert transcoder(x).shape == (10, 3)
    assert torch.all((transcoder.encode(x) > 0).sum(dim=-1) <= 4)
    torch.testing.assert_close(transcoder.decoder.weight.norm(dim=0), torch.ones(12))
    with pytest.raises(ValueError):
        transcoder(torch.zeros(3, 3))


def test_group_isolation_and_policy_identity_are_required(tmp_path):
    config = TranscoderConfig(2, 3, 12, 6)
    paths = make_shards(tmp_path, overlap=True)
    with pytest.raises(ValueError, match="leakage"):
        validate_shard_splits([paths["train"]], [paths["dev"]], policy_fingerprint="teacher@hash", layer_path="layers.0.mlp", config=config)
    paths = make_shards(tmp_path, overlap=False)
    with pytest.raises(ValueError, match="fingerprint"):
        validate_shard_splits([paths["train"]], [paths["dev"]], policy_fingerprint="other", layer_path="layers.0.mlp", config=config)
    with pytest.raises(ValueError, match="never test"):
        write_activation_shard(tmp_path / "test.pt", torch.ones(1, 2), torch.ones(1, 3), group_ids=["x"], prefix_hashes=["h"], split="test", policy_fingerprint="p", layer_path="mlp")


def test_stream_training_dev_metrics_checkpoint_and_seed(tmp_path):
    paths = make_shards(tmp_path)
    config = TranscoderConfig(2, 3, 12, 6)
    initial = TopKTranscoder(config, seed=11)
    initial_dev = reconstruction_metrics(initial, iter_activation_batches([paths["dev"]], 8, seed=11, shuffle=False))
    first, metadata = train_layer_transcoder(config, [paths["train"]], [paths["dev"]],
        policy_fingerprint="teacher@hash", layer_path="layers.0.mlp", output_path=tmp_path / "tc.pt",
        seed=11, epochs=20, batch_size=16, learning_rate=0.03, max_dev_fvu=2.0)
    assert metadata["dev"]["output_mse"] < initial_dev["output_mse"]
    assert metadata["dev"]["fvu_undefined"] == 0
    assert metadata["fidelity_gate_passed"]
    assert all(epoch["train_rows"] == 96 for epoch in metadata["history"])
    loaded, reloaded_metadata = load_transcoder(tmp_path / "tc.pt", expected_policy_fingerprint="teacher@hash", expected_layer_path="layers.0.mlp")
    assert loaded.checkpoint_hash() == first.checkpoint_hash() == metadata["transcoder_hash"]
    second, _ = train_layer_transcoder(config, [paths["train"]], [paths["dev"]],
        policy_fingerprint="teacher@hash", layer_path="layers.0.mlp", output_path=tmp_path / "tc2.pt",
        seed=11, epochs=20, batch_size=16, learning_rate=0.03, max_dev_fvu=2.0)
    assert first.checkpoint_hash() == second.checkpoint_hash()
    with pytest.raises(ValueError, match="different policy"):
        load_transcoder(tmp_path / "tc.pt", expected_policy_fingerprint="wrong")


def test_fvu_matches_output_variance_and_undefined_is_explicit():
    tc = TopKTranscoder(TranscoderConfig(1, 1, 1, 1))
    with torch.no_grad():
        tc.encoder.weight.fill_(2)
        tc.encoder.bias.zero_()
        tc.decoder.weight.fill_(1)
        tc.decoder.bias.zero_()
    x = torch.tensor([[1.0], [2.0], [3.0]])
    assert reconstruction_metrics(tc, iter([(x, 2 * x)]))["output_fvu"] == 0
    # yhat=2x, y=x: SSE=14, centered output SS=2 => FVU=7.
    assert reconstruction_metrics(tc, iter([(x, x)]))["output_fvu"] == pytest.approx(7)
    constant = reconstruction_metrics(tc, iter([(torch.ones(3, 1), torch.ones(3, 1))]))
    assert constant["fvu_undefined"] == 1


def test_initialization_does_not_consume_policy_rng():
    torch.manual_seed(42)
    before = torch.get_rng_state().clone()
    TopKTranscoder(TranscoderConfig(3, 4, 8, 2), seed=9)
    assert torch.equal(before, torch.get_rng_state())
