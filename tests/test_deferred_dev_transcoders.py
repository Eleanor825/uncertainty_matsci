"""Tiny CPU arithmetic/recovery proof, not policy or materials results."""
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from matdiscovery.accounting import file_sha256
from matdiscovery.deferred_dev_transcoders import precompute_layer, precompute_parallel, finalize_development
from matdiscovery.transcoders import TranscoderConfig, train_layer_transcoder, write_activation_shard


def inputs(root, layer):
    generator = torch.Generator().manual_seed(73)
    result = []
    for split in ("train", "dev"):
        x = torch.rand(33, 3, generator=generator)
        y = x @ torch.tensor([[.4, -.2], [.2, .5], [-.1, .6]]) + .1
        path = root / layer.replace(".", "_") / (split + ".pt")
        write_activation_shard(path, x, y, group_ids=[split] * len(x),
            prefix_hashes=[split + str(i) for i in range(len(x))], split=split,
            policy_fingerprint="unit-policy-no-real-weights", layer_path=layer)
        result.append(path)
    return result


@pytest.mark.parametrize("epochs,workers", [(16, 2), (64, 4)])
def test_spawn_workers_exact_all_epoch_and_original_dev_selection_parity(tmp_path, monkeypatch, epochs, workers):
    import matdiscovery.transcoders as original
    torch.set_num_threads(2)
    config = TranscoderConfig(3, 2, 8, 4)
    tasks, references, captured = [], [], []
    real_metrics = original.reconstruction_metrics
    def capture(model, batches):
        captured.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
        return real_metrics(model, batches)
    monkeypatch.setattr(original, "reconstruction_metrics", capture)
    for index in range(workers):
        layer = f"model.language_model.layers.{index}.mlp"
        train, dev = inputs(tmp_path, layer)
        common = dict(policy_fingerprint="unit-policy-no-real-weights", layer_path=layer,
            seed=1729 + index, epochs=epochs, batch_size=8, learning_rate=.03, device="cpu", max_dev_fvu=.5)
        captured.clear()
        model, metadata = train_layer_transcoder(config, [train], [dev], output_path=tmp_path / f"serial_{index}.pt", **common)
        references.append((metadata, list(captured), model.checkpoint_hash(), dev))
        tasks.append({"config": asdict(config), "train_paths": [str(train)],
                      "output_dir": str(tmp_path / f"pre_{index}"), "kwargs": {**common, "cpu_threads": 2}})
    results = precompute_parallel(tasks, max_workers=workers)
    for task, record, (expected, epoch_states, state_hash, dev) in zip(tasks, results, references):
        assert record["development_read"] is False and record["selected_checkpoint"] is None
        assert len(record["epochs"]) == len(epoch_states) == epochs
        for epoch, original_state in zip(record["epochs"], epoch_states):
            actual = torch.load(epoch["path"], map_location="cpu", weights_only=True)["state_dict"]
            assert all(torch.equal(actual[key], value) for key, value in original_state.items())
        output = tmp_path / (Path(task["output_dir"]).name + "_selected.pt")
        model, metadata = finalize_development(task["output_dir"], [dev], output_path=output)
        assert metadata == expected  # Includes every MSE/FVU, best epoch and full original provenance.
        assert model.checkpoint_hash() == state_hash


def test_completed_precompute_is_verified_without_rewrite_and_partial_or_tamper_rejected(tmp_path):
    layer = "model.language_model.layers.0.mlp"; train, _ = inputs(tmp_path, layer)
    config = TranscoderConfig(3, 2, 8, 4)
    kwargs = dict(policy_fingerprint="unit-policy-no-real-weights", layer_path=layer,
        epochs=16, seed=1729, device="cpu", batch_size=8, learning_rate=.03)
    folder = tmp_path / "pre"
    result = precompute_layer(config, [train], folder, **kwargs)
    before = {p: (file_sha256(p), p.stat().st_mtime_ns) for p in folder.iterdir() if p.is_file()}
    assert precompute_layer(config, [train], folder, **kwargs) == result
    assert before == {p: (file_sha256(p), p.stat().st_mtime_ns) for p in folder.iterdir() if p.is_file()}
    (folder / "epoch_001.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Epoch bytes"):
        precompute_layer(config, [train], folder, **kwargs)
    partial = tmp_path / "partial"; partial.mkdir(); (partial / "started.json").write_text("{}")
    with pytest.raises(ValueError, match="never retrain"):
        precompute_layer(config, [train], partial, **kwargs)


def test_test_shards_are_never_accepted_and_failed_fvu_is_preserved(tmp_path):
    layer = "model.language_model.layers.0.mlp"; train, dev = inputs(tmp_path, layer)
    config = TranscoderConfig(3, 2, 8, 4)
    common = dict(policy_fingerprint="unit-policy-no-real-weights", layer_path=layer,
        seed=1729, device="cpu", epochs=16, max_dev_fvu=0.)
    with pytest.raises(ValueError, match="training layer"):
        precompute_layer(config, [dev], tmp_path / "wrong", **common)
    precompute_layer(config, [train], tmp_path / "pre", **common)
    _, result = finalize_development(tmp_path / "pre", [dev], output_path=tmp_path / "failed.pt")
    assert result["fidelity_gate_passed"] is False and result["dev"]["output_fvu"] > 0
    with pytest.raises(ValueError, match="never overwrite"):
        finalize_development(tmp_path / "pre", [dev], output_path=tmp_path / "failed.pt")
