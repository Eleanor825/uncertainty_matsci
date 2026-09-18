"""CPU64 reference parity, train-only statistics and raw-export contracts."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

from matdiscovery.accounting import file_sha256
from matdiscovery.esopt import tensor_state_hash
from matdiscovery import normalized_transcoders as production
from matdiscovery.transcoders import (TranscoderConfig, TopKTranscoder, write_activation_shard,
    iter_activation_batches, reconstruction_metrics, load_transcoder)
from torch_runtime_fixture import restore_torch_runtime

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "technical_not_main/tc_scale_diagnostic_v2"


def reference_modules():
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module
    helper = load("frozen_affine_helper", REFERENCE / "verify_affine_fold.py")
    previous = sys.modules.get("verify_affine_fold")
    sys.modules["verify_affine_fold"] = helper
    try:
        diagnostic = load("frozen_normalized_diagnostic", REFERENCE / "run_normalized_diagnostic.py")
    finally:
        if previous is None: sys.modules.pop("verify_affine_fold", None)
        else: sys.modules["verify_affine_fold"] = previous
    return helper, diagnostic


def sources(root):
    generator = torch.Generator().manual_seed(712)
    paths = {"train": [], "dev": []}
    for index, (split, rows) in enumerate((("train", 11), ("train", 7), ("train", 9), ("dev", 13))):
        x = torch.randn(rows, 3, generator=generator) * .37 + torch.tensor([1.1, -.7, .3])
        y = x @ torch.tensor([[.02, -.03], [.04, .01], [-.03, .04]]) + torch.tensor([.8, -.2])
        path = root / f"{split}_{index}.pt"
        write_activation_shard(path, x, y, group_ids=[split] * rows,
            prefix_hashes=[f"{split}-{index}-{i}" for i in range(rows)], split=split,
            policy_fingerprint="unit-raw-policy", layer_path="unit.mlp")
        paths[split].append(path)
    return paths


def original_reference(config, train, dev, *, seed, batch_size, learning_rate):
    """Frozen diagnostic loop with only its tensor device/dimensions parameterized."""
    helper, diagnostic = reference_modules()
    stats = helper.train_statistics(train)
    model = TopKTranscoder(config, seed=seed)
    raw = helper.fold_to_original_topk(model, stats)
    mx, my = stats["mean_x"].float(), stats["mean_y"].float()
    sx, sy = stats["scale_x"], stats["scale_y"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.)
    history, best, best_state, best_epoch = [], float("inf"), None, None
    for epoch in range(64):
        model.train(); loss_sum, rows = 0., 0
        for x, y in iter_activation_batches(train, batch_size, seed=seed + epoch, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model((x - mx) / sx), (y - my) / sy)
            loss.backward(); optimizer.step(); model.normalize_decoder_()
            loss_sum += loss.item() * len(x); rows += len(x)
        model.eval()
        with torch.no_grad():
            raw.encoder.weight.copy_((sy / sx) * model.encoder.weight)
            raw.encoder.bias.copy_(sy * (model.encoder.bias - model.encoder.weight @ mx / sx))
            raw.decoder.weight.copy_(model.decoder.weight)
            raw.decoder.bias.copy_(sy * model.decoder.bias + my)
            diagnostic.validate_raw_mapping(model, raw, mx, my, sx, sy)
            audit = diagnostic.audit_fold(model, raw, x, mx, my, sx, sy)
            train_raw = reconstruction_metrics(raw, iter_activation_batches(train, batch_size, seed=seed, shuffle=False))
            dev_raw = reconstruction_metrics(raw, iter_activation_batches(dev, batch_size, seed=seed, shuffle=False))
        history.append({"epoch": epoch, "train_rows": rows, "train_normalized_batch_mse": loss_sum / rows,
            "train_raw_end_epoch": train_raw, "dev": dev_raw, "fold_audit": audit,
            "fold_max_absolute_error": audit["output_max_absolute_error"], "train_output_mse": train_raw["output_mse"]})
        if dev_raw["output_mse"] < best:
            best, best_epoch = dev_raw["output_mse"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()}
    raw.load_state_dict(best_state)
    return raw, history, best_epoch, stats


def test_frozen_statistics_fold_and_boundary_functions_have_identical_AST():
    assert file_sha256(REFERENCE / "run_normalized_diagnostic.py") == production.DIAGNOSTIC_REFERENCE_SHA256
    assert file_sha256(REFERENCE / "verify_affine_fold.py") == production.STATISTICS_REFERENCE_SHA256
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(Path(path).read_text()).body if isinstance(node, ast.FunctionDef)}
    actual = functions(production.__file__)
    expected = functions(REFERENCE / "run_normalized_diagnostic.py") | functions(REFERENCE / "verify_affine_fold.py")
    for name in ("train_statistics", "fold_to_original_topk", "validate_raw_mapping", "audit_fold"):
        assert actual[name] == expected[name]


def test_cpu_complete64_exact_reference_raw_selection_and_original_loader(tmp_path):
    torch.set_num_threads(2)
    paths = sources(tmp_path)
    config = TranscoderConfig(3, 2, 8, 4)
    reference, history, selected, stats = original_reference(config, paths["train"], paths["dev"], seed=1730, batch_size=8, learning_rate=.0004)
    output = tmp_path / "bank/layer_00.pt"
    model, metadata = production.train_normalized_layer(config, paths["train"], paths["dev"],
        policy_fingerprint="unit-raw-policy", layer_path="unit.mlp", output_path=output,
        seed=1730, epochs=64, batch_size=8, learning_rate=.0004, device="cpu", registration={"unit": True})
    assert metadata["history"] == history and metadata["selected_epoch"] == selected
    assert model.checkpoint_hash() == reference.checkpoint_hash()
    assert all(torch.equal(model.state_dict()[key], reference.state_dict()[key]) for key in model.state_dict())
    assert metadata["normalization"]["mean_x"] == stats["mean_x"].tolist()
    assert metadata["normalization"]["mean_y"] == stats["mean_y"].tolist()
    assert metadata["normalization"]["scale_x"] == stats["scale_x"] and metadata["normalization"]["scale_y"] == stats["scale_y"]
    assert all(row["train_rows"] == 27 and row["dev"]["rows"] == 13 for row in history)
    assert any(row["train_output_mse"] != row["train_normalized_batch_mse"] for row in history)
    assert {x.name for x in output.parent.iterdir()} == {"layer_00.pt", "layer_00.pt.json"}
    restored, restored_metadata = load_transcoder(output, expected_policy_fingerprint="unit-raw-policy", expected_layer_path="unit.mlp")
    assert restored_metadata == metadata and restored.checkpoint_hash() == model.checkpoint_hash()
    assert production.verify_normalized_metadata(metadata, config=config, policy_fingerprint="unit-raw-policy",
        layer_path="unit.mlp", train_paths=paths["train"], dev_paths=paths["dev"], expected_registration={"unit": True}) == metadata
    for mutate in (lambda m: m["normalization"]["mean_x"].__setitem__(0, 99.),
                   lambda m: m["history"][0].__setitem__("train_output_mse", 77.),
                   lambda m: m.__setitem__("fit_recipe", "unknown")):
        bad = copy.deepcopy(metadata); mutate(bad)
        with pytest.raises(ValueError): production.verify_normalized_metadata(bad)
    with pytest.raises(ValueError, match="Existing/partial"):
        production.train_normalized_layer(config, paths["train"], paths["dev"], policy_fingerprint="unit-raw-policy",
            layer_path="unit.mlp", output_path=output, seed=1730, device="cpu")


def test_statistics_reject_dev_and_zero_variance_and_keep_every_train_row(tmp_path):
    paths = sources(tmp_path)
    stats = production.train_statistics(paths["train"])
    assert stats["rows"] == 27 and len(stats["sources"]) == 3
    with pytest.raises(ValueError, match="training shards only"):
        production.train_statistics(paths["train"] + paths["dev"])
    path = tmp_path / "constant.pt"
    write_activation_shard(path, torch.ones(3, 3), torch.ones(3, 2), group_ids=["train"] * 3,
        prefix_hashes=["c"] * 3, split="train", policy_fingerprint="unit", layer_path="unit")
    with pytest.raises(ValueError, match="Undefined centered RMS"):
        production.train_statistics([path])


def test_raw_sources_reject_test_role_before_model_or_optimizer(tmp_path, monkeypatch):
    paths = sources(tmp_path)
    path = paths["dev"][0]
    shard = torch.load(path, weights_only=True); shard["metadata"]["split"] = "test"; torch.save(shard, path)
    monkeypatch.setattr(production, "TopKTranscoder", lambda *a, **kw: pytest.fail("must reject before a model is created"))
    with pytest.raises(ValueError, match="test is forbidden"):
        production.train_normalized_layer(TranscoderConfig(3, 2, 8, 4), paths["train"], paths["dev"],
            policy_fingerprint="unit-raw-policy", layer_path="unit.mlp", output_path=tmp_path / "bad.pt", seed=0, device="cpu")


def test_fold_positive_scalars_unit_decoder_and_boundary_audit_is_truthful():
    helper, diagnostic = reference_modules()
    torch.manual_seed(11)
    model = TopKTranscoder(TranscoderConfig(7, 5, 16, 4), seed=41)
    stats = {"mean_x": torch.randn(7), "mean_y": torch.randn(5), "scale_x": .37, "scale_y": .019}
    raw = production.fold_to_original_topk(model, stats)
    x = torch.randn(31, 7)
    audit = production.audit_fold(model, raw, x, stats["mean_x"], stats["mean_y"], .37, .019)
    assert audit == diagnostic.audit_fold(model, raw, x, stats["mean_x"], stats["mean_y"], .37, .019)
    assert audit["claim"] == "real_arithmetic_equivalence_FP32_cutoff_discontinuities_recorded"
    assert torch.equal(raw.decoder.weight, model.decoder.weight)
    assert audit["support_changes_outside_rounding_band"] == audit["sign_changes_outside_rounding_band"] == 0
    for sx, sy in ((0, 1), (1, -1), (float("nan"), 1)):
        with pytest.raises(ValueError): production.fold_to_original_topk(model, {**stats, "scale_x": sx, "scale_y": sy})
