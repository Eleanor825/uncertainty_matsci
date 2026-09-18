"""Real tiny CPU fits test the full32 publication contract, never materials."""
from pathlib import Path

import pytest
import torch

from matdiscovery.accounting import file_sha256
from matdiscovery.parallel_normalized_bank import build_bank, inventory, verify_bank_recipe, provider
from matdiscovery.representation_training import TranscoderStageConfig, train_transcoders_from_collections
from matdiscovery.transcoders import TranscoderConfig
from test_deferred_transcoder_publisher import collections
from torch_runtime_fixture import restore_torch_runtime


def test_full32_normalized_cpu_fits_original_filter_and_read_only_resume(tmp_path):
    torch.set_num_threads(2)
    items = collections(tmp_path / "collections")
    manifests = [x["manifest"] for x in items]
    config = TranscoderStageConfig(feature_dim=8, top_k=4, epochs=64, batch_size=8,
        learning_rate=.03, max_dev_fvu=.5, device="cpu")
    output, runs = tmp_path / "bank", tmp_path / "runs"
    registration = {"core_fingerprint": "unit-no-material-experiment", "fit_recipe": "train_centered_scalar_rms_fp32_export_v1"}
    bank = build_bank(manifests, output, runs, model_key="qwen35_4b", config=config, registration=registration)
    assert bank["complete"] and bank["passed_layers"] == 32
    assert bank["development_exact_prefix_filter"]["duplicate_dev_activation_rows"] == 4 * 32
    verify_bank_recipe(bank, core_fingerprint=registration["core_fingerprint"])
    before = inventory(output)
    assert train_transcoders_from_collections(manifests, output, model_key="qwen35_4b", config=config, resume=True) == bank
    assert inventory(output) == before
    with pytest.raises(ValueError, match="Fresh registered"):
        build_bank(manifests, output, runs, model_key="qwen35_4b", config=config, registration=registration)
    (runs / "layers/layer_00/selected.pt.json").write_text("{}")
    with pytest.raises(ValueError, match="selected model changed"):
        verify_bank_recipe(bank, core_fingerprint=registration["core_fingerprint"])


def test_provider_restores_original_function_after_invalid_admission(tmp_path):
    from matdiscovery import transcoders
    original = transcoders.train_layer_transcoder
    with pytest.raises(FileNotFoundError):
        with provider(tmp_path / "run", {"core_fingerprint": "unit"}):
            transcoders.train_layer_transcoder(TranscoderConfig(3, 2, 8, 4), [], [],
                output_path=tmp_path / "bank/layer_00.pt", policy_fingerprint="unit",
                layer_path="model.language_model.layers.0.mlp")
    assert transcoders.train_layer_transcoder is original
