"""Mathematical and persistence checks; these are NOT material-discovery runs."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from matdiscovery.esopt import AgenticESOpt, cosine_sigma, iter_noise, population_zscores, stable_parameter_seed


class TinyHFPolicy(torch.nn.Module):
    """Tiny test double exposing only the canonical HF policy contract."""
    def __init__(self, dtype=torch.float32, tied=False):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny_causal_lm")
        self.embed_tokens = torch.nn.Embedding(4, 3, dtype=dtype)
        self.projection = torch.nn.Linear(3, 3, dtype=dtype)
        self.lm_head = torch.nn.Linear(3, 4, bias=False, dtype=dtype)
        if tied:
            self.lm_head.weight = self.embed_tokens.weight
        self.register_buffer("model_version", torch.tensor(1))
        self.eval()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def optimizer(model, **kwargs):
    return AgenticESOpt(model, policy_model_id="test/tiny@fixed", noise_chunk_size=5, **kwargs)


def noise_tensor(seed, name, parameter, chunk_size=5):
    return torch.cat([noise for _, noise in iter_noise(seed, name, parameter.numel(), chunk_size)]).reshape(parameter.shape)


def test_full_update_matches_population_formula_without_sigma_division():
    torch.manual_seed(10)
    model = TinyHFPolicy()
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    es = optimizer(model)
    seeds, rewards, alpha = [11, 22, 33], [0.0, 1.0, 4.0], 0.02
    z = population_zscores(rewards)
    record = es.step(seeds, rewards, alpha=alpha, sigma=0.001)
    for name, parameter in model.named_parameters():
        expected_delta = sum(weight * noise_tensor(seed, name, parameter) for weight, seed in zip(z, seeds)) * (alpha / len(seeds))
        torch.testing.assert_close(parameter, initial[name] + expected_delta, rtol=1e-6, atol=1e-7)
    assert record["state_hash_before"] != record["state_hash_after"]
    assert record["ddof"] == 0
    assert es.total_parameters == sum(p.numel() for p in model.parameters())
    assert all(record["parameter_delta_l2"][name] > 0 for name in initial)


def test_sigma_is_perturbation_scale_not_an_update_denominator():
    torch.manual_seed(3)
    first = TinyHFPolicy()
    second = copy.deepcopy(first)
    a, b = optimizer(first), optimizer(second)
    a.step([1, 2], [1, 0], alpha=0.005, sigma=0.1)
    b.step([1, 2], [1, 0], alpha=0.005, sigma=0.00001)
    assert a.model_state_hash() == b.model_state_hash()


def test_population_zscore_and_constant_reward_noop():
    assert population_zscores([1, 3]) == pytest.approx([-1, 1])
    assert population_zscores([2, 2, 2]) == [0, 0, 0]
    es = optimizer(TinyHFPolicy())
    before = es.model_state_hash()
    es.step([1, 2], [8, 8], alpha=0.2, sigma=0.3)
    assert before == es.model_state_hash()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_trajectory_noise_is_one_sided_and_cpu_restore_is_bit_exact(dtype):
    torch.manual_seed(7)
    model = TinyHFPolicy(dtype=dtype)
    es = optimizer(model)
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    original_hash = es.model_state_hash()
    for iteration in range(4):
        with es.perturbation(seed=123, sigma=0.17):
            perturbed_hash = es.model_state_hash()
            assert original_hash != perturbed_hash
            for name, parameter in model.named_parameters():
                expected = original[name].clone()
                expected.add_(noise_tensor(123, name, parameter).to(dtype), alpha=0.17)
                assert torch.equal(parameter, expected)
            # Multiple turns observe the same perturbation, without resampling.
            assert perturbed_hash == es.model_state_hash()
            with pytest.raises(RuntimeError):
                es.step([1, 2], [0, 1], alpha=0.1, sigma=0.1)
            with pytest.raises(RuntimeError):
                es.begin_perturbation(seed=456, sigma=0.1)
        assert es.model_state_hash() == original_hash
        assert all(torch.equal(parameter, original[name]) for name, parameter in model.named_parameters())
    with pytest.raises(RuntimeError, match="trajectory failed"):
        with es.perturbation(seed=5, sigma=0.02):
            raise RuntimeError("trajectory failed")
    assert es.model_state_hash() == original_hash
    assert es.active_perturbation is None


def test_seed_and_noise_are_stable_across_processes():
    command = "from matdiscovery.esopt import stable_parameter_seed, iter_noise; import json; print(json.dumps([stable_parameter_seed(42,'projection.weight'), [n.tolist() for _, n in iter_noise(42,'projection.weight',9,5)]]))"
    outputs = []
    for python_hash_seed in ("1", "98765"):
        env = dict(os.environ, PYTHONHASHSEED=python_hash_seed)
        outputs.append(subprocess.check_output([sys.executable, "-c", command], env=env, text=True).strip())
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])[0] == stable_parameter_seed(42, "projection.weight")


def test_scope_manifest_tied_weights_and_auxiliary_rejection():
    tied = TinyHFPolicy(tied=True)
    full = optimizer(tied)
    names = [item["name"] for item in full.parameter_manifest()]
    assert "embed_tokens.weight" in names and "projection.bias" in names
    embedding = next(item for item in full.parameter_manifest() if item["name"] == "embed_tokens.weight")
    assert set(embedding["aliases"]) == {"embed_tokens.weight", "lm_head.weight"}
    linear = optimizer(TinyHFPolicy(), parameter_scope="all_linear")
    assert [item.name for item in linear.manifest] == ["projection.weight"]
    with pytest.raises(ValueError, match="no parameters"):
        optimizer(TinyHFPolicy(), parameter_scope="lora")
    wrapped = TinyHFPolicy()
    wrapped.transcoders = torch.nn.Linear(3, 3)
    with pytest.raises(ValueError, match="forbidden"):
        optimizer(wrapped)
    with pytest.raises(ValueError, match="canonical HF"):
        optimizer(torch.nn.Linear(3, 3))


def test_history_replay_and_checkpoint_resume(tmp_path):
    torch.manual_seed(17)
    base = TinyHFPolicy(dtype=torch.bfloat16)
    trained = copy.deepcopy(base)
    es = optimizer(trained)
    es.step([2, 3, 4], [1, 0, 2], alpha=0.03, sigma=0.1, metadata={"split": "train"})
    es.step([5, 6, 7], [0, 2, 1], alpha=0.01, sigma=0.05)
    history = tmp_path / "history.json"
    checkpoint = tmp_path / "policy.pt"
    es.save_history(history)
    hashes = es.save_checkpoint(checkpoint)
    assert len(hashes["file_sha256"]) == 64
    assert hashes["model_state_hash"] == es.model_state_hash()
    replay = optimizer(copy.deepcopy(base))
    replay.replay_history(history)
    assert replay.model_state_hash() == es.model_state_hash()
    resumed = optimizer(copy.deepcopy(base))
    resumed.load_checkpoint(checkpoint)
    assert resumed.model_state_hash() == es.model_state_hash()
    assert resumed.history == es.history
    next_args = dict(seeds=[8, 9], rewards=[0, 1], alpha=0.01, sigma=0.01)
    resumed.step(**next_args)
    es.step(**next_args)
    assert resumed.model_state_hash() == es.model_state_hash()
    wrong = optimizer(TinyHFPolicy(dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="base checkpoint hash"):
        wrong.replay_history(history)


def test_incompatible_and_tampered_checkpoint_rejected(tmp_path):
    model = TinyHFPolicy()
    es = optimizer(model)
    path = tmp_path / "policy.pt"
    es.save_checkpoint(path)
    incompatible = AgenticESOpt(copy.deepcopy(model), policy_model_id="other@revision", noise_chunk_size=5)
    with pytest.raises(ValueError, match="policy_model_id"):
        incompatible.load_checkpoint(path)
    data = torch.load(path, weights_only=True)
    data["state_dict"]["projection.weight"][0, 0] += 1
    torch.save(data, path)
    with pytest.raises(ValueError, match="content hash"):
        es.load_checkpoint(path)


def test_invalid_inputs_and_cosine_endpoints():
    assert cosine_sigma(0.1, 0.01, 0, 5) == pytest.approx(0.1)
    assert cosine_sigma(0.1, 0.01, 4, 5) == pytest.approx(0.01)
    assert cosine_sigma(0.1, 0.01, 0, 1) == 0.1
    es = optimizer(TinyHFPolicy())
    before = es.model_state_hash()
    for seeds, rewards in [([1, 1], [0, 1]), ([1], []), ([1, 2], [0, float("nan")])]:
        with pytest.raises(ValueError):
            es.step(seeds, rewards, alpha=0.1, sigma=0.1)
    with pytest.raises(ValueError):
        es.step([1, 2], [0, 1], alpha=0.1, sigma=0.1, metadata={"bad": float("nan")})
    assert before == es.model_state_hash()


def test_unrecorded_changes_and_failed_updates_do_not_silently_corrupt_weights():
    model = TinyHFPolicy(dtype=torch.bfloat16)
    es = optimizer(model)
    before = es.model_state_hash()
    with pytest.raises((ValueError, RuntimeError)):
        es.step([1, 2], [0, 1], alpha=1e100, sigma=0.1)
    assert es.model_state_hash() == before
    assert not es.history
    with torch.no_grad():
        model.projection.weight[0, 0] += 1
    with pytest.raises(ValueError, match="outside"):
        es.step([1, 2], [0, 1], alpha=0.01, sigma=0.1)


def test_noise_does_not_consume_global_torch_rng():
    torch.manual_seed(12)
    rng = torch.get_rng_state().clone()
    list(iter_noise(7, "projection.weight", 20, 5))
    assert torch.equal(rng, torch.get_rng_state())
