"""CPU-only contracts; synthetic labels are not scientific study evidence."""
import copy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

from matdiscovery.accounting import file_sha256, fingerprint
from matdiscovery.failure_risk import (
    HEADS, FAILURE_TYPES, FailureAwareRiskModel, FailureRiskError,
    FailureRiskTrainingConfig, FailureTypeFeatures, _episode_weights,
    masked_binary_cross_entropy,
)
from matdiscovery.uncertainty import CalibratedRiskModel, RiskTrainingConfig


@pytest.fixture
def inputs(tmp_path):
    names = ["sampling.entropy", "sampling.valid_json_action"]
    tx = np.column_stack((np.linspace(-2, 2, 12), np.arange(12) % 2)).astype(float)
    dx = np.column_stack((np.linspace(-1, 1, 8), np.arange(8) % 2)).astype(float)
    tx[2, 0] = np.nan
    ty, dy = np.full((12, len(HEADS)), np.nan), np.full((8, len(HEADS)), np.nan)
    ty[:, 0], dy[:, 0] = np.arange(12) % 2, np.arange(8) % 2
    tool, screen, unstable, not_new = (HEADS.index(n) for n in ("tool_execution_failure", "screening_unavailable", "unstable", "not_new"))
    ty[:, tool], dy[:, tool] = (np.arange(12) % 3 == 0), (np.arange(8) % 3 == 0)
    ty[:, screen], dy[:, screen] = 0, 0  # One-class train/dev.
    ty[:, unstable], dy[:, unstable] = np.arange(12) % 2, 0  # One-class dev.
    ty[:, not_new], dy[:, not_new] = 1, np.arange(8) % 2  # One-class train.
    groups = [f"t-group-{i // 4}" for i in range(12)]
    dev_groups = [f"d-group-{i // 4}" for i in range(8)]
    episodes = [f"t-episode-{i // 2}" for i in range(12)]
    dev_episodes = [f"d-episode-{i // 2}" for i in range(8)]
    runtime = {"schema": "unit_only_runtime", "dtype": "torch.float32", "device": "cpu"}
    configuration = {"policy_runtime": runtime, "enable_thinking": False}
    cfg_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    source = {"policy_runtime": runtime, "policy_configuration": configuration,
              "policy_configuration_fingerprint": cfg_hash, "model_key": "unit_only",
              "checkpoint_hash": "unit_only_initial", "benchmarks": ["made"],
              "test_used_for_fit": False, "test_used_for_threshold_selection": False,
              "source_files": [], "total_decision_rows": 20}
    source["dataset_fingerprint"] = fingerprint(source)
    # The typed rows intentionally exceed the primary future-labelled rows.
    primary = CalibratedRiskModel().fit(tx[:8], ty[:8, 0], dx[:6], dy[:6, 0],
        feature_names=names, train_groups=groups[:8], dev_groups=dev_groups[:6],
        train_episode_ids=episodes[:8], config=RiskTrainingConfig(label_kind="future_failure",
        hidden_width=4, epochs=2, patience=2, batch_size=4, seed=8))
    primary.provenance["collection_provenance"] = source
    path = tmp_path / "entropy_risk.pt"
    primary.save(path)
    return {"primary": primary, "train_x": tx, "train_y_types": ty, "dev_x": dx,
            "dev_y_types": dy, "feature_names": names, "train_groups": groups,
            "dev_groups": dev_groups, "train_episode_ids": episodes,
            "dev_episode_ids": dev_episodes, "primary_checkpoint": path,
            "expected_primary_sha256": file_sha256(path), "expected_policy_runtime": runtime,
            "provenance": source, "train_splits": ["train"] * 12, "dev_splits": ["dev"] * 8,
            "test_groups": ["heldout-family"],
            "config": FailureRiskTrainingConfig(epochs=3, batch_size=5, patience=2, seed=41)}


def fitted(inputs, **changes):
    return FailureAwareRiskModel.fit(**(inputs | changes))


def load_saved(path, receipt, inputs, **changes):
    args = dict(primary=inputs["primary"], primary_checkpoint=inputs["primary_checkpoint"],
        expected_sha256=receipt["checkpoint_sha256"], expected_primary_sha256=inputs["expected_primary_sha256"],
        expected_policy_runtime=inputs["expected_policy_runtime"], expected_provenance=inputs["provenance"])
    return FailureAwareRiskModel.load(path, **(args | changes))


def test_unknown_labels_have_exact_zero_loss_gradient():
    labels = torch.full((2, len(HEADS)), float("nan"))
    labels[0, [0, 2, 4, 6]] = torch.tensor([0., 1., 0., 1.])
    labels[1, [0, 1, 4, 6]] = torch.tensor([1., 0., 1., 0.])
    logits = torch.zeros_like(labels, requires_grad=True)
    weights = torch.ones_like(logits)
    first = masked_binary_cross_entropy(logits, labels, weights)
    gradient, = torch.autograd.grad(first, logits)
    assert torch.equal(gradient[~torch.isfinite(labels)], torch.zeros_like(gradient[~torch.isfinite(labels)]))
    changed = logits.detach().clone()
    changed[~torch.isfinite(labels)] = 1000.
    assert masked_binary_cross_entropy(changed, labels, weights) == first.detach()
    all_unknown = torch.full_like(labels, float("nan"))
    zero = masked_binary_cross_entropy(logits, all_unknown, weights)
    zero.backward()
    assert zero.item() == 0 and torch.equal(logits.grad, torch.zeros_like(logits))


def test_primary_is_unchanged_and_extra_known_type_rows_are_allowed(inputs):
    primary = inputs["primary"]
    before = primary.predict_proba(inputs["dev_x"], inputs["feature_names"])
    parameters = {k: v.clone() for k, v in primary.model.state_dict().items()}
    model = fitted(inputs)
    assert model.features is primary.features and model.model is primary.model
    assert model.provenance is primary.provenance
    np.testing.assert_array_equal(model.predict_proba(inputs["dev_x"], inputs["feature_names"]), before)
    assert all(torch.equal(v, primary.model.state_dict()[k]) for k, v in parameters.items())
    assert model.failure_provenance["training_rows"] == 12 > primary.provenance["train_rows"]
    assert model.typed_model.layers[0].out_features == model.typed_model.layers[2].out_features == 64
    assert model.typed_model.layers[-1].out_features == len(HEADS)
    assert all(p.device.type == "cpu" and not p.requires_grad for p in model.typed_model.parameters())


def test_per_head_coverage_one_class_and_unknown_heads_are_explicit(inputs):
    model = fitted(inputs)
    probabilities = model.predict_failure_types(inputs["dev_x"], inputs["feature_names"])
    available = {"generation_invalid", "tool_execution_failure"}
    assert model.head_availability == {n: n in available for n in HEADS}
    for i, name in enumerate(FAILURE_TYPES):
        if name in available:
            assert probabilities[name].shape == (8,)
            assert np.isfinite(probabilities[name]).all()
            assert ((probabilities[name] >= 0) & (probabilities[name] <= 1)).all()
        else:
            assert probabilities[name] is None and model.temperatures[name] is None
    missing = model.head_support["scientific_evaluation_failure"]
    assert missing["train"]["observed"] == 0 and missing["train"]["unknown"] == 12
    assert missing["dev"]["coverage"] == 0
    assert "dev_positive_support_missing" in model.head_support["unstable"]["reasons"]
    assert model.require_control_ready() is model


def test_generation_only_and_no_heads_cannot_control(inputs, tmp_path):
    y, dy = inputs["train_y_types"].copy(), inputs["dev_y_types"].copy()
    y[:, 1:], dy[:, 1:] = np.nan, np.nan
    model = fitted(inputs, train_y_types=y, dev_y_types=dy)
    assert model.head_availability["generation_invalid"] and not model.control_ready
    with pytest.raises(FailureRiskError, match="non-generation"):
        model.require_control_ready()
    path = tmp_path / "generation-only.pt"
    receipt = model.save(path)
    assert not load_saved(path, receipt, inputs).control_ready
    with pytest.raises(FailureRiskError, match="non-generation"):
        load_saved(path, receipt, inputs, require_control_ready=True)
    y[:], dy[:] = np.nan, np.nan
    empty = fitted(inputs, train_y_types=y, dev_y_types=dy)
    assert empty.selected_epoch == 0 and empty.history == []
    assert all(v is None for v in empty.predict_failure_types(inputs["dev_x"], inputs["feature_names"]).values())


def test_candidate_generation_failure_is_distinct_from_json_validity(inputs):
    y = np.full_like(inputs["train_y_types"], np.nan)
    dy = np.full_like(inputs["dev_y_types"], np.nan)
    json_head, candidate_head = HEADS.index("generation_invalid"), HEADS.index("candidate_generation_failure")
    y[:, json_head], dy[:, json_head] = 0, 0  # Every proposal is well-formed JSON.
    y[:, candidate_head], dy[:, candidate_head] = np.arange(len(y)) % 2, np.arange(len(dy)) % 2
    model = fitted(inputs, train_y_types=y, dev_y_types=dy)
    output = model.predict_failure_types(inputs["dev_x"], inputs["feature_names"])
    assert output["generation_invalid"] is None
    assert output["candidate_generation_failure"].shape == (8,)
    assert model.require_control_ready() is model
    assert model.head_support["candidate_generation_failure"]["train"]["positive"] == 6
    assert "RPC" in model.failure_provenance["candidate_generation_failure_interpretation"]


def test_train_only_features_and_prototypes_ignore_development_values(inputs):
    first = fitted(inputs)
    second = fitted(inputs, dev_x=inputs["dev_x"] * 100 + 1000,
                    dev_y_types=inputs["dev_y_types"][::-1].copy())
    assert first.typed_features.state() == second.typed_features.state()
    before = copy.deepcopy(first.typed_features.state())
    first.predict_failure_types(np.full((3, 2), np.nan), inputs["feature_names"])
    assert first.typed_features.state() == before
    assert first.failure_provenance["epoch_selection_split"] == first.failure_provenance["temperature_calibration_split"] == "dev"
    assert first.failure_provenance["test_used_for_fit"] is False
    assert first.failure_provenance["test_used_for_calibration"] is False


def test_episode_weighted_error_prototype_is_not_long_trajectory_mean():
    x = np.array([[-1.], [1.], [10.], [10.], [10.]])
    y = np.full((5, len(HEADS)), np.nan)
    head = HEADS.index("tool_execution_failure")
    y[:, head] = [0, 1, 1, 1, 1]
    groups, episodes = ["g"] * 5, ["a", "b", "c", "c", "c"]
    f = FailureTypeFeatures().fit(x, y, ["sampling.x"], groups, episodes, np.ones(5))
    assert f.mean[0] == pytest.approx((-1 + 1 + 10) / 3)
    expected = ((1 - f.mean[0]) / f.std[0] + (10 - f.mean[0]) / f.std[0]) / 2
    assert f.error_prototypes[head, 0] == pytest.approx(expected)
    assert not f.has_error_prototype[0]
    weights = _episode_weights(groups, episodes, np.ones(5), y[:, head] == 1)
    assert weights.tolist() == pytest.approx([0, .5, 1/6, 1/6, 1/6])
    f2 = FailureTypeFeatures().fit(np.concatenate([x, x[-1:]]), np.concatenate([y, y[-1:]]),
        ["sampling.x"], groups + ["g"], episodes + ["c"], np.ones(6))
    np.testing.assert_allclose(f.error_prototypes, f2.error_prototypes, atol=1e-14)


@pytest.mark.parametrize("change,match", [
    ({"train_splits": ["train"] * 11 + ["test"]}, "Test/undeclared"),
    ({"dev_splits": ["dev"] * 7 + ["train"]}, "Test/undeclared"),
    ({"dev_groups": ["t-group-0"] * 8}, "leakage"),
    ({"test_groups": ["d-group-0"]}, "leakage"),
    ({"feature_names": ["sampling.valid_json_action", "sampling.entropy"]}, "exactly equal"),
    ({"expected_primary_sha256": "0" * 64}, "Primary checkpoint SHA256"),
    ({"expected_policy_runtime": {"dtype": "torch.bfloat16"}}, "runtime differs"),
    ({"train_sample_weights": [0.] * 12}, "strictly positive"),
])
def test_leakage_schema_weights_and_identity_fail_closed(inputs, change, match):
    with pytest.raises((FailureRiskError, ValueError), match=match):
        fitted(inputs, **change)


def test_test_tainted_or_changed_provenance_is_rejected(inputs):
    source = copy.deepcopy(inputs["provenance"])
    source["test_used_for_fit"] = True
    with pytest.raises(FailureRiskError, match="exclude test"):
        fitted(inputs, provenance=source)
    source = copy.deepcopy(inputs["provenance"])
    source["total_decision_rows"] += 1
    with pytest.raises(FailureRiskError, match="dataset fingerprint"):
        fitted(inputs, provenance=source)
    source["dataset_fingerprint"] = fingerprint({k: v for k, v in source.items() if k != "dataset_fingerprint"})
    with pytest.raises(FailureRiskError, match="provenance"):
        fitted(inputs, provenance=source)


def test_labels_are_observed_binary_or_nan(inputs):
    for value in (2, np.inf):
        labels = inputs["train_y_types"].copy()
        labels[0, 1] = value
        with pytest.raises(FailureRiskError, match="observed 0/1"):
            fitted(inputs, train_y_types=labels)
    with pytest.raises(FailureRiskError, match=f"N by {len(HEADS)}"):
        fitted(inputs, dev_y_types=inputs["dev_y_types"][:, :5])


def test_primary_in_memory_tampering_is_rejected(inputs):
    with torch.no_grad():
        next(inputs["primary"].model.parameters()).add_(.1)
    with pytest.raises(FailureRiskError, match="in-memory state"):
        fitted(inputs)


def test_checkpoint_roundtrip_and_external_inner_tamper_guards(inputs, tmp_path):
    model = fitted(inputs)
    path = tmp_path / "entropy_risk.failure_heads.pt"
    receipt = model.save(path)
    restored = load_saved(path, receipt, inputs, require_control_ready=True)
    assert restored.head_support == model.head_support
    assert restored.failure_provenance == model.failure_provenance
    assert restored.typed_features.state() == model.typed_features.state()
    before = model.predict_failure_types(inputs["dev_x"], inputs["feature_names"])
    after = restored.predict_failure_types(inputs["dev_x"], inputs["feature_names"])
    for name in FAILURE_TYPES:
        if before[name] is None:
            assert after[name] is None
        else:
            np.testing.assert_array_equal(before[name], after[name])
    with pytest.raises(FileExistsError):
        model.save(path)
    with pytest.raises(FailureRiskError, match="schema/order"):
        restored.predict_failure_types(inputs["dev_x"], inputs["feature_names"][::-1])
    with pytest.raises(FailureRiskError, match="Typed checkpoint SHA256"):
        load_saved(path, receipt, inputs, expected_sha256="0" * 64)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["model"]["layers.4.bias"][1] += 1
    torch.save(state, path)
    with pytest.raises(FailureRiskError, match="Typed checkpoint SHA256"):
        load_saved(path, receipt, inputs)
    with pytest.raises(FailureRiskError, match="content checksum"):
        load_saved(path, receipt, inputs, expected_sha256=file_sha256(path))


def test_loaded_checkpoint_rejects_changed_runtime_primary_and_source(inputs, tmp_path):
    path = tmp_path / "heads.pt"
    receipt = fitted(inputs).save(path)
    with pytest.raises(FailureRiskError, match="runtime differs"):
        load_saved(path, receipt, inputs, expected_policy_runtime={"dtype": "bf16"})
    changed = copy.deepcopy(inputs["provenance"])
    changed["benchmarks"] = ["crystalgym"]
    changed["dataset_fingerprint"] = fingerprint({k: v for k, v in changed.items() if k != "dataset_fingerprint"})
    with pytest.raises(FailureRiskError, match="provenance"):
        load_saved(path, receipt, inputs, expected_provenance=changed)
    inputs["primary_checkpoint"].write_bytes(b"changed primary checkpoint")
    with pytest.raises(FailureRiskError, match="Primary checkpoint SHA256"):
        load_saved(path, receipt, inputs)


def test_fitting_is_deterministic_without_consuming_global_rng(inputs):
    state = torch.get_rng_state().clone()
    first = fitted(inputs)
    assert torch.equal(torch.get_rng_state(), state)
    second = fitted(inputs)
    assert first.history == second.history and first.temperatures == second.temperatures
    assert all(torch.equal(v, second.typed_model.state_dict()[k]) for k, v in first.typed_model.state_dict().items())


def test_architecture_and_hyperparameters_are_explicit():
    assert HEADS == ("generation_invalid", "candidate_generation_failure", "tool_execution_failure",
                     "screening_unavailable", "scientific_evaluation_failure", "unstable", "not_new")
    with pytest.raises(FailureRiskError, match="64-wide"):
        FailureRiskTrainingConfig(hidden_width=8)
    with pytest.raises(FailureRiskError):
        replace(FailureRiskTrainingConfig(), epochs=0)
