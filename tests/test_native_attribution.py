"""Cut-Jacobian finite differences on tiny native networks only.

Passing these tests does NOT validate Qwen/Gemma/DeepSeek or any material task.
Each real HF backend must separately pass NativeAttributor.validate_backend.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from matdiscovery.esopt import tensor_state_hash
from matdiscovery.graph_features import extract_graph_features
from matdiscovery.native_attribution import (LEGACY_CONTRACT, NativeAttributor, PolicyStamp, TranscoderBinding, UnsupportedAttribution)
from matdiscovery.transcoders import TopKTranscoder, TranscoderConfig


class TinyLayer(torch.nn.Module):
    def __init__(self, factor, nonlinear=False):
        super().__init__()
        self.mlp = torch.nn.Linear(1, 1, bias=False)
        self.mlp.weight.data.fill_(factor)
        self.nonlinear = nonlinear

    def forward(self, x):
        if self.nonlinear:
            x = x + 0.1 * x.cumsum(dim=1)
            normalized = x / torch.sqrt(x.square() + 0.7)
        else:
            normalized = x
        return x + self.mlp(normalized)


class TinyNativeHF(torch.nn.Module):
    def __init__(self, nonlinear=False):
        super().__init__()
        self.config = SimpleNamespace(model_type="tiny_native", _attn_implementation="tiny_eager")
        self.embed_tokens = torch.nn.Embedding(4, 1)
        self.embed_tokens.weight.data.copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        self.layers = torch.nn.ModuleList([TinyLayer(2, nonlinear), TinyLayer(3, nonlinear)])
        self.lm_head = torch.nn.Linear(1, 2, bias=False)
        self.lm_head.weight.data.copy_(torch.tensor([[1.0], [-1.0]]))
        self.eval()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, use_cache=False, **kwargs):
        assert use_cache is False
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return SimpleNamespace(logits=self.lm_head(x))


def setup(nonlinear=False, **kwargs):
    policy = TinyNativeHF(nonlinear)
    fingerprint = tensor_state_hash(dict(policy.state_dict()))
    stamp = PolicyStamp("version-0", fingerprint, "tiny-test-only", 0)
    version = [stamp.state_id]
    bindings = []
    for layer, factor in enumerate((2, 3)):
        transcoder = TopKTranscoder(TranscoderConfig(1, 1, 1, 1))
        with torch.no_grad():
            transcoder.encoder.weight.fill_(factor)
            transcoder.encoder.bias.zero_()
            transcoder.decoder.weight.fill_(1)
            transcoder.decoder.bias.zero_()
        path = f"layers.{layer}.mlp"
        metadata = {"fidelity_gate_passed": True, "max_dev_fvu": 0.01,
                    "transcoder_hash": transcoder.checkpoint_hash(),
                    "provenance": {"layer_path": path, "policy_fingerprint": fingerprint},
                    "fixture_only": True}
        bindings.append(TranscoderBinding(path, transcoder, metadata))
    attributor = NativeAttributor(policy, bindings, state_id_getter=lambda: version[0],
        architecture_review={"verified": True, "source": "this synthetic test fixture; no production-model claim"},
        max_nodes=32, max_feature_nodes=8, max_logits=1, node_influence_mass=1.0, edge_row_mass=1.0,
        target_mode="after_step_next_token_v1", **kwargs)
    return policy, bindings, attributor, stamp, version


def test_forward_parity_direct_edges_and_cut_intervention_agree():
    policy, bindings, attributor, stamp, version = setup()
    ids = torch.tensor([[0, 1]])
    original_state = tensor_state_hash(dict(policy.state_dict()))
    with pytest.raises(UnsupportedAttribution, match="validate_backend"):
        attributor.attribute(ids, stamp)
    report = attributor.validate_backend(ids, stamp, epsilon=1e-3)
    assert report["passed"]
    result = attributor.attribute(ids, stamp)
    assert result.metadata["contract"] == LEGACY_CONTRACT
    assert result.metadata["original_crv_exact_reproduction"] is False
    assert result.metadata["forward_parity_max_abs"] == 0
    assert result.graph.graph_kind == LEGACY_CONTRACT
    assert {record["kind"] for record in result.node_records} >= {"feature", "token", "logit", "reconstruction_error", "omitted_features", "decoder_bias"}
    # First layer feature at final token is 2*x = 4. The next MLP Jacobian is CUT.
    source = next(i for i, node in enumerate(result.node_records) if node["kind"] == "feature" and node["layer"] == 0 and node["position"] == 1)
    target = next(i for i, node in enumerate(result.node_records) if node["kind"] == "logit" and node["token_id"] == 0)
    edge = np.flatnonzero((result.graph.sources == source) & (result.graph.targets == target))
    assert edge.size == 1
    assert result.graph.weights[edge[0]] == pytest.approx(4.0)
    trace = attributor.capture(ids, stamp)
    actual = (attributor.intervention_value(trace, result, source, target, 1e-3) - attributor.intervention_value(trace, result, source, target, -1e-3)) / 2e-3
    assert actual == pytest.approx(4.0, rel=1e-3)
    # In the unconstrained policy the later factor-3 MLP contributes an extra x4.
    def uncut(scale):
        handle = policy.layers[0].mlp.register_forward_hook(lambda module, args, output: output + torch.tensor([[[0.0], [4.0 * scale]]]))
        try:
            return float(policy(ids).logits[0, -1, 0])
        finally:
            handle.remove()
    total_derivative = (uncut(1e-3) - uncut(-1e-3)) / 2e-3
    assert total_derivative == pytest.approx(16.0, rel=1e-3)
    assert tensor_state_hash(dict(policy.state_dict())) == original_state
    assert not any("transcoder" in name for name, _ in policy.named_modules())
    features = extract_graph_features(result.graph).values
    assert features["input_logit_path_missing"] == 0
    assert features["edge_count"] > 0


def test_native_norm_and_causal_mixer_jacobians_are_not_frozen():
    _, _, attributor, stamp, _ = setup(nonlinear=True)
    ids = torch.tensor([[0, 2, 1]])
    report = attributor.validate_backend(ids, stamp, epsilon=2e-3, atol=2e-4, rtol=0.08)
    assert report["passed"]
    result = attributor.attribute(ids, stamp)
    assert result.metadata["gradient_rules"]["norm_scales"] == "native local Jacobian; not frozen"
    assert result.metadata["full_prefix_tokens"] == 3


def test_stale_weights_bad_fidelity_and_cached_prefix_fail_closed():
    _, bindings, attributor, stamp, version = setup()
    ids = torch.tensor([[0, 1]])
    with pytest.raises(UnsupportedAttribution, match="cache"):
        attributor.capture(ids, stamp, model_kwargs={"use_cache": True})
    version[0] = "mutated"
    with pytest.raises(UnsupportedAttribution, match="version changed"):
        attributor.capture(ids, stamp)
    version[0] = stamp.state_id
    with torch.no_grad():
        bindings[0].transcoder.decoder.weight.mul_(10)
    with pytest.raises(UnsupportedAttribution, match="fidelity gate"):
        attributor.capture(ids, stamp)


def test_complete_prefix_node_budget_is_not_silently_truncated():
    _, _, attributor, stamp, _ = setup()
    attributor.max_nodes = 5
    with pytest.raises(UnsupportedAttribution, match="no silent prefix truncation"):
        attributor.validate_backend(torch.tensor([[0, 1]]), stamp)


def test_unknown_architecture_and_missing_ple_source_are_rejected():
    policy, bindings, _, stamp, _ = setup()
    with pytest.raises(UnsupportedAttribution, match="source review"):
        NativeAttributor(policy, bindings, state_id_getter=lambda: stamp.state_id, architecture_review={})
    policy.embed_tokens_per_layer = torch.nn.Embedding(4, 2)
    with pytest.raises(UnsupportedAttribution, match="PLE"):
        NativeAttributor(policy, bindings, state_id_getter=lambda: stamp.state_id,
                         architecture_review={"verified": True, "source": "fixture"})


def test_official_qwen35_native_gdn_operator_backward_finite_difference():
    """Tiny CPU operator test using installed official HF code; no weight download.

    This checks the native fallback's differentiability, not trained Qwen4B/9B.
    The real model still needs the separate attributor backend preflight.
    """
    pytest.importorskip("transformers")
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    generator = torch.Generator().manual_seed(31)
    query = torch.randn(1, 4, 1, 4, generator=generator, requires_grad=True)
    key = torch.randn(1, 4, 1, 4, generator=generator)
    value = torch.randn(1, 4, 1, 3, generator=generator)
    beta = torch.full((1, 4, 1), 0.3)
    decay = torch.full((1, 4, 1), -0.2)
    direction = torch.randn(query.shape, generator=generator)
    def function(q):
        output, state = torch_chunk_gated_delta_rule(q, key, value, decay, beta,
            chunk_size=4, output_final_state=False, use_qk_l2norm_in_kernel=True)
        assert state is None
        return output.sum()
    gradient = torch.autograd.grad(function(query), query)[0]
    analytical = float((gradient * direction).sum())
    epsilon = 1e-3
    with torch.no_grad():
        numerical = float((function(query + epsilon * direction) - function(query - epsilon * direction)) / (2 * epsilon))
    assert np.isfinite(analytical)
    assert analytical == pytest.approx(numerical, rel=0.02, abs=1e-4)


def test_official_gdn_full_teacher_forcing_is_causal_and_matches_recurrent_state_path():
    """Actual installed HF operator, tiny tensors; no pretrained model claim."""
    pytest.importorskip("transformers")
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule,
    )
    generator = torch.Generator().manual_seed(43)
    query = torch.randn(1, 4, 1, 4, generator=generator, requires_grad=True)
    key = torch.randn(1, 4, 1, 4, generator=generator, requires_grad=True)
    value = torch.randn(1, 4, 1, 3, generator=generator, requires_grad=True)
    decay, beta = torch.full((1, 4, 1), -0.2), torch.full((1, 4, 1), 0.3)
    options = {"output_final_state": True, "use_qk_l2norm_in_kernel": True}
    complete, final_state = torch_chunk_gated_delta_rule(query, key, value, decay, beta, chunk_size=4, **options)
    recurrent, recurrent_state = torch_recurrent_gated_delta_rule(query, key, value, decay, beta, **options)
    torch.testing.assert_close(complete, recurrent, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(final_state, recurrent_state, rtol=1e-5, atol=1e-6)
    prefix, prefix_state = torch_recurrent_gated_delta_rule(query[:, :2], key[:, :2], value[:, :2], decay[:, :2], beta[:, :2], **options)
    suffix, _ = torch_recurrent_gated_delta_rule(query[:, 2:], key[:, 2:], value[:, 2:], decay[:, 2:], beta[:, 2:], initial_state=prefix_state, **options)
    torch.testing.assert_close(torch.cat([prefix, suffix], 1), complete, rtol=1e-5, atol=1e-6)
    gradients = torch.autograd.grad(complete[:, :2].sum(), [query, key, value])
    for gradient in gradients:
        torch.testing.assert_close(gradient[:, 2:], torch.zeros_like(gradient[:, 2:]), rtol=0, atol=1e-7)
