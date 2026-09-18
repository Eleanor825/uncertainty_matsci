"""Unit fixtures only: emitted-token alignment and causal action-score attribution."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from matdiscovery.action_targets import ActionTargetError, build_action_targets
from matdiscovery.esopt import tensor_state_hash
from matdiscovery.graph_features import GraphSchemaError, extract_graph_features
from matdiscovery.native_attribution import CONTRACT, NativeAttributor, PolicyStamp, TranscoderBinding, UnsupportedAttribution
from matdiscovery.transcoders import TopKTranscoder, TranscoderConfig


class CharacterTokenizer:
    all_special_ids = [0, 1]

    def decode(self, tokens, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "".join({0: "<pad>", 1: "<|im_end|>"}.get(token, chr(token - 10) if token >= 10 else "P") for token in tokens)


def encoded(action, *, suffix=""):
    body = json.dumps(action, separators=(",", ":"))
    ids = [3, 4] + [ord(character) + 10 for character in body] + [1] + [ord(character) + 10 for character in suffix]
    return torch.tensor([ids], dtype=torch.long), body


def targets_for(action, benchmark="crystalgym", *, suffix=""):
    ids, body = encoded(action, suffix=suffix)
    targets = build_action_targets(ids, prompt_token_count=2, tokenizer=CharacterTokenizer(),
                                   benchmark=benchmark, parsed_action=action)
    return ids, targets, body


def test_only_action_value_tokens_targeted_with_correct_shift_and_eos_exclusion():
    action = {"action": "Li", "rationale": "A later explanation must not be a target."}
    ids, targets, body = targets_for(action)
    assert "".join(targets.token_texts) == "Li"
    assert targets.token_positions == tuple(range(2 + body.index("Li"), 4 + body.index("Li")))
    assert targets.prediction_positions == tuple(position - 1 for position in targets.token_positions)
    assert 1 not in targets.token_ids
    assert targets.semantic_paths == ("/action",)
    assert targets.causal_horizon < 2 + body.index("rationale")
    assert targets.source_prefix_hash == tensor_state_hash({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    assert "probability" in targets.to_dict()["probability_semantics"]


def test_made_targets_tool_and_argument_values_but_no_keys_or_rationale():
    action = {"tool": "generate_structures", "arguments": {"compositions": ["Al2O3"], "num_candidates": 32}, "rationale": "not this"}
    _, targets, _ = targets_for(action, "made")
    assert "".join(targets.token_texts) == "generate_structuresAl2O332"
    assert targets.semantic_paths == ("/tool", "/arguments/compositions/0", "/arguments/num_candidates")
    assert all(text not in {"{", "}", '"', ":", ",", "[", "]"} for text in targets.token_texts)


def test_json_token_boundaries_may_include_adjacent_punctuation_but_not_field_names():
    class Pieces:
        all_special_ids = [9]
        pieces = {1: '{"action":', 2: '"Li"', 3: '}', 9: '<end>', 5: '{"action":"Li"}'}
        def decode(self, tokens, **kwargs):
            return "".join(self.pieces.get(token, "P") for token in tokens)
    ids = torch.tensor([[0, 1, 2, 3, 9]])
    target = build_action_targets(ids, prompt_token_count=1, tokenizer=Pieces(), benchmark="crystalgym", parsed_action={"action": "Li"})
    assert target.token_ids == (2,)
    with pytest.raises(ActionTargetError, match="inseparably"):
        build_action_targets(torch.tensor([[0, 5, 9]]), prompt_token_count=1, tokenizer=Pieces(), benchmark="crystalgym", parsed_action={"action": "Li"})


def test_ambiguous_json_or_unstable_token_mapping_fails_closed():
    action = {"action": "Li"}
    ids, body = encoded(action, suffix=json.dumps(action, separators=(",", ":")))
    with pytest.raises(ActionTargetError, match="ambiguous"):
        build_action_targets(ids, prompt_token_count=2, tokenizer=CharacterTokenizer(), benchmark="crystalgym", parsed_action=action)
    class Unstable(CharacterTokenizer):
        def decode(self, tokens, **kwargs):
            value = super().decode(tokens, **kwargs)
            return value[:-1] + "\ufffd" if tokens and tokens[-1] == ord("L") + 10 else value
    ids, _ = encoded(action)
    with pytest.raises(ActionTargetError, match="boundaries"):
        build_action_targets(ids, prompt_token_count=2, tokenizer=Unstable(), benchmark="crystalgym", parsed_action=action)


def test_score_uses_unwarped_logits_and_rejects_top_p_filtered_scores():
    _, target, _ = targets_for({"action": "Li"})
    raw = torch.linspace(-2, 2, 256).repeat(1, len(target.token_ids), 1)
    expected = torch.stack([raw[0, i].log_softmax(-1)[token] for i, token in enumerate(target.token_ids)]).mean()
    assert float(target.mean_logprob(raw)) == pytest.approx(float(expected))
    assert float(target.mean_logprob(raw)) != pytest.approx(float(target.mean_logprob(raw / 0.6)))
    filtered = raw.clone()
    filtered[:, :, :32] = -float("inf")
    with pytest.raises(ActionTargetError, match="processed/filtered"):
        target.mean_logprob(filtered)


class CausalLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Linear(2, 2, bias=False)
        self.mlp.weight.data.copy_(torch.eye(2) * 0.5)
    def forward(self, x):
        x = x + 0.03 * x.cumsum(1)
        return x + self.mlp(x)


class TinyCausalPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="unit_test_causal_scan")
        self.embedding = torch.nn.Embedding(256, 2)
        numbers = torch.arange(256).float()
        self.embedding.weight.data.copy_(torch.stack([0.2 + numbers / 1000, 0.3 + (numbers % 7) / 100], -1))
        self.layers = torch.nn.ModuleList([CausalLayer(), CausalLayer()])
        self.head = torch.nn.Linear(2, 256, bias=False)
        self.head.weight.data.copy_(torch.randn(256, 2, generator=torch.Generator().manual_seed(12)) * 0.1)
        self.eval()
    def get_input_embeddings(self):
        return self.embedding
    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0):
        assert use_cache is False
        values = self.embedding(input_ids)
        for layer in self.layers:
            values = layer(values)
        selector = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return SimpleNamespace(logits=self.head(values[:, selector]))


def native_setup():
    policy = TinyCausalPolicy()
    stamp = PolicyStamp("unit_test_v2", tensor_state_hash(dict(policy.state_dict())), "tiny-unit-test-only", 0)
    bindings = []
    for layer in range(2):
        transcoder = TopKTranscoder(TranscoderConfig(2, 2, 2, 2))
        with torch.no_grad():
            transcoder.encoder.weight.copy_(torch.eye(2) * 0.5)
            transcoder.encoder.bias.zero_()
            transcoder.decoder.weight.copy_(torch.eye(2))
            transcoder.decoder.bias.zero_()
        path = f"layers.{layer}.mlp"
        bindings.append(TranscoderBinding(path, transcoder, {"fidelity_gate_passed": True, "max_dev_fvu": 0.01,
            "transcoder_hash": transcoder.checkpoint_hash(), "provenance": {"layer_path": path, "policy_fingerprint": stamp.checkpoint_hash}}))
    attributor = NativeAttributor(policy, bindings, state_id_getter=lambda: stamp.state_id,
        architecture_review={"verified": True, "source": "synthetic unit test causal model only"},
        max_nodes=512, max_feature_nodes=16, node_influence_mass=1, edge_row_mass=1)
    return policy, attributor, stamp


def test_full_prefix_teacher_forcing_matches_each_causal_prefix_and_future_gradients_zero():
    policy, attributor, stamp = native_setup()
    ids, targets, _ = targets_for({"action": "Li", "rationale": "unrelated later tokens"})
    trace = attributor.capture(ids, stamp, action_targets=targets, model_kwargs={"logits_to_keep": torch.tensor([0])})
    assert trace.prefix_hash == targets.source_prefix_hash
    assert trace.model_kwargs["logits_to_keep"].tolist() == list(targets.prediction_positions)
    assert trace.parity_max_abs == 0
    for index, position in enumerate(targets.token_positions):
        logits = policy(ids[:, :position], logits_to_keep=1).logits[0, -1]
        torch.testing.assert_close(trace.reference_logits[0, index], logits, rtol=1e-6, atol=1e-6)
    score = targets.mean_logprob(trace.logits)
    sources = [*trace.embedding_leaves.values(), *trace.mlp_leaves.values()]
    gradients = torch.autograd.grad(score, sources, allow_unused=True, retain_graph=True)
    assert any(gradient is not None and bool(gradient[:, :targets.causal_horizon + 1].abs().sum() > 0) for gradient in gradients)
    for gradient in gradients:
        if gradient is not None:
            assert torch.count_nonzero(gradient[:, targets.causal_horizon + 1:]) == 0


def test_v2_single_score_sink_cut_finite_differences_and_future_append_invariance():
    _, attributor, stamp = native_setup()
    action = {"action": "Li", "rationale": "later text"}
    ids, target, _ = targets_for(action)
    report = attributor.validate_backend(ids, stamp, action_targets=target, epsilon=1e-2, atol=3e-4, rtol=0.08)
    assert report["passed"] and report["required_direct_action_sink_source_kinds"] == ["token", "feature"]
    result = attributor.attribute(ids, stamp, action_targets=target)
    assert result.metadata["contract"] == CONTRACT
    assert result.metadata["sink_count"] == 1
    assert result.metadata["sink_influence_seed_is_probability"] is False
    assert result.graph.selected_logit_probabilities is None
    assert sum(node["kind"] == "score" for node in result.node_records) == 1
    assert all(node.get("position", 0) <= target.causal_horizon for node in result.node_records)
    feature_values = extract_graph_features(result.graph).values
    assert feature_values["score_node_count"] == 1
    assert feature_values["logit_probabilities_missing"] == 1
    longer, later_target, _ = targets_for(action, suffix="future suffix")
    assert later_target.target_spec_hash == target.target_spec_hash
    assert later_target.source_prefix_hash != target.source_prefix_hash
    later = attributor.attribute(longer, stamp, action_targets=later_target)
    assert later.metadata["objective_value"] == pytest.approx(result.metadata["objective_value"], abs=1e-6)
    np.testing.assert_allclose(later.graph.weights, result.graph.weights, atol=1e-6, rtol=1e-6)
    assert later.metadata["future_source_gradient_max_abs"] == 0
    result.graph.selected_logit_probabilities = np.array([1.0])
    with pytest.raises(GraphSchemaError, match="masquerade"):
        result.graph.validate()


def test_v2_requires_explicit_action_targets_and_stale_target_hashes_fail():
    _, attributor, stamp = native_setup()
    ids, target, _ = targets_for({"action": "Li"})
    with pytest.raises(UnsupportedAttribution, match="ActionTargets"):
        attributor.capture(ids, stamp)
    with pytest.raises(ActionTargetError, match="original parsed"):
        build_action_targets(ids, prompt_token_count=2, tokenizer=CharacterTokenizer(), benchmark="crystalgym")
    altered = ids.clone()
    altered[0, 0] += 1
    with pytest.raises(ActionTargetError, match="source prefix"):
        attributor.capture(altered, stamp, action_targets=target)
