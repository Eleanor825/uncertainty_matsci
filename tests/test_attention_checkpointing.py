"""CPU official-HF tiny Qwen tests only; no pretrained/TC/oracle result claims."""
import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from matdiscovery.attention_checkpointing import AttentionCheckpointError, install_attention_checkpointing
from matdiscovery.esopt import tensor_state_hash


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_model(vocab_size=32):
    config = Qwen3_5TextConfig(vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2,
        linear_num_value_heads=2, layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]},
        attention_dropout=0.0, max_position_embeddings=128)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(317)
        return Qwen3_5ForCausalLM(config).eval().cpu()


def capture_cut(model, ids):
    """Same-valued native MLP cut fixture; it supplies no fake transcoders."""
    flags = [p.requires_grad for p in model.parameters()]
    leaves, handles = {}, []
    def embed_hook(module, args, output):
        leaves["embedding"] = output.detach().requires_grad_(True)
        return leaves["embedding"]
    try:
        for p in model.parameters():
            p.requires_grad_(False)
        handles.append(model.get_input_embeddings().register_forward_hook(embed_hook))
        for index, layer in enumerate(model.model.layers):
            def mlp_hook(module, args, output, index=index):
                leaves[f"mlp{index}"] = output.detach().requires_grad_(True)
                return leaves[f"mlp{index}"]
            handles.append(layer.mlp.register_forward_hook(mlp_hook))
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        return output, leaves
    finally:
        for hook in handles:
            hook.remove()
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)


def cut_intervention(model, ids, leaves, source, position, direction, epsilon):
    handles = []
    def replacement(key):
        def hook(module, args, output):
            value = leaves[key].detach().clone()
            if key == source:
                value[0, position] += epsilon * direction
            return value
        return hook
    try:
        handles.append(model.get_input_embeddings().register_forward_hook(replacement("embedding")))
        for index, layer in enumerate(model.model.layers):
            handles.append(layer.mlp.register_forward_hook(replacement(f"mlp{index}")))
        with torch.no_grad():
            return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, 4, 7].item()
    finally:
        for handle in handles:
            handle.remove()


def test_official_qwen_attention_and_gdn_checkpoint_preserve_cut_forward_and_repeated_vjps():
    original = tiny_model()
    recomputed = copy.deepcopy(original)
    # Restoration must preserve preexisting frozen parameters as well as True.
    next(original.parameters()).requires_grad_(False)
    next(recomputed.parameters()).requires_grad_(False)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    before = tensor_state_hash(dict(recomputed.state_dict()))
    original_names = list(recomputed.state_dict())
    flags = [p.requires_grad for p in recomputed.parameters()]
    handle = install_attention_checkpointing(recomputed)
    stable_metadata = handle.metadata()
    assert len(stable_metadata["attention_paths"]) == 2
    mlp_calls = []
    spies = [layer.mlp.register_forward_hook(lambda *args: mlp_calls.append("mlp")) for layer in recomputed.model.layers]
    try:
        with torch.no_grad():
            native = original(ids, use_cache=False).logits
            checkpoint_no_grad = recomputed(ids, use_cache=False).logits
        torch.testing.assert_close(native, checkpoint_no_grad, rtol=0, atol=0)
        normal_logits, normal_leaves = capture_cut(original, ids)
        checkpoint_logits, checkpoint_leaves = capture_cut(recomputed, ids)
        torch.testing.assert_close(checkpoint_logits, normal_logits, rtol=0, atol=0)
        torch.testing.assert_close(checkpoint_logits, native, rtol=0, atol=0)
        assert handle.counters["checkpointed_forwards"] == 2
        calls_before_backward = len(mlp_calls)
        # Multiple targets and repeated VJPs after all capture hooks/flags have
        # been restored must never re-run either MLP.
        for position, token in [(2, 3), (4, 7), (4, 7), (1, 11)]:
            reference = torch.autograd.grad(normal_logits[0, position, token], tuple(normal_leaves.values()), retain_graph=True)
            actual = torch.autograd.grad(checkpoint_logits[0, position, token], tuple(checkpoint_leaves.values()), retain_graph=True)
            for want, got in zip(reference, actual):
                torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7)
                assert torch.count_nonzero(got[:, position + 1:]) == 0
            assert [p.requires_grad for p in recomputed.parameters()] == flags
            assert len(mlp_calls) == calls_before_backward
        assert handle.counters["recomputations"] >= 8
        assert handle.metadata() == stable_metadata
        assert tensor_state_hash(dict(recomputed.state_dict())) == before
        assert list(recomputed.state_dict()) == original_names
        assert all(p.grad is None for p in recomputed.parameters())
    finally:
        for spy in spies:
            spy.remove()
        handle.remove()


def test_official_qwen_checkpoint_cut_vjp_matches_finite_difference_for_embedding_and_mlp():
    model = tiny_model()
    handle = install_attention_checkpointing(model)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    logits, leaves = capture_cut(model, ids)
    gradients = torch.autograd.grad(logits[0, 4, 7], tuple(leaves.values()), retain_graph=True)
    for source, gradient in zip(leaves, gradients):
        if source not in {"embedding", "mlp0"}:
            continue
        position = 2
        direction = gradient[0, position].detach()
        direction = direction / direction.norm()
        analytical = float((gradient[0, position] * direction).sum())
        assert abs(analytical) > 1e-4
        for epsilon in (1e-4, 3e-4, 1e-3):
            positive = cut_intervention(model, ids, leaves, source, position, direction, epsilon)
            negative = cut_intervention(model, ids, leaves, source, position, direction, -epsilon)
            assert (positive - negative) / (2 * epsilon) == pytest.approx(analytical, rel=0.03, abs=1e-5)
    handle.remove()


def test_no_grad_trainable_and_real_cache_paths_are_native_bypasses():
    original = tiny_model()
    model = copy.deepcopy(original)
    handle = install_attention_checkpointing(model)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        model(ids, use_cache=False)
    assert handle.counters["bypassed_no_grad"] == 2
    model(ids, use_cache=False)
    assert handle.counters["bypassed_trainable_or_training"] == 2
    for p in model.parameters():
        p.requires_grad_(False)
    model(ids, use_cache=False)
    assert handle.counters["bypassed_no_input_gradient"] == 2
    inputs = model.get_input_embeddings()(ids).detach().requires_grad_(True)
    cached = model(inputs_embeds=inputs, use_cache=True)
    reference = original(inputs_embeds=inputs, use_cache=True)
    torch.testing.assert_close(cached.logits, reference.logits, rtol=0, atol=0)
    next_input = inputs[:, :1].detach().requires_grad_(True)
    next_cached = model(inputs_embeds=next_input, past_key_values=cached.past_key_values, use_cache=True)
    next_reference = original(inputs_embeds=next_input, past_key_values=reference.past_key_values, use_cache=True)
    torch.testing.assert_close(next_cached.logits, next_reference.logits, rtol=0, atol=0)
    assert handle.counters["bypassed_cache"] == 4
    assert handle.counters["checkpointed_forwards"] == 0
    handle.remove()


def test_recompute_exception_restores_parameter_flags():
    model = tiny_model()
    handle = install_attention_checkpointing(model)
    def raise_on_recompute(*args):
        if handle.counters["recomputations"]:
            raise RuntimeError("unit-test-recompute-failure")
    spy = model.model.layers[1].self_attn.q_proj.register_forward_pre_hook(raise_on_recompute)
    flags = [p.requires_grad for p in model.parameters()]
    logits, leaves = capture_cut(model, torch.tensor([[1, 2, 3, 4, 5]]))
    with pytest.raises(RuntimeError, match="unit-test-recompute-failure"):
        torch.autograd.grad(logits.sum(), tuple(leaves.values()))
    assert [p.requires_grad for p in model.parameters()] == flags
    spy.remove()
    handle.remove()


def test_scope_duplicate_install_and_mutated_weights_fail_closed():
    model = tiny_model()
    with pytest.raises(AttentionCheckpointError, match="every native"):
        install_attention_checkpointing(model, attention_paths=["model.layers.0"])
    handle = install_attention_checkpointing(model)
    with pytest.raises(AttentionCheckpointError, match="already installed"):
        install_attention_checkpointing(model)
    logits, leaves = capture_cut(model, torch.tensor([[1, 2, 3, 4, 5]]))
    with torch.no_grad():
        model.model.layers[1].self_attn.q_proj.weight.add_(0)
    with pytest.raises(AttentionCheckpointError, match="changed before recomputation"):
        torch.autograd.grad(logits.sum(), tuple(leaves.values()))
    handle.remove()


def test_native_attributor_many_targets_with_exact_linear_unit_mlp_boundaries():
    """Actual NativeAttributor integration, synthetic analytically exact TCs.

    The previous tests retain the official random nonlinear MLPs and use no TC.
    This test alone replaces tiny MLPs with 0.5*identity so the positive/negative
    identity TC is exact by construction. It makes no learned fidelity claim.
    """
    from matdiscovery.native_attribution import NativeAttributor, PolicyStamp, TranscoderBinding
    from matdiscovery.transcoders import TopKTranscoder, TranscoderConfig
    from test_action_targets import targets_for

    model = tiny_model(vocab_size=256)
    for layer in model.model.layers:
        layer.mlp = torch.nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            layer.mlp.weight.copy_(torch.eye(16) * 0.5)
    model.eval()
    other = copy.deepcopy(model)
    handle = install_attention_checkpointing(other)
    other._matdiscovery_runtime_config = {"attention_checkpointing": handle.metadata()}

    def attributor(policy):
        stamp = PolicyStamp("unit_test_only", tensor_state_hash(dict(policy.state_dict())), "unit_test_only", 0)
        bindings = []
        for index in range(2):
            tc = TopKTranscoder(TranscoderConfig(16, 16, 32, 32))
            with torch.no_grad():
                tc.encoder.weight.copy_(torch.cat([torch.eye(16), -torch.eye(16)], 0))
                tc.encoder.bias.zero_()
                tc.decoder.weight.copy_(torch.cat([torch.eye(16), -torch.eye(16)], 1) * 0.5)
                tc.decoder.bias.zero_()
            path = f"model.layers.{index}.mlp"
            bindings.append(TranscoderBinding(path, tc, {"fidelity_gate_passed": True, "max_dev_fvu": 1e-5,
                "classification": "unit_test_analytic_identity_only_not_learned_fidelity",
                "transcoder_hash": tc.checkpoint_hash(), "provenance": {"layer_path": path, "policy_fingerprint": stamp.checkpoint_hash}}))
        return NativeAttributor(policy, bindings, state_id_getter=lambda: stamp.state_id,
            architecture_review={"verified": True, "source": "unit-only linear MLP replacement plus actual HF attention/GDN"},
            max_nodes=512, max_feature_nodes=64, node_influence_mass=1, edge_row_mass=1), stamp

    reference, stamp = attributor(model)
    checked, checked_stamp = attributor(other)
    ids, targets, _ = targets_for({"action": "Li", "rationale": "future"})
    before_flags = [p.requires_grad for p in other.parameters()]
    validation = checked.validate_backend(ids, checked_stamp, action_targets=targets, epsilon=1e-3)
    assert validation["passed"]
    expected = reference._assemble(reference.capture(ids, stamp, action_targets=targets))
    actual = checked.attribute(ids, checked_stamp, action_targets=targets)
    import numpy as np
    np.testing.assert_array_equal(actual.graph.sources, expected.graph.sources)
    np.testing.assert_array_equal(actual.graph.targets, expected.graph.targets)
    np.testing.assert_allclose(actual.graph.weights, expected.graph.weights, rtol=1e-6, atol=1e-7)
    assert actual.metadata["backward_targets"] == 65
    assert actual.metadata["future_source_gradient_max_abs"] == 0
    assert handle.counters["recomputations"] > 65
    assert [p.requires_grad for p in other.parameters()] == before_flags
    handle.remove()
