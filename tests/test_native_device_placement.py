"""Unit tests of resident CPU transcoders; synthetic networks, never study results."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from matdiscovery.esopt import tensor_state_hash
from matdiscovery.native_attribution import NativeAttributor, UnsupportedAttribution
from test_action_targets import native_setup, targets_for


def make_placement(policy_device="cpu", *, cpu_token_modules=False, transcoder_device="cpu",
                   constant_storage_device=None, max_feature_nodes=128, residual_fixture=False):
    policy, original, stamp = native_setup()
    policy.to(policy_device)
    policy.get_output_embeddings = lambda: policy.head
    if cpu_token_modules:
        policy.embedding.cpu()
        policy.head.cpu()
        policy.embedding.register_forward_pre_hook(lambda module, args: (args[0].to("cpu"),))
        policy.embedding.register_forward_hook(lambda module, args, output: output.to(policy_device))
        policy.head.register_forward_pre_hook(lambda module, args: (args[0].to("cpu"),))
        # Keep the tiny unit-test score on CPU to exercise the entire backward
        # transfer. The production adapter may copy its output back to CUDA.
    for binding in original.bindings:
        binding.transcoder.to(transcoder_device)
        if residual_fixture:
            with torch.no_grad():
                binding.transcoder.decoder.bias.fill_(0.0001)
            binding.training_metadata["transcoder_hash"] = binding.transcoder.checkpoint_hash()
    attributor = NativeAttributor(policy, original.bindings, state_id_getter=lambda: stamp.state_id,
        architecture_review={"verified": True, "source": "synthetic device unit test only"},
        max_nodes=512, max_feature_nodes=max_feature_nodes, node_influence_mass=1, edge_row_mass=1,
        constant_storage_device=constant_storage_device)
    return policy, attributor, stamp


def test_runtime_inventory_is_bound_to_backend_validation_and_exposes_tc_precision():
    _, attributor, stamp = make_placement()
    runtime = attributor.runtime_metadata()
    assert runtime["schema"] == "native_attribution_runtime_precision_devices_v1"
    assert all(row["device"] == "cpu" and row["dtype"] == "torch.float32"
               for rows in runtime["transcoder_tensors"].values() for row in rows)
    ids, targets, _ = targets_for({"action": "Li"})
    attributor.bindings[0].transcoder.double()
    with pytest.raises(UnsupportedAttribution, match="Backend"):
        attributor.capture(ids, stamp, action_targets=targets)


@pytest.mark.parametrize("kind", ["meta", "mixed_precision"])
def test_transcoders_require_real_consistent_resident_tensors(kind):
    policy, original, stamp = native_setup()
    if kind == "meta":
        original.bindings[0].transcoder.to("meta")
    else:
        original.bindings[0].transcoder.decoder.double()
    with pytest.raises(UnsupportedAttribution, match="resident|one precision"):
        NativeAttributor(policy, original.bindings, state_id_getter=lambda: stamp.state_id,
            architecture_review={"verified": True, "source": "synthetic unit test"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real CPU/CUDA transfer needs CUDA")
@pytest.mark.parametrize("cpu_token_modules", [False, True])
def test_cpu_transcoders_preserve_cuda_cut_edges_and_bidirectional_autograd(cpu_token_modules):
    # All tensors are tiny; this does not load a model or call an evaluator.
    policy, attributor, stamp = make_placement("cuda:0", cpu_token_modules=cpu_token_modules)
    before = tensor_state_hash(dict(policy.state_dict()))
    ids, targets, _ = targets_for({"action": "Li", "rationale": "future text"})
    ids = ids.to("cuda:0")
    trace = attributor.capture(ids, stamp, action_targets=targets)
    assert trace.parity_max_abs == 0
    assert all(z.device.type == "cpu" for z in trace.activations.values())
    assert all(z.device.type == "cpu" for z in trace.reconstructions.values())
    assert all(z.device.type == "cuda" for z in trace.mlp_leaves.values())
    assert trace.logits.device.type == ("cpu" if cpu_token_modules else "cuda")
    score = targets.mean_logprob(trace.logits)
    sources = [*trace.embedding_leaves.values(), *trace.mlp_leaves.values()]
    gradients = torch.autograd.grad(score, sources, retain_graph=True)
    assert all(g.device == source.device for g, source in zip(gradients, sources))
    assert all(torch.count_nonzero(g[:, targets.causal_horizon + 1:]) == 0 for g in gradients)
    report = attributor.validate_backend(ids, stamp, action_targets=targets, epsilon=1e-2, atol=3e-4, rtol=0.08)
    assert report["passed"]
    assert report["runtime"] == attributor.runtime_metadata()
    result = attributor.attribute(ids, stamp, action_targets=targets)
    assert result.metadata["runtime"]["transcoder_tensors"] == report["runtime"]["transcoder_tensors"]
    # CPU target-feature encoding must retain a path back to earlier GPU leaves.
    assert any(result.node_records[int(s)]["kind"] == "feature" and result.node_records[int(t)]["kind"] == "feature"
               for s, t in zip(result.graph.sources, result.graph.targets))
    assert result.metadata["future_source_gradient_max_abs"] == 0
    assert tensor_state_hash(dict(policy.state_dict())) == before
    assert all(p.grad is None for binding in attributor.bindings for p in binding.transcoder.parameters())

    # Identical tiny weights with the TC bank on CUDA provide a placement-only
    # comparison. Moving constants must not change the mathematical edge set.
    _, on_cuda, other_stamp = make_placement("cuda:0", cpu_token_modules=cpu_token_modules, transcoder_device="cuda:0")
    other = on_cuda._assemble(on_cuda.capture(ids, other_stamp, action_targets=targets))
    np.testing.assert_array_equal(result.graph.sources, other.graph.sources)
    np.testing.assert_array_equal(result.graph.targets, other.graph.targets)
    np.testing.assert_allclose(result.graph.weights, other.graph.weights, atol=2e-6, rtol=2e-5)
    assert result.metadata["objective_value"] == pytest.approx(other.metadata["objective_value"], abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Paired CPU/CUDA constant storage needs CUDA")
def test_cpu_constants_preserve_full_graph_and_all_residual_interventions():
    """A tiny nonzero-residual unit fixture isolates only constant placement."""
    models = []
    ids, targets, _ = targets_for({"action": "Li", "rationale": "future text"})
    ids = ids.cuda()
    for storage in (None, "cuda:0"):
        policy, attributor, stamp = make_placement("cuda:0", cpu_token_modules=True,
            constant_storage_device=storage, max_feature_nodes=24, residual_fixture=True)
        before = tensor_state_hash(dict(policy.state_dict()))
        trace = attributor.capture(ids, stamp, action_targets=targets)
        result = attributor._assemble(trace)
        models.append((policy, attributor, stamp, trace, result, before))
    cpu, gpu = models
    cpu_trace, gpu_trace = cpu[3], gpu[3]
    cpu_result, gpu_result = cpu[4], gpu[4]
    assert all(t.device.type == "cpu" and not t.requires_grad for t in cpu_trace.reconstructions.values())
    assert all(t.device.type == "cuda" and t.requires_grad for t in cpu_trace.mlp_leaves.values())
    assert all(t.device.type == "cuda" and t.requires_grad for t in cpu_trace.mlp_inputs.values())
    assert all(t.device.type == "cuda" for t in gpu_trace.reconstructions.values())
    assert cpu_trace.fidelity == gpu_trace.fidelity
    assert cpu_trace.prefix_hash == gpu_trace.prefix_hash
    for path in cpu_trace.reconstructions:
        torch.testing.assert_close(cpu_trace.reconstructions[path], gpu_trace.reconstructions[path].cpu(), rtol=0, atol=0)
    np.testing.assert_array_equal(cpu_result.graph.sources, gpu_result.graph.sources)
    np.testing.assert_array_equal(cpu_result.graph.targets, gpu_result.graph.targets)
    np.testing.assert_allclose(cpu_result.graph.weights, gpu_result.graph.weights, atol=2e-6, rtol=2e-5)
    for runtime in cpu_result.metadata["constant_runtime"].values():
        assert runtime["reconstruction_device"] == runtime["selected_contributions_device"] == "cpu"
        assert set(runtime["residual_devices"].values()) == {"cpu"}
        assert runtime["residual_projection_device"] == "cpu"
        assert runtime["device_transfers"] == runtime["projections"] > 0
    assert all(row["device_transfers"] == 0 for row in gpu_result.metadata["constant_runtime"].values())
    # All three source kinds are actually present, including error and bias;
    # this does not merely compare an exact TC's zero residuals.
    for kind in ("reconstruction_error", "omitted_features", "decoder_bias"):
        edges = [i for i, (s, t) in enumerate(zip(cpu_result.graph.sources, cpu_result.graph.targets))
                 if cpu_result.node_records[int(s)]["kind"] == kind
                 and cpu_result.node_records[int(t)]["kind"] == "feature"]
        assert edges
        index = max(edges, key=lambda i: abs(cpu_result.graph.weights[i]))
        source, target = int(cpu_result.graph.sources[index]), int(cpu_result.graph.targets[index])
        epsilon = 0.01
        slopes = []
        for _, attributor, _, trace, result, _ in models:
            positive = attributor.intervention_value(trace, result, source, target, epsilon)
            negative = attributor.intervention_value(trace, result, source, target, -epsilon)
            slopes.append((positive - negative) / (2 * epsilon))
        assert slopes[0] == pytest.approx(slopes[1], abs=2e-6, rel=2e-5)
        assert slopes[0] == pytest.approx(cpu_result.graph.weights[index], abs=2e-6, rel=0.03)
    for policy, _, _, _, _, before in models:
        assert tensor_state_hash(dict(policy.state_dict())) == before
