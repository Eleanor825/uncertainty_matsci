"""Graph direction, pruning and feature contracts; no model-experiment claims."""
from types import SimpleNamespace

import numpy as np
import pytest

from matdiscovery.graph_features import GraphPayload, GraphSchemaError, extract_graph_features, from_tracer_graph


def chain(weights=(-2.0, 4.0)):
    return GraphPayload(
        node_types=np.array(["token", "feature", "logit"]),
        sources=np.array([0, 1]), targets=np.array([1, 2]), weights=np.array(weights),
        node_mask=np.ones(3, dtype=bool), edge_mask=np.ones(2, dtype=bool), n_layers=1,
        node_layers=np.array([-1, 0, -1]), activation_values=np.array([0, 3, 0]),
        node_influence=np.array([1, 2, 1]), selected_logit_probabilities=np.array([0.75]),
        provenance={"policy_checkpoint_hash": "fixture-only"},
    )


def test_correct_direction_signed_influence_and_positive_distance():
    result = extract_graph_features(chain())
    values = result.values
    assert values["input_logit_min_hops"] == 2
    assert values["input_logit_path_missing"] == 0
    assert values["input_logit_reachable_logit_fraction"] == 1
    assert values["input_logit_min_positive_distance"] == pytest.approx(1 / (2 + 1e-8) + 1 / (4 + 1e-8))
    assert values["edge_signed_sum"] == 2
    assert values["edge_positive_sum"] == 4
    assert values["edge_negative_abs_sum"] == 2
    assert values["edge_negative_fraction"] == 0.5
    assert values["activation_std"] == 0
    assert values["layer_0_feature_count"] == 1
    assert values["edge_abs_hhi"] == pytest.approx(5 / 9)
    assert values["edge_abs_effective_count"] == pytest.approx(9 / 5)
    assert values["selected_logit_entropy"] == pytest.approx(-0.75 * np.log(0.75))
    assert result.metadata["schema_version"] == "crv_corrected_action_sink_v2"
    assert np.isfinite(result.vector()).all()
    reverse = chain()
    reverse.sources, reverse.targets = reverse.targets, reverse.sources
    assert extract_graph_features(reverse).values["input_logit_path_missing"] == 1


def test_both_node_and_edge_masks_apply():
    graph = chain()
    graph.edge_mask[1] = False
    values = extract_graph_features(graph).values
    assert values["edge_count"] == 1
    assert values["input_logit_path_missing"] == 1
    assert values["weak_components"] == 2
    graph = chain()
    graph.node_mask[1] = False
    values = extract_graph_features(graph).values
    assert values["node_count"] == 2
    assert values["edge_count"] == 0
    assert values["activation_mean"] == 0
    assert values["weak_components"] == 2


def test_tracer_target_source_adapter_with_real_node_layout():
    # F=1, L=1, T=2 -> feature0, errors1/2, tokens3/4, logit5.
    matrix = np.zeros((6, 6))
    matrix[0, 3] = 2
    matrix[5, 0] = -4
    matrix[5, 4] = 8  # Explicitly edge-pruned shortcut.
    matrix[5, 2] = 99  # Explicitly node-pruned residual.
    node_mask = np.array([True, True, False, True, True, True])
    edge_mask = np.ones((6, 6), dtype=bool)
    edge_mask[5, 4] = False
    tracer = SimpleNamespace(
        adjacency_matrix=matrix, selected_features=np.array([0]), active_features=np.array([[0, 1, 27]]),
        activation_values=np.array([3.0]), n_pos=2, cfg=SimpleNamespace(n_layers=1),
        logit_tokens=np.array([42]), logit_probabilities=np.array([0.7]),
    )
    payload = from_tracer_graph(tracer, node_mask=node_mask, edge_mask=edge_mask,
                                node_influence=np.array([2, 0, 5, 1, 1, 1]), dense_chunk_cells=6)
    assert set(zip(payload.sources, payload.targets, payload.weights)) == {(3, 0, 2), (0, 5, -4)}
    values = extract_graph_features(payload).values
    assert values["input_logit_min_hops"] == 2
    assert values["sampled_reachable_pair_fraction"] == 0.5
    assert values["error_influence_fraction"] == 0
    assert values["node_count"] == 5
    assert values["edge_count"] == 2
    with pytest.raises(GraphSchemaError, match="both"):
        from_tracer_graph(tracer, node_mask=node_mask)


def test_missing_graph_is_explicit_and_schema_is_stable():
    missing = extract_graph_features(None, n_layers=1, missing_reason="attribution_oom")
    available = extract_graph_features(chain())
    assert missing.values.keys() == available.values.keys()
    assert missing.values["graph_missing"] == 1
    assert missing.metadata["missing_reason"] == "attribution_oom"
    assert available.values["graph_missing"] == 0
    assert np.isfinite(missing.vector()).all()
    graph = chain()
    graph.node_mask[:] = False
    empty = extract_graph_features(graph)
    assert empty.values["graph_missing"] == 0
    assert empty.values["empty_after_pruning"] == 1


def test_path_budgets_do_not_return_partial_statistics_as_complete():
    result = extract_graph_features(chain(), max_path_sources=1000, max_path_edge_visits=1)
    assert result.values["path_budget_exhausted"] == 1
    assert result.values["input_logit_path_missing"] == 1
    assert result.values["sampled_path_missing"] == 1
    assert result.values["sampled_path_source_count"] == 0
    assert result.values["weighted_path_budget_exhausted"] == 1
    assert result.values["weighted_path_missing"] == 1
    limited = extract_graph_features(chain(), max_path_sources=0)
    assert limited.values["sampled_path_missing"] == 1
    assert limited.values["input_logit_min_hops"] == 2


def test_sampled_source_count_is_bounded_on_large_token_set():
    # Many sources, one hub and one logit; never perform all-pairs paths.
    n_sources = 200
    payload = GraphPayload(
        node_types=np.array(["token"] * n_sources + ["feature", "logit"]),
        sources=np.array(list(range(n_sources)) + [n_sources]),
        targets=np.array([n_sources] * n_sources + [n_sources + 1]),
        weights=np.ones(n_sources + 1), node_mask=np.ones(n_sources + 2, dtype=bool),
        edge_mask=np.ones(n_sources + 1, dtype=bool), n_layers=1,
    )
    values = extract_graph_features(payload, max_path_sources=3, max_path_edge_visits=10_000).values
    assert values["sampled_path_source_count"] == 3
    assert values["sampled_reachable_pair_fraction"] == 1
    assert values["sampled_reachable_mean_hops"] == 2
    assert values["activation_missing"] == 1
    assert values["node_influence_missing"] == 1


@pytest.mark.parametrize("mutation", ["proxy", "nan", "indices", "duplicate", "mask"])
def test_bad_graph_contracts_fail_closed(mutation):
    graph = chain()
    if mutation == "proxy":
        graph.graph_kind = "hidden_activation_proxy"
    elif mutation == "nan":
        graph.weights[0] = np.nan
    elif mutation == "indices":
        graph.targets[0] = 10
    elif mutation == "duplicate":
        graph.sources[:] = 0
        graph.targets[:] = 1
    elif mutation == "mask":
        graph.edge_mask = np.array([1, 0])
    with pytest.raises(GraphSchemaError):
        extract_graph_features(graph)


def test_bfloat16_tracer_activations_are_adapted_without_numpy_dtype_failure():
    torch = pytest.importorskip("torch")
    # F=1, L=1, T=1 -> feature0, error1, token2, logit3.
    matrix = torch.zeros(4, 4)
    matrix[0, 2], matrix[3, 0] = 1, 2
    graph = SimpleNamespace(adjacency_matrix=matrix, selected_features=torch.tensor([0]),
                            active_features=torch.tensor([[0, 0, 1]]), activation_values=torch.tensor([2.5], dtype=torch.bfloat16),
                            n_pos=1, cfg=SimpleNamespace(n_layers=1), logit_tokens=torch.tensor([1]))
    payload = from_tracer_graph(graph, node_mask=torch.ones(4, dtype=torch.bool), edge_mask=torch.ones(4, 4, dtype=torch.bool))
    assert extract_graph_features(payload).values["activation_mean"] == 2.5
