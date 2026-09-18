"""Corrected, bounded-cost CRV features from real attribution graphs.

Tracer matrices use A[target, source]. Edge lists below always use explicit
source/target arrays. Both node and edge masks are applied. Signed influence
is preserved; path distances use positive inverse absolute influence or hops.
No hidden-activation array is silently promoted to an attribution graph.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "crv_corrected_action_sink_v2"
NODE_TYPES = {"feature", "error", "token", "logit", "score"}


class GraphSchemaError(ValueError):
    pass


@dataclass
class GraphPayload:
    """Explicit sparse graph schema. Node indices refer to the unpruned graph.

    node_influence means tracer cumulative influence, not task-error labels.
    activation_values is aligned to graph nodes (only feature entries are used).
    Masks must be provided, including when every node/edge is intentionally kept.
    """
    node_types: np.ndarray
    sources: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    node_mask: np.ndarray
    edge_mask: np.ndarray
    n_layers: int
    node_layers: np.ndarray | None = None
    activation_values: np.ndarray | None = None
    node_influence: np.ndarray | None = None
    selected_logit_probabilities: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    graph_kind: str = "crv_attribution_graph"

    def validate(self) -> None:
        if self.graph_kind not in {"crv_attribution_graph", "native_local_jacobian_mlp_cut_v1", "native_local_jacobian_action_logprob_mlp_cut_v2"}:
            raise GraphSchemaError("Only declared attribution-graph variants use this feature schema; activation proxies are forbidden.")
        if not isinstance(self.n_layers, int) or self.n_layers < 1:
            raise GraphSchemaError("n_layers must be positive.")
        self.node_types = np.asarray(self.node_types, dtype=str)
        if self.node_types.ndim != 1 or not set(self.node_types).issubset(NODE_TYPES):
            raise GraphSchemaError("node_types must contain feature/error/token/logit/score.")
        if self.graph_kind == "native_local_jacobian_action_logprob_mlp_cut_v2":
            if np.count_nonzero(self.node_types == "score") != 1 or np.count_nonzero(self.node_types == "logit"):
                raise GraphSchemaError("An action-logprob graph requires exactly one score sink and no raw-logit sinks")
            if self.selected_logit_probabilities is not None:
                raise GraphSchemaError("Mean action log probability cannot masquerade as a selected-logit probability distribution")
        n = self.node_types.size
        for key in ("sources", "targets"):
            value = np.asarray(getattr(self, key))
            if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
                raise GraphSchemaError(f"{key} must be a 1D integer index array.")
            value = value.astype(np.int64, copy=False)
            if value.size and (value.min() < 0 or value.max() >= n):
                raise GraphSchemaError(f"{key} contains an out-of-range node.")
            setattr(self, key, value)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        e = self.sources.size
        if self.targets.size != e or self.weights.shape != (e,) or not np.isfinite(self.weights).all():
            raise GraphSchemaError("Edges require equal-length, finite weights/source/target arrays.")
        for key, length in (("node_mask", n), ("edge_mask", e)):
            mask = np.asarray(getattr(self, key))
            if mask.shape != (length,) or mask.dtype != np.bool_:
                raise GraphSchemaError(f"{key} must be an explicit boolean mask.")
            setattr(self, key, mask)
        if self.node_layers is not None:
            self.node_layers = np.asarray(self.node_layers)
            if self.node_layers.shape != (n,) or not np.issubdtype(self.node_layers.dtype, np.integer):
                raise GraphSchemaError("node_layers must be integer and aligned to nodes.")
            feature_layers = self.node_layers[self.node_types == "feature"]
            if feature_layers.size and ((feature_layers < 0).any() or (feature_layers >= self.n_layers).any()):
                raise GraphSchemaError("Feature layer outside n_layers.")
        for key in ("activation_values", "node_influence"):
            if getattr(self, key) is not None:
                value = np.asarray(getattr(self, key), dtype=np.float64)
                if value.shape != (n,):
                    raise GraphSchemaError(f"{key} must align to nodes.")
                relevant = self.node_mask.copy()
                if key == "activation_values":
                    relevant &= self.node_types == "feature"
                if not np.isfinite(value[relevant]).all():
                    raise GraphSchemaError(f"{key} contains nonfinite selected values.")
                setattr(self, key, value)
        if self.selected_logit_probabilities is not None:
            p = np.asarray(self.selected_logit_probabilities, dtype=np.float64)
            if p.ndim != 1 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or p.sum() > 1 + 1e-5:
                raise GraphSchemaError("Selected logit probabilities must be finite subprobabilities.")
            self.selected_logit_probabilities = p


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value)


def from_tracer_graph(
    graph: Any,
    *,
    node_mask: Any | None = None,
    edge_mask: Any | None = None,
    node_influence: Any | None = None,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.98,
    dense_chunk_cells: int = 1_000_000,
    provenance: Mapping[str, Any] | None = None,
) -> GraphPayload:
    """Adapt a real circuit-tracer Graph without duplicating its dense N*N matrix.

    Supply BOTH masks, or let the installed tracer compute both. Dense scanning
    is inherently O(N*N) for the upstream representation; memory is chunked.
    For large graphs prefer building GraphPayload from a streamed sparse graph.
    """
    if graph is None:
        raise GraphSchemaError("Use extract_graph_features(None) to mark missing graphs.")
    required = ("adjacency_matrix", "selected_features", "active_features", "activation_values", "n_pos", "cfg", "logit_tokens")
    if any(not hasattr(graph, key) for key in required):
        raise GraphSchemaError("Input does not implement the circuit-tracer Graph contract.")
    if (node_mask is None) != (edge_mask is None):
        raise GraphSchemaError("Supply both node and edge masks together.")
    computed_masks = node_mask is None
    if computed_masks:
        try:
            from circuit_tracer.graph import prune_graph
        except ImportError as exc:
            raise GraphSchemaError("Install the pinned CRV tracer or supply explicit pruning masks.") from exc
        result = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
        node_mask, edge_mask, cumulative = result
        if node_influence is None:
            node_influence = cumulative
    adjacency = graph.adjacency_matrix
    n = int(adjacency.shape[0])
    if len(adjacency.shape) != 2 or tuple(adjacency.shape) != (n, n):
        raise GraphSchemaError("Tracer adjacency must be a square A[target, source] matrix.")
    keep = _numpy(node_mask)
    if keep.shape != (n,) or keep.dtype != np.bool_ or tuple(edge_mask.shape) != (n, n):
        raise GraphSchemaError("Pruning masks do not match the dense graph.")
    if dense_chunk_cells < n:
        raise GraphSchemaError("dense_chunk_cells must fit at least one full adjacency row.")
    selected = _numpy(graph.selected_features).astype(np.int64)
    f = selected.size
    layers = int(graph.cfg.n_layers)
    positions = int(graph.n_pos)
    k = len(graph.logit_tokens)
    error_end = f + layers * positions
    token_end = error_end + positions
    if token_end + k != n:
        raise GraphSchemaError("Node layout disagrees with feature/error/token/logit counts.")
    types = np.empty(n, dtype="U7")
    types[:f], types[f:error_end], types[error_end:token_end], types[token_end:] = "feature", "error", "token", "logit"
    node_layers = np.full(n, -1, dtype=np.int64)
    active_features = _numpy(graph.active_features)
    activations = np.zeros(n, dtype=np.float64)
    if f:
        node_layers[:f] = active_features[selected, 0]
        activations[:f] = _numpy(graph.activation_values)[selected]
    source_parts, target_parts, weight_parts = [], [], []
    row_batch = max(1, dense_chunk_cells // n)
    for start in range(0, n, row_batch):
        stop = min(start + row_batch, n)
        if not keep[start:stop].any():
            continue
        block = _numpy(adjacency[start:stop])
        edge_keep = _numpy(edge_mask[start:stop])
        if edge_keep.dtype != np.bool_:
            raise GraphSchemaError("Tracer edge_mask must be boolean.")
        valid = edge_keep & keep[start:stop, None] & keep[None, :]
        rows, columns = np.nonzero(valid & (block != 0))
        # CRITICAL: matrix ROW is TARGET, COLUMN is SOURCE.
        source_parts.append(columns)
        target_parts.append(rows + start)
        weight_parts.append(block[rows, columns])
    concatenate = lambda parts, dtype: np.concatenate(parts).astype(dtype, copy=False) if parts else np.array([], dtype=dtype)
    sources = concatenate(source_parts, np.int64)
    payload = GraphPayload(
        node_types=types, sources=sources, targets=concatenate(target_parts, np.int64),
        weights=concatenate(weight_parts, np.float64), node_mask=keep,
        edge_mask=np.ones(sources.size, dtype=bool), n_layers=layers,
        node_layers=node_layers, activation_values=activations,
        node_influence=None if node_influence is None else _numpy(node_influence),
        selected_logit_probabilities=None if not hasattr(graph, "logit_probabilities") else _numpy(graph.logit_probabilities),
        provenance={**dict(provenance or {}), "adapter": "tracer_target_source_v1",
                    "mask_origin": "tracer_prune_graph" if computed_masks else "explicit_masks",
                    **({"node_threshold": node_threshold, "edge_threshold": edge_threshold} if computed_masks else {})},
    )
    payload.validate()
    return payload


@dataclass
class GraphFeatureResult:
    values: dict[str, float]
    metadata: dict[str, Any]

    def vector(self, names: list[str] | None = None) -> np.ndarray:
        """Use a frozen explicit names list when fitting a model."""
        names = sorted(self.values) if names is None else names
        return np.array([self.values[name] for name in names], dtype=np.float64)


def _defaults(n_layers: int) -> dict[str, float]:
    names = [
        "graph_missing", "empty_after_pruning", "activation_missing", "node_influence_missing", "layer_metadata_missing", "logit_probabilities_missing",
        "node_count", "feature_node_count", "error_node_count", "token_node_count", "logit_node_count", "score_node_count", "edge_count", "density", "weak_components",
        "activation_mean", "activation_max", "activation_std", "activation_q50", "activation_q90", "activation_q99",
        "edge_signed_sum", "edge_signed_mean", "edge_signed_std", "edge_positive_sum", "edge_negative_abs_sum", "edge_abs_sum", "edge_negative_fraction",
        "edge_abs_top1_share", "edge_abs_top10_share", "edge_abs_hhi", "edge_abs_effective_count",
        "node_influence_top1_share", "node_influence_top10_share", "node_influence_hhi", "node_influence_effective_count", "error_influence_fraction",
        "degree_mean", "degree_max", "degree_std", "selected_logit_entropy", "selected_logit_probability_mass", "top_logit_probability",
        "input_logit_reachability_missing", "input_logit_path_missing", "input_logit_min_hops", "input_logit_reachable_logit_fraction", "path_budget_exhausted",
        "sampled_hop_statistics_missing", "sampled_path_missing", "sampled_path_source_count", "sampled_reachable_pair_fraction", "sampled_reachable_mean_hops", "sampled_reachable_max_hops",
        "weighted_path_missing", "input_logit_min_positive_distance", "weighted_path_budget_exhausted",
    ]
    result = dict.fromkeys(names, 0.0)
    for layer in range(n_layers):
        result[f"layer_{layer}_feature_count"] = 0.0
        result[f"layer_{layer}_feature_fraction"] = 0.0
        result[f"layer_{layer}_activation_mean"] = 0.0
    for name in ("graph_missing", "activation_missing", "node_influence_missing", "layer_metadata_missing", "logit_probabilities_missing", "input_logit_reachability_missing", "input_logit_path_missing", "sampled_hop_statistics_missing", "sampled_path_missing", "weighted_path_missing"):
        result[name] = 1.0
    return result


def _concentration(values: np.ndarray) -> dict[str, float]:
    values = np.abs(values)
    total = float(values.sum())
    if not values.size or total == 0:
        return {"top1_share": 0.0, "top10_share": 0.0, "hhi": 0.0, "effective_count": 0.0}
    probabilities = values / total
    topk = min(10, values.size)
    top = np.partition(probabilities, -topk)[-topk:]
    hhi = float(np.dot(probabilities, probabilities))
    return {"top1_share": float(probabilities.max()), "top10_share": float(top.sum()), "hhi": hhi, "effective_count": 1 / hhi}


def _bfs(adjacency: list[list[tuple[int, float]]], sources: np.ndarray, budget: int) -> tuple[np.ndarray, bool, int]:
    distances = np.full(len(adjacency), -1, dtype=np.int64)
    queue = deque(int(x) for x in sources)
    distances[sources] = 0
    visits = 0
    while queue:
        current = queue.popleft()
        for target, _ in adjacency[current]:
            if visits >= budget:
                return distances, False, visits
            visits += 1
            if distances[target] < 0:
                distances[target] = distances[current] + 1
                queue.append(target)
    return distances, True, visits


def _positive_distance(adjacency: list[list[tuple[int, float]]], sources: np.ndarray,
                       targets: set[int], epsilon: float, budget: int) -> tuple[float | None, bool]:
    distances = np.full(len(adjacency), np.inf)
    distances[sources] = 0
    queue = [(0.0, int(x)) for x in sources]
    heapq.heapify(queue)
    visits = 0
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances[current]:
            continue
        if current in targets:
            return distance, True
        for target, weight in adjacency[current]:
            if visits >= budget:
                return None, False
            visits += 1
            candidate = distance + 1 / (abs(weight) + epsilon)
            if candidate < distances[target]:
                distances[target] = candidate
                heapq.heappush(queue, (candidate, target))
    return None, True


def extract_graph_features(
    graph: GraphPayload | None,
    *,
    n_layers: int | None = None,
    max_path_sources: int = 16,
    max_path_edge_visits: int = 2_000_000,
    distance_epsilon: float = 1e-8,
    missing_reason: str | None = None,
) -> GraphFeatureResult:
    """Compute finite features with bounded sampled paths, never all-pairs paths.

    Unavailable values are zero-filled ONLY together with explicit missing flags.
    The caller must preserve these flags. A failure should pass None plus reason,
    not fabricate GraphPayload from hidden states.
    """
    if max_path_sources < 0 or max_path_edge_visits < 0 or not np.isfinite(distance_epsilon) or distance_epsilon <= 0:
        raise ValueError("Path budgets must be nonnegative and epsilon positive.")
    if graph is not None:
        graph.validate()
        if n_layers is not None and n_layers != graph.n_layers:
            raise GraphSchemaError("n_layers disagrees with the graph.")
        n_layers = graph.n_layers
    else:
        n_layers = 32 if n_layers is None else n_layers
    if n_layers < 1:
        raise ValueError("n_layers must be positive.")
    values = _defaults(n_layers)
    metadata: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "n_layers": n_layers,
                              "distance_definition": "1/(abs(attribution)+epsilon)", "distance_epsilon": distance_epsilon,
                              "max_path_sources": max_path_sources, "max_path_edge_visits": max_path_edge_visits}
    if graph is None:
        metadata["missing_reason"] = missing_reason or "not_extracted"
        return GraphFeatureResult(values, metadata)
    metadata["provenance"] = dict(graph.provenance)
    metadata["graph_kind"] = graph.graph_kind
    metadata["path_metric_target"] = "terminal logit or action-score sink; historical input_logit_* field names retained explicitly in schema v2"
    values["graph_missing"] = 0.0
    keep = graph.node_mask
    retained = np.flatnonzero(keep)
    n = retained.size
    values["node_count"] = float(n)
    values["empty_after_pruning"] = float(n == 0)
    for kind in NODE_TYPES:
        values[kind + "_node_count"] = float(np.count_nonzero(keep & (graph.node_types == kind)))
    selected_edges = graph.edge_mask & keep[graph.sources] & keep[graph.targets] & (graph.weights != 0)
    sources, targets, weights = graph.sources[selected_edges], graph.targets[selected_edges], graph.weights[selected_edges]
    # Duplicate pairs would change density and concentration; reject ambiguous input.
    if np.any(sources == targets):
        raise GraphSchemaError("Self edges are not valid CRV direct-effect graph edges.")
    if sources.size:
        pairs = np.rec.fromarrays([sources, targets])
        if np.unique(pairs).size != sources.size:
            raise GraphSchemaError("Duplicate source/target edges must be aggregated before extraction.")
    e = weights.size
    values["edge_count"] = float(e)
    values["density"] = float(e / (n * (n - 1))) if n > 1 else 0.0
    features = keep & (graph.node_types == "feature")
    if graph.activation_values is not None:
        values["activation_missing"] = 0.0
        activation = graph.activation_values[features]
        if activation.size:
            values.update({"activation_mean": float(activation.mean()), "activation_max": float(activation.max()),
                           "activation_std": float(activation.std(ddof=0)),
                           **{f"activation_q{q}": float(np.percentile(activation, q)) for q in (50, 90, 99)}})
    if graph.node_layers is not None:
        values["layer_metadata_missing"] = 0.0
        for layer in range(n_layers):
            mask = features & (graph.node_layers == layer)
            count = int(mask.sum())
            values[f"layer_{layer}_feature_count"] = float(count)
            values[f"layer_{layer}_feature_fraction"] = count / max(int(features.sum()), 1)
            if count and graph.activation_values is not None:
                values[f"layer_{layer}_activation_mean"] = float(graph.activation_values[mask].mean())
    if e:
        values.update({"edge_signed_sum": float(weights.sum()), "edge_signed_mean": float(weights.mean()), "edge_signed_std": float(weights.std(ddof=0)),
                       "edge_positive_sum": float(weights[weights > 0].sum()), "edge_negative_abs_sum": float(-weights[weights < 0].sum()),
                       "edge_abs_sum": float(np.abs(weights).sum()), "edge_negative_fraction": float(np.mean(weights < 0))})
        values.update({"edge_abs_" + key: value for key, value in _concentration(weights).items()})
    if graph.node_influence is not None:
        values["node_influence_missing"] = 0.0
        influence = np.abs(graph.node_influence[keep])
        values.update({"node_influence_" + key: value for key, value in _concentration(influence).items()})
        total = float(influence.sum())
        values["error_influence_fraction"] = float(np.abs(graph.node_influence[keep & (graph.node_types == "error")]).sum() / total) if total else 0.0
    if graph.selected_logit_probabilities is not None:
        values["logit_probabilities_missing"] = 0.0
        probs = graph.selected_logit_probabilities
        values["selected_logit_probability_mass"] = float(probs.sum())
        values["selected_logit_entropy"] = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
        values["top_logit_probability"] = float(probs.max()) if probs.size else 0.0
    if n == 0:
        return GraphFeatureResult(values, metadata)
    local = np.full(keep.size, -1, dtype=np.int64)
    local[retained] = np.arange(n)
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    parent = list(range(n))
    degree = np.zeros(n, dtype=np.int64)
    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    for source, target, weight in zip(local[sources], local[targets], weights):
        source, target = int(source), int(target)
        adjacency[source].append((target, float(weight)))
        degree[source] += 1
        degree[target] += 1
        left, right = find(source), find(target)
        if left != right:
            parent[left] = right
    values["weak_components"] = float(len({find(x) for x in range(n)}))
    values.update({"degree_mean": float(degree.mean()), "degree_max": float(degree.max()), "degree_std": float(degree.std(ddof=0))})
    inputs = local[np.flatnonzero(keep & (graph.node_types == "token"))]
    logits = local[np.flatnonzero(keep & np.isin(graph.node_types, ["logit", "score"]))]
    if inputs.size and logits.size:
        distances, complete, visits = _bfs(adjacency, inputs, max_path_edge_visits)
        values["path_budget_exhausted"] = float(not complete)
        if complete:
            values["input_logit_reachability_missing"] = 0.0
            reachable = distances[logits] >= 0
            values["input_logit_reachable_logit_fraction"] = float(reachable.mean())
            # A finished search with no path is a valid measured absence, but has
            # no path length; preserve the missing-length flag instead of -1.
            if reachable.any():
                values["input_logit_path_missing"] = 0.0
                values["input_logit_min_hops"] = float(distances[logits][reachable].min())
        # Pre-limit source count by worst-case edge visits; no all-pairs explosion.
        sample_count = min(inputs.size, max_path_sources, max_path_edge_visits // max(e, 1))
        if sample_count:
            sampled = inputs[np.linspace(0, inputs.size - 1, sample_count, dtype=int)]
            all_hops: list[np.ndarray] = []
            pair_count = 0
            remaining = max_path_edge_visits
            all_complete = True
            for source in sampled:
                distance, complete, used = _bfs(adjacency, np.array([source]), remaining)
                remaining -= used
                if not complete:
                    all_complete = False
                    break
                reachable = distance[logits] >= 0
                pair_count += int(reachable.sum())
                all_hops.append(distance[logits][reachable])
            if all_complete:
                values["sampled_path_missing"] = 0.0
                values["sampled_path_source_count"] = float(sample_count)
                values["sampled_reachable_pair_fraction"] = pair_count / (sample_count * logits.size)
                if pair_count:
                    values["sampled_hop_statistics_missing"] = 0.0
                    hops = np.concatenate(all_hops)
                    values["sampled_reachable_mean_hops"] = float(hops.mean())
                    values["sampled_reachable_max_hops"] = float(hops.max())
        distance, complete = _positive_distance(adjacency, inputs, set(int(x) for x in logits), distance_epsilon, max_path_edge_visits)
        values["weighted_path_budget_exhausted"] = float(not complete)
        if distance is not None:
            values["weighted_path_missing"] = 0.0
            values["input_logit_min_positive_distance"] = distance
    metadata["retained_edges"] = int(e)
    metadata["retained_nodes"] = int(n)
    if not np.isfinite(list(values.values())).all():
        raise GraphSchemaError("Feature extraction produced nonfinite values.")
    return GraphFeatureResult(values, metadata)
