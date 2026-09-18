#!/usr/bin/env python3
"""Full train/dev transcoder and native-graph stages; scientific tests are excluded."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matdiscovery.representation_training import (
    GraphStageConfig, RepresentationError, TranscoderStageConfig,
    generate_graph_features, train_transcoders_from_collections,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    train = stages.add_parser("train-transcoders", help="Train every canonical MLP layer from all declared activation shards")
    graph = stages.add_parser("graph-features", help="Replay every complete train/dev decision prefix")
    for stage in (train, graph):
        stage.add_argument("--manifest", action="append", type=Path, required=True)
        stage.add_argument("--model-key", choices=("qwen35_4b", "qwen35_9b"), required=True)
    train.add_argument("--output", type=Path, required=True)
    tc_defaults = TranscoderStageConfig()
    for name in ("feature_multiplier", "feature_dim", "top_k", "epochs", "batch_size", "seed"):
        train.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(tc_defaults, name))
    for name in ("learning_rate", "max_dev_fvu"):
        train.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(tc_defaults, name))
    train.add_argument("--device", default="cpu")
    graph.add_argument("--model-manifest", type=Path, required=True)
    graph.add_argument("--checkpoint-dir", type=Path, required=True)
    graph.add_argument("--transcoder-manifest", type=Path, required=True)
    graph.add_argument("--report", type=Path, required=True)
    graph.add_argument("--resume", action="store_true")
    graph.add_argument("--sparse-artifact-dir", type=Path)
    graph_defaults = GraphStageConfig()
    for name in ("max_nodes", "max_feature_nodes", "max_backward_targets", "max_logits", "max_path_sources", "max_path_edge_visits", "validation_max_edges"):
        graph.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(graph_defaults, name))
    for name in ("validation_epsilon", "validation_rtol", "validation_atol"):
        graph.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(graph_defaults, name))
    graph.add_argument("--device", default="cuda:0")
    graph.add_argument("--transcoder-device", default=None,
                       help="Resident TC bank device, e.g. cpu; defaults to the policy device")
    graph.add_argument("--constant-storage-device", default=None,
                       help="Defaults to CPU for CPU transcoders; stores reconstruction/residual constants without moving autograd leaves")
    graph.add_argument("--dtype", choices=("float32", "bfloat16"), default=graph_defaults.dtype)
    graph.add_argument("--attn-implementation", choices=("eager", "sdpa"), default=graph_defaults.attn_implementation)
    graph.add_argument("--sdpa-backend", choices=("auto", "math", "efficient"), default=graph_defaults.sdpa_backend)
    graph.add_argument("--torch-cpu-threads", type=int, default=None,
                       help="Defaults to the captured collection runtime; an explicit value must match it")
    graph.add_argument("--cpu-embedding-and-lm-head", action="store_true",
                       help="Keep original tied embeddings/output head on CPU with differentiable transfers")
    graph.add_argument("--attention-checkpointing", action="store_true",
                       help="Recompute only native attention/GDN during frozen-parameter attribution")
    from matdiscovery.native_attribution import SOURCE_SELECTION_RULES
    graph.add_argument("--source-selection-rule", choices=SOURCE_SELECTION_RULES, default=graph_defaults.source_selection_rule,
                       help="Opt in to the reviewed Qwen final-MLP structural causal-position filter")
    args = parser.parse_args(argv)
    try:
        if args.stage == "train-transcoders":
            config = TranscoderStageConfig(**{k: getattr(args, k) for k in TranscoderStageConfig.__dataclass_fields__})
            report = train_transcoders_from_collections(args.manifest, args.output, model_key=args.model_key, config=config)
        else:
            config = GraphStageConfig(**{k: getattr(args, k) for k in GraphStageConfig.__dataclass_fields__})
            report = generate_graph_features(
                args.manifest, model_key=args.model_key, model_manifest=args.model_manifest,
                checkpoint_dir=args.checkpoint_dir, transcoder_manifest=args.transcoder_manifest,
                report_path=args.report, config=config, resume=args.resume,
                sparse_artifact_dir=args.sparse_artifact_dir,
            )
        print(json.dumps({k: v for k, v in report.items() if k in {"status", "complete", "ready_for_graphs", "passed_layers", "expected_layers", "successful_graphs", "unavailable_graphs", "expected_decisions", "completed_decisions"}}))
        return 0 if report["status"] == "succeeded" else 1
    except (RepresentationError, ValueError, RuntimeError, OSError, ImportError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
