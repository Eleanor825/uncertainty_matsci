#!/usr/bin/env python3
"""Fit all declared risk arms from full collection manifests, never test records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matdiscovery.fit_risk import RiskDataError, fit_risk_models


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True,
                        help="Complete collection manifest with runtime provenance; repeat to consume all collections")
    parser.add_argument("--graph-file", action="append", type=Path, default=[],
                        help="Additional graph JSONL; adjacent graph_features.jsonl is auto-discovered")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--checkpoint-hash", help="Require this initial verified model checkpoint hash")
    parser.add_argument("--label-kind", choices=("future_failure", "immediate_error"), default="future_failure")
    parser.add_argument("--seed", type=int, default=1729)
    for name in ("hidden-width", "epochs", "batch-size", "patience"):
        parser.add_argument(f"--{name}", type=int)
    for name in ("learning-rate", "weight-decay"):
        parser.add_argument(f"--{name}", type=float)
    args = parser.parse_args(argv)
    fields = ("seed", "hidden_width", "epochs", "batch_size", "patience", "learning_rate", "weight_decay")
    options = {name: getattr(args, name) for name in fields if getattr(args, name) is not None}
    try:
        report = fit_risk_models(
            args.manifest, args.output, model_key=args.model_key, label_kind=args.label_kind,
            graph_paths=args.graph_file, expected_checkpoint_hash=args.checkpoint_hash,
            hyperparameters=options,
        )
    except RiskDataError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 1
    print(json.dumps({"status": report["status"], "report": str(args.output.resolve() / "fit_report.json"),
                      "methods": {k: v["status"] for k, v in report["methods"].items()},
                      "classification": report["classification"]}))
    return 0 if report["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
