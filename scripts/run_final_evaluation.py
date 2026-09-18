#!/usr/bin/env python3
"""Execute a scheduling batch of the full 2160-job final evaluation matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from matdiscovery.accounting import METHODS, MODEL_KEYS, build_final_manifest
from matdiscovery.final_evaluation import FinalEvaluationRunner, select_final_jobs
from matdiscovery.policy import declared_runtime_options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model-key", choices=MODEL_KEYS, required=True)
    parser.add_argument("--benchmark", choices=("made", "crystalgym"), required=True)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=range(1, 6))
    parser.add_argument("--property", dest="property_name", choices=("bm", "density", "band_gap"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--risk-root", type=Path, required=True, help="ROOT/model_key/benchmark/{method}.pt")
    parser.add_argument("--transcoder-root", type=Path, required=True, help="ROOT/model_key/benchmark/transcoder_manifest.json")
    parser.add_argument("--es-root", type=Path, required=True, help="ROOT/model_key/benchmark/property_or_audc/es_method/seed_N")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--device", help="Must agree with the locked main protocol")
    parser.add_argument("--cuda-memory-fraction", type=float, help="Must agree with the locked main protocol")
    args = parser.parse_args(argv)
    if args.cuda_memory_fraction is not None and not 0 < args.cuda_memory_fraction <= 1:
        parser.error("--cuda-memory-fraction must be in (0, 1]")
    filters = {k: getattr(args, k) for k in ("model_key", "benchmark", "method", "seed", "property_name")}
    if args.plan_only:
        manifest = build_final_manifest(args.project)
        jobs = select_final_jobs(manifest, **filters)
        runtime = declared_runtime_options(json.loads((args.project / "configs/main_protocol.json").read_text()),
                                           device=args.device, cuda_memory_fraction=args.cuda_memory_fraction)
        print(json.dumps({"status": "plan_only_no_physical_execution", "manifest_fingerprint": manifest["manifest_fingerprint"],
            "whole_study_expected_counts": manifest["expected_counts"], "batch_expected_jobs": len(jobs),
            "batch_expected_counts": {k: sum(j["expected_counts"][k] for j in jobs) for k in ("episodes", "candidate_oracle_attempts", "dft_episode_attempts")},
            "filters": filters, "policy_runtime_declared": runtime, "batch_does_not_reduce_study_scope": True}, indent=2))
        return 0
    runner = FinalEvaluationRunner(args.project, args.output, risk_root=args.risk_root, transcoder_root=args.transcoder_root,
        es_root=args.es_root, device=args.device, cuda_memory_fraction=args.cuda_memory_fraction)
    print(json.dumps(runner.run(**filters, resume=args.resume), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
