#!/usr/bin/env python3
"""Freeze the reviewed causal selection repair, reuse the bank, run the full core.

Both the full CPU regression and the real unchanged-tolerance GPU FD receipt
must exist before preparation. This does not repeat V3 training or collection.
"""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--pause-closure", type=Path, required=True)
    parser.add_argument("--validated-probe", type=Path, required=True)
    parser.add_argument("--implementation-validation", type=Path, required=True)
    parser.add_argument("--diagnostic-history", type=Path, nargs="+", required=True)
    parser.add_argument("--declare-only", action="store_true")
    parser.add_argument("--fast-subset", action="store_true", help="Use the separately registered B10 new-rollout subset; retain historical B50 data")
    args = parser.parse_args()
    from matdiscovery.graph_recovery import prepare_causal_workspace
    core = prepare_causal_workspace(args.source_project, args.project,
        predecessor_workspace=args.predecessor, pause_closure_path=args.pause_closure,
        validated_probe_path=args.validated_probe, diagnostic_history_paths=args.diagnostic_history,
        implementation_validation_path=args.implementation_validation, fast_subset=args.fast_subset)
    root = Path(core["workspace"])
    command = [str(root / "environments/policy-py312/bin/python"), "-B",
        str(root / "scripts/run_core_study.py"), "--project", str(root)]
    if args.declare_only:
        command.append("--declare-only")
    environment = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    return subprocess.run(command, cwd=root, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
