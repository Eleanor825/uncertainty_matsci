"""Opt-in real MADE CPU diagnostic: NEVER a main experiment or pytest test.

Requires a fresh installed MADE environment and a previously frozen training
snapshot. It keeps the official B=50 configuration but intentionally stops after
at most one candidate step. Initialization oracle calls and CPU/wall costs are
recorded under technical_not_main/, separately from scientific main results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import selectors
import subprocess
import time
import traceback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--system", default="Co-Os-Ti")
    parser.add_argument("--rpc-timeout-seconds", default=1800, type=int)
    parser.add_argument("--scorer-only", action="store_true", help="One real surrogate batch, zero candidate ground-truth steps.")
    args = parser.parse_args()
    root = args.project.resolve()
    index = json.loads((root / "data/raw/materials_project/index.json").read_text())
    snapshot = index["snapshots"][args.system]
    if snapshot.get("split") != "train":
        raise ValueError("This technical diagnostic may only use a training snapshot.")
    assets = json.loads((root / "configs/assets.runtime.json").read_text())
    asset_specs = {}
    for key in ("orb_checkpoint", "mace_checkpoint", "chemeleon_checkpoint", "element_reference_energies"):
        asset_specs[key] = {"path": assets[key]}
        if key + "_sha256" in assets:
            asset_specs[key]["sha256"] = assets[key + "_sha256"]
    run = root / "technical_not_main" / ("made_cpu_rpc_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    run.mkdir(parents=True)
    initialization = {
        "benchmark": "made", "vendor_root": str(root / "vendor/MADE"),
        "work_dir": str(run / "environment"), "seed": 20260915, "budget": 50,
        "elements": args.system.split("-"), "mp_cache_path": snapshot["path"],
        "mp_cache_sha256": snapshot["sha256"], "assets": asset_specs,
        "device": "cpu", "stability_tolerance": 0.1,
    }
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", PYTHONPATH=str(root / "src"))
    summary = {
        "classification": "technical_not_main", "scientific_main_result": False,
        "device": "cpu", "cuda_visible_devices": "", "run_path": str(run),
        "task": args.system, "configured_budget": 50, "maximum_candidate_step_attempts_authorized": 0 if args.scorer_only else 1,
        "maximum_surrogate_score_calls": 1 if args.scorer_only else 0,
        "excluded_from_uncertainty_and_es_training": True,
        "snapshot": snapshot, "operations": [],
    }
    (run / "diagnostic_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    log = (run / "server.stderr.log").open("w")
    process = subprocess.Popen(
        [str(root / "environments/made-py312/bin/python"), "-m", "matdiscovery.env_server", "--benchmark", "made"],
        cwd=root, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=log, text=True, bufsize=1,
    )
    summary["server_pid"] = process.pid
    print(json.dumps({"classification": "technical_not_main", "run_path": str(run), "server_pid": process.pid}), flush=True)
    request_number = candidate_steps = 0
    started = time.monotonic()

    def rpc(operation, arguments):
        nonlocal request_number, candidate_steps
        request_number += 1
        if operation == "step":
            candidate_steps += 1
            if candidate_steps > (0 if args.scorer_only else 1):
                raise RuntimeError("Diagnostic candidate-step quota exceeded.")
        request = {"id": request_number, "op": operation, "args": arguments}
        with (run / "rpc_requests.jsonl").open("a") as handle:
            handle.write(json.dumps(request) + "\n")
        start = time.monotonic()
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(args.rpc_timeout_seconds):
                raise TimeoutError(f"RPC {operation} exceeded the diagnostic timeout.")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"Server exited before {operation}: {process.poll()}")
        response = json.loads(line)
        with (run / "rpc_responses.jsonl").open("a") as handle:
            handle.write(json.dumps(response) + "\n")
        record = {"op": operation, "elapsed_seconds": time.monotonic() - start,
                  "ok": response.get("ok"), "id": response.get("id")}
        if not response.get("ok"):
            record["error"] = response.get("error")
        summary["operations"].append(record)
        print(json.dumps(record), flush=True)
        if not response.get("ok"):
            raise RuntimeError(f"RPC {operation} failed: {response.get('error')}")
        return response["result"]

    try:
        result = rpc("init", initialization)
        summary["initial_counters"] = result["observation"]["counts"]
        generation = rpc("tool", {"name": "generate_structures", "arguments": {
            "generator_name": "random", "compositions": "".join(args.system.split("-")), "num_candidates": 2}})
        accepted = [item for item in generation["output"]["records"] if item["accepted"]]
        if not accepted:
            raise RuntimeError("Both diagnostic candidates rejected; no oracle step was run.")
        chosen = accepted[0]
        if args.scorer_only:
            result = rpc("tool", {"name": "score_buffer", "arguments": {"scorer_name": "oracle", "composition": chosen["composition"]}})
            summary["surrogate_result"] = result["output"]
            summary["final_counters"] = result["status"]["counts"]
            actual_calls = result["status"]["counts"].get("surrogate_oracle_attempts", 0)
            if actual_calls != len(accepted) or result["output"][0]["surrogate_oracle_attempts"] != actual_calls:
                raise RuntimeError("Real surrogate evaluation accounting does not match the fresh uncached candidates.")
            rpc("tool", {"name": "select_for_evaluation", "arguments": {"composition": chosen["composition"], "scorer_name": "oracle"}})
        else:
            rpc("tool", {"name": "select_for_evaluation", "arguments": {
                "composition": chosen["composition"], "structure_hash": chosen["hash"],
                "reason": "technical_not_main_single_cpu_oracle_diagnostic"}})
            result = rpc("step", {})
            summary["candidate_result"] = {"official_observation": result["official_observation"],
                                           "official_metrics": result["official_metrics"],
                                           "counts": result["observation"]["counts"]}
        rpc("close", {})
        summary["status"] = "technical_rpc_and_single_real_surrogate_batch_passed" if args.scorer_only else "technical_rpc_and_single_real_oracle_step_passed"
    except Exception as exc:
        summary["status"] = "technical_diagnostic_failed"
        summary["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc(file=log)
        # A timeout might leave an oracle operation in progress; do not enqueue an
        # apparently successful close behind it or run a second candidate attempt.
        if process.poll() is None:
            process.terminate()
    finally:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        summary.update(elapsed_seconds=time.monotonic() - started, candidate_step_attempts=candidate_steps,
                       server_exit_code=process.returncode,
                       resource_usage={"child_user_cpu_seconds": usage.ru_utime,
                                       "child_system_cpu_seconds": usage.ru_stime,
                                       "max_rss_kib_linux": usage.ru_maxrss})
        (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        log.close()
        print(json.dumps({key: summary.get(key) for key in ("classification", "status", "run_path", "elapsed_seconds", "candidate_step_attempts", "failure", "resource_usage")}), flush=True)
    if summary["status"] not in {"technical_rpc_and_single_real_oracle_step_passed", "technical_rpc_and_single_real_surrogate_batch_passed"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
