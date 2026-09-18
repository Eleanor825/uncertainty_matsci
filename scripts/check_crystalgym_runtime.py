#!/usr/bin/env python3
"""One real CPU CrystalGym terminal diagnostic; never a main result.

Uses a training prototype and deterministic legal Li actions. Default bulk
modulus checks both real pw.x SCF and ev.x EOS fitting in one official episode.
All physics/reward settings remain the pinned benchmark's settings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import selectors
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--prototype-index", default=8354, type=int, choices=(630,2271,8354,8666,8906))
    parser.add_argument("--property", default="bm", choices=("bm", "density", "band_gap"))
    parser.add_argument("--actions", help="Explicit comma-separated legal elements, exactly one per site; diagnostic input only")
    parser.add_argument("--timeout", default=1800, type=int)
    args = parser.parse_args()
    action_sequence = args.actions.split(",") if args.actions else None
    root = args.project.resolve()
    runtime_path = root/"configs/qe.runtime.json"
    runtime = json.loads(runtime_path.read_text())
    run = root/"technical_not_main"/("crystalgym_cpu_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
    run.mkdir(parents=True)
    environment = dict(os.environ)
    environment.update(runtime["environment"])
    environment["PYTHONPATH"] = str(root/"src")
    environment["DGLBACKEND"] = "pytorch"
    environment["PATH"] = os.pathsep.join(runtime["path_prepend"]+[environment.get("PATH", "")])
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(runtime["ld_library_path_prepend"]+[environment.get("LD_LIBRARY_PATH", "")])
    configuration = {"benchmark": "crystalgym", "vendor_root": str(root/"vendor/crystal-gym"),
        "work_dir": str(run/"environment"), "seed": 20260915, "budget": 1,
        "budget_unit": "dft_episode_attempts", "property": args.property,
        "target": {"bm": 500.0, "density": 5.0, "band_gap": 2.0}[args.property],
        "split": "train", "prototype_index": args.prototype_index,
        "assets": {"qe_dir": runtime["qe_dir"], "pseudo_dir": runtime["sssp"]["directory"]}}
    summary = {"classification": "technical_not_main", "scientific_main_result": False,
               "excluded_from_uncertainty_and_es_training": True, "configuration": configuration,
               "run_path": str(run), "operations": [], "maximum_terminal_attempts": 1,
               "diagnostic_actions": action_sequence or "Li_at_every_site"}
    (run/"diagnostic_manifest.json").write_text(json.dumps(summary, indent=2)+"\n")
    log = (run/"server.stderr.log").open("w")
    process = subprocess.Popen([str(root/"environments/crystalgym-py311/bin/python"), "-m", "matdiscovery.env_server", "--benchmark", "crystalgym"],
        cwd=root, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
        start_new_session=True)
    print(json.dumps({"classification": "technical_not_main", "run_path": str(run), "server_pid": process.pid}), flush=True)
    started, number, terminal_attempts = time.monotonic(), 0, 0

    def rpc(op, arguments):
        nonlocal number
        number += 1
        request = {"id": number, "op": op, "args": arguments}
        with (run/"rpc_requests.jsonl").open("a") as handle:
            handle.write(json.dumps(request)+"\n")
        before = time.monotonic()
        process.stdin.write(json.dumps(request)+"\n")
        process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(args.timeout):
                raise TimeoutError(f"{op} exceeded diagnostic wall-time limit")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"RPC process exited during {op}: {process.poll()}")
        response = json.loads(line)
        with (run/"rpc_responses.jsonl").open("a") as handle:
            handle.write(json.dumps(response)+"\n")
        record = {"op": op, "ok": response.get("ok"), "seconds": time.monotonic()-before}
        if not response.get("ok"):
            record["error"] = response.get("error")
        summary["operations"].append(record)
        print(json.dumps(record), flush=True)
        if not response.get("ok"):
            raise RuntimeError(f"RPC error: {response.get('error')}")
        return response["result"]

    try:
        initialized = rpc("init", configuration)
        observation = initialized["observation"]
        summary["runtime_metadata"] = initialized["metadata"]
        if action_sequence is not None:
            legal = {item["element"] for item in observation["legal_actions"]}
            if len(action_sequence) != observation["num_sites"] or not set(action_sequence) <= legal:
                raise ValueError("Diagnostic action sequence must match every site and the actual legal action set")
        while not observation["episode_done"]:
            if observation["filled_count"] + 1 == observation["num_sites"]:
                terminal_attempts += 1
                if terminal_attempts > 1:
                    raise RuntimeError("Refusing a second terminal DFT attempt")
            element = action_sequence[observation["filled_count"]] if action_sequence is not None else "Li"
            result = rpc("step", {"action": element})
            observation = result["observation"]
        summary["scientific_result"] = result["scientific_result"]
        summary["counters"] = observation["counts"]
        outputs = list((run/"environment/calculations").rglob("espresso.pwo"))
        summary["qe_outputs"] = [str(p) for p in outputs]
        banners = [line.strip() for p in outputs for line in p.read_text(errors="replace").splitlines() if "Program PWSCF" in line]
        summary["version_banners"] = banners
        eos_files = list((run/"environment/calculations").rglob("length_energy.dat"))
        summary["eos_energy_volume_points"] = [len(p.read_text().splitlines()) for p in eos_files]
        rpc("close", {})
        if not banners or not any("7.3.1" in line for line in banners):
            raise RuntimeError("Real QE output did not confirm the required 7.3.1 version")
        if summary["scientific_result"]["official_error_flag"] != 0:
            raise RuntimeError("Official DFT endpoint returned a failure flag; retain evidence, do not report successful scientific evaluation")
        summary["status"] = "real_official_terminal_dft_verified"
        runtime["runtime_dft_verified"] = True
        runtime["technical_diagnostic_path"] = str(run)
        runtime["verified_property"] = args.property
        runtime["verified_prototype_index"] = args.prototype_index
        runtime["version_banners"] = banners
        runtime["runtime_verification_note"] = "One CPU technical_not_main episode, excluded from all main results and method training. Other properties/prototypes are not claimed verified by this diagnostic."
        runtime_path.write_text(json.dumps(runtime, indent=2)+"\n")
        installation_path = root/"logs/install-crystalgym-verified.json"
        if installation_path.is_file():
            installation = json.loads(installation_path.read_text())
            installation.update(status="installed_dependency_and_real_terminal_runtime_verified",
                                technical_diagnostic_path=str(run), verified_property=args.property,
                                verification_scope="One technical_not_main training-prototype episode; not main experiment completion.")
            installation_path.write_text(json.dumps(installation, indent=2)+"\n")
    except Exception as exc:
        summary["status"] = "technical_diagnostic_failed"
        summary["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        if process.poll() is None:
            # The isolated session contains only this diagnostic's server/QE tree.
            # Stop its MPI children too if a technical timeout occurs.
            os.killpg(process.pid, signal.SIGTERM)
    finally:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        summary.update(elapsed_seconds=time.monotonic()-started, terminal_attempts=terminal_attempts,
                       server_exit_code=process.returncode,
                       cpu_user_seconds=usage.ru_utime, cpu_system_seconds=usage.ru_stime,
                       max_rss_kib_linux=usage.ru_maxrss)
        (run/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        log.close()
        print(json.dumps({key: summary.get(key) for key in ["classification", "status", "run_path", "elapsed_seconds", "terminal_attempts", "failure", "scientific_result"]}), flush=True)
    if summary["status"] != "real_official_terminal_dft_verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
