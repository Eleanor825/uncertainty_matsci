#!/usr/bin/env python3
"""Finite experiment-stage supervisor. Scientific failures halt, never auto-replay."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def write(path: Path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    path = root / "logs/pipeline_status.json"
    status = json.loads(path.read_text()) if path.exists() else {"stages": {}, "study_complete": False}
    with (root / "logs/pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status.update(supervisor_pid=os.getpid(), study_complete=False)
        while True:
            queue = json.loads((root / "configs/stage_queue.json").read_text())
            next_stage = None
            for stage in queue["stages"]:
                previous = status["stages"].get(stage["id"])
                if previous:
                    if previous["command_fingerprint"] != digest(stage):
                        raise RuntimeError("An already-started stage was changed; explicit reconciliation is required")
                    if previous["status"] != "succeeded":
                        raise RuntimeError("An interrupted/failed stage requires reconciliation; it will not be replayed")
                    continue
                next_stage = stage
                break
            if next_stage is None:
                status.update(state="finished" if queue.get("sealed") else "waiting_for_declared_next_stages", heartbeat=time.time())
                status["study_complete"] = bool(queue.get("sealed") and queue.get("final_acceptance_verified"))
                write(path, status)
                if queue.get("sealed"):
                    return 0
                time.sleep(15)
                continue
            stage = next_stage
            if stage.get("requires_sealed") and not queue.get("sealed"):
                status.update(state="waiting_for_sealed_stage_plan", heartbeat=time.time(), study_complete=False)
                write(path, status)
                time.sleep(15)
                continue
            env = os.environ.copy()
            env.update(PYTHONPATH=str(root / "src"), PYTHONHASHSEED="0", PYTHONNOUSERSITE="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            logfile = root / "logs" / (stage["id"] + ".log")
            record = {"status": "started", "started_at": time.time(), "command_fingerprint": digest(stage), "command": stage["command"], "log": str(logfile)}
            status["stages"][stage["id"]] = record
            status.update(state="running", active_stage=stage["id"])
            write(path, status)
            with logfile.open("a") as log:
                process = subprocess.Popen(stage["command"], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                record["pid"] = process.pid
                while process.poll() is None:
                    status["heartbeat"] = time.time()
                    write(path, status)
                    time.sleep(5)
                record.update(returncode=process.returncode, finished_at=time.time(), status="succeeded" if process.returncode == 0 else "failed")
                write(path, status)
                if process.returncode:
                    status.update(state="halted_requires_reconciliation", study_complete=False)
                    write(path, status)
                    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
