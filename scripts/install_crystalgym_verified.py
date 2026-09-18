#!/usr/bin/env python3
"""Install unchanged resolved CrystalGym requirements using verified local wheels.

Optionally takes over only this project's still-running bootstrap process. A
fresh local-disk environment is exposed at the existing project environment path
to avoid thousands of shared-filesystem metadata operations. No old env is reused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--environment-base", required=True, type=Path)
    parser.add_argument("--take-over-pid", type=int)
    args = parser.parse_args()
    root = args.project.resolve()
    wheelhouse = root/"cache/crystalgym-wheels"
    manifest = json.loads((wheelhouse/"pypi-wheel-manifest.json").read_text())
    for item in manifest:
        path = wheelhouse/item["filename"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Official PyPI wheel verification failed: {path}")
    print(json.dumps({"verified_pypi_wheels": len(manifest), "total_bytes": sum(x["size_bytes"] for x in manifest)}), flush=True)
    record = {"classification": "clean_environment_installation", "started_at": datetime.now(timezone.utc).isoformat(),
              "wheel_manifest_sha256": sha256(wheelhouse/"pypi-wheel-manifest.json"),
              "official_requirements_sha256": sha256(root/"vendor/crystal-gym/requirements.txt"),
              "resolved_requirements_sha256": sha256(root/"cache/crystalgym-resolved.requirements.txt"),
              "operations": [], "status": "running"}
    if args.take_over_pid and (Path("/proc")/str(args.take_over_pid)).exists():
        parent = Path("/proc")/str(args.take_over_pid)
        command = (parent/"cmdline").read_bytes().replace(b"\0", b" ").decode()
        if str(root/"scripts/bootstrap_remote.py") not in command or "--stage crystalgym" not in command:
            raise RuntimeError("Refusing to stop a process outside this project's CrystalGym bootstrap")

        def descendants(pid):
            path = Path("/proc")/str(pid)/"task"/str(pid)/"children"
            children = [int(x) for x in path.read_text().split()] if path.exists() else []
            return [p for child in children for p in descendants(child)] + [pid]

        pids = descendants(args.take_over_pid)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for _ in range(30):
            if not any((Path("/proc")/str(pid)).exists() for pid in pids):
                break
            time.sleep(0.5)
        record["stopped_owned_bootstrap_pids"] = pids
    environment = args.environment_base.resolve()/"crystalgym-py311"
    link = root/"environments/crystalgym-py311"
    environment.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() and not link.is_symlink():
        backup = link.with_name(link.name+".incomplete_preserved_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        link.rename(backup)
        record["preserved_incomplete_environment"] = str(backup)
    if not link.exists():
        link.symlink_to(environment, target_is_directory=True)
    if link.resolve() != environment:
        raise RuntimeError("Existing project environment symlink points outside the requested fresh environment")
    env = dict(os.environ)
    env.update(UV_LINK_MODE="hardlink", PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    log_path = root/"logs/install-crystalgym-verified.log"
    state_path = root/"logs/install-crystalgym-verified.json"

    def run(command):
        start = time.monotonic()
        print(json.dumps({"command": command, "log": str(log_path)}), flush=True)
        with log_path.open("a") as stream:
            result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record["operations"].append({"command": command, "returncode": result.returncode, "seconds": time.monotonic()-start})
        state_path.write_text(json.dumps(record, indent=2)+"\n")
        if result.returncode:
            record["status"] = "failed"
            state_path.write_text(json.dumps(record, indent=2)+"\n")
            raise RuntimeError(f"Installation command failed: {command}; see {log_path}")

    if not (environment/"bin/python").exists():
        run(["uv", "venv", "--python", "3.11", str(environment)])
    run(["uv", "pip", "install", "--python", str(environment/"bin/python"),
         "-r", str(root/"cache/crystalgym-resolved.requirements.txt"),
         *[str(wheelhouse/item["filename"]) for item in manifest]])
    run(["uv", "pip", "install", "--python", str(environment/"bin/python"), "--no-deps", "-e", str(root/"vendor/crystal-gym")])
    run(["uv", "pip", "check", "--python", str(environment/"bin/python")])
    frozen = subprocess.check_output(["uv", "pip", "freeze", "--python", str(environment/"bin/python")], cwd=root, env=env, text=True)
    (root/"logs/crystalgym-verified-installed.txt").write_text(frozen)
    record.update(status="installed_dependency_check_passed_runtime_validation_pending", environment=str(environment), project_link=str(link),
                  finished_at=datetime.now(timezone.utc).isoformat())
    state_path.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"status": record["status"], "environment": str(environment), "state": str(state_path)}), flush=True)


if __name__ == "__main__":
    main()
