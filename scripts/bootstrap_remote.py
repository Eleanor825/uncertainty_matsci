#!/usr/bin/env python3
"""Create clean project environments; record every command and its exit status."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("made", "policy", "crystalgym"), required=True)
    parser.add_argument("--environment-base", type=Path, help="Optional fresh local-disk prefix; project retains a symlink")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    name = {"made": "made-py312", "policy": "policy-py312", "crystalgym": "crystalgym-py311"}[args.stage]
    environment = (args.environment_base or root / "environments") / name
    environment.parent.mkdir(parents=True, exist_ok=True)
    project_link = root / "environments" / name
    if args.environment_base and not project_link.exists():
        project_link.symlink_to(environment, target_is_directory=True)
    python = "3.11" if args.stage == "crystalgym" else "3.12"
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(environment)
    env["UV_LINK_MODE"] = "hardlink" if args.environment_base else "copy"
    env["PYTHONNOUSERSITE"] = "1"
    commands = [["uv", "venv", "--python", python, str(environment)]]
    if args.stage == "made":
        commands += [["uv", "sync", "--frozen", "--no-dev", "--project", str(root / "vendor/MADE")]]
    elif args.stage == "policy":
        commands += [["uv", "pip", "install", "--python", str(environment / "bin/python"), "-e", str(root) + "[policy,dev]"]]
    else:
        commands += [
            ["uv", "pip", "install", "--python", str(environment / "bin/python"), "-r", str(root / "vendor/crystal-gym/requirements.txt")],
            ["uv", "pip", "install", "--python", str(environment / "bin/python"), "--no-deps", "-e", str(root / "vendor/crystal-gym")],
        ]
    record = {"stage": args.stage, "environment": str(environment), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "commands": [], "status": "running"}
    status = root / "logs" / f"bootstrap-{args.stage}.json"
    status.parent.mkdir(parents=True, exist_ok=True)
    for command in commands:
        print(json.dumps({"command": command}), flush=True)
        result = subprocess.run(command, cwd=root, env=env)
        record["commands"].append({"argv": command, "returncode": result.returncode})
        if result.returncode:
            record["status"] = "failed"
            record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            status.write_text(json.dumps(record, indent=2) + "\n")
            return result.returncode
        status.write_text(json.dumps(record, indent=2) + "\n")
    freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(environment / "bin/python")], capture_output=True, text=True, env=env)
    (root / "configs" / f"{args.stage}-installed.txt").write_text(freeze.stdout)
    record["status"] = "installed_not_runtime_verified"
    record["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    status.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
