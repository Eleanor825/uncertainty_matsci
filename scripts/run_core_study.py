#!/usr/bin/env python3
"""Run the separately registered MADE core through the existing durable supervisor.

Only domain modules can publish successful scientific receipts. This wrapper
binds their source/protocol, commands and receipt bytes; it never fabricates a
result, changes budgets or automatically repeats an interrupted stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


STAGES = (
    ("import", "core_collection", ("--stage", "import")),
    ("collect", "core_collection", ("--stage", "collect")),
    ("transcoders", "core_representation", ("--stage", "transcoders")),
    ("graphs", "core_representation", ("--stage", "graphs")),
    ("risk", "core_representation", ("--stage", "risk")),
    ("esopt", "core_es", ()),
    ("final", "core_final", ()),
    ("report", "core_report", ()),
)
NAMES = tuple(row[0] for row in STAGES)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_workspace(root):
    from matdiscovery.core_protocol import read_core
    # This is the independent protocol/source/corpus registration reader used
    # by the domain modules, not the original full-study acceptance function.
    read_core(root)
    sources = [root / "configs/deadline_core_protocol.json",
               root / "configs/main_protocol.json"]
    sources += sorted((root / "src/matdiscovery").glob("*.py"))
    sources += [root / "scripts/run_core_study.py", root / "scripts/run_stage_queue.py"]
    return {str(path.relative_to(root)): sha(path) for path in sources}


def stage_command(root, name):
    choices = {key: (module, tail) for key, module, tail in STAGES}
    if name not in choices:
        raise ValueError("Unregistered core stage")
    module, tail = choices[name]
    return [str(root / "environments/policy-py312/bin/python"), "-B", "-m",
            "matdiscovery." + module, "--project", str(root), *tail]


def receipt_path(root, name):
    return root / "experiments/core_stage_receipts" / (name + ".json")


def verified_receipt(root, name):
    path = receipt_path(root, name)
    result = read(path)
    if result.get("complete") is not True:
        raise RuntimeError("Core domain stage did not publish complete verified evidence: " + name)
    return {"path": str(path), "sha256": sha(path)}


def build_queue(root):
    python = str(root / "environments/policy-py312/bin/python")
    return {"schema": "registered_MADE_core_stage_queue_v1", "sealed": True,
        "final_acceptance_verified": False,
        "completion_scope": "deadline_MADE_core_only_not_original_full_study",
        "original_full_study_complete": False,
        "stages": [{"id": "core-" + name, "requires_sealed": True,
                    "command": [python, "-B", str(root / "scripts/run_core_study.py"),
                                "--project", str(root), "--run-stage", name]}
                   for name in NAMES]}


def declare_queue(root):
    validate_workspace(root)
    queue, path = build_queue(root), root / "configs/stage_queue.json"
    if path.exists():
        if read(path) != queue:
            raise RuntimeError("Existing core queue differs; no silent replacement or stage reset")
    else:
        write(path, queue)
    return path


def run_stage(root, name):
    before = validate_workspace(root)
    for prior in NAMES[:NAMES.index(name)]:
        verified_receipt(root, prior)
    command = stage_command(root, name)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(root / "src"), PYTHONHASHSEED="0",
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    started = time.time()
    result = subprocess.run(command, cwd=root, env=environment)
    if result.returncode:
        return result.returncode
    if before != validate_workspace(root):
        raise RuntimeError("Frozen core source or protocol changed during execution")
    proof = verified_receipt(root, name)
    index = root / "experiments/core_stage_verification" / (name + ".json")
    record = {"schema": "core_stage_receipt_verification_v1", "stage": name,
        "complete": True, "domain_receipt": proof, "source_sha256": before,
        "command": command, "started_at": started, "finished_at": time.time(),
        "scope": "registered_deadline_core_only", "original_full_study_complete": False}
    if index.exists():
        old = read(index)
        if old.get("domain_receipt") != proof or old.get("source_sha256") != before:
            raise RuntimeError("Existing verified stage differs; reconciliation is required")
    else:
        write(index, record)
    print(json.dumps({"stage": name, "verified_receipt": proof,
                      "original_full_study_complete": False}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--prepare-from", type=Path)
    parser.add_argument("--reconciliation", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--declare-only", action="store_true")
    mode.add_argument("--run-stage", choices=NAMES)
    args = parser.parse_args(argv)
    root = args.project.resolve()
    if args.prepare_from is not None:
        if args.reconciliation is None or args.run_stage:
            parser.error("Preparation requires reconciliation and cannot execute an individual stage")
        from matdiscovery.core_protocol import prepare_core_workspace
        prepare_core_workspace(args.prepare_from.resolve(), root,
                               reconciliation_path=args.reconciliation.resolve())
    if args.run_stage:
        return run_stage(root, args.run_stage)
    queue = declare_queue(root)
    print(json.dumps({"core_queue": str(queue), "stages": list(NAMES),
                      "original_full_study_complete": False}), flush=True)
    if args.declare_only:
        return 0
    # Reuse the tested supervisor: exclusive workspace lock, durable stage/PID
    # records, five-second heartbeats, and no automatic replay of failed stages.
    supervisor = root / "scripts/run_stage_queue.py"
    command = [str(root / "environments/policy-py312/bin/python"), "-B", str(supervisor),
               "--project", str(root)]
    result = subprocess.run(command, cwd=root)
    if result.returncode:
        return result.returncode
    proof = verified_receipt(root, "report")
    print(json.dumps({"core_complete": True, "report_receipt": proof,
                      "original_full_study_complete": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
