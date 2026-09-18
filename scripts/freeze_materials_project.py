#!/usr/bin/env python3
"""Fetch complete official MP entries once, then freeze per-system snapshots."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import sys
import time


def credential(path: Path | None) -> str:
    if os.environ.get("MATERIALS_PROJECT_API_KEY"):
        return os.environ["MATERIALS_PROJECT_API_KEY"]
    if path:
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix("export ")
            if line.startswith("MATERIALS_PROJECT_API_KEY="):
                words = shlex.split(line.split("=", 1)[1], comments=True)
                if words:
                    return words[0]
    raise RuntimeError("Materials Project credential is unavailable")


def atomic_json(path: Path, value) -> str:
    content = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(content)
    temporary.replace(path)
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path)
    parser.add_argument("--systems", type=Path, help="JSON with splits mapping to lists of element lists")
    args = parser.parse_args()
    from mp_api.client import MPRester
    # The live MP API now serializes these classes under pymatgen.core.entries.
    # Preserve MADE's pinned evaluator rather than upgrading its pymatgen version.
    # This affects import resolution only; class deserialization still validates
    # the full entry, structure, energy adjustments and numeric values.
    compatibility = []
    try:
        importlib.import_module("pymatgen.core.entries")
    except ModuleNotFoundError:
        old_entries = importlib.import_module("pymatgen.entries.computed_entries")
        sys.modules["pymatgen.core.entries"] = old_entries
        compatibility.append({"api_module": "pymatgen.core.entries", "local_module": "pymatgen.entries.computed_entries", "operation": "module_alias_only_no_numeric_changes"})

    root = args.project.resolve()
    output = root / "data" / "raw" / "materials_project"
    output.mkdir(parents=True, exist_ok=True)
    if args.systems:
        splits = json.loads(args.systems.read_text())["splits"]
    else:
        systems = []
        for complexity in ("ternary", "quaternary", "quinary"):
            path = root / "vendor/MADE/data/systems_10_mp_20" / f"systems_{complexity}_n10_maxatoms20_intermetallic_smact.json"
            systems.extend(json.loads(path.read_text()))
        splits = {"test": systems}
    seen = {}
    for split, systems in splits.items():
        for elements in systems:
            system = "-".join(sorted(elements))
            if system in seen and seen[system] != split:
                raise ValueError(f"Chemical system present in two splits: {system}")
            seen[system] = split
    index_path = output / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"schema": "mp_frozen_snapshots_v1", "snapshots": {}, "failures": {}}
    key = credential(args.credential_file)
    with MPRester(api_key=key, mute_progress_bars=True) as mpr:
        database_version = mpr.get_database_version()
        for system, split in sorted(seen.items()):
            path = output / f"{system}.json"
            existing = index["snapshots"].get(system)
            if path.exists() and existing and hashlib.sha256(path.read_bytes()).hexdigest() == existing["sha256"]:
                print(json.dumps({"system": system, "status": "verified_existing", "entries": existing["entries"]}), flush=True)
                continue
            elements = system.split("-")
            for attempt in range(3):
                try:
                    entries = mpr.get_entries_in_chemsys(elements=elements, additional_criteria={"thermo_types": ["GGA_GGA+U"]})
                    if not entries or any(getattr(entry, "structure", None) is None for entry in entries):
                        raise RuntimeError("MP returned empty or structure-incomplete entries")
                    value = {"schema": "mp_entries_snapshot_v1", "source": "materials_project", "database_version": database_version, "retrieved_at": datetime.now(timezone.utc).isoformat(), "elements": elements, "thermo_types": ["GGA_GGA+U"], "complete": True, "serialization_compatibility": compatibility, "entries": [entry.as_dict() for entry in entries]}
                    digest = atomic_json(path, value)
                    index["snapshots"][system] = {"path": str(path), "sha256": digest, "entries": len(entries), "split": split, "database_version": database_version}
                    index["failures"].pop(system, None)
                    atomic_json(index_path, index)
                    print(json.dumps({"system": system, "status": "frozen", "entries": len(entries), "sha256": digest}), flush=True)
                    break
                except Exception as exc:
                    message = str(exc).replace(key, "[REDACTED]")
                    if attempt == 2:
                        index["failures"][system] = {"exception": type(exc).__name__, "message": message, "split": split}
                        atomic_json(index_path, index)
                        print(json.dumps({"system": system, "status": "failed", "exception": type(exc).__name__}), flush=True)
                    else:
                        time.sleep(2 ** (attempt + 1))
    expected = set(seen)
    complete = expected.issubset(index["snapshots"]) and not (expected & set(index["failures"]))
    summary = {"expected_systems": len(expected), "completed_systems": len(expected & set(index["snapshots"])), "complete": complete}
    atomic_json(output / "completion.json", summary)
    print(json.dumps(summary), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
