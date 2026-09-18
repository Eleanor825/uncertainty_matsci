#!/usr/bin/env python3
"""Execute a research stage from an immutable source/configuration snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.project.resolve()
    entry = Path(args.entry)
    if entry.name != args.entry or entry.suffix != ".py" or not (root / "scripts" / entry).is_file():
        raise ValueError("Entry must name an existing project Python script")
    files = sorted(p for directory in ("src", "scripts", "configs") for p in (root / directory).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".json", ".yaml", ".toml", ".txt"}
        and p.name != "stage_queue.json")
    hashes = {str(p.relative_to(root)): sha(p) for p in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    parent = root / "frozen_sources"
    parent.mkdir(exist_ok=True)
    destination = parent / ("stage-" + digest[:24])
    source_manifest = {"original_project": str(root), "source_fingerprint": digest, "files": hashes}
    if not destination.exists():
        temporary = Path(tempfile.mkdtemp(prefix=".stage-partial-", dir=parent))
        for relative, expected in hashes.items():
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative, target)
            if sha(target) != expected or sha(root / relative) != expected:
                raise RuntimeError("Project source changed while taking the execution snapshot")
        for directory in ("data", "vendor", "environments", "experiments", "logs", "tools"):
            (temporary / directory).symlink_to(root / directory, target_is_directory=True)
        (temporary / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
        temporary.rename(destination)
    if json.loads((destination / "source_manifest.json").read_text()) != source_manifest:
        raise RuntimeError("Immutable source provenance differs from this project and snapshot")
    for relative, expected in hashes.items():
        if sha(destination / relative) != expected:
            raise RuntimeError("An immutable source snapshot was modified")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if any(value == "--project" or value.startswith("--project=") for value in arguments):
        raise ValueError("The wrapper supplies the immutable --project path")
    print(json.dumps({"frozen_execution_root": str(destination), "source_fingerprint": digest, "entry": args.entry}), flush=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(destination / "src"), PYTHONHASHSEED="0", PYTHONNOUSERSITE="1")
    os.execve(sys.executable, [sys.executable, str(destination / "scripts" / entry),
        "--project", str(destination), *arguments], env)


if __name__ == "__main__":
    main()
