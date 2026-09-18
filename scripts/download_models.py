#!/usr/bin/env python3
"""Download the audited official checkpoints without accepting unverified files.

Examples:
  python scripts/download_models.py --dry-run
  python scripts/download_models.py --models qwen35_4b --source modelscope
  python scripts/download_models.py --metadata-only --source modelscope

Only the currently selected audited official repository IDs are accepted. HF and
ModelScope use pinned commits, and every byte must match the immutable HF
manifest's LFS SHA256 or Git blob SHA1. No Hugging Face token is required.
Audited DeepSeek candidates are intentionally not downloaded until their
gradient backend and resource plan are selected and verified.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from urllib.parse import quote, urlencode
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_REPOSITORIES = {
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-12B-it",
    "Qwen/Qwen3.8-27B",
}
NETWORK_FAILURES = {5, 6, 7, 28, 35, 51, 60}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_relative_path(name: str) -> Path:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe manifest path: {name!r}")
    return Path(*path.parts)


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    keys = set()
    for model in manifest["models"]:
        if model["model_id"] not in ALLOWED_REPOSITORIES:
            raise ValueError(f"Repository is not an audited official source: {model['model_id']}")
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError("HF revision must be a full immutable commit SHA")
        if not re.fullmatch(r"[a-z0-9_]+", model["key"]) or model["key"] in keys:
            raise ValueError("Invalid or duplicate model key")
        keys.add(model["key"])
        if model.get("gated") is not False:
            raise ValueError("This downloader only accepts the audited non-gated checkpoints")
        if model.get("metadata_integrity_status") != "fixed_revision_official_hashes_verified":
            raise ValueError("Metadata checksums have not been audited")
        names = set()
        for item in model["metadata_files"] + model["weight_files"]:
            safe_relative_path(item["path"])
            if item["path"] in names:
                raise ValueError("Duplicate file in manifest")
            names.add(item["path"])
            if not isinstance(item["size_bytes"], int) or item["size_bytes"] <= 0:
                raise ValueError("Missing positive file size")
            valid_sha256 = bool(re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")))
            valid_blob_sha1 = bool(re.fullmatch(r"[0-9a-f]{40}", item.get("git_blob_sha1", "")))
            if not valid_sha256 and not valid_blob_sha1:
                raise ValueError(f"No official checksum for {item['path']}")
            if item["path"].endswith(".safetensors") and not valid_sha256:
                raise ValueError("Weights require the official LFS SHA256")
            if item.get("modelscope_revision") and not re.fullmatch(r"[0-9a-f]{40}", item["modelscope_revision"]):
                raise ValueError("ModelScope per-file revision must be an immutable commit")
            if item.get("inline_base64") and item["path"].endswith(".safetensors"):
                raise ValueError("Weight payloads cannot be embedded in the manifest")
        if sum(f["size_bytes"] for f in model["weight_files"]) != model["weights_total_bytes"]:
            raise ValueError("Weight sizes disagree with the model total")


def verify_file(path: Path, item: dict) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    actual_size = path.stat().st_size
    if actual_size != item["size_bytes"]:
        return False, f"size mismatch: {actual_size} != {item['size_bytes']}"
    digest = hashlib.sha256() if "sha256" in item else hashlib.sha1()
    if "sha256" not in item:
        digest.update(f"blob {actual_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_hash = digest.hexdigest()
    expected_hash = item.get("sha256", item.get("git_blob_sha1"))
    if actual_hash != expected_hash:
        return False, f"checksum mismatch: {actual_hash} != {expected_hash}"
    return True, actual_hash


def quarantine(path: Path, reason: str) -> str:
    target = path.with_name(path.name + ".invalid-" + uuid.uuid4().hex[:12])
    path.rename(target)
    print(f"QUARANTINED {path.name}: {reason}; kept at {target.name}", flush=True)
    return str(target)


def looks_like_invalid_weight(path: Path, item: dict) -> bool:
    if not item["path"].endswith(".safetensors") or not path.is_file():
        return False
    with path.open("rb") as stream:
        prefix = stream.read(256)
    if len(prefix) < 8:
        return False
    stripped = prefix.lstrip().lower()
    if stripped.startswith((b"<!doctype html", b"<html", b"<?xml")):
        return True
    # A safetensors file starts with an eight-byte little-endian JSON header size.
    header_size = int.from_bytes(prefix[:8], "little")
    return header_size <= 1 or header_size > min(100_000_000, item["size_bytes"] - 8)


def source_url(source: str, model: dict, item: dict) -> str:
    repository = quote(model["model_id"], safe="/")
    filename = quote(item["path"], safe="/")
    if source == "huggingface":
        return f"https://huggingface.co/{repository}/resolve/{model['revision']}/{filename}?download=true"
    if source == "modelscope":
        revision = item.get("modelscope_revision")
        if not revision:
            raise ValueError(f"No verified fixed ModelScope revision for {item['path']}")
        query = urlencode({"Revision": revision, "FilePath": item["path"]})
        return f"https://modelscope.cn/api/v1/models/{repository}/repo?{query}"
    raise ValueError("Only official Hugging Face and ModelScope sources are allowed")


def run_curl(url: str, partial: Path, connect_timeout: int, retries: int) -> tuple[int, str]:
    # curl validates Range/Content-Range when resuming. --fail rejects HTTP errors;
    # final length + cryptographic verification also rejects successful HTML pages.
    command = [
        "curl", "--http1.1", "--ipv4", "--fail", "--location",
        "--proto", "=https", "--proto-redir", "=https",
        "--connect-timeout", str(connect_timeout), "--retry", str(retries),
        "--speed-limit", "1024", "--speed-time", "120",
        "--retry-delay", "2", "--retry-connrefused", "--show-error", "--silent",
        "--continue-at", "-", "--output", str(partial), url,
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    # Do not log signed redirect URLs or authentication state.
    return result.returncode, result.stderr.strip()[-1200:]


def promote_verified(partial: Path, destination: Path, item: dict) -> None:
    good, reason = verify_file(partial, item)
    if not good:
        raise RuntimeError(f"Refusing to promote {partial.name}: {reason}")
    try:
        # Atomic creation with no overwrite, unlike rename/replace on POSIX.
        os.link(partial, destination)
    except FileExistsError:
        existing_good, _ = verify_file(destination, item)
        if not existing_good:
            raise RuntimeError(f"Destination appeared during download: {destination}")
    partial.unlink()


def download_file(
    model: dict, item: dict, directory: Path, sources: list[str],
    disabled_sources: set[str], connect_timeout: int, retries: int,
) -> dict:
    destination = directory / safe_relative_path(item["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    good, reason = verify_file(destination, item)
    if good:
        print(f"VERIFIED EXISTING {model['key']}/{item['path']}", flush=True)
        return {"path": item["path"], "status": "verified_existing", "hash": reason, "verified_at": now()}
    if destination.exists():
        quarantine(destination, reason)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        good, reason = verify_file(partial, item)
        if good:
            promote_verified(partial, destination, item)
            return {"path": item["path"], "status": "verified_resumed", "hash": reason, "verified_at": now()}
        if partial.stat().st_size >= item["size_bytes"] or looks_like_invalid_weight(partial, item):
            quarantine(partial, reason)
    if item.get("inline_base64"):
        # Small canonical metadata is embedded only when the official mirror has
        # a different bookkeeping/card/license file. It was fetched from the
        # fixed HF revision and is still checked before becoming a complete file.
        if partial.exists():
            quarantine(partial, "replace partial metadata with checksum-pinned canonical bytes")
        payload = base64.b64decode(item["inline_base64"], validate=True)
        partial.write_bytes(payload)
        good, reason = verify_file(partial, item)
        if not good:
            quarantine(partial, reason)
            raise RuntimeError(f"Inline canonical metadata failed verification: {item['path']}")
        promote_verified(partial, destination, item)
        return {"path": item["path"], "status": "canonical_inline_verified", "source": "fixed_hf_revision",
                "hash": reason, "size_bytes": item["size_bytes"], "verified_at": now()}
    errors = []
    for source in sources:
        if source in disabled_sources:
            continue
        for attempt in range(2):
            offset = partial.stat().st_size if partial.exists() else 0
            print(f"DOWNLOAD {model['key']}/{item['path']} source={source} resume_bytes={offset}", flush=True)
            code, detail = run_curl(source_url(source, model, item), partial, connect_timeout, retries)
            # A transfer may be complete even if the final connection close failed.
            good, reason = verify_file(partial, item)
            if good:
                promote_verified(partial, destination, item)
                print(f"VERIFIED {model['key']}/{item['path']}", flush=True)
                return {"path": item["path"], "status": "downloaded_verified", "source": source,
                        "hash": reason, "size_bytes": item["size_bytes"], "verified_at": now()}
            error = f"{source}: curl={code}; {reason}; {detail}"
            errors.append(error)
            if code in NETWORK_FAILURES:
                disabled_sources.add(source)
            invalid_complete = partial.exists() and partial.stat().st_size >= item["size_bytes"]
            invalid_payload = looks_like_invalid_weight(partial, item)
            if invalid_complete or invalid_payload:
                quarantine(partial, reason if invalid_complete else "not a safetensors payload")
                if attempt == 0 and source not in disabled_sources:
                    continue
                break
            if code in (33, 36) and offset and attempt == 0:
                # A source that cannot resume gets one clean retry. Keep the old
                # partial as evidence instead of overwriting it.
                if partial.exists():
                    quarantine(partial, "source did not support the requested byte range")
                continue
            break
    if not errors:
        errors.append("All requested sources are unavailable after earlier network failures")
    raise RuntimeError(f"No verified download for {model['key']}/{item['path']}: " + " | ".join(errors))


def write_receipt(directory: Path, receipt: dict) -> None:
    temp = directory / (".download_receipt-" + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, directory / "download_receipt.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs/model_manifest.json")
    parser.add_argument("--dest-root", type=Path, default=PROJECT_ROOT / "assets/models")
    parser.add_argument("--models", nargs="+", help="Current manifest model keys; defaults to the selected scope")
    parser.add_argument("--source", choices=["auto", "huggingface", "modelscope"], default="auto")
    parser.add_argument("--metadata-only", action="store_true", help="Download configs/tokenizers/cards only, no weights")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without network or filesystem changes")
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args(argv)
    if args.connect_timeout < 1 or args.retries < 0:
        parser.error("Timeout must be positive and retries nonnegative")
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    selected = args.models or manifest["selection"]["ordered_model_keys"]
    by_key = {m["key"]: m for m in manifest["models"]}
    if len(set(selected)) != len(selected) or any(key not in by_key for key in selected):
        parser.error("Models must be unique known manifest keys")
    sources = ["huggingface", "modelscope"] if args.source == "auto" else [args.source]
    models = [by_key[key] for key in selected]
    plan = [{"key": m["key"], "model_id": m["model_id"], "revision": m["revision"],
             "files": len(m["metadata_files"]) + (0 if args.metadata_only else len(m["weight_files"])),
             "bytes": sum(f["size_bytes"] for f in m["metadata_files"]) + (0 if args.metadata_only else m["weights_total_bytes"]),
             "destination": str(args.dest_root / m["key"])} for m in models]
    print(json.dumps({"plan": plan, "sources": sources, "metadata_only": args.metadata_only}, indent=2), flush=True)
    if args.dry_run:
        return 0
    if not shutil.which("curl"):
        raise RuntimeError("curl is required for validated IPv4 HTTPS range downloads")
    disabled_sources: set[str] = set()
    failed = False
    for model in models:
        directory = args.dest_root / model["key"]
        directory.mkdir(parents=True, exist_ok=True)
        receipt = {"model_id": model["model_id"], "revision": model["revision"], "started_at": now(),
                   "scope": "metadata_only" if args.metadata_only else "complete_checkpoint", "files": [], "status": "in_progress"}
        with (directory / ".download.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another downloader holds the model lock: {directory}")
            try:
                files = model["metadata_files"] + ([] if args.metadata_only else model["weight_files"])
                for item in files:
                    receipt["files"].append(download_file(model, item, directory, sources, disabled_sources,
                                                          args.connect_timeout, args.retries))
                    write_receipt(directory, receipt)
                receipt["status"] = "metadata_verified" if args.metadata_only else "complete_verified"
            except (RuntimeError, OSError) as exc:
                failed = True
                receipt["status"] = "failed"
                receipt["error"] = str(exc)
                print(f"FAILED {model['key']}: {exc}", file=sys.stderr, flush=True)
            finally:
                receipt["finished_at"] = now()
                write_receipt(directory, receipt)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
