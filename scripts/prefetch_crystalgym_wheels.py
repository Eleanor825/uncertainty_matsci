#!/usr/bin/env python3
"""Fetch large resolved Linux CPython 3.11 wheels from official PyPI with hashes.

This is a transport accelerator, not a dependency-version override. Input must
be the exact uv-compiled official CrystalGym requirements. No wheel is installed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--parts", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    requirements = []
    for line in args.requirements.read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^ ;]+)", line.strip())
        if match and (match[1] in {"torch", "triton", "scipy"} or match[1].startswith("nvidia-")):
            requirements.append((match[1], match[2]))
    records = []
    for name, version in requirements:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
            data = json.load(response)
        candidates = [item for item in data["urls"] if item["filename"].endswith(".whl") and
                      "x86_64" in item["filename"] and "manylinux" in item["filename"] and
                      ("cp311-cp311" in item["filename"] or "py3-none" in item["filename"])]
        if not candidates:
            continue  # e.g. small nvidia-ml-py3 source package already built by uv.
        item = sorted(candidates, key=lambda x: x["filename"])[0]
        if item["size"] < 20*1024*1024:
            continue
        records.append({"package": name, "version": version, "filename": item["filename"],
                        "url": item["url"], "size_bytes": item["size"], "sha256": item["digests"]["sha256"]})
    (args.output/"pypi-wheel-manifest.json").write_text(json.dumps(records, indent=2)+"\n")

    def fetch(item):
        target = args.output/item["filename"]
        start = time.monotonic()
        if target.exists() and sha256(target) == item["sha256"]:
            return {"file": target.name, "status": "verified_existing"}
        size, chunk = item["size_bytes"], 16*1024*1024
        parts = list(range((size+chunk-1)//chunk))

        def part(index):
            lo, hi = index*chunk, min(size, (index+1)*chunk)-1
            path = args.output/(item["filename"]+f".part{index:04d}")
            if path.exists() and path.stat().st_size == hi-lo+1:
                return path
            for attempt in range(4):
                try:
                    request = urllib.request.Request(item["url"], headers={"Range": f"bytes={lo}-{hi}"})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        expected = f"bytes {lo}-{hi}/{size}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected:
                            raise RuntimeError("Server did not honor exact byte range")
                        with path.open("wb") as output:
                            while True:
                                block = response.read(1024*1024)
                                if not block:
                                    break
                                output.write(block)
                    if path.stat().st_size != hi-lo+1:
                        raise RuntimeError("Truncated wheel range")
                    return path
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)

        with ThreadPoolExecutor(max_workers=args.parts) as pool:
            part_paths = list(pool.map(part, parts))
        assembling = target.with_name(target.name+".assembling")
        with assembling.open("wb") as output:
            for path in part_paths:
                with path.open("rb") as source:
                    while True:
                        block = source.read(8*1024*1024)
                        if not block:
                            break
                        output.write(block)
        if sha256(assembling) != item["sha256"]:
            raise RuntimeError(f"Official PyPI SHA256 mismatch: {target.name}")
        assembling.replace(target)
        for path in part_paths:
            path.unlink()
        return {"file": target.name, "status": "downloaded_sha256_verified", "bytes": size, "seconds": time.monotonic()-start}

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for task in as_completed([pool.submit(fetch, item) for item in records]):
            print(json.dumps(task.result()), flush=True)


if __name__ == "__main__":
    main()
