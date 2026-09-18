#!/usr/bin/env python3
"""Derive observed MADE label sidecars outside immutable complete collections."""
import argparse
import json
from pathlib import Path

from matdiscovery.failure_labels import FailureLabelError, derive_failure_labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Separate root, normally experiments/failure_labels")
    args = parser.parse_args(argv)
    reports, seen = [], set()
    for manifest in args.manifest:
        try:
            result = derive_failure_labels(manifest)
            identity = (result["model_key"], result["job_id"])
            if identity in seen:
                raise FailureLabelError("The same collection was supplied more than once")
            seen.add(identity)
            output = args.output_root / result["model_key"] / "made" / (result["job_id"] + ".json")
            # Re-verification makes a concurrent source change fail closed.
            result = derive_failure_labels(manifest, output_path=output)
            reports.append({"manifest": str(manifest), "status": "succeeded", "output": str(output.resolve()),
                            "rows": len(result["records"]), "sidecar_fingerprint": result["sidecar_fingerprint"]})
        except (FailureLabelError, OSError, ValueError, KeyError, TypeError) as exc:
            reports.append({"manifest": str(manifest), "status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
    passed = sum(row["status"] == "succeeded" for row in reports)
    print(json.dumps({"status": "succeeded" if passed == len(reports) else "failed",
                      "expected_collections": len(reports), "completed_collections": passed,
                      "physical_oracle_calls": 0, "collections": reports}, ensure_ascii=False))
    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
