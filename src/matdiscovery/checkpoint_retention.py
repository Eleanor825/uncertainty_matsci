"""Retire redundant completed-study ES snapshots, preserving exact replay proof.

Only this project's generated policy.pt files are eligible. Initial pretrained
assets, current/best/final weights, histories, markers and raw results remain.
No in-progress training snapshot is retired and an unexplained missing tensor
never becomes a retirement record.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

from .accounting import file_sha256, fingerprint, write_json_atomic


SCHEMA = "completed_es_checkpoint_retirement_v1"


def _read(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _completion_identity(summary):
    return {name: summary[name] for name in (
        "run_fingerprint", "required_generations", "completed_generations", "expected_evaluations", "completed_evaluations",
        "best_generation", "best_actual_model_state_hash", "final_actual_model_state_hash", "initial_checkpoint_manifest_hash")}


def _validated_completion(root):
    summary = _read(root / "training_summary.json")
    manifest = _read(root / "run_manifest.json")
    end = manifest["identity"]["config"]["generations"]
    best = summary["best_generation"]
    _require(summary["status"] == "complete" and summary["completed_generations"] == summary["required_generations"] == end
             and summary["completed_evaluations"] == summary["expected_evaluations"] and summary["test_data_used"] is False,
             "Only fully completed ES conditions can retire generated checkpoints")
    _require(type(best) is int and 1 <= best <= end and summary["run_fingerprint"] == manifest["run_fingerprint"] == fingerprint(manifest["identity"]),
             "Completed ES identity or selected checkpoint is invalid")
    proof = summary["clean_reload"]
    final = root / "checkpoints" / f"generation_{end:04d}" / "policy.pt"
    _require(summary["clean_reload_checked_generation"] == end and proof["verified"] is True and proof["fresh_instance"] is True
             and proof["allclose"] is True and proof["loaded_state_hash"] == proof["expected_state_hash"] == summary["final_actual_model_state_hash"]
             and proof["checkpoint_file_sha256"] == file_sha256(final), "Retirement requires retained final weights and their actual clean-reload proof")
    markers = {}
    for generation in range(end + 1):
        folder = root / "checkpoints" / f"generation_{generation:04d}"
        marker = _read(folder / "complete.json")
        _require(marker["status"] == "complete" and marker["generation"] == generation and marker["run_fingerprint"] == summary["run_fingerprint"],
                 "Every original ES generation must retain its complete marker")
        _require(set(marker["files"]) == {"policy.pt", "policy.pt.json", "driver_state.json", "es_history.json"}, "Incomplete ES commit marker")
        for name in ("policy.pt.json", "driver_state.json", "es_history.json"):
            _require(file_sha256(folder / name) == marker["files"][name], "Checkpoint history/metadata changed before retirement")
        markers[generation] = marker
    for generation, expected in ((best, summary["best_actual_model_state_hash"]), (end, summary["final_actual_model_state_hash"])):
        path = root / "checkpoints" / f"generation_{generation:04d}" / "policy.pt"
        _require(markers[generation]["actual_model_state_hash"] == expected and file_sha256(path) == markers[generation]["files"]["policy.pt"],
                 "Best and final checkpoints must remain present and checksum-correct")
    return summary, manifest, markers


def verify_retired_checkpoint(root: str | Path, generation: int, marker: dict, *, verify_retained_weights: bool = True) -> dict:
    """Allow a missing old tensor only with an explicit immutable retirement proof."""
    root = Path(root).resolve()
    receipt = _read(root / "checkpoint_retirement.json")
    _require(receipt["schema"] == SCHEMA and receipt["status"] == "retirement_committed"
             and receipt["retirement_fingerprint"] == fingerprint({k: v for k, v in receipt.items() if k != "retirement_fingerprint"}),
             "Checkpoint retirement receipt is unknown or modified")
    summary = _read(root / "training_summary.json")
    _require(summary["status"] == "complete" and _completion_identity(summary) == receipt["completion_identity"]
             and receipt["run_fingerprint"] == marker["run_fingerprint"]
             and generation not in receipt["retained_generations"], "Retirement cannot excuse a missing active/best/final checkpoint")
    row = receipt["retired"].get(str(generation))
    folder = root / "checkpoints" / f"generation_{generation:04d}"
    _require(row is not None and row["policy_pt_sha256"] == marker["files"]["policy.pt"]
             and row["actual_model_state_hash"] == marker["actual_model_state_hash"]
             and row["complete_marker_sha256"] == file_sha256(folder / "complete.json")
             and row["relative_path"] == f"checkpoints/generation_{generation:04d}/policy.pt",
             "Missing checkpoint has no matching pre-deletion verified retirement entry")
    # Batch checkpoint auditors may hash retained weights once themselves; this
    # avoids re-reading tens of GB for every retired generation's metadata.
    if verify_retained_weights:
        for retained in receipt["retained_generations"]:
            keep = root / "checkpoints" / f"generation_{retained:04d}"
            keep_marker = _read(keep / "complete.json")
            _require((keep / "policy.pt").is_file() and file_sha256(keep / "policy.pt") == keep_marker["files"]["policy.pt"],
                     "Retirement is valid only while best/final snapshots remain checksum-correct")
    return row


def retire_completed_es_checkpoints(root: str | Path) -> dict:
    """Commit a retirement receipt, then unlink only verified redundant policy.pt.

    Re-running after an interruption finishes the already declared retirement.
    Rebuilding uses the pinned initial pretrained model and the retained exact
    seeded ES history, verifying each tensor hash with AgenticESOpt.replay_history.
    """
    root = Path(root).resolve()
    _require(root.is_dir(), "Training directory does not exist")
    with (root / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Cannot retire checkpoints during a running ES session") from exc
        summary, manifest, markers = _validated_completion(root)
        retained = sorted({summary["best_generation"], summary["required_generations"]})
        receipt_path = root / "checkpoint_retirement.json"
        if receipt_path.exists():
            receipt = _read(receipt_path)
            _require(receipt["completion_identity"] == _completion_identity(summary), "Retirement belongs to another completed training state")
            for generation in markers:
                if generation not in retained:
                    verify_retired_checkpoint(root, generation, markers[generation], verify_retained_weights=False)
        else:
            retired = {}
            for generation, marker in markers.items():
                if generation in retained:
                    continue
                path = root / "checkpoints" / f"generation_{generation:04d}" / "policy.pt"
                _require(path.is_file() and not path.is_symlink() and file_sha256(path) == marker["files"]["policy.pt"],
                         "Unexplained missing/corrupt/generated checkpoint cannot be declared retired")
                retired[str(generation)] = {"relative_path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
                    "policy_pt_sha256": marker["files"]["policy.pt"], "actual_model_state_hash": marker["actual_model_state_hash"],
                    "complete_marker_sha256": file_sha256(path.parent / "complete.json")}
            receipt = {"schema": SCHEMA, "status": "retirement_committed", "run_fingerprint": summary["run_fingerprint"],
                       "completion_identity": _completion_identity(summary), "retained_generations": retained, "retired": retired,
                       "initial_checkpoint_manifest_hash": summary["initial_checkpoint_manifest_hash"],
                       "initial_actual_model_state_hash": manifest["initial_actual_model_state_hash"],
                       "rebuild": "Load exact pinned initial weights, then AgenticESOpt.replay_history(generation_N/es_history.json); validate every step hash. No physical actions are rerun.",
                       "retirement_scope": "generated policy.pt only; all metadata, hashes, seed/reward history and raw trajectories retained"}
            receipt["retirement_fingerprint"] = fingerprint(receipt)
            write_json_atomic(receipt_path, receipt)  # durable authorization BEFORE deletion
        removed_now = 0
        for row in receipt["retired"].values():
            path = root / row["relative_path"]
            if path.exists():
                _require(not path.is_symlink() and file_sha256(path) == row["policy_pt_sha256"], "Checkpoint changed after its retirement receipt was committed")
                path.unlink()
                removed_now += row["size_bytes"]
        completion = {"status": "complete", "retirement_fingerprint": receipt["retirement_fingerprint"],
                      "retained_generations": retained, "retired_generations": sorted(map(int, receipt["retired"])),
                      "retired_bytes": sum(row["size_bytes"] for row in receipt["retired"].values()), "removed_bytes_this_call": removed_now}
        write_json_atomic(root / "checkpoint_retirement_completion.json", completion)
        return completion
