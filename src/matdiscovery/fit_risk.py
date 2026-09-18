"""Fit the three declared risk predictors from complete train/dev collections.

Input manifests use ``decision_files`` and optional ``graph_feature_files`` lists.
Entries are paths or {path, sha256}; relative paths are relative to the manifest.
Complete manifests with verified policy runtime provenance are required. All listed files/rows are read,
and adjacent graph_features.jsonl files are joined by decision_id. Test records,
cross-split groups, non-initial checkpoints, and stale replay graphs fail closed.

Future-failure labels describe the next scientific assessment/terminal outcome.
They are never claims that an individual decision caused the eventual failure.
Null labels are excluded from supervised fitting with explicit counts/reasons.
Missing graphs remain rows with null graph values and a missingness indicator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .accounting import canonical_json, file_sha256, fingerprint, write_json_atomic


RISK_METHODS = ("entropy_risk", "hidden_risk", "graph_risk")
FEATURE_PREFIXES = {
    "entropy_risk": ("sampling.",),
    "hidden_risk": ("sampling.", "hidden.", "action."),
    "graph_risk": ("sampling.", "hidden.", "graph.", "action."),
}
LABELS = {
    "future_failure": "label_future_failure",
    "immediate_error": "label_immediate_error",
}
LABEL_MEANINGS = {
    "future_failure": "Observed next scientific evaluation or terminal failure risk; not a causal error label for each decision.",
    "immediate_error": "Observed immediate action/interface error; not proof of material or causal validity.",
}
GRAPH_SUCCESS = {"ok", "success", "succeeded", "available", "complete"}


class RiskDataError(ValueError):
    """The supplied data cannot support the declared train/dev risk fit."""


@dataclass
class RiskDataset:
    records: list[dict]
    provenance: dict


def _json(raw: bytes | str, source: Path):
    def nonfinite(value):
        raise RiskDataError(f"Nonfinite JSON value {value} in {source}; use explicit null")
    try:
        return json.loads(raw, parse_constant=nonfinite)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise RiskDataError(f"Invalid JSON in {source}: {exc}") from exc


def _path(entry, base: Path) -> tuple[Path, str | None]:
    if isinstance(entry, str):
        name, expected_hash = entry, None
    elif isinstance(entry, Mapping) and isinstance(entry.get("path"), str):
        name, expected_hash = entry["path"], entry.get("sha256")
    else:
        raise RiskDataError("Manifest file entries must be paths or {path, sha256} objects")
    path = Path(name)
    return (path if path.is_absolute() else base / path).resolve(), expected_hash


def _read(path: Path, kind: str, inventory: dict, expected_hash: str | None = None) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RiskDataError(f"Cannot read declared {kind} file {path}: {exc}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise RiskDataError(f"Declared file hash mismatch: {path}")
    if str(path) in inventory and inventory[str(path)]["sha256"] != digest:
        raise RiskDataError(f"Input changed while reading: {path}")
    inventory[str(path)] = {"path": str(path), "sha256": digest, "size_bytes": len(raw), "kind": kind}
    return raw


def _jsonl(path: Path, kind: str, inventory: dict, expected_hash=None) -> list[dict]:
    raw = _read(path, kind, inventory, expected_hash)
    rows = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        row = _json(line, path)
        if not isinstance(row, dict):
            raise RiskDataError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append({**row, "_source_path": str(path), "_source_line": line_number})
    inventory[str(path)]["rows"] = len(rows)
    return rows


def _initial_stamp(stamp: Any, context: str) -> dict:
    required = {"state_id", "checkpoint_hash", "model_id", "generation", "perturbation_seed", "perturbation_sigma"}
    if not isinstance(stamp, Mapping) or not required.issubset(stamp):
        raise RiskDataError(f"{context} needs the complete initial PolicyStamp")
    for name in ("state_id", "checkpoint_hash", "model_id"):
        if not isinstance(stamp[name], str) or not stamp[name]:
            raise RiskDataError(f"Invalid {name} in {context}")
    if type(stamp["generation"]) is not int or stamp["generation"] != 0:
        raise RiskDataError(f"{context} must come from generation 0 of the initial checkpoint")
    if stamp["perturbation_seed"] is not None or stamp["perturbation_sigma"] is not None:
        raise RiskDataError(f"{context} must not come from a perturbed policy")
    return dict(stamp)


def _features(values: Any, *, graph_only: bool = False) -> dict:
    if not isinstance(values, dict):
        raise RiskDataError("Every decision/graph record needs a feature dictionary")
    prefixes = ("graph.",) if graph_only else FEATURE_PREFIXES["graph_risk"]
    for name, value in values.items():
        if not isinstance(name, str) or not name.startswith(prefixes) or name.endswith("."):
            raise RiskDataError(f"Unapproved feature prefix (possible outcome leakage): {name!r}")
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise RiskDataError(f"Feature {name} must be finite numeric or explicit null")
    return dict(values)


def _prefix(record: Mapping) -> str | None:
    value = record.get("prefix_hash", record.get("policy_stamp", {}).get("prefix_hash"))
    return value if isinstance(value, str) and value else None


def _check_graph_identity(graph: Mapping, decision: Mapping) -> None:
    original = decision["policy_stamp"]
    source_state = graph.get("source_policy_state_id", graph.get("policy_state_id"))
    if source_state != original["state_id"]:
        raise RiskDataError(f"Graph source policy state differs for decision {decision['decision_id']}")
    if _prefix(decision) is None or graph.get("prefix_hash") != _prefix(decision):
        raise RiskDataError(f"Stale/different graph prefix for decision {decision['decision_id']}")
    if graph.get("checkpoint_hash", original["checkpoint_hash"]) != original["checkpoint_hash"]:
        raise RiskDataError("Graph checkpoint differs from the decision checkpoint")
    generation = decision["generation"]
    if (graph.get("policy_runtime") != generation["policy_runtime"]
            or graph.get("source_policy_configuration_fingerprint") != generation["configuration_fingerprint"]):
        raise RiskDataError("Graph runtime/precision or source configuration differs from its decision")
    replay_id = graph.get("replay_policy_state_id")
    # A source tag cannot conceal a different graph-producing session under the
    # legacy policy_state_id key: that session also needs verified replay proof.
    if "source_policy_state_id" in graph and graph.get("policy_state_id") not in (None, source_state):
        if replay_id is not None and graph["policy_state_id"] != replay_id:
            raise RiskDataError("Graph-producing policy_state_id and replay_policy_state_id disagree")
        replay_id = graph["policy_state_id"]
    replay_stamp = graph.get("replay_policy_stamp")
    if replay_stamp is not None:
        replay_stamp = _initial_stamp(replay_stamp, "Graph replay")
        if replay_id is not None and replay_id != replay_stamp["state_id"]:
            raise RiskDataError("Graph replay state_id and replay PolicyStamp disagree")
    elif replay_id is not None and replay_id != original["state_id"]:
        flat = {"state_id": replay_id}
        for name in ("checkpoint_hash", "model_id"):
            if name not in graph:
                raise RiskDataError("Cross-session graph replay requires checkpoint_hash and model_id")
            flat[name] = graph[name]
        for name in ("generation", "perturbation_seed", "perturbation_sigma"):
            if f"replay_{name}" not in graph:
                raise RiskDataError("Cross-session replay needs proof of generation 0 and no perturbation")
            flat[name] = graph[f"replay_{name}"]
        replay_stamp = _initial_stamp(flat, "Graph replay")
    if replay_stamp is not None:
        if any(replay_stamp[k] != original[k] for k in ("checkpoint_hash", "model_id")):
            raise RiskDataError("Graph replay did not use the same verified initial checkpoint")


def _merge_graph(decision: dict, graph: Mapping | None) -> None:
    inline = {k: v for k, v in decision["features"].items() if k.startswith("graph.")}
    if graph is not None:
        _check_graph_identity(graph, decision)
        features = _features(graph.get("features"), graph_only=True)
        status = graph.get("graph_status")
        if not isinstance(status, str) or not status:
            raise RiskDataError("Offline graph needs an explicit graph_status")
        if decision.get("graph_status") in GRAPH_SUCCESS:
            for name in inline.keys() & features.keys():
                if inline[name] is not None and inline[name] != features[name]:
                    raise RiskDataError(f"Offline graph conflicts with an already recorded graph feature: {name}")
        decision["features"] = {k: v for k, v in decision["features"].items() if not k.startswith("graph.")}
        decision["features"].update(features)
        decision["_graph_source"] = graph["_source_path"]
    else:
        features = inline
        status = decision.get("graph_status")
        if status is None:
            status = "available" if features.get("graph.graph_missing") == 0 and _prefix(decision) else "missing"
        if not isinstance(status, str):
            raise RiskDataError("graph_status must be a string")
        decision["_graph_source"] = "inline" if features else None
    substantive = any(value is not None for name, value in features.items() if name != "graph.graph_missing" and not name.endswith("_missing"))
    available = status in GRAPH_SUCCESS and substantive
    if status in GRAPH_SUCCESS and features.get("graph.graph_missing") == 1:
        raise RiskDataError("A successful graph cannot simultaneously have graph_missing=1")
    if available and _prefix(decision) is None:
        raise RiskDataError("Available graph features require the captured prefix hash")
    if not available:
        for name in list(decision["features"]):
            if name.startswith("graph."):
                decision["features"][name] = None
    decision["features"]["graph.graph_missing"] = float(not available)
    decision["_graph_available"] = available
    decision["_graph_status"] = status if substantive or status not in GRAPH_SUCCESS else "missing_numeric_graph_features"


def _group_hash(values: Iterable[str]) -> str:
    return fingerprint(sorted(set(values)))


def _episode_identity(row: Mapping) -> str:
    return canonical_json([row["benchmark"], row["model_key"], row["group_id"], str(row["episode_id"])])


def _coverage(rows: Sequence[Mapping]) -> dict:
    return {split: {
        "rows": len(subset := [r for r in rows if r["split"] == split]),
        "available": sum(r["_graph_available"] for r in subset),
        "missing": sum(not r["_graph_available"] for r in subset),
        "status_counts": dict(Counter(r["_graph_status"] for r in subset)),
    } for split in ("train", "dev")}


def load_risk_dataset(
    manifest_paths: Sequence[str | Path], *, model_key: str,
    label_kind: str = "future_failure", graph_paths: Sequence[str | Path] = (),
    expected_checkpoint_hash: str | None = None,
) -> RiskDataset:
    """Read every supplied collection, verify identities, and retain unlabelled rows."""
    if label_kind not in LABELS or not manifest_paths or not model_key:
        raise RiskDataError("Require manifests, model_key and a declared supported label_kind")
    inventory, decision_files, graph_files = {}, {}, {}
    manifests = []
    common_runtime = common_configuration = common_configuration_fingerprint = None
    for supplied in manifest_paths:
        path = Path(supplied).resolve()
        if path.suffix == ".jsonl":
            raise RiskDataError("Provide complete collection manifests with policy runtime provenance, not standalone decision JSONL")
        manifest = _json(_read(path, "collection_manifest", inventory), path)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("decision_files"), list):
            raise RiskDataError("Collection manifest requires a decision_files list")
        if not manifest["decision_files"]:
            raise RiskDataError(f"Collection manifest has no decision files: {path}")
        runtime, configuration = manifest.get("policy_runtime"), manifest.get("policy_configuration")
        configuration_fingerprint = manifest.get("policy_configuration_fingerprint")
        if (manifest.get("complete") is not True or not isinstance(runtime, dict) or not runtime
                or not isinstance(configuration, dict) or not configuration):
            raise RiskDataError("Risk fitting requires complete collection policy runtime/configuration provenance")
        configuration_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        if configuration_fingerprint != configuration_hash or configuration.get("policy_runtime") != runtime:
            raise RiskDataError("Collection runtime/configuration fingerprint is inconsistent")
        if common_runtime is None:
            common_runtime, common_configuration, common_configuration_fingerprint = runtime, configuration, configuration_fingerprint
        elif (runtime, configuration, configuration_fingerprint) != (common_runtime, common_configuration, common_configuration_fingerprint):
            raise RiskDataError("Collections mix policy runtime/precision/backend/configuration")
        manifests.append({"path": str(path), "content": manifest})
        for field, target in (("decision_files", decision_files), ("graph_feature_files", graph_files)):
            entries = manifest.get(field, [])
            if not isinstance(entries, list):
                raise RiskDataError(f"Manifest {field} must be a list")
            for entry in entries:
                resolved, digest = _path(entry, path.parent)
                if resolved in target and (field == "decision_files" or target[resolved] != digest):
                    raise RiskDataError(f"Duplicate/conflicting declared input file: {resolved}")
                target[resolved] = digest
    for supplied in graph_paths:
        graph_files.setdefault(Path(supplied).resolve(), None)
    for path in decision_files:
        adjacent = path.parent / "graph_features.jsonl"
        if adjacent.exists():
            graph_files.setdefault(adjacent.resolve(), None)
    records, by_id, policies, checkpoints = [], {}, {}, set()
    for path, digest in sorted(decision_files.items()):
        for row in _jsonl(path, "decisions", inventory, digest):
            for name in ("decision_id", "benchmark", "model_key", "split", "task_id", "group_id", "input_ids_file"):
                if not isinstance(row.get(name), str) or not row[name]:
                    raise RiskDataError(f"Decision needs a nonempty {name}: {path}:{row['_source_line']}")
            if row["split"] not in {"train", "dev"}:
                raise RiskDataError(f"Test/undeclared split records are forbidden for fitting: {row['decision_id']}")
            if row["model_key"] != model_key:
                raise RiskDataError("Supplied collections must all belong to the requested model_key; no silent filtering")
            if not isinstance(row.get("episode_id"), (int, str)) or isinstance(row["episode_id"], bool):
                raise RiskDataError("Every decision requires an episode_id")
            if row["decision_id"] in by_id:
                raise RiskDataError(f"Duplicate decision_id: {row['decision_id']}")
            for label in LABELS.values():
                if label not in row or (row[label] is None and label == "label_immediate_error"):
                    raise RiskDataError(f"Decision needs an observed immediate label and explicit future label/null: {label}")
                if row[label] is not None and (type(row[label]) is not int or row[label] not in (0, 1)):
                    raise RiskDataError(f"Label {label} must be binary 0/1 or explicit null")
            row["policy_stamp"] = _initial_stamp(row.get("policy_stamp"), "Decision")
            generation = row.get("generation")
            if (not isinstance(generation, dict) or generation.get("policy_runtime") != common_runtime
                    or generation.get("configuration_fingerprint") != common_configuration_fingerprint):
                raise RiskDataError("Decision runtime/precision or configuration differs from its collection")
            if _prefix(row) is None:
                raise RiskDataError("Every decision needs its complete prefix_hash for train/dev leakage checks")
            stamp = row["policy_stamp"]
            checkpoints.add((stamp["model_id"], stamp["checkpoint_hash"]))
            policies[stamp["state_id"]] = stamp
            if expected_checkpoint_hash is not None and stamp["checkpoint_hash"] != expected_checkpoint_hash:
                raise RiskDataError("Collection does not match the explicitly requested initial checkpoint hash")
            row["features"] = _features(row.get("features"))
            ids_path, _ = _path(row["input_ids_file"], path.parent)
            if str(ids_path) not in inventory:
                _read(ids_path, "captured_input_ids", inventory, row.get("input_ids_sha256"))
            elif row.get("input_ids_sha256") not in (None, inventory[str(ids_path)]["sha256"]):
                raise RiskDataError("Conflicting captured input_ids file hash")
            row["_input_ids_path"] = str(ids_path)
            by_id[row["decision_id"]] = row
            records.append(row)
    if not records:
        raise RiskDataError("No decision records were supplied; no model can be fitted")
    if len(checkpoints) != 1:
        raise RiskDataError("Train/dev and cross-benchmark collections must use the same initial checkpoint")
    groups = {split: {r["group_id"] for r in records if r["split"] == split} for split in ("train", "dev")}
    if overlap := groups["train"] & groups["dev"]:
        raise RiskDataError(f"Train/dev group_id leakage: {sorted(overlap)[:5]}")
    graphs = {}
    for path, digest in sorted(graph_files.items()):
        for graph in _jsonl(path, "graph_features", inventory, digest):
            decision_id = graph.get("decision_id")
            if not isinstance(decision_id, str) or decision_id not in by_id:
                raise RiskDataError(f"Graph has no matching supplied decision: {decision_id!r}")
            if decision_id in graphs:
                raise RiskDataError(f"Duplicate graph decision_id: {decision_id}")
            graphs[decision_id] = graph
    for row in records:
        _merge_graph(row, graphs.get(row["decision_id"]))
    records.sort(key=lambda r: r["decision_id"])
    train_prefixes = {_prefix(row) for row in records if row["split"] == "train"}
    duplicates = [row for row in records if row["split"] == "dev" and _prefix(row) in train_prefixes]
    for row in records:
        row["_excluded_from_fit_reason"] = (
            "duplicate_dev_prefix_in_train" if row["split"] == "dev" and _prefix(row) in train_prefixes else None
        )
    label_field = LABELS[label_kind]
    label_counts = {}
    for split in ("train", "dev"):
        subset = [r for r in records if r["split"] == split]
        labelled = [r for r in subset if r[label_field] is not None]
        fit_rows = [r for r in labelled if r["_excluded_from_fit_reason"] is None]
        reasons = Counter(str(r.get(f"{label_field}_reason") or r.get("label_missing_reason") or "reason_not_recorded") for r in subset if r[label_field] is None)
        label_counts[split] = {
            "all_rows": len(subset), "labelled_rows": len(labelled),
            "fit_rows": len(fit_rows),
            "null_label_rows": len(subset) - len(labelled), "null_label_reasons": dict(reasons),
            "class_counts": {str(k): v for k, v in sorted(Counter(r[label_field] for r in labelled).items())},
            "fit_class_counts": {str(k): v for k, v in sorted(Counter(r[label_field] for r in fit_rows).items())},
            "duplicate_dev_prefix_rows": sum(r["_excluded_from_fit_reason"] is not None for r in subset),
            "labelled_duplicate_dev_prefix_rows": len(labelled) - len(fit_rows),
            "all_groups": len(groups[split]), "all_groups_hash": _group_hash(groups[split]),
            "fit_groups": len({r["group_id"] for r in fit_rows}),
            "fit_groups_hash": _group_hash(r["group_id"] for r in fit_rows),
            "groups_without_fit_rows": sorted(groups[split] - {r["group_id"] for r in fit_rows}),
            "fit_episodes": len({_episode_identity(r) for r in fit_rows}),
            "fit_episode_ids_hash": _group_hash(_episode_identity(r) for r in fit_rows),
            "fit_decision_ids_hash": _group_hash(r["decision_id"] for r in fit_rows),
        }
    model_id, checkpoint_hash = next(iter(checkpoints))
    provenance = {
        "schema_version": 2, "model_key": model_key, "model_id": model_id,
        "policy_runtime": common_runtime, "policy_configuration": common_configuration,
        "policy_configuration_fingerprint": common_configuration_fingerprint,
        "checkpoint_hash": checkpoint_hash, "policy_stamps": [policies[k] for k in sorted(policies)],
        "benchmarks": sorted({r["benchmark"] for r in records}),
        "source_files": [inventory[k] for k in sorted(inventory)],
        "collection_manifests": manifests, "total_decision_rows": len(records),
        "source_decision_ids_hash": _group_hash(r["decision_id"] for r in records),
        "label_kind": label_kind, "label_field": label_field,
        "label_meaning": LABEL_MEANINGS[label_kind], "label_counts": label_counts,
        "graph_coverage_all_rows": _coverage(records),
        "graph_coverage_labelled_rows": _coverage([r for r in records if r[label_field] is not None]),
        "graph_coverage_fit_rows": _coverage([r for r in records if r[label_field] is not None and r["_excluded_from_fit_reason"] is None]),
        "development_exact_prefix_filter": {
            "rule": "Exclude dev decisions whose complete prefix_hash occurs in any train decision; retain every raw record and all train rows.",
            "duplicate_dev_prefix_rows": len(duplicates),
            "unique_duplicate_dev_prefixes": len({_prefix(r) for r in duplicates}),
            "retained_development_groups": sorted({r["group_id"] for r in records if r["split"] == "dev" and r["_excluded_from_fit_reason"] is None}),
            "groups_without_fit_rows": label_counts["dev"]["groups_without_fit_rows"],
            "original_collection_rows_retained": True, "physical_experiment_budgets_reduced": False,
            "crystalgym_claim": "Within-prototype held-out trajectory calibration; not structural generalization.",
        },
        "all_declared_files_read": True, "sample_cap": None,
        "test_used_for_fit": False, "test_used_for_threshold_selection": False,
        "task_thresholds_modified": False, "null_labels_treated_as_negative": False,
    }
    provenance["dataset_fingerprint"] = fingerprint(provenance)
    return RiskDataset(records, provenance)


def prepare_risk_data(dataset: RiskDataset, method: str) -> dict:
    """Train-only feature schema; preserve labelled rows with partially missing features."""
    import numpy as np
    if method not in RISK_METHODS:
        raise RiskDataError(f"Unsupported declared risk method: {method}")
    field = dataset.provenance["label_field"]
    rows = {split: [r for r in dataset.records if r["split"] == split and r[field] is not None and r["_excluded_from_fit_reason"] is None] for split in ("train", "dev")}
    if dataset.provenance["label_counts"]["dev"]["groups_without_fit_rows"]:
        raise RiskDataError("Development groups have no supervised rows after exact-prefix filtering/null-label accounting: " + ", ".join(dataset.provenance["label_counts"]["dev"]["groups_without_fit_rows"]))
    for split, subset in rows.items():
        if not subset:
            raise RiskDataError(f"No observed {dataset.provenance['label_kind']} labels in {split}")
        if {r[field] for r in subset} != {0, 1}:
            raise RiskDataError(f"{split} has only one observed class; a calibrated risk fit requires both classes")
    names = sorted({name for r in rows["train"] for name in r["features"] if name.startswith(FEATURE_PREFIXES[method])})
    required = ["sampling."]
    if method != "entropy_risk":
        required.append("hidden.")
    if method == "graph_risk":
        for split, subset in rows.items():
            if not any(r["_graph_available"] for r in subset):
                raise RiskDataError(f"Graph features are completely missing in labelled {split}; no graph model is fitted")
        required.append("graph.")
    for prefix in required:
        substantive = [name for name in names if name.startswith(prefix) and name != "graph.graph_missing"]
        for split, subset in rows.items():
            if not any(r["features"].get(name) is not None for r in subset for name in substantive):
                raise RiskDataError(f"No observed {prefix} features from the training schema in {split}")
    arrays = {}
    for split, subset in rows.items():
        arrays[f"{split}_x"] = np.asarray([[np.nan if r["features"].get(name) is None else r["features"][name] for name in names] for r in subset], dtype=np.float64)
        arrays[f"{split}_y"] = np.asarray([r[field] for r in subset], dtype=np.int64)
        arrays[f"{split}_groups"] = [r["group_id"] for r in subset]
        arrays[f"{split}_episode_ids"] = [_episode_identity(r) for r in subset]
    dev_only = sorted({name for r in rows["dev"] for name in r["features"] if name.startswith(FEATURE_PREFIXES[method])} - set(names))
    return {**arrays, "feature_names": names, "development_only_features_not_used": dev_only,
            "feature_schema_fitted_on": "labelled_train_records_only"}


def _verify_sources(dataset: RiskDataset) -> None:
    for item in dataset.provenance["source_files"]:
        if file_sha256(item["path"]) != item["sha256"]:
            raise RiskDataError(f"Input changed after training began: {item['path']}")


def _save_checkpoint(model, path: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".pt", dir=path.parent)
    os.close(fd)
    try:
        model.save(temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fit_risk_models(
    manifest_paths: Sequence[str | Path], output_dir: str | Path, *, model_key: str,
    label_kind: str = "future_failure", graph_paths: Sequence[str | Path] = (),
    config=None, hyperparameters: Mapping | None = None,
    expected_checkpoint_hash: str | None = None,
) -> dict:
    """Fit all three arms; write per-method failure reports instead of fake models.

    Returns a report with status succeeded/partially_failed/failed. Checkpoints
    exist only for successfully fitted methods. The CLI exits nonzero unless all
    three succeed. Existing output reports/checkpoints are not overwritten.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ["fit_report.json", "input_provenance.json"] + [f"{m}{suffix}" for m in RISK_METHODS for suffix in (".pt", ".schema.json", ".fit.json")]
    if any((output / name).exists() for name in names):
        raise RiskDataError("Risk output already exists; use a new output directory to preserve fit provenance")
    report = {"schema_version": 1, "model_key": model_key, "label_kind": label_kind,
              "classification": "train_development_risk_fit_not_test_results", "methods": {},
              "test_used_for_fit": False, "scientific_test_results_generated": False}
    try:
        dataset = load_risk_dataset(manifest_paths, model_key=model_key, label_kind=label_kind,
                                    graph_paths=graph_paths, expected_checkpoint_hash=expected_checkpoint_hash)
    except (RiskDataError, OSError, ValueError) as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json_atomic(output / "fit_report.json", report)
        return report
    write_json_atomic(output / "input_provenance.json", dataset.provenance)
    report.update(dataset_fingerprint=dataset.provenance["dataset_fingerprint"],
                  label_counts=dataset.provenance["label_counts"],
                  graph_coverage=dataset.provenance["graph_coverage_fit_rows"],
                  development_exact_prefix_filter=dataset.provenance["development_exact_prefix_filter"])
    for method in RISK_METHODS:
        result = {
            "method": method, "label_kind": label_kind, "status": "failed",
            "dataset_fingerprint": dataset.provenance["dataset_fingerprint"],
            "input_provenance_path": str(output / "input_provenance.json"),
            "label_counts": dataset.provenance["label_counts"],
            "graph_coverage": dataset.provenance["graph_coverage_fit_rows"],
        }
        try:
            data = prepare_risk_data(dataset, method)
            from .uncertainty import CalibratedRiskModel, RiskTrainingConfig, risk_metrics
            if config is not None and hyperparameters:
                raise RiskDataError("Pass a RiskTrainingConfig or hyperparameters, not both")
            training_config = config or RiskTrainingConfig(label_kind=label_kind, **dict(hyperparameters or {}))
            if training_config.label_kind != label_kind:
                raise RiskDataError("RiskTrainingConfig and requested label_kind differ")
            if not training_config.include_error_similarity:
                raise RiskDataError("The declared risk arms include train-only error-prototype similarity")
            if (training_config.epochs < 1 or training_config.batch_size < 1 or training_config.patience < 1
                    or training_config.hidden_width < 1 or training_config.learning_rate <= 0
                    or training_config.weight_decay < 0):
                raise RiskDataError("Training hyperparameters must specify positive dimensions/epochs/rate")
            model = CalibratedRiskModel().fit(
                data["train_x"], data["train_y"], data["dev_x"], data["dev_y"],
                feature_names=data["feature_names"], train_groups=data["train_groups"],
                dev_groups=data["dev_groups"], train_episode_ids=data["train_episode_ids"],
                config=training_config,
            )
            probabilities = model.predict_proba(data["dev_x"], data["feature_names"])
            development_metrics = risk_metrics(data["dev_y"], probabilities)
            schema = {
                "schema_version": 1, "method": method, "feature_names": data["feature_names"],
                "allowed_prefixes": list(FEATURE_PREFIXES[method]),
                "feature_schema_fitted_on": data["feature_schema_fitted_on"],
                "development_only_features_not_used": data["development_only_features_not_used"],
                "missing_feature_encoding": "train-only median imputation and explicit missing indicators",
                "error_similarity": "TrainOnlyFeatures training-error prototype; never development/test prototypes",
                "preprocessing_state": model.features.state(), "training_config": asdict(training_config),
            }
            schema["feature_schema_fingerprint"] = fingerprint(schema)
            model.provenance.update({
                "collection_provenance": dataset.provenance,
                "method": method, "feature_schema_fingerprint": schema["feature_schema_fingerprint"],
                "feature_schema": schema, "development_metrics": development_metrics,
                "development_metric_unit": "labelled decision; correlated within episode, not independent test samples",
                "training_episode_weighting": "inverse labelled decisions per benchmark/model/group/episode",
                "source_code_sha256": {"fit_risk.py": file_sha256(__file__),
                                       "uncertainty.py": file_sha256(Path(__file__).with_name("uncertainty.py"))},
                "python_version": sys.version,
            })
            _verify_sources(dataset)
            checkpoint = output / f"{method}.pt"
            _save_checkpoint(model, checkpoint)
            write_json_atomic(output / f"{method}.schema.json", schema)
            result.update(
                status="succeeded", checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                training_config=asdict(training_config), feature_schema_fingerprint=schema["feature_schema_fingerprint"],
                feature_count=len(data["feature_names"]), selected_epoch=model.selected_epoch,
                history=model.history, temperature=model.temperature, development_metrics=development_metrics,
                train_rows=len(data["train_y"]), development_rows=len(data["dev_y"]),
                provenance=model.provenance,
            )
        except (RiskDataError, ValueError, RuntimeError, OSError, TypeError, ImportError) as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
        write_json_atomic(output / f"{method}.fit.json", result)
        report["methods"][method] = result
    succeeded = sum(item["status"] == "succeeded" for item in report["methods"].values())
    report["status"] = "succeeded" if succeeded == len(RISK_METHODS) else "partially_failed" if succeeded else "failed"
    write_json_atomic(output / "fit_report.json", report)
    return report
