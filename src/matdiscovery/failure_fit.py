"""Fit observed failure-type heads from complete, unchanged MADE collections."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .accounting import file_sha256, fingerprint, write_json_atomic
from .failure_labels import HEADS, derive_failure_labels, load_failure_labels, row_identity_hash
from .failure_risk import FAILURE_TYPES, FailureAwareRiskModel, FailureRiskTrainingConfig
from .fit_risk import RISK_METHODS, RiskDataError, _episode_identity, _verify_sources, load_risk_dataset
from .uncertainty import CalibratedRiskModel, risk_metrics


def prepare_failure_data(dataset, labelled, feature_names):
    """Use every eligible proposal, including those without a future label.

    Post-action evidence supplies targets only. Prediction features come solely
    from the previously verified, pre-action feature records and primary schema.
    """
    if tuple(HEADS) != tuple(FAILURE_TYPES):
        raise RiskDataError("Failure label and predictor schemas differ")
    identities = [row["decision_id"] for row in dataset.records]
    if len(identities) != len(set(identities)) or set(labelled) != set(identities):
        raise RiskDataError("Typed labels must cover every declared proposal exactly once")
    names = list(feature_names)
    arrays, counts = {}, {}
    for split in ("train", "dev"):
        eligible = []
        excluded = Counter()
        for row in dataset.records:
            if row["split"] != split:
                continue
            label = labelled[row["decision_id"]]
            if label["row_identity_hash"] != row_identity_hash(row):
                raise RiskDataError("Typed label belongs to another decision/prefix/policy")
            if set(label["labels"]) != set(HEADS):
                raise RiskDataError("Incomplete failure-type label schema")
            if row["_excluded_from_fit_reason"] is not None:
                excluded[row["_excluded_from_fit_reason"]] += 1
                continue
            eligible.append(row)
        if not eligible:
            raise RiskDataError("No eligible typed-risk proposals in " + split)
        arrays[split + "_x"] = np.asarray([
            [np.nan if row["features"].get(name) is None else row["features"][name] for name in names]
            for row in eligible], dtype=np.float64)
        arrays[split + "_y"] = np.asarray([
            [np.nan if labelled[row["decision_id"]]["labels"][head] is None
             else labelled[row["decision_id"]]["labels"][head] for head in HEADS]
            for row in eligible], dtype=np.float64)
        arrays[split + "_groups"] = [row["group_id"] for row in eligible]
        arrays[split + "_episodes"] = [_episode_identity(row) for row in eligible]
        arrays[split + "_splits"] = [row["split"] for row in eligible]
        counts[split] = {"included_proposals": len(eligible), "excluded_reasons": dict(excluded),
                         "without_overall_future_label": sum(row["label_future_failure"] is None for row in eligible)}
    return arrays, counts


def fit_failure_models(manifest_paths, output_dir, label_output, *, model_key, hyperparameters=None, corpus_contract=None):
    """Bind three typed predictors to their already fitted overall predictors.

    Sidecars are stored outside immutable collection directories. Every source
    and row is checked; missing type outcomes remain masked, not negative.
    """
    output, labels_root = Path(output_dir).resolve(), Path(label_output).resolve()
    paths = [Path(path).resolve() for path in manifest_paths]
    expected_count = 210
    if corpus_contract is not None:
        from .core_protocol import validate_corpus_contract
        validate_corpus_contract(corpus_contract, manifest_paths=paths, model_key=model_key)
        expected_count = corpus_contract["expected_collection_jobs"]
    if len(paths) != expected_count or len(set(paths)) != expected_count:
        raise RiskDataError("MADE typed fitting requires the complete registered train/development collection matrix (default 210)")
    report_path = output / "failure_fit_report.json"
    if report_path.exists() or any(output.glob("*.failure_heads.*")):
        raise RiskDataError("Typed fit evidence already exists; do not overwrite or silently retry it")
    dataset = load_risk_dataset(paths, model_key=model_key)
    if dataset.provenance["benchmarks"] != ["made"]:
        raise RiskDataError("This failure taxonomy is validated for MADE only")
    primary_report = json.loads((output / "fit_report.json").read_text())
    if (primary_report["status"] != "succeeded" or primary_report["model_key"] != model_key
            or primary_report["dataset_fingerprint"] != dataset.provenance["dataset_fingerprint"]):
        raise RiskDataError("Typed fitting needs all three verified primary predictors from this same corpus")
    config = FailureRiskTrainingConfig(**dict(hyperparameters or {}))
    labels_root.mkdir(parents=True, exist_ok=True)
    labelled, sources = {}, []
    for path in paths:
        sidecar_path = labels_root / (path.parent.name + ".json")
        sidecar = (load_failure_labels(sidecar_path, manifest_path=path) if sidecar_path.exists()
                   else derive_failure_labels(path, output_path=sidecar_path))
        if sidecar["model_key"] != model_key:
            raise RiskDataError("Failure label source model differs")
        sources.append({"path": str(sidecar_path), "sha256": file_sha256(sidecar_path),
                        "source_manifest": sidecar["source_manifest"],
                        "sidecar_fingerprint": sidecar["sidecar_fingerprint"]})
        for row in sidecar["records"]:
            if row["decision_id"] in labelled:
                raise RiskDataError("Duplicated failure-labelled decision")
            labelled[row["decision_id"]] = row
    evidence = {"schema": "complete_MADE_failure_label_inventory_v1", "sources": sources,
                "heads": list(HEADS), "expected_collection_jobs": expected_count,
                "model_key": model_key, "dataset_fingerprint": dataset.provenance["dataset_fingerprint"],
                "labelled_proposals": len(labelled), "test_used_for_fit": False,
                "targets_never_added_to_prediction_features": True}
    if corpus_contract is not None:
        evidence["corpus_contract"] = corpus_contract
    evidence["fingerprint"] = fingerprint(evidence)
    write_json_atomic(output / "failure_label_inventory.json", evidence)
    report = {"schema": "failure_type_fit_report_v1", "model_key": model_key, "benchmark": "made",
              "heads": list(HEADS), "classification": "train_dev_fit_not_test_results",
              "label_inventory": {"path": str(output / "failure_label_inventory.json"),
                                  "sha256": file_sha256(output / "failure_label_inventory.json")},
              "methods": {}, "test_used_for_fit": False, "scientific_test_results_generated": False}
    for method in RISK_METHODS:
        checkpoint = output / (method + ".pt")
        primary_receipt = json.loads((output / (method + ".fit.json")).read_text())
        result = {"method": method, "status": "failed", "heads": list(HEADS)}
        try:
            if primary_receipt["status"] != "succeeded" or primary_receipt["checkpoint_sha256"] != file_sha256(checkpoint):
                raise RiskDataError("Primary checkpoint lacks its successful matching fit receipt")
            primary = CalibratedRiskModel.load(checkpoint)
            if primary.provenance["collection_provenance"] != dataset.provenance:
                raise RiskDataError("Primary and typed predictor source data differ")
            arrays, counts = prepare_failure_data(dataset, labelled, primary.features.names)
            typed = FailureAwareRiskModel.fit(primary, arrays["train_x"], arrays["train_y"],
                arrays["dev_x"], arrays["dev_y"], feature_names=primary.features.names,
                train_groups=arrays["train_groups"], dev_groups=arrays["dev_groups"],
                train_episode_ids=arrays["train_episodes"], dev_episode_ids=arrays["dev_episodes"],
                train_splits=arrays["train_splits"], dev_splits=arrays["dev_splits"],
                primary_checkpoint=checkpoint, expected_primary_sha256=file_sha256(checkpoint),
                expected_policy_runtime=dataset.provenance["policy_runtime"],
                provenance=dataset.provenance, config=config)
            typed.require_control_ready()
            typed.failure_provenance.update(label_inventory=evidence,
                source_fit_code_sha256=file_sha256(__file__),
                exact_prefix_exclusion=dataset.provenance["development_exact_prefix_filter"])
            predictions = typed.predict_failure_types(arrays["dev_x"], primary.features.names)
            development = {}
            for index, head in enumerate(HEADS):
                observed = np.isfinite(arrays["dev_y"][:, index])
                probability = predictions[head]
                development[head] = {"observed_rows": int(observed.sum()),
                    "unknown_rows": int((~observed).sum()), "available": probability is not None,
                    "descriptive_metrics": risk_metrics(arrays["dev_y"][observed, index], probability[observed])
                    if probability is not None and observed.any() else None}
            _verify_sources(dataset)
            if any(file_sha256(item["path"]) != item["sha256"] for item in sources):
                raise RiskDataError("Failure label sidecar changed during fitting")
            saved = typed.save(output / (method + ".failure_heads.pt"))
            result.update(saved, status="succeeded", feature_names=list(primary.features.names),
                source_provenance=dataset.provenance, label_inventory=report["label_inventory"],
                training_config=asdict(config), proposal_counts=counts, development_metrics=development,
                source_code_sha256=file_sha256(__file__))
        except (ValueError, RuntimeError, OSError, TypeError) as error:
            result.update(error_type=type(error).__name__, error=str(error))
        write_json_atomic(output / (method + ".failure_heads.fit.json"), result)
        report["methods"][method] = result
    report["status"] = "succeeded" if all(item["status"] == "succeeded" for item in report["methods"].values()) else "failed"
    write_json_atomic(report_path, report)
    return report
