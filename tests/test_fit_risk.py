"""Synthetic risk-data contract checks, never material-discovery experiment results."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from matdiscovery.accounting import file_sha256
from matdiscovery.fit_risk import (
    RiskDataError, fit_risk_models, load_risk_dataset, prepare_risk_data,
)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def collection(tmp_path):
    """Each split has its own initial-policy session and disjoint episode groups."""
    files, all_rows = {}, {}
    runtime = {"synthetic_fixture_only": True, "dtype": "float32", "device": "cpu"}
    configuration = {"policy_runtime": runtime, "synthetic_fixture_only": True}
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    for split, count in (("train", 7), ("dev", 4)):
        directory = tmp_path / split
        directory.mkdir()
        rows, graphs = [], []
        for index in range(count):
            decision_id = f"synthetic-{split}-{index}"
            ids = directory / f"{decision_id}.ids.json"
            ids.write_text(json.dumps({"test_fixture_only": True, "input_ids": [[1, index, 3]]}))
            stamp = {"model_id": "test/tiny", "checkpoint_hash": "a" * 64,
                     "state_id": f"initial-{split}-session", "generation": 0,
                     "perturbation_seed": None, "perturbation_sigma": None}
            row = {
                "decision_id": decision_id, "benchmark": "made", "model_key": "synthetic_model",
                "split": split, "task_id": f"synthetic-task-{index // 2}",
                "group_id": f"{split}-group-{index // 2}", "episode_id": "0",
                "features": {"sampling.entropy": index / 10, "hidden.layer_0_norm": 2 + index,
                             "action.token_count": 3}, "input_ids_file": ids.name,
                "input_ids_sha256": file_sha256(ids), "policy_stamp": stamp,
                "generation": {"policy_runtime": runtime, "configuration_fingerprint": config_hash},
                "prefix_hash": ("b" if split == "train" else "c") * 63 + str(index),
                "label_immediate_error": index % 2,
                "label_future_failure": None if split == "train" and index == 6 else index % 2,
                "label_future_failure_reason": "next_scientific_evaluation_not_observed" if index == 6 else None,
            }
            if split == "dev":
                row["features"]["sampling.development_only"] = 1e9
            rows.append(row)
            graphs.append({"decision_id": decision_id,
                           "features": {"graph.graph_missing": 0, "graph.node_count": 10 + index},
                           "graph_status": "succeeded", "policy_state_id": stamp["state_id"],
                           "policy_runtime": runtime, "source_policy_configuration_fingerprint": config_hash,
                           "prefix_hash": row["prefix_hash"]})
        decisions, graph_file = directory / "decisions.jsonl", directory / "graph_features.jsonl"
        write_jsonl(decisions, rows)
        write_jsonl(graph_file, graphs)
        manifest = directory / "collection.json"
        manifest.write_text(json.dumps({"schema_version": 2, "complete": True, "decision_files": [decisions.name],
            "policy_runtime": runtime, "policy_configuration": configuration, "policy_configuration_fingerprint": config_hash}))
        files[split] = {"manifest": manifest, "decisions": decisions, "graphs": graph_file, "rows": rows, "graph_rows": graphs}
        all_rows[split] = rows
    return files


def load(collection, **kwargs):
    return load_risk_dataset([collection[s]["manifest"] for s in ("train", "dev")], model_key="synthetic_model", **kwargs)


def test_full_read_null_label_accounting_and_initial_sessions(collection):
    dataset = load(collection)
    assert len(dataset.records) == 11
    assert dataset.provenance["sample_cap"] is None
    assert len(dataset.provenance["policy_stamps"]) == 2
    counts = dataset.provenance["label_counts"]
    assert counts["train"]["labelled_rows"] == 6
    assert counts["train"]["null_label_rows"] == 1
    assert counts["train"]["null_label_reasons"] == {"next_scientific_evaluation_not_observed": 1}
    assert dataset.provenance["graph_coverage_all_rows"]["train"]["available"] == 7
    assert dataset.provenance["graph_coverage_labelled_rows"]["train"]["available"] == 6
    assert dataset.provenance["test_used_for_fit"] is False


def test_train_only_ordered_feature_schemas_and_episode_ids(collection):
    dataset = load(collection)
    entropy = prepare_risk_data(dataset, "entropy_risk")
    hidden = prepare_risk_data(dataset, "hidden_risk")
    graph = prepare_risk_data(dataset, "graph_risk")
    assert entropy["feature_names"] == ["sampling.entropy"]
    assert hidden["feature_names"] == ["action.token_count", "hidden.layer_0_norm", "sampling.entropy"]
    assert "graph.node_count" in graph["feature_names"]
    assert entropy["train_x"].shape == (6, 1)
    assert entropy["dev_x"].shape == (4, 1)
    assert entropy["development_only_features_not_used"] == ["sampling.development_only"]
    assert len(set(entropy["train_episode_ids"])) == 3  # episode_id='0' is namespaced by group
    assert entropy["train_y"].tolist() == [0, 1, 0, 1, 0, 1]


@pytest.mark.parametrize("mutation,match", [
    ("test", "Test/undeclared"), ("group_overlap", "leakage"),
    ("checkpoint", "same initial checkpoint"), ("generation", "generation 0"),
    ("perturbation", "perturbed"), ("model_key", "no silent filtering"),
    ("outcome_feature", "outcome leakage"), ("nonfinite", "Nonfinite"),
])
def test_rejects_leakage_and_mixed_policy_states(collection, mutation, match):
    rows = copy.deepcopy(collection["dev"]["rows"])
    if mutation == "test":
        rows[0]["split"] = "test"
    elif mutation == "group_overlap":
        rows[0]["group_id"] = "train-group-0"
    elif mutation == "checkpoint":
        rows[0]["policy_stamp"]["checkpoint_hash"] = "d" * 64
    elif mutation == "generation":
        rows[0]["policy_stamp"]["generation"] = 1
    elif mutation == "perturbation":
        rows[0]["policy_stamp"]["perturbation_seed"] = 12
    elif mutation == "model_key":
        rows[0]["model_key"] = "different_model"
    elif mutation == "outcome_feature":
        rows[0]["features"]["reward.final"] = 1.0
    else:
        rows[0]["features"]["sampling.entropy"] = float("nan")
    write_jsonl(collection["dev"]["decisions"], rows)
    with pytest.raises(RiskDataError, match=match):
        load(collection)


@pytest.mark.parametrize("mutation,match", [("state", "source policy state"), ("prefix", "graph prefix"), ("weights", "checkpoint")])
def test_rejects_stale_graphs(collection, mutation, match):
    rows = copy.deepcopy(collection["train"]["graph_rows"])
    if mutation == "state":
        rows[0]["policy_state_id"] = "old-state"
    elif mutation == "prefix":
        rows[0]["prefix_hash"] = "old-prefix"
    else:
        rows[0]["checkpoint_hash"] = "d" * 64
    write_jsonl(collection["train"]["graphs"], rows)
    with pytest.raises(RiskDataError, match=match):
        load(collection)


def test_cross_session_graph_replay_requires_verified_initial_stamp(collection):
    rows = copy.deepcopy(collection["train"]["graph_rows"])
    row = rows[0]
    row["source_policy_state_id"] = row.pop("policy_state_id")
    row["replay_policy_state_id"] = "new-replay-session"
    row["checkpoint_hash"] = "a" * 64
    write_jsonl(collection["train"]["graphs"], rows)
    with pytest.raises(RiskDataError, match="Cross-session"):
        load(collection)
    row["replay_policy_stamp"] = {**collection["train"]["rows"][0]["policy_stamp"], "state_id": "new-replay-session"}
    write_jsonl(collection["train"]["graphs"], rows)
    assert load(collection).provenance["graph_coverage_labelled_rows"]["train"]["available"] == 6
    row["replay_policy_stamp"]["generation"] = 2
    write_jsonl(collection["train"]["graphs"], rows)
    with pytest.raises(RiskDataError, match="generation 0"):
        load(collection)


def test_source_state_tag_cannot_hide_an_unverified_graph_session(collection):
    rows = copy.deepcopy(collection["train"]["graph_rows"])
    rows[0]["source_policy_state_id"] = rows[0]["policy_state_id"]
    rows[0]["policy_state_id"] = "different-graph-producing-session"
    write_jsonl(collection["train"]["graphs"], rows)
    with pytest.raises(RiskDataError, match="Cross-session"):
        load(collection)


def test_missing_graphs_retain_rows_but_cannot_train_graph_arm(collection):
    for split in ("train", "dev"):
        collection[split]["graphs"].unlink()
    dataset = load(collection)
    assert prepare_risk_data(dataset, "hidden_risk")["train_x"].shape[0] == 6
    assert dataset.provenance["graph_coverage_labelled_rows"]["train"]["missing"] == 6
    with pytest.raises(RiskDataError, match="completely missing"):
        prepare_risk_data(dataset, "graph_risk")


def test_partial_graph_failure_is_missing_not_zero_proxy_or_dropped_row(collection):
    rows = copy.deepcopy(collection["train"]["graph_rows"])
    rows[0].update(graph_status="cuda_out_of_memory", features={"graph.node_count": 0, "graph.graph_missing": 1})
    write_jsonl(collection["train"]["graphs"], rows)
    dataset = load(collection)
    data = prepare_risk_data(dataset, "graph_risk")
    first = next(r for r in dataset.records if r["decision_id"] == rows[0]["decision_id"])
    assert first["features"]["graph.node_count"] is None
    assert first["features"]["graph.graph_missing"] == 1
    assert len(data["train_y"]) == 6
    assert np.isnan(data["train_x"][:, data["feature_names"].index("graph.node_count")]).sum() == 1


def test_single_class_failure_writes_no_checkpoint(collection, tmp_path):
    rows = copy.deepcopy(collection["train"]["rows"])
    for row in rows:
        row["label_future_failure"] = 0
    write_jsonl(collection["train"]["decisions"], rows)
    output = tmp_path / "failed-fit"
    report = fit_risk_models([collection[s]["manifest"] for s in ("train", "dev")], output, model_key="synthetic_model")
    assert report["status"] == "failed"
    assert all("one observed class" in item["error"] for item in report["methods"].values())
    assert not list(output.glob("*.pt"))
    assert (output / "input_provenance.json").exists()


def test_missing_token_evidence_and_declared_source_hashes_are_checked(collection):
    manifest = collection["train"]["manifest"]
    content = json.loads(manifest.read_text())
    manifest.write_text(json.dumps({**content, "decision_files": [{"path": "decisions.jsonl", "sha256": "e" * 64}]}))
    with pytest.raises(RiskDataError, match="hash mismatch"):
        load(collection)
    manifest.write_text(json.dumps(content))
    (collection["train"]["decisions"].parent / collection["train"]["rows"][0]["input_ids_file"]).unlink()
    with pytest.raises(RiskDataError, match="captured_input_ids"):
        load(collection)


def test_risk_dataset_rejects_mixed_runtime_even_for_same_checkpoint(collection):
    rows = copy.deepcopy(collection["dev"]["rows"])
    rows[0]["generation"]["policy_runtime"]["dtype"] = "bfloat16"
    write_jsonl(collection["dev"]["decisions"], rows)
    with pytest.raises(RiskDataError, match="runtime/precision"):
        load(collection)


def test_risk_dataset_rejects_graph_from_another_runtime(collection):
    graphs = copy.deepcopy(collection["train"]["graph_rows"])
    graphs[0]["policy_runtime"]["dtype"] = "bfloat16"
    write_jsonl(collection["train"]["graphs"], graphs)
    with pytest.raises(RiskDataError, match="Graph runtime/precision"):
        load(collection)


def test_runtime_provenance_reaches_risk_dataset(collection):
    data = load(collection)
    manifest = json.loads(collection["train"]["manifest"].read_text())
    assert data.provenance["policy_runtime"] == manifest["policy_runtime"]
    assert data.provenance["policy_configuration_fingerprint"] == manifest["policy_configuration_fingerprint"]


def test_duplicate_decisions_and_unmatched_graphs_are_not_silently_deduplicated(collection):
    rows = collection["train"]["rows"]
    write_jsonl(collection["train"]["decisions"], rows + [rows[0]])
    with pytest.raises(RiskDataError, match="Duplicate decision_id"):
        load(collection)
    write_jsonl(collection["train"]["decisions"], rows)
    graph = copy.deepcopy(collection["train"]["graph_rows"][0])
    graph["decision_id"] = "not-in-collections"
    write_jsonl(collection["train"]["graphs"], collection["train"]["graph_rows"] + [graph])
    with pytest.raises(RiskDataError, match="no matching"):
        load(collection)


def test_cli_rejects_test_records_and_exits_nonzero(collection, tmp_path):
    rows = copy.deepcopy(collection["dev"]["rows"])
    rows[0]["split"] = "test"
    write_jsonl(collection["dev"]["decisions"], rows)
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_risk_models.py"
    command = [sys.executable, str(script), "--manifest", str(collection["train"]["manifest"]),
               "--manifest", str(collection["dev"]["manifest"]), "--model-key", "synthetic_model",
               "--output", str(tmp_path / "cli-failure")]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "failed"
    report = json.loads((tmp_path / "cli-failure" / "fit_report.json").read_text())
    assert "Test/undeclared" in report["error"]


def test_real_existing_risk_model_fit_schema_provenance_and_determinism(collection, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    from matdiscovery.uncertainty import CalibratedRiskModel, RiskTrainingConfig
    config = RiskTrainingConfig(label_kind="future_failure", seed=31, epochs=3, hidden_width=4,
                                batch_size=4, patience=3)
    paths = [collection[s]["manifest"] for s in ("train", "dev")]
    reports = [fit_risk_models(paths, tmp_path / f"fit-{i}", model_key="synthetic_model", config=config) for i in range(2)]
    assert all(report["status"] == "succeeded" for report in reports)
    data = prepare_risk_data(load(collection), "graph_risk")
    models = [CalibratedRiskModel.load(report["methods"]["graph_risk"]["checkpoint"]) for report in reports]
    for model in models:
        assert model.provenance["collection_provenance"]["label_counts"]["train"]["null_label_rows"] == 1
        assert model.features.names == data["feature_names"]
        assert model.config.seed == 31
        assert model.provenance["test_used_for_fit"] is False
    np.testing.assert_array_equal(models[0].predict_proba(data["dev_x"], data["feature_names"]),
                                  models[1].predict_proba(data["dev_x"], data["feature_names"]))
    assert reports[0]["methods"]["graph_risk"]["history"] == reports[1]["methods"]["graph_risk"]["history"]


def test_exact_train_prefixes_are_removed_only_from_dev_fit_rows(collection):
    rows = copy.deepcopy(collection["dev"]["rows"])
    graphs = copy.deepcopy(collection["dev"]["graph_rows"])
    rows[0]["prefix_hash"] = collection["train"]["rows"][0]["prefix_hash"]
    graphs[0]["prefix_hash"] = rows[0]["prefix_hash"]
    write_jsonl(collection["dev"]["decisions"], rows)
    write_jsonl(collection["dev"]["graphs"], graphs)
    dataset = load(collection)
    data = prepare_risk_data(dataset, "entropy_risk")
    assert len(dataset.records) == 11  # Original collection is preserved.
    assert len(data["train_y"]) == 6
    assert len(data["dev_y"]) == 3
    assert dataset.provenance["label_counts"]["dev"]["labelled_rows"] == 4
    assert dataset.provenance["label_counts"]["dev"]["fit_rows"] == 3
    audit = dataset.provenance["development_exact_prefix_filter"]
    assert audit["duplicate_dev_prefix_rows"] == 1
    assert audit["retained_development_groups"] == ["dev-group-0", "dev-group-1"]
    assert audit["physical_experiment_budgets_reduced"] is False


def test_exact_prefix_filter_fails_when_a_dev_group_becomes_empty(collection):
    rows = copy.deepcopy(collection["dev"]["rows"])
    graphs = copy.deepcopy(collection["dev"]["graph_rows"])
    for index in (0, 1):
        rows[index]["prefix_hash"] = collection["train"]["rows"][index]["prefix_hash"]
        graphs[index]["prefix_hash"] = rows[index]["prefix_hash"]
    write_jsonl(collection["dev"]["decisions"], rows)
    write_jsonl(collection["dev"]["graphs"], graphs)
    dataset = load(collection)
    assert dataset.provenance["label_counts"]["dev"]["groups_without_fit_rows"] == ["dev-group-0"]
    with pytest.raises(RiskDataError, match="Development groups"):
        prepare_risk_data(dataset, "entropy_risk")
