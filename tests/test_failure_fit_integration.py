"""CPU integration fixtures only; no scientific collection, LLM, GPU or oracle runs."""
import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.failure_fit import prepare_failure_data
from matdiscovery.failure_labels import HEADS, HORIZONS, FailureLabelError, derive_failure_labels, row_identity_hash, tool_failure_labels
from matdiscovery.failure_risk import FailureAwareRiskModel, FailureRiskTrainingConfig
from matdiscovery.fit_risk import RiskDataError, prepare_risk_data
from matdiscovery.rollouts import DiscoveryRollout
from matdiscovery.rpc import EnvironmentClient
from matdiscovery.training_jobs import load_frozen_controllers
from matdiscovery.uncertainty import CalibratedRiskModel, RiskTrainingConfig
from test_failure_labels import fixture as raw_label_fixture, write_rows
from test_fit_risk import collection as raw_risk_fixture, load as load_dataset, write_jsonl


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def prepared(tmp_path):
    files = raw_risk_fixture.__wrapped__(tmp_path)
    for split, info in files.items():
        for index, row in enumerate(info["rows"]):
            row.update(local_decision_id=f"d{index:07d}-c0", episode_index=0)
        if split == "dev":
            info["rows"][0]["prefix_hash"] = files["train"]["rows"][0]["prefix_hash"]
            info["graph_rows"][0]["prefix_hash"] = info["rows"][0]["prefix_hash"]
            write_jsonl(info["graphs"], info["graph_rows"])
        write_jsonl(info["decisions"], info["rows"])
    dataset = load_dataset(files)
    labelled = {}
    for row in dataset.records:
        index = int(row["decision_id"].rsplit("-", 1)[1])
        labels = dict.fromkeys(HEADS)
        labels["generation_invalid"] = 1 if row["label_future_failure"] is None else index % 2
        if row["label_future_failure"] is not None:
            labels["tool_execution_failure"] = int(index % 3 == 0)
        labelled[row["decision_id"]] = {
            "decision_id": row["decision_id"], "row_identity_hash": row_identity_hash(row), "labels": labels,
            # Deliberately tempting posterior content must not enter train_x.
            "features": {"sampling.entropy": 1e12}, "evidence": {"property_value": 1e15},
            "classification": "synthetic_unit_labels_only",
        }
    names = prepare_risk_data(dataset, "entropy_risk")["feature_names"]
    return dataset, labelled, names


def test_prepare_uses_predecision_features_and_keeps_known_heads_of_future_null_proposals(prepared):
    dataset, labelled, names = prepared
    arrays, counts = prepare_failure_data(dataset, labelled, names)
    assert names == ["sampling.entropy"]
    assert arrays["train_x"].shape == (7, 1)
    np.testing.assert_allclose(arrays["train_x"][:, 0], np.arange(7) / 10)
    assert counts["train"]["without_overall_future_label"] == 1
    assert arrays["train_y"][-1, HEADS.index("generation_invalid")] == 1
    assert np.isnan(arrays["train_y"][-1, 1:]).all()
    assert arrays["dev_x"].shape == (3, 1)
    assert counts["dev"]["excluded_reasons"] == {"duplicate_dev_prefix_in_train": 1}
    assert len(dataset.records) == 11  # Neither raw train nor raw dev was removed.
    changed = copy.deepcopy(labelled)
    for label in changed.values():
        label["features"]["sampling.entropy"] = -1e20
        label["evidence"]["future_failure"] = True
    again, _ = prepare_failure_data(dataset, changed, names)
    np.testing.assert_array_equal(arrays["train_x"], again["train_x"])


@pytest.mark.parametrize("mutation", ["identity", "missing", "extra", "duplicate_dataset"])
def test_prepare_rejects_mismatched_or_duplicate_proposal_identity(prepared, mutation):
    dataset, labelled, names = copy.deepcopy(prepared)
    identifier = dataset.records[0]["decision_id"]
    if mutation == "identity": labelled[identifier]["row_identity_hash"] = "0" * 64
    elif mutation == "missing": del labelled[identifier]
    elif mutation == "extra": labelled["not-a-proposal"] = copy.deepcopy(labelled[identifier])
    else: dataset.records.append(copy.deepcopy(dataset.records[0]))
    with pytest.raises(RiskDataError):
        prepare_failure_data(dataset, labelled, names)


def test_full_fit_join_rejects_duplicated_sidecar_rows_before_training(prepared, tmp_path, monkeypatch):
    """The 210 entries are mock identities, never complete scientific jobs."""
    import matdiscovery.failure_fit as module
    dataset, labelled, _ = prepared
    output = tmp_path / "unit-fit"
    output.mkdir()
    write_json_atomic(output / "fit_report.json", {"status": "succeeded", "model_key": "synthetic_model",
        "dataset_fingerprint": dataset.provenance["dataset_fingerprint"], "classification": "unit_test_only"})
    paths = [tmp_path / f"unit-job-{i}" / "collection_manifest.json" for i in range(210)]
    monkeypatch.setattr(module, "load_risk_dataset", lambda *args, **kwargs: dataset)
    called = []
    def duplicate_sidecar(manifest, *, output_path):
        called.append(manifest)
        body = {"model_key": "synthetic_model", "records": [next(iter(labelled.values()))],
                "source_manifest": {"path": str(manifest), "sha256": "0" * 64},
                "sidecar_fingerprint": "unit-only", "classification": "unit_test_only"}
        write_json_atomic(output_path, body)
        return body
    monkeypatch.setattr(module, "derive_failure_labels", duplicate_sidecar)
    with pytest.raises(RiskDataError, match="Duplicated failure-labelled"):
        module.fit_failure_models(paths, output, tmp_path / "unit-labels", model_key="synthetic_model")
    assert len(called) == 2
    assert not (output / "failure_label_inventory.json").exists()
    assert not list(output.glob("*.failure_heads.pt"))


@pytest.fixture
def saved_controllers(prepared, tmp_path):
    dataset, labelled, names = prepared
    primary_data = prepare_risk_data(dataset, "entropy_risk")
    primary = CalibratedRiskModel().fit(primary_data["train_x"], primary_data["train_y"],
        primary_data["dev_x"], primary_data["dev_y"], feature_names=names,
        train_groups=primary_data["train_groups"], dev_groups=primary_data["dev_groups"],
        train_episode_ids=primary_data["train_episode_ids"],
        config=RiskTrainingConfig(label_kind="future_failure", hidden_width=4, epochs=2, batch_size=4, patience=2, seed=37))
    source = dataset.provenance
    schema = {"feature_names": names, "classification": "unit_test_only"}
    primary.provenance.update(method="entropy_risk", collection_provenance=source, feature_schema=schema)
    checkpoint = tmp_path / "entropy_risk.pt"
    primary.save(checkpoint)
    write_json_atomic(checkpoint.with_suffix(".schema.json"), schema)
    write_json_atomic(checkpoint.with_suffix(".fit.json"), {"method": "entropy_risk", "status": "succeeded",
        "checkpoint_sha256": file_sha256(checkpoint), "classification": "unit_test_only"})
    # All 210 unique files are tiny integrity fixtures. They are not asserted to
    # contain physical observations; this tests the controller loading boundary.
    label_files = []
    for index in range(210):
        path = tmp_path / "unit-label-inventory" / f"sidecar_{index:03d}.json"
        write_json_atomic(path, {"classification": "unit_test_only", "index": index})
        label_files.append({"path": str(path), "sha256": file_sha256(path)})
    inventory = {"schema": "complete_MADE_failure_label_inventory_v1", "classification": "unit_test_only",
        "sources": label_files, "heads": list(HEADS), "expected_collection_jobs": 210,
        "model_key": source["model_key"], "dataset_fingerprint": source["dataset_fingerprint"],
        "labelled_proposals": len(labelled), "test_used_for_fit": False}
    inventory["fingerprint"] = fingerprint(inventory)
    inventory_path = tmp_path / "failure_label_inventory.json"
    write_json_atomic(inventory_path, inventory)
    arrays, counts = prepare_failure_data(dataset, labelled, names)
    typed = FailureAwareRiskModel.fit(primary, arrays["train_x"], arrays["train_y"], arrays["dev_x"], arrays["dev_y"],
        feature_names=names, train_groups=arrays["train_groups"], dev_groups=arrays["dev_groups"],
        train_episode_ids=arrays["train_episodes"], dev_episode_ids=arrays["dev_episodes"],
        train_splits=arrays["train_splits"], dev_splits=arrays["dev_splits"],
        primary_checkpoint=checkpoint, expected_primary_sha256=file_sha256(checkpoint),
        expected_policy_runtime=source["policy_runtime"], provenance=source,
        config=FailureRiskTrainingConfig(epochs=2, batch_size=4, patience=2, seed=39))
    typed.require_control_ready()
    typed.failure_provenance["label_inventory"] = inventory
    typed_path = tmp_path / "entropy_risk.failure_heads.pt"
    receipt = typed.save(typed_path)
    receipt.update(status="succeeded", method="entropy_risk", heads=list(HEADS), source_provenance=source,
        label_inventory={"path": str(inventory_path), "sha256": file_sha256(inventory_path)}, classification="unit_test_only")
    receipt_path = tmp_path / "entropy_risk.failure_heads.fit.json"
    write_json_atomic(receipt_path, receipt)
    policy = SimpleNamespace(checkpoint_hash=source["checkpoint_hash"],
        configuration_fingerprint=source["policy_configuration_fingerprint"],
        runtime_precision_record=lambda: copy.deepcopy(source["policy_runtime"]))
    return SimpleNamespace(primary=primary, typed=typed, policy=policy, source=source, checkpoint=checkpoint,
        typed_path=typed_path, receipt=receipt, receipt_path=receipt_path, inventory=inventory,
        inventory_path=inventory_path, names=names, arrays=arrays, counts=counts, label_files=label_files)


def load_controllers(bundle):
    return load_frozen_controllers(bundle.policy, model_key=bundle.source["model_key"], benchmark="made",
        method="entropy_risk", risk_checkpoint=bundle.checkpoint, transcoder_manifest=None,
        graph_config={}, device="cpu", failure_control={"enabled_benchmarks": ["made"]})


def test_actual_cpu_primary_and_typed_models_roundtrip_through_controller_loader(saved_controllers):
    b = saved_controllers
    risk, graph, files = load_controllers(b)
    assert isinstance(risk, FailureAwareRiskModel) and graph is None
    assert risk.control_ready and risk.head_availability["tool_execution_failure"]
    assert risk.failure_provenance["training_rows"] == 7 > risk.provenance["train_rows"]
    np.testing.assert_array_equal(risk.predict_proba(b.arrays["dev_x"], b.names), b.primary.predict_proba(b.arrays["dev_x"], b.names))
    expected = b.typed.predict_failure_types(b.arrays["dev_x"], b.names)
    actual = risk.predict_failure_types(b.arrays["dev_x"], b.names)
    for head in HEADS:
        if expected[head] is None: assert actual[head] is None
        else: np.testing.assert_array_equal(actual[head], expected[head])
    assert all(Path(item["path"]) in files for item in b.label_files)
    assert all(not p.requires_grad and p.device.type == "cpu" for p in risk.typed_model.parameters())


@pytest.mark.parametrize("mutation", ["primary_binding", "runtime", "typed_hash", "inventory_hash",
    "inventory_semantics", "duplicate_sidecar", "sidecar_bytes"])
def test_loader_rejects_wrong_primary_runtime_or_label_inventory(saved_controllers, mutation):
    b = saved_controllers
    if mutation == "runtime":
        b.policy.runtime_precision_record = lambda: {"dtype": "torch.bfloat16", "classification": "unit_test_only"}
    elif mutation == "primary_binding": b.receipt["primary_checkpoint_sha256"] = "0" * 64
    elif mutation == "typed_hash": b.receipt["checkpoint_sha256"] = "0" * 64
    elif mutation == "inventory_hash": b.receipt["label_inventory"]["sha256"] = "0" * 64
    elif mutation == "sidecar_bytes": Path(b.label_files[0]["path"]).write_text("changed unit-only evidence")
    else:
        if mutation == "duplicate_sidecar": b.inventory["sources"][1] = b.inventory["sources"][0]
        else: b.inventory["labelled_proposals"] += 1
        b.inventory["fingerprint"] = fingerprint({k: v for k, v in b.inventory.items() if k != "fingerprint"})
        write_json_atomic(b.inventory_path, b.inventory)
        b.receipt["label_inventory"]["sha256"] = file_sha256(b.inventory_path)
    write_json_atomic(b.receipt_path, b.receipt)
    with pytest.raises(ValueError): load_controllers(b)


@pytest.mark.parametrize("success", [True, False])
def test_real_environment_client_check_false_preserves_scientific_response_id(tmp_path, success):
    """Exercise real request() with a synthetic pipe; never start env_server."""
    response = {"id": 0, "ok": success, **({"result": {"official_observation": {"is_stable": True}}} if success else {"error": {"code": "unit-only"}})}
    read_fd, write_fd = os.pipe()
    os.write(write_fd, (json.dumps(response) + "\n").encode()); os.close(write_fd)
    client = EnvironmentClient.__new__(EnvironmentClient)
    client.closed, client.sequence, client.timeout = False, 0, 1
    with os.fdopen(read_fd) as stream, (tmp_path / "unit-rpc.jsonl").open("a+") as audit:
        client.process = SimpleNamespace(stdin=io.StringIO(), stdout=stream)
        client.audit = audit
        actual = client.request("step", check=False)
        assert actual == response and actual["id"] == 0


@pytest.mark.parametrize("corrupt_field", ["observed_failure_types", "failure_type_observation_rpc_ids"])
def test_online_observed_labels_and_evidence_match_independent_sidecar_derivation(tmp_path, corrupt_field):
    data = raw_label_fixture(tmp_path, failed_first=True)
    # Also cover an executed generator whose acceptance outcome is unknown:
    # the tool exception is observed, not a fictitious zero-candidate result.
    data["rpc"][3]["payload"] = {"id": 1, "ok": False, "error": {"code": "unit-generator-exception"}}
    write_rows(data["directory"] / "rpc/rpc.jsonl", data["rpc"])
    expected = derive_failure_labels(data["manifest_path"])
    records = copy.deepcopy(data["rows"])
    indexed = {row["decision_id"]: row for row in records}
    by_sequence = {}
    for row in records:
        sequence = int(row["local_decision_id"].split("-")[0][1:])
        by_sequence.setdefault(sequence, []).append(row)
        row["observed_failure_types"] = dict.fromkeys(HEADS)
        row["observed_failure_types"]["generation_invalid"] = int(not row["generation"]["success"])
        row["failure_type_observation_rpc_ids"] = {head: [] for head in HEADS}
        row["failure_type_horizons"] = dict(HORIZONS)
    responses = {row["payload"]["id"]: row["payload"] for row in data["rpc"] if row["direction"] == "response"}
    output = tmp_path / "unit-online-labels"; output.mkdir()
    waiting = []
    for aligned in expected["execution_alignment"]:
        if aligned["decision_id"] is not None:
            waiting.extend(by_sequence[aligned["sequence"]])
            row = indexed[aligned["decision_id"]]
            response = responses[aligned["tool_rpc_id"]]
            measured = tool_failure_labels(row["generation"]["parsed_action"], response)
            row["observed_failure_types"].update(measured)
            for head, value in measured.items():
                if value is not None or (head == "candidate_generation_failure" and row["generation"]["parsed_action"]["tool"] in {"generate_structures", "create_structure"}):
                    row["failure_type_observation_rpc_ids"][head] = [response["id"]]
        if "scientific_rpc_id" in aligned:
            response = responses[aligned["scientific_rpc_id"]]
            DiscoveryRollout._label_rows(output, waiting, int(not response["ok"]), scientific_response=response)
    actual = {row["decision_id"]: row for row in [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]}
    for derived in expected["records"]:
        online = actual[derived["decision_id"]]
        assert online["observed_failure_types"] == derived["labels"]
        assert online["failure_type_observation_rpc_ids"] == derived["source_rpc_ids"]
    # New collectors carry the online fields. The independent sidecar must now
    # verify them, while the original no-fields fixture above remains valid.
    write_rows(data["directory"] / "decisions.jsonl", [actual[row["decision_id"]] for row in data["rows"]])
    data["manifest"]["decision_files"][0]["sha256"] = file_sha256(data["directory"] / "decisions.jsonl")
    write_json_atomic(data["manifest_path"], data["manifest"])
    verified = derive_failure_labels(data["manifest_path"])
    assert verified["online_observation_records_checked"] == {"observed_failure_types": len(records), "failure_type_observation_rpc_ids": len(records)}
    first = actual[data["rows"][0]["decision_id"]]
    # A rejected proposal has neither an observed scientific label nor its RPC.
    first[corrupt_field]["unstable"] = 0 if corrupt_field == "observed_failure_types" else [1]
    write_rows(data["directory"] / "decisions.jsonl", [actual[row["decision_id"]] for row in data["rows"]])
    data["manifest"]["decision_files"][0]["sha256"] = file_sha256(data["directory"] / "decisions.jsonl")
    write_json_atomic(data["manifest_path"], data["manifest"])
    with pytest.raises(FailureLabelError, match="Online " + corrupt_field):
        derive_failure_labels(data["manifest_path"])
