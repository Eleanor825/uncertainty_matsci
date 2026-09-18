"""Synthetic representation-stage checks; these are not real-model research results."""

import json
from pathlib import Path

import pytest

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.representation_training import (
    BANK_SCHEMA, GRAPH_SCHEMA, GRAPH_TARGET_CONTRACT, QWEN_MLP_PATHS, GraphStageConfig, RepresentationError,
    TranscoderStageConfig, _bank_fingerprint, _load_resume, _replay_records,
    filter_development_shard, read_collections, train_transcoders_from_collections, validate_transcoder_bank,
    verify_graph_record,
)


def collection_files(tmp_path, *, tensor_tokens=False):
    result = []
    runtime = {"classification": "unit_test_only", "dtype": "float32", "device": "cpu"}
    configuration = {"classification": "unit_test_only", "policy_runtime": runtime}
    configuration_fingerprint = fingerprint(configuration)
    for split in ("train", "dev"):
        directory = tmp_path / split
        directory.mkdir()
        stamp = {"model_id": "Qwen/Qwen3.5-4B", "checkpoint_hash": "a" * 64,
                 "state_id": f"synthetic-{split}-initial-session", "generation": 0,
                 "perturbation_seed": None, "perturbation_sigma": None}
        tokens = directory / "synthetic-tokens.pt"
        if tensor_tokens:
            import torch
            from matdiscovery.esopt import tensor_state_hash
            ids = torch.tensor([[0, 1, 2]])
            torch.save({"input_ids": ids, "policy_stamp": stamp}, tokens)
            prefix = tensor_state_hash({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        else:
            tokens.write_text("synthetic fixture; never passed to a model")
            prefix = ("b" if split == "train" else "c") * 64
        row = {"decision_id": f"synthetic-{split}-decision", "benchmark": "made",
               "model_key": "qwen35_4b", "split": split, "task_id": split + "-task",
               "group_id": split + "-group", "episode_id": "0", "input_ids_file": tokens.name,
               "prefix_hash": prefix, "policy_stamp": stamp,
               "generation": {"policy_runtime": dict(runtime), "configuration_fingerprint": configuration_fingerprint},
               # These deliberately invalid label values must never be read by representation stages.
               "label_future_failure": {"forbidden_to_consume": True}, "label_immediate_error": "irrelevant"}
        decisions = directory / "decisions.jsonl"
        decisions.write_text(json.dumps(row) + "\n")
        manifest = directory / "collection_manifest.json"
        data = {"job_id": "synthetic-" + split, "complete": True,
                "policy_runtime": dict(runtime), "policy_configuration": configuration,
                "policy_configuration_fingerprint": configuration_fingerprint,
                "decision_files": [{"path": decisions.name, "sha256": file_sha256(decisions)}],
                "activation_shards": []}
        write_json_atomic(manifest, data)
        result.append({"manifest": manifest, "manifest_data": data, "decisions": decisions,
                       "record": row, "tokens": tokens, "stamp": stamp})
    return result


def update_decision(item, field, value):
    item["record"][field] = value
    item["decisions"].write_text(json.dumps(item["record"]) + "\n")
    item["manifest_data"]["decision_files"][0]["sha256"] = file_sha256(item["decisions"])
    write_json_atomic(item["manifest"], item["manifest_data"])


def test_all_collection_identities_are_read_without_outcome_labels(tmp_path):
    items = collection_files(tmp_path)
    result = read_collections([item["manifest"] for item in items], model_key="qwen35_4b")
    assert len(result["records"]) == 2
    assert result["future_labels_used"] is False
    assert result["sample_cap"] is None
    assert all("label_future_failure" not in row for row in result["records"])
    assert all(len(item["missing_activation_layers"]) == 32 for item in result["collections"])
    reversed_result = read_collections([item["manifest"] for item in reversed(items)], model_key="qwen35_4b")
    assert result == reversed_result


@pytest.mark.parametrize("mutation", ["test", "model", "group", "checksum", "incomplete"])
def test_representation_input_boundaries_are_strict(tmp_path, mutation):
    items = collection_files(tmp_path)
    if mutation == "test":
        update_decision(items[1], "split", "test")
    elif mutation == "model":
        update_decision(items[1], "model_key", "qwen35_9b")
    elif mutation == "group":
        update_decision(items[1], "group_id", "train-group")
    elif mutation == "checksum":
        items[1]["decisions"].write_text("modified")
    else:
        items[1]["manifest_data"]["complete"] = False
        write_json_atomic(items[1]["manifest"], items[1]["manifest_data"])
    with pytest.raises((RepresentationError, ValueError)):
        read_collections([item["manifest"] for item in items], model_key="qwen35_4b")


def graph_record(source, *, succeeded=True):
    replay = {**source["policy_stamp"], "state_id": "another-synthetic-initial-session"}
    record = {"schema": GRAPH_SCHEMA, "decision_id": source["decision_id"],
              "source_fingerprint": source["source_fingerprint"],
              "source_policy_state_id": source["policy_stamp"]["state_id"],
              "replay_policy_state_id": replay["state_id"], "replay_policy_stamp": replay,
              "checkpoint_hash": replay["checkpoint_hash"], "prefix_hash": source["prefix_hash"],
              "transcoder_bank_fingerprint": "synthetic-bank", "graph_pipeline_fingerprint": "synthetic-pipeline",
              "target_contract": GRAPH_TARGET_CONTRACT,
              "policy_runtime": source["policy_runtime"],
              "source_policy_configuration_fingerprint": source["source_policy_configuration_fingerprint"],
              "complete": True, "graph_status": "succeeded" if succeeded else "unavailable",
              "backend_validation_passed": succeeded,
              "features": {"graph.graph_missing": 0.0 if succeeded else 1.0},
              "artifacts": []}
    if succeeded:
        record.update(target_spec_hash="synthetic-target-spec", action_targets={
            "target_spec_hash": "synthetic-target-spec", "source_prefix_hash": source["prefix_hash"]})
    if not succeeded:
        record.update(error="synthetic fidelity failure", failure_kind="prefix_fidelity_failed")
    record["record_fingerprint"] = fingerprint(record)
    return record


@pytest.mark.parametrize("field,new", [
    ("prefix_hash", "different"), ("source_policy_state_id", "different"),
    ("transcoder_bank_fingerprint", "old-bank"), ("graph_pipeline_fingerprint", "old-code"),
    ("schema", "old-schema"), ("complete", False),
    ("target_contract", "native_local_jacobian_mlp_cut_v1"),
])
def test_resume_never_accepts_changed_identity_or_old_schema(tmp_path, field, new):
    items = collection_files(tmp_path)
    source = read_collections([x["manifest"] for x in items], model_key="qwen35_4b")["records"][0]
    row = graph_record(source)
    row[field] = new
    row["record_fingerprint"] = fingerprint({k: v for k, v in row.items() if k != "record_fingerprint"})
    with pytest.raises(RepresentationError):
        verify_graph_record(row, source, bank_fingerprint="synthetic-bank", pipeline_fingerprint="synthetic-pipeline")


def test_terminal_unavailable_records_resume_without_becoming_success(tmp_path):
    items = collection_files(tmp_path)
    sources = read_collections([x["manifest"] for x in items], model_key="qwen35_4b")["records"]
    row = graph_record(sources[0], succeeded=False)
    journal = Path(sources[0]["decision_file"]).parent / "graph_features.jsonl"
    journal.write_text(json.dumps(row) + "\n")
    resumed = _load_resume(sources, "synthetic-bank", "synthetic-pipeline", True)
    assert resumed[sources[0]["decision_id"]]["graph_status"] == "unavailable"
    with pytest.raises(RepresentationError, match="already exists"):
        _load_resume(sources, "synthetic-bank", "synthetic-pipeline", False)
    journal.write_text(json.dumps(row) + "\n{" )
    with pytest.raises(RepresentationError, match="reconciliation"):
        _load_resume(sources, "synthetic-bank", "synthetic-pipeline", True)


def test_full_bank_manifest_rejects_missing_layers_and_modified_weights(tmp_path):
    layers = []
    for index, path in enumerate(QWEN_MLP_PATHS):
        checkpoint = tmp_path / f"synthetic_{index}.bin"
        checkpoint.write_text("synthetic metadata integrity fixture; not a model")
        layers.append({"layer_path": path, "status": "succeeded", "checkpoint": str(checkpoint),
                       "checkpoint_sha256": file_sha256(checkpoint), "metadata": {"fidelity_gate_passed": True,
                       "max_dev_fvu": 0.5, "dev": {"output_fvu": 0.1, "fvu_undefined": 0.0}}})
    bank = {"schema": BANK_SCHEMA, "model_key": "qwen35_4b", "checkpoint_hash": "a" * 64,
            "policy_runtime": {"classification": "unit_test_only"},
            "policy_configuration": {"policy_runtime": {"classification": "unit_test_only"}},
            "policy_configuration_fingerprint": fingerprint({"policy_runtime": {"classification": "unit_test_only"}}),
            "complete": True, "ready_for_graphs": True, "status": "succeeded",
            "expected_layers": 32, "completed_layers": 32, "passed_layers": 32,
            "configuration": {"max_dev_fvu": 0.5},
            "expected_layer_paths": list(QWEN_MLP_PATHS), "layers": layers}
    path = tmp_path / "synthetic_manifest.json"
    bank["bank_fingerprint"] = _bank_fingerprint(bank)
    write_json_atomic(path, bank)
    assert len(validate_transcoder_bank(path, model_key="qwen35_4b", checkpoint_hash="a" * 64)["layers"]) == 32
    Path(layers[0]["checkpoint"]).write_text("corrupted")
    with pytest.raises(RepresentationError, match="checkpoint"):
        validate_transcoder_bank(path, model_key="qwen35_4b", checkpoint_hash="a" * 64)
    bank["layers"].pop()
    bank["bank_fingerprint"] = _bank_fingerprint(bank)
    write_json_atomic(path, bank)
    with pytest.raises(RepresentationError, match="32"):
        validate_transcoder_bank(path, model_key="qwen35_4b", checkpoint_hash="a" * 64)


def test_default_stage_protocol_is_explicit_and_not_a_success_claim():
    tc, graph = TranscoderStageConfig(), GraphStageConfig()
    assert (tc.feature_multiplier, tc.top_k, tc.epochs, tc.batch_size, tc.learning_rate, tc.max_dev_fvu) == (2, 64, 16, 256, 4e-4, 0.5)
    assert (graph.max_nodes, graph.max_feature_nodes, graph.max_backward_targets, graph.max_logits) == (16384, 256, 266, 10)
    assert len(QWEN_MLP_PATHS) == 32
    with pytest.raises(RepresentationError):
        TranscoderStageConfig(max_dev_fvu=float("nan")).validate()


@pytest.mark.parametrize("field,value", [("transcoder_device", "meta"), ("constant_storage_device", "meta"), ("device", "disk"),
    ("dtype", "float16"), ("attn_implementation", "unverified"), ("cpu_embedding_and_lm_head", 1)])
def test_graph_runtime_options_are_explicit_and_resident(field, value):
    with pytest.raises(RepresentationError):
        GraphStageConfig(**{field: value}).validate()
    GraphStageConfig(dtype="float32", transcoder_device="cpu", cpu_embedding_and_lm_head=True).validate()


def test_graph_cli_forwards_resident_precision_options_without_loading_a_model(tmp_path):
    import runpy
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_representations.py"
    main = runpy.run_path(str(script))["main"]
    observed = []
    main.__globals__["generate_graph_features"] = lambda *args, **kwargs: (observed.append(kwargs) or {"status": "succeeded"})
    result = main(["graph-features", "--manifest", "unit_manifest.json", "--model-key", "qwen35_4b",
        "--model-manifest", "unit_models.json", "--checkpoint-dir", "unit_no_model", "--transcoder-manifest", "unit_bank.json",
        "--report", str(tmp_path / "unit_report.json"), "--dtype", "float32", "--transcoder-device", "cpu", "--constant-storage-device", "cpu",
        "--cpu-embedding-and-lm-head", "--attention-checkpointing", "--attn-implementation", "sdpa", "--sdpa-backend", "efficient", "--torch-cpu-threads", "8"])
    assert result == 0
    config = observed[0]["config"]
    assert config.dtype == "float32" and config.transcoder_device == "cpu" and config.cpu_embedding_and_lm_head
    assert config.sdpa_backend == "efficient" and config.torch_cpu_threads == 8
    assert config.attention_checkpointing is True
    assert config.constant_storage_device == "cpu"
    assert not (tmp_path / "unit_report.json").exists()


def test_replay_runtime_mismatch_is_a_failure_before_any_forward(tmp_path):
    from matdiscovery.native_attribution import PolicyStamp
    items = collection_files(tmp_path)
    sources = read_collections([x["manifest"] for x in items], model_key="qwen35_4b")["records"]
    class WrongRuntimeAdapter:
        model_stamp = PolicyStamp("synthetic-replay", "a" * 64, "Qwen/Qwen3.5-4B", 0)
        def runtime_precision_record(self):
            return {"dtype": "bfloat16"}
        def prepare_trace_inputs(self, *args, **kwargs):
            raise AssertionError("Mismatched runtime must fail before model preparation")
    report = _replay_records(sources, WrongRuntimeAdapter(), object(), config=GraphStageConfig(),
        bank_fingerprint="synthetic-bank", pipeline_fingerprint="synthetic-pipeline", existing={},
        report_path=tmp_path / "synthetic-report.json")
    assert report["status"] == "failed" and report["unavailable_graphs"] == len(sources)
    for row in _load_resume(sources, "synthetic-bank", "synthetic-pipeline", True).values():
        assert "runtime precision/backend/placement" in row["error"]


@pytest.mark.parametrize("mutation", ["missing_runtime", "row_runtime", "row_configuration", "mixed_runtime", "configuration_hash"])
def test_collection_precision_and_configuration_provenance_cannot_be_mixed(tmp_path, mutation):
    items = collection_files(tmp_path)
    item = items[1]
    if mutation == "missing_runtime":
        item["manifest_data"].pop("policy_runtime")
    elif mutation == "row_runtime":
        item["record"]["generation"]["policy_runtime"] = {"dtype": "bfloat16"}
        update_decision(item, "generation", item["record"]["generation"])
    elif mutation == "row_configuration":
        item["record"]["generation"]["configuration_fingerprint"] = "old-bf16-config"
        update_decision(item, "generation", item["record"]["generation"])
    elif mutation == "mixed_runtime":
        item["manifest_data"]["policy_runtime"] = {"dtype": "bfloat16"}
    else:
        item["manifest_data"]["policy_configuration_fingerprint"] = "unverified"
    write_json_atomic(item["manifest"], item["manifest_data"])
    with pytest.raises(RepresentationError, match="runtime|configuration"):
        read_collections([x["manifest"] for x in items], model_key="qwen35_4b")


def test_existing_transcoder_trainer_processes_all_32_layers_without_llm(tmp_path):
    torch = pytest.importorskip("torch")
    from matdiscovery.transcoders import write_activation_shard
    items = collection_files(tmp_path, tensor_tokens=True)
    # Give train/dev different complete prefixes, as the existing trainer requires.
    from matdiscovery.esopt import tensor_state_hash
    dev_ids = torch.tensor([[2, 1, 0]])
    torch.save({"input_ids": dev_ids, "policy_stamp": items[1]["stamp"]}, items[1]["tokens"])
    update_decision(items[1], "prefix_hash", tensor_state_hash({"input_ids": dev_ids, "attention_mask": torch.ones_like(dev_ids)}))
    for item in items:
        generator = torch.Generator().manual_seed(41 if item["record"]["split"] == "train" else 42)
        for index, layer in enumerate(QWEN_MLP_PATHS):
            x = torch.randn(8, 2, generator=generator)
            y = x * 0.4 + 0.1
            path = item["manifest"].parent / f"synthetic_layer_{index:02d}.pt"
            meta = write_activation_shard(path, x, y, group_ids=[item["record"]["group_id"]] * 8,
                                          prefix_hashes=[item["record"]["prefix_hash"]] * 8,
                                          split=item["record"]["split"], policy_fingerprint="a" * 64, layer_path=layer)
            item["manifest_data"]["activation_shards"].append({"path": str(path), "layer_path": layer,
                                                              "split": item["record"]["split"], "tensor_hash": meta["tensor_hash"]})
        write_json_atomic(item["manifest"], item["manifest_data"])
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        bank = train_transcoders_from_collections(
            [x["manifest"] for x in items], tmp_path / "synthetic-bank", model_key="qwen35_4b",
            config=TranscoderStageConfig(top_k=2, epochs=1, batch_size=4, max_dev_fvu=100.0),
        )
    finally:
        torch.set_num_threads(before)
    assert bank["complete"] is True
    assert bank["passed_layers"] == 32
    assert len(bank["activation_shards"]) == 64
    assert all(row["metadata"]["config"]["feature_dim"] == 4 for row in bank["layers"])
    assert all(row["metadata"]["provenance"]["rows"] == {"train": 8, "dev": 8} for row in bank["layers"])
    validate_transcoder_bank(tmp_path / "synthetic-bank" / "transcoder_manifest.json", model_key="qwen35_4b", checkpoint_hash="a" * 64)


def test_replay_fidelity_failure_is_durable_and_keeps_full_prefix(tmp_path):
    torch = pytest.importorskip("torch")
    from matdiscovery.native_attribution import PolicyStamp
    from matdiscovery.esopt import tensor_state_hash
    items = collection_files(tmp_path, tensor_tokens=True)
    action = {"tool": "generate_structures", "arguments": {"num_candidates": 1}}
    body = json.dumps(action, separators=(",", ":"))
    ids = torch.tensor([[2] + [ord(character) + 10 for character in body] + [1]])
    for item in items:
        torch.save({"input_ids": ids, "policy_stamp": item["stamp"]}, item["tokens"])
        item["record"]["generation"].update(prompt_token_count=1, completion_count=ids.shape[1] - 1, parsed_action=action)
        update_decision(item, "prefix_hash", tensor_state_hash({"input_ids": ids, "attention_mask": torch.ones_like(ids)}))
    sources = read_collections([x["manifest"] for x in items], model_key="qwen35_4b")["records"]
    current = PolicyStamp("synthetic-replay", "a" * 64, "Qwen/Qwen3.5-4B", 0)
    observed = []
    class UnitTokenizer:
        all_special_ids = [1]
        def decode(self, tokens, **kwargs):
            return "".join("<end>" if token == 1 else chr(token - 10) for token in tokens)
    class SyntheticAdapter:
        model_stamp = current
        tokenizer = UnitTokenizer()
        def runtime_precision_record(self):
            return sources[0]["policy_runtime"]
        def prepare_trace_inputs(self, ids, *, stamp, attention_mask):
            observed.append(ids.clone())
            return ids, {"attention_mask": attention_mask, "use_cache": False, "logits_to_keep": 1}
    class DeliberatelyFailingValidation:
        def validate_backend(self, *args, **kwargs):
            raise RuntimeError("synthetic fixture current-policy fidelity gate failed: FVU=9")
        def attribute(self, *args, **kwargs):
            raise AssertionError("Attribution must not run before backend validation passes")
    report = _replay_records(sources, SyntheticAdapter(), DeliberatelyFailingValidation(),
                             config=GraphStageConfig(device="cpu"), bank_fingerprint="synthetic-bank",
                             pipeline_fingerprint="synthetic-pipeline", existing={}, report_path=tmp_path / "synthetic-report.json")
    assert report["complete"] is True and report["status"] == "failed"
    assert report["unavailable_graphs"] == 2
    assert report["failures_by_kind"] == {"prefix_fidelity_failed": 2}
    assert all(value.shape == ids.shape for value in observed)
    assert not list(tmp_path.rglob("*.npz"))
    cached = _load_resume(sources, "synthetic-bank", "synthetic-pipeline", True)
    assert len(cached) == 2
    assert all(row["features"] == {"graph.graph_missing": 1.0} for row in cached.values())


def test_filtered_dev_shards_preserve_original_data_and_new_fingerprint(tmp_path):
    torch = pytest.importorskip("torch")
    from matdiscovery.transcoders import _load_shard, write_activation_shard
    path = tmp_path / "synthetic_original_dev.pt"
    x = torch.arange(8).reshape(4, 2).float()
    metadata = write_activation_shard(path, x, x + 1, group_ids=["dev-a", "dev-a", "dev-b", "dev-b"],
                                      prefix_hashes=["shared", "unique-a", "shared", "unique-b"],
                                      split="dev", policy_fingerprint="a" * 64, layer_path=QWEN_MLP_PATHS[0])
    original_hash = file_sha256(path)
    filtered = filter_development_shard(_load_shard(path), source_path=path,
                                        output_path=tmp_path / "filtered_dev.pt", train_prefixes={"shared"})
    assert file_sha256(path) == original_hash
    assert filtered["audit"]["duplicate_dev_prefix_rows"] == 2
    assert filtered["audit"]["retained_groups"] == ["dev-a", "dev-b"]
    assert filtered["metadata"]["tensor_hash"] != metadata["tensor_hash"]
    assert filtered["metadata"]["prefix_hashes"] == ["unique-a", "unique-b"]
    empty = filter_development_shard(_load_shard(path), source_path=path,
                                     output_path=tmp_path / "must_not_exist.pt",
                                     train_prefixes={"shared", "unique-a", "unique-b"})
    assert empty["path"] is None
    assert empty["audit"]["retained_groups"] == []
    assert not (tmp_path / "must_not_exist.pt").exists()
