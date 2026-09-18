"""Complete initial-policy transcoder fitting and auditable offline graph replay.

Training streams every declared activation shard through the existing trainer.
Graph replay reads an explicit identity/token projection of each decision; future
outcome labels are not consulted. Complete prefixes and all 32 native Qwen MLP
boundaries are mandatory. Fidelity/Jacobian failures produce unavailable records,
never hidden-state proxy graphs. No dense adjacency matrix is serialized.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Mapping, Sequence

from .accounting import canonical_json, file_sha256, fingerprint, write_json_atomic
from .fit_risk import RiskDataError, _initial_stamp


QWEN_MLP_PATHS = tuple(f"model.language_model.layers.{i}.mlp" for i in range(32))
BANK_SCHEMA = "matdiscovery_full_qwen_transcoder_bank_v2"
GRAPH_SCHEMA = "matdiscovery_offline_native_action_graph_features_v2"
GRAPH_TARGET_CONTRACT = "native_local_jacobian_action_logprob_mlp_cut_v2"
TC_RECOVERY_CONTRACT = "published_passed_layer_only_no_inflight_retraining_v1"


class RepresentationError(ValueError):
    pass


@dataclass(frozen=True)
class TranscoderStageConfig:
    feature_multiplier: int = 2
    feature_dim: int | None = None
    top_k: int = 64
    epochs: int = 16
    batch_size: int = 256
    learning_rate: float = 4e-4
    max_dev_fvu: float = 0.5
    seed: int = 1729
    device: str = "cpu"

    def validate(self):
        for name in ("feature_multiplier", "top_k", "epochs", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise RepresentationError(f"{name} must be a positive integer")
        if self.feature_dim is not None and (type(self.feature_dim) is not int or self.feature_dim < 1):
            raise RepresentationError("feature_dim must be positive when explicitly supplied")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise RepresentationError("learning_rate must be finite and positive")
        if not math.isfinite(self.max_dev_fvu) or self.max_dev_fvu < 0:
            raise RepresentationError("max_dev_fvu must be finite and nonnegative")
        if type(self.seed) is not int or self.seed < 0:
            raise RepresentationError("seed must be a nonnegative integer")


@dataclass(frozen=True)
class GraphStageConfig:
    max_nodes: int = 16384
    max_feature_nodes: int = 256
    max_backward_targets: int = 266
    max_logits: int = 10
    max_path_sources: int = 16
    max_path_edge_visits: int = 2_000_000
    validation_epsilon: float = 1e-3
    validation_rtol: float = 0.05
    validation_atol: float = 1e-4
    validation_max_edges: int = 4
    device: str = "cuda:0"
    transcoder_device: str | None = None
    constant_storage_device: str | None = None
    dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    sdpa_backend: str = "auto"
    cpu_embedding_and_lm_head: bool = False
    attention_checkpointing: bool = False
    torch_cpu_threads: int | None = None
    source_selection_rule: str = "global_activation"

    def validate(self):
        from .native_attribution import SOURCE_SELECTION_RULES
        if self.source_selection_rule not in SOURCE_SELECTION_RULES:
            raise RepresentationError("Unknown attribution source-selection rule")
        for name in ("max_nodes", "max_feature_nodes", "max_backward_targets", "max_logits", "validation_max_edges"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise RepresentationError(f"{name} must be positive")
        if self.validation_max_edges < 2:
            raise RepresentationError("Validation must cover token and feature edges")
        for name in ("max_path_sources", "max_path_edge_visits"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise RepresentationError(f"{name} must be a nonnegative integer")
        if self.validation_epsilon <= 0 or not math.isfinite(self.validation_epsilon):
            raise RepresentationError("Finite positive validation epsilon is required")
        if any(not math.isfinite(x) or x < 0 for x in (self.validation_rtol, self.validation_atol)):
            raise RepresentationError("Finite nonnegative validation tolerances are required")
        for name in ("device", "transcoder_device", "constant_storage_device"):
            value = getattr(self, name)
            if value is None and name in {"transcoder_device", "constant_storage_device"}:
                continue
            if not isinstance(value, str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", value):
                raise RepresentationError(f"{name} must select a resident CPU or CUDA device")
        if self.dtype not in {"float32", "bfloat16"} or self.attn_implementation not in {"eager", "sdpa"}:
            raise RepresentationError("Select an explicit supported policy precision and attention implementation")
        if type(self.cpu_embedding_and_lm_head) is not bool:
            raise RepresentationError("cpu_embedding_and_lm_head must be boolean")
        if type(self.attention_checkpointing) is not bool:
            raise RepresentationError("attention_checkpointing must be boolean")
        if self.sdpa_backend not in {"auto", "math", "efficient"} or (self.sdpa_backend != "auto" and self.attn_implementation != "sdpa"):
            raise RepresentationError("Explicit SDPA kernels require the SDPA attention implementation")
        if self.torch_cpu_threads is not None and (type(self.torch_cpu_threads) is not int or self.torch_cpu_threads < 1):
            raise RepresentationError("torch_cpu_threads must be a positive integer when supplied")


def _read_json(path: Path):
    try:
        raw = path.read_bytes()
        def invalid(value):
            raise RepresentationError(f"Nonfinite JSON {value} in {path}")
        value = json.loads(raw, parse_constant=invalid)
        return value, hashlib.sha256(raw).hexdigest()
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise RepresentationError(f"Cannot read {path}: {exc}") from exc


def _resolve(entry: Mapping | str, base: Path) -> Path:
    name = entry if isinstance(entry, str) else entry.get("path")
    if not isinstance(name, str) or not name:
        raise RepresentationError("Every collection file entry requires a path")
    path = Path(name)
    return (path if path.is_absolute() else base / path).resolve()


def _configuration_hash(configuration: Mapping) -> str:
    # Match QwenPolicyAdapter's UTF-8 canonical configuration digest exactly.
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read_collections(manifest_paths: Sequence[str | Path], *, model_key: str) -> dict:
    """Read every declared decision identity without consuming its outcome labels."""
    if not manifest_paths or model_key not in {"qwen35_4b", "qwen35_9b"}:
        raise RepresentationError("Provide collections for exactly one selected smaller Qwen model")
    manifests, records, shards, source_files = [], [], [], []
    decisions_seen, paths_seen, shard_paths_seen, jobs_seen, checkpoints, states = set(), set(), set(), set(), set(), {}
    groups = {"train": set(), "dev": set()}
    common_runtime, common_configuration, common_configuration_fingerprint = None, None, None
    for supplied in manifest_paths:
        path = Path(supplied).resolve()
        manifest, digest = _read_json(path)
        if not isinstance(manifest, dict) or manifest.get("complete") is not True:
            raise RepresentationError(f"Collection must be explicitly complete: {path}")
        runtime, configuration = manifest.get("policy_runtime"), manifest.get("policy_configuration")
        configuration_fingerprint = manifest.get("policy_configuration_fingerprint")
        if not isinstance(runtime, dict) or not runtime or not isinstance(configuration, dict) or not configuration:
            raise RepresentationError("Collection requires nonempty policy runtime and configuration provenance")
        if not isinstance(configuration_fingerprint, str) or not configuration_fingerprint:
            raise RepresentationError("Collection requires a nonempty policy configuration fingerprint")
        if configuration_fingerprint != _configuration_hash(configuration):
            raise RepresentationError("Collection policy configuration fingerprint does not verify")
        if configuration.get("policy_runtime") != runtime:
            raise RepresentationError("Collection policy runtime differs from its verified full configuration")
        if common_runtime is None:
            common_runtime, common_configuration = runtime, configuration
            common_configuration_fingerprint = configuration_fingerprint
        elif (runtime != common_runtime or configuration != common_configuration
              or configuration_fingerprint != common_configuration_fingerprint):
            raise RepresentationError("All collections must share identical policy runtime and configuration provenance")
        if not isinstance(manifest.get("job_id"), str) or manifest["job_id"] in jobs_seen:
            raise RepresentationError("Collection requires a unique job_id")
        jobs_seen.add(manifest["job_id"])
        entries = manifest.get("decision_files")
        if not isinstance(entries, list) or not entries:
            raise RepresentationError("Collection must declare all decision_files")
        local_records = []
        for entry in entries:
            decision_path = _resolve(entry, path.parent)
            if decision_path in paths_seen:
                raise RepresentationError("A decision file was supplied more than once")
            paths_seen.add(decision_path)
            raw = decision_path.read_bytes()
            decision_hash = hashlib.sha256(raw).hexdigest()
            if not isinstance(entry, Mapping) or entry.get("sha256") != decision_hash:
                raise RepresentationError(f"Decision file needs a matching declared SHA256: {decision_path}")
            source_files.append({"path": str(decision_path), "sha256": decision_hash, "kind": "decisions"})
            for line in raw.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                # Explicitly project identities; generation rewards/labels are never read.
                keys = ("decision_id", "benchmark", "model_key", "split", "task_id", "group_id", "episode_id", "input_ids_file", "prefix_hash", "policy_stamp")
                if not isinstance(item, dict) or any(key not in item for key in keys):
                    raise RepresentationError("Decision is missing a required identity/token field")
                row = {key: item[key] for key in keys}
                generation = item.get("generation", {})
                if (not isinstance(generation, dict) or generation.get("policy_runtime") != runtime
                        or generation.get("configuration_fingerprint") != configuration_fingerprint):
                    raise RepresentationError("Decision runtime or configuration fingerprint differs from its collection")
                row["policy_runtime"] = runtime
                row["action_generation"] = {
                    key: generation[key] for key in ("prompt_token_count", "completion_count", "parsed_action")
                    if isinstance(generation, dict) and key in generation
                }
                if isinstance(generation, dict) and "configuration_fingerprint" in generation:
                    row["source_policy_configuration_fingerprint"] = generation["configuration_fingerprint"]
                if row["split"] not in groups:
                    raise RepresentationError("Test records and unknown splits are forbidden in representation stages")
                if row["model_key"] != model_key:
                    raise RepresentationError("Mixed model keys are forbidden; use one model per process")
                if not isinstance(row["decision_id"], str) or row["decision_id"] in decisions_seen:
                    raise RepresentationError("Duplicate or invalid decision_id")
                decisions_seen.add(row["decision_id"])
                for name in ("benchmark", "task_id", "group_id", "prefix_hash"):
                    if not isinstance(row[name], str) or not row[name]:
                        raise RepresentationError(f"Decision needs a nonempty {name}")
                stamp = _initial_stamp(row["policy_stamp"], "Representation source decision")
                row["policy_stamp"] = stamp
                states[stamp["state_id"]] = stamp
                checkpoints.add((stamp["model_id"], stamp["checkpoint_hash"]))
                groups[row["split"]].add(row["group_id"])
                token_path = _resolve(row["input_ids_file"], decision_path.parent)
                token_hash = file_sha256(token_path)
                if item.get("input_ids_sha256") not in (None, token_hash):
                    raise RepresentationError("Captured input_ids file checksum differs")
                row.update(input_ids_file=str(token_path), input_ids_sha256=token_hash,
                           decision_file=str(decision_path), collection_job_id=manifest["job_id"])
                row["source_fingerprint"] = fingerprint({k: v for k, v in row.items() if k not in {"decision_file", "input_ids_file"}})
                records.append(row)
                local_records.append(row)
        layer_paths = set()
        for entry in manifest.get("activation_shards", []):
            if not isinstance(entry, dict) or not {"path", "layer_path", "split", "tensor_hash"}.issubset(entry):
                raise RepresentationError("Activation shard entry is missing its identity")
            if entry["layer_path"] not in QWEN_MLP_PATHS or entry["layer_path"] in layer_paths:
                raise RepresentationError("Each job must aggregate at most one shard per canonical MLP layer")
            layer_paths.add(entry["layer_path"])
            if entry["split"] not in groups:
                raise RepresentationError("Test activation shards are forbidden")
            shard_path = _resolve(entry, path.parent)
            if shard_path in shard_paths_seen:
                raise RepresentationError("The same activation shard file was declared more than once")
            shard_paths_seen.add(shard_path)
            shards.append({**entry, "path": str(shard_path), "job_id": manifest["job_id"],
                           "source_groups": sorted({r["group_id"] for r in local_records if r["split"] == entry["split"]}),
                           "source_prefixes": sorted({r["prefix_hash"] for r in local_records if r["split"] == entry["split"]})})
        manifests.append({"path": str(path), "sha256": digest, "job_id": manifest["job_id"],
                          "decision_rows": len(local_records), "activation_layers": sorted(layer_paths),
                          "missing_activation_layers": sorted(set(QWEN_MLP_PATHS) - layer_paths)})
    if not records or len(checkpoints) != 1:
        raise RepresentationError("All collections must share one nonempty verified initial model checkpoint")
    if groups["train"] & groups["dev"]:
        raise RepresentationError("Train/dev group_id leakage in collection records")
    model_id, checkpoint_hash = next(iter(checkpoints))
    records.sort(key=lambda row: row["decision_id"])
    manifest_info = sorted(manifests, key=lambda item: item["job_id"])
    return {"model_key": model_key, "model_id": model_id, "checkpoint_hash": checkpoint_hash,
            "policy_runtime": common_runtime, "policy_configuration": common_configuration,
            "policy_configuration_fingerprint": common_configuration_fingerprint,
            "records": records, "activation_shards": sorted(shards, key=lambda s: (s["layer_path"], s["split"], s["path"])),
            "collections": manifest_info, "source_files": sorted(source_files, key=lambda x: x["path"]),
            "source_policy_stamps": [states[k] for k in sorted(states)],
            "collection_fingerprint": fingerprint(manifest_info),
            "total_decisions": len(records), "test_used": False, "future_labels_used": False, "sample_cap": None}


def _bank_fingerprint(bank: Mapping) -> str:
    return fingerprint({k: v for k, v in bank.items() if k != "bank_fingerprint"})


def filter_development_shard(shard: Mapping, *, source_path: str | Path,
                             output_path: str | Path, train_prefixes: set[str], verify_existing: bool = False) -> dict:
    """Exclude exact train-prefix rows; resume verifies derived files without rewriting."""
    import torch
    from .transcoders import _load_shard, write_activation_shard
    from .esopt import tensor_state_hash
    metadata = shard["metadata"]
    if metadata["split"] != "dev":
        raise RepresentationError("Exact-prefix filtering applies only to dev shards")
    keep = torch.tensor([prefix not in train_prefixes for prefix in metadata["prefix_hashes"]], dtype=torch.bool)
    retained = int(keep.sum())
    audit = {
        "source_path": str(source_path), "source_tensor_hash": metadata["tensor_hash"],
        "original_rows": metadata["rows"], "retained_rows": retained,
        "duplicate_dev_prefix_rows": metadata["rows"] - retained,
        "original_groups": sorted(set(metadata["group_ids"])),
        "retained_groups": sorted({group for group, selected in zip(metadata["group_ids"], keep.tolist()) if selected}),
        "original_shard_modified": False,
    }
    if not retained:
        return {"path": None, "metadata": None, "audit": audit}
    if retained == metadata["rows"]:
        return {"path": str(source_path), "metadata": metadata, "audit": audit}
    groups = [group for group, selected in zip(metadata["group_ids"], keep.tolist()) if selected]
    prefixes = [prefix for prefix, selected in zip(metadata["prefix_hashes"], keep.tolist()) if selected]
    if verify_existing:
        try:
            existing = _load_shard(output_path)
            filtered = {**metadata, "rows": retained, "group_ids": groups, "prefix_hashes": prefixes,
                        "tensor_hash": tensor_state_hash({"inputs": shard["inputs"][keep], "outputs": shard["outputs"][keep]})}
            sidecar, _ = _read_json(Path(str(output_path) + ".json"))
            if existing["metadata"] != filtered or sidecar != filtered:
                raise RepresentationError("Filtered dev shard differs from the exact source-row projection; reconciliation required")
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            raise RepresentationError("Missing or changed filtered dev shard; reconciliation required") from exc
    else:
        if Path(output_path).exists() or Path(str(output_path) + ".json").exists():
            raise RepresentationError("Unpublished filtered dev output exists; reconciliation required")
        filtered = write_activation_shard(
            output_path, shard["inputs"][keep], shard["outputs"][keep], group_ids=groups,
            prefix_hashes=prefixes, split="dev", policy_fingerprint=metadata["policy_fingerprint"],
            layer_path=metadata["layer_path"],
        )
    audit.update(filtered_path=str(Path(output_path).resolve()), filtered_tensor_hash=filtered["tensor_hash"],
                 filtered_file_sha256=file_sha256(output_path))
    return {"path": str(Path(output_path).resolve()), "metadata": filtered, "audit": audit}


def _read_layer_resume(output: Path, resume: bool) -> dict | None:
    files = [path for path in output.rglob("*") if path.is_file()]
    if not files:
        return None
    if not resume:
        raise RepresentationError("Transcoder output already exists; use strict resume or reconcile it")
    manifest = output / "transcoder_manifest.json"
    try:
        bank, _ = _read_json(manifest)
    except (OSError, ValueError) as exc:
        raise RepresentationError("Orphan/partial transcoder output lacks a published manifest; reconciliation required") from exc
    if (not isinstance(bank, dict) or bank.get("schema") != BANK_SCHEMA
            or bank.get("bank_fingerprint") != _bank_fingerprint(bank)
            or bank.get("recovery_contract") != TC_RECOVERY_CONTRACT):
        raise RepresentationError("Unknown or changed transcoder recovery manifest; reconciliation required")
    if "in_progress_layer" not in bank or bank["in_progress_layer"] is not None:
        raise RepresentationError("An in-progress layer has no published completion; reconciliation required, never automatic retraining")
    layers = bank.get("layers")
    if (not isinstance(layers, list) or len(layers) > 32 or bank.get("completed_layers") != len(layers)
            or bank.get("passed_layers") != len(layers) or bank.get("source_integrity_error")):
        raise RepresentationError("Incomplete/failed layer accounting; reconciliation required")
    for index, layer in enumerate(layers):
        if (not isinstance(layer, dict) or layer.get("layer_index") != index
                or layer.get("layer_path") != QWEN_MLP_PATHS[index] or layer.get("status") != "succeeded"):
            raise RepresentationError("Only a consecutive prefix of passed complete layers can resume; reconciliation required")
    if bank.get("complete") is True and (len(layers) != 32 or bank.get("status") != "succeeded" or bank.get("ready_for_graphs") is not True):
        raise RepresentationError("Completed bank failed its fidelity/integrity gate; reconciliation required")
    allowed = {manifest.resolve()}
    for index in range(len(layers)):
        allowed.update((output / name).resolve() for name in (f"layer_{index:02d}.pt", f"layer_{index:02d}.pt.json"))
    for item in bank.get("development_exact_prefix_filter", {}).get("dev_shard_audits", []):
        if item.get("filtered_path"):
            path = Path(item["filtered_path"]).resolve()
            if output not in path.parents:
                raise RepresentationError("Filtered recovery artifact is outside bank directory")
            allowed.update((path, Path(str(path) + ".json")))
    if {path.resolve() for path in files} != allowed:
        raise RepresentationError("Missing, orphan, or partial transcoder files; reconciliation required")
    return bank


def _verify_completed_layer(layer, index, *, output, config, dimensions, by_layer, checkpoint_hash):
    """Recompute current full source provenance; accept only bytes-bound complete epochs."""
    from .transcoders import TranscoderConfig, load_transcoder, validate_shard_splits
    path = QWEN_MLP_PATHS[index]
    input_dim, output_dim = dimensions[path]
    tc_config = TranscoderConfig(input_dim, output_dim, config.feature_dim or config.feature_multiplier * input_dim, config.top_k)
    checkpoint = output / f"layer_{index:02d}.pt"
    sidecar = Path(str(checkpoint) + ".json")
    if (_resolve(layer.get("checkpoint"), output) != checkpoint or file_sha256(checkpoint) != layer.get("checkpoint_sha256")
            or file_sha256(sidecar) != layer.get("metadata_sha256")):
        raise RepresentationError("Published layer checkpoint/metadata bytes changed; reconciliation required")
    transcoder, metadata = load_transcoder(checkpoint, expected_policy_fingerprint=checkpoint_hash, expected_layer_path=path)
    del transcoder
    json_metadata, _ = _read_json(sidecar)
    if metadata != layer.get("metadata") or metadata != json_metadata or metadata.get("transcoder_hash") != layer.get("transcoder_hash"):
        raise RepresentationError("Published layer metadata differs from checkpoint; reconciliation required")
    expected = {"method": "per_mlp_topk_input_to_output_v1", "config": asdict(tc_config), "seed": config.seed + index,
                "epochs": config.epochs, "learning_rate": config.learning_rate, "batch_size": config.batch_size,
                "max_dev_fvu": config.max_dev_fvu}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RepresentationError("Completed layer changed the fixed training configuration; reconciliation required")
    provenance = validate_shard_splits(by_layer[path]["train"], by_layer[path]["dev"],
        policy_fingerprint=checkpoint_hash, layer_path=path, config=tc_config)
    if metadata.get("provenance") != provenance:
        raise RepresentationError("Completed layer source/provenance differs from all current shards; reconciliation required")
    history = metadata.get("history")
    if not isinstance(history, list) or len(history) != config.epochs:
        raise RepresentationError("Completed layer lacks all declared epochs; reconciliation required")
    for epoch, record in enumerate(history):
        if (not isinstance(record, dict) or record.get("epoch") != epoch or record.get("train_rows") != provenance["rows"]["train"]
                or not isinstance(record.get("dev"), dict) or record["dev"].get("rows") != provenance["rows"]["dev"]):
            raise RepresentationError("Completed layer epoch history omitted rows or epochs; reconciliation required")
        for value in (record.get("train_output_mse"), record["dev"].get("output_mse"), record["dev"].get("output_fvu")):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise RepresentationError("Completed layer has invalid reconstruction history; reconciliation required")
    best_epoch = min(range(config.epochs), key=lambda i: history[i]["dev"]["output_mse"])
    dev = history[best_epoch]["dev"]
    if (metadata.get("selected_epoch") != best_epoch or metadata.get("dev") != dev
            or metadata.get("fidelity_gate_passed") is not True or dev.get("fvu_undefined") != 0
            or dev["output_fvu"] > config.max_dev_fvu):
        raise RepresentationError("Completed layer failed the unchanged development selection/fidelity rule; reconciliation required")


def train_transcoders_from_collections(
    manifest_paths: Sequence[str | Path], output_dir: str | Path, *, model_key: str,
    config: TranscoderStageConfig | None = None, resume: bool = False,
) -> dict:
    """Train all 32 layers; resume only clean boundaries between published layers.

    `in_progress_layer` is published BEFORE entering the unchanged full trainer.
    An interrupted in-flight layer, orphan/partial checkpoint, failed fidelity,
    or conflicting source/config requires explicit reconciliation: never adopt
    unpublished weights, silently retrain a failed layer, resample, or select a
    better retry. Published passed layers are reverified before the next layer;
    a complete bank is verified and returned without rewriting its fingerprint.
    This is layer-boundary recovery, not optimizer/batch-level continuation.
    """
    config = config or TranscoderStageConfig()
    config.validate()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    previous = _read_layer_resume(output, resume)
    collection = read_collections(manifest_paths, model_key=model_key)
    training_sources = {name: file_sha256(Path(__file__).with_name(name)) for name in
                        ("representation_training.py", "transcoders.py", "esopt.py")}
    if previous is not None and (previous.get("configuration") != asdict(config)
            or previous.get("collection_fingerprint") != collection["collection_fingerprint"]
            or previous.get("training_source_sha256") != training_sources):
        raise RepresentationError("Transcoder collection/configuration/training source changed; reconciliation required")
    from .transcoders import TranscoderConfig, _load_shard, train_layer_transcoder
    by_layer = defaultdict(lambda: {"train": [], "dev": []})
    dimensions, shard_provenance, filter_audits = {}, [], []
    retained_dev_groups, retained_dev_rows = defaultdict(set), Counter()
    train_prefixes = {r["prefix_hash"] for r in collection["records"] if r["split"] == "train"}
    all_dev_groups = {r["group_id"] for r in collection["records"] if r["split"] == "dev"}
    duplicate_dev_decisions = [r for r in collection["records"] if r["split"] == "dev" and r["prefix_hash"] in train_prefixes]
    for entry in collection["activation_shards"]:
        shard = _load_shard(entry["path"])
        metadata = shard["metadata"]
        x, y = shard["inputs"], shard["outputs"]
        if (x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]
                or metadata["rows"] != x.shape[0] or x.shape[0] < 1
                or metadata["input_dim"] != x.shape[1] or metadata["output_dim"] != y.shape[1]
                or len(metadata["group_ids"]) != x.shape[0] or len(metadata["prefix_hashes"]) != x.shape[0]):
            raise RepresentationError("Activation shard metadata does not describe every paired token row")
        for field in ("layer_path", "split", "tensor_hash"):
            if metadata[field] != entry[field]:
                raise RepresentationError(f"Activation shard {field} differs from its collection manifest")
        if metadata["policy_fingerprint"] != collection["checkpoint_hash"]:
            raise RepresentationError("Activation shard was not captured from the initial checkpoint")
        if not set(metadata["group_ids"]).issubset(entry["source_groups"]) or not set(metadata["prefix_hashes"]).issubset(entry["source_prefixes"]):
            raise RepresentationError("Activation shard groups/prefixes are absent from its source decisions")
        layer = entry["layer_path"]
        shape = (metadata["input_dim"], metadata["output_dim"])
        if layer in dimensions and dimensions[layer] != shape:
            raise RepresentationError("Mixed activation dimensions for one MLP layer")
        dimensions[layer] = shape
        training_path, training_metadata = entry["path"], metadata
        if entry["split"] == "dev":
            filtered_path = output / "filtered_dev_shards" / f"layer_{QWEN_MLP_PATHS.index(layer):02d}" / (fingerprint(entry["job_id"]) + ".pt")
            filtered = filter_development_shard(shard, source_path=entry["path"], output_path=filtered_path,
                                                train_prefixes=train_prefixes, verify_existing=previous is not None)
            training_path, training_metadata = filtered["path"], filtered["metadata"]
            filter_audits.append({"layer_path": layer, "job_id": entry["job_id"], **filtered["audit"]})
            if training_metadata is not None:
                retained_dev_groups[layer].update(training_metadata["group_ids"])
                retained_dev_rows[layer] += training_metadata["rows"]
        if training_path is not None:
            by_layer[layer][entry["split"]].append(training_path)
        shard_provenance.append({**entry, "sha256": file_sha256(entry["path"]), "rows": metadata["rows"],
                                 "fit_shard_path": training_path,
                                 "fit_shard_tensor_hash": training_metadata["tensor_hash"] if training_metadata else None})
        del shard, x, y
    bank = {
        "schema": BANK_SCHEMA, "status": "failed", "complete": False,
        "recovery_contract": TC_RECOVERY_CONTRACT, "training_source_sha256": training_sources,
        "in_progress_layer": None, "completed_layers": 0, "passed_layers": 0,
        "model_key": model_key, "model_id": collection["model_id"],
        "checkpoint_hash": collection["checkpoint_hash"], "expected_layer_paths": list(QWEN_MLP_PATHS),
        "policy_runtime": collection["policy_runtime"], "policy_configuration": collection["policy_configuration"],
        "policy_configuration_fingerprint": collection["policy_configuration_fingerprint"],
        "expected_layers": 32, "configuration": asdict(config),
        "collection_fingerprint": collection["collection_fingerprint"],
        "collections": collection["collections"], "activation_shards": shard_provenance,
        "development_exact_prefix_filter": {
            "rule": "Remove dev activation rows with complete prefixes present in any train collection; raw collections and train shards remain complete.",
            "duplicate_dev_decision_rows": len(duplicate_dev_decisions),
            "unique_duplicate_dev_prefixes": len({r["prefix_hash"] for r in duplicate_dev_decisions}),
            "duplicate_dev_activation_rows": sum(item["duplicate_dev_prefix_rows"] for item in filter_audits),
            "training_prefixes_hash": fingerprint(sorted(train_prefixes)),
            "dev_shard_audits": filter_audits,
            "retained_groups_by_layer": {layer: sorted(retained_dev_groups[layer]) for layer in QWEN_MLP_PATHS},
            "empty_groups_by_layer": {layer: sorted(all_dev_groups - retained_dev_groups[layer]) for layer in QWEN_MLP_PATHS},
            "physical_experiment_budgets_reduced": False,
            "crystalgym_claim": "Within-prototype held-out trajectory calibration, not structural generalization.",
        },
        "all_supplied_shards_consumed": True, "sample_cap": None,
        "test_used_for_training_or_fidelity": False, "layers": [],
    }
    if previous is not None:
        progress_fields = {"status", "complete", "ready_for_graphs", "layers", "completed_layers", "passed_layers", "bank_fingerprint"}
        if ({key: value for key, value in previous.items() if key not in progress_fields}
                != {key: value for key, value in bank.items() if key not in progress_fields}):
            raise RepresentationError("Published bank source/filter/provenance no longer matches; reconciliation required")
        for index, layer in enumerate(previous["layers"]):
            _verify_completed_layer(layer, index, output=output, config=config, dimensions=dimensions,
                                    by_layer=by_layer, checkpoint_hash=collection["checkpoint_hash"])
        if previous.get("complete") is True:
            for item in shard_provenance + collection["source_files"] + collection["collections"]:
                if file_sha256(item["path"]) != item["sha256"]:
                    raise RepresentationError("Source changed during complete-bank recovery verification; reconciliation required")
            validate_transcoder_bank(output / "transcoder_manifest.json", model_key=model_key,
                                    checkpoint_hash=collection["checkpoint_hash"])
            return previous
        bank = previous
    for index, layer in enumerate(QWEN_MLP_PATHS):
        if index < len(bank["layers"]):
            continue
        bank["in_progress_layer"] = {"layer_index": index, "layer_path": layer}
        bank["bank_fingerprint"] = _bank_fingerprint(bank)
        write_json_atomic(output / "transcoder_manifest.json", bank)
        result = {"layer_path": layer, "layer_index": index, "status": "failed"}
        try:
            if layer not in dimensions or not by_layer[layer]["train"] or not by_layer[layer]["dev"]:
                raise RepresentationError("Every canonical layer requires all available train and dev shards")
            empty_groups = all_dev_groups - retained_dev_groups[layer]
            if empty_groups or retained_dev_rows[layer] < 2:
                raise RepresentationError("Insufficient dev rows or entire dev groups empty after exact-prefix filtering: " + ", ".join(sorted(empty_groups)))
            input_dim, output_dim = dimensions[layer]
            tc_config = TranscoderConfig(input_dim, output_dim, config.feature_dim or config.feature_multiplier * input_dim, config.top_k)
            checkpoint = output / f"layer_{index:02d}.pt"
            transcoder, metadata = train_layer_transcoder(
                tc_config, by_layer[layer]["train"], by_layer[layer]["dev"],
                policy_fingerprint=collection["checkpoint_hash"], layer_path=layer,
                output_path=checkpoint, seed=config.seed + index, epochs=config.epochs,
                batch_size=config.batch_size, learning_rate=config.learning_rate,
                device=config.device, max_dev_fvu=config.max_dev_fvu,
            )
            result.update(status="succeeded" if metadata["fidelity_gate_passed"] else "failed_fidelity",
                          checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                          metadata_sha256=file_sha256(Path(str(checkpoint) + ".json")),
                          transcoder_hash=metadata["transcoder_hash"], metadata=metadata)
            del transcoder
        except (RepresentationError, ValueError, RuntimeError, OSError) as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
        bank["layers"].append(result)
        bank["in_progress_layer"] = None
        bank["completed_layers"] = len(bank["layers"])
        bank["passed_layers"] = sum(item["status"] == "succeeded" for item in bank["layers"])
        bank["bank_fingerprint"] = _bank_fingerprint(bank)
        write_json_atomic(output / "transcoder_manifest.json", bank)
    bank["complete"] = len(bank["layers"]) == 32
    bank["ready_for_graphs"] = all(item["status"] == "succeeded" for item in bank["layers"])
    for item in shard_provenance + collection["source_files"] + collection["collections"]:
        try:
            if file_sha256(item["path"]) != item["sha256"]:
                raise RepresentationError(f"Source changed during transcoder training: {item['path']}")
        except OSError as exc:
            bank["source_integrity_error"] = str(exc)
            bank["ready_for_graphs"] = False
            break
        except RepresentationError as exc:
            bank["source_integrity_error"] = str(exc)
            bank["ready_for_graphs"] = False
            break
    bank["status"] = "succeeded" if bank["ready_for_graphs"] else "failed"
    bank["bank_fingerprint"] = _bank_fingerprint(bank)
    write_json_atomic(output / "transcoder_manifest.json", bank)
    return bank


def validate_transcoder_bank(path: str | Path, *, model_key: str, checkpoint_hash: str,
                            policy_runtime: Mapping | None = None,
                            policy_configuration_fingerprint: str | None = None) -> dict:
    """Verify all declared layer checkpoint bytes before graph construction."""
    path = Path(path).resolve()
    bank, _ = _read_json(path)
    if not isinstance(bank, dict) or bank.get("schema") != BANK_SCHEMA or bank.get("bank_fingerprint") != _bank_fingerprint(bank):
        raise RepresentationError("Unknown or modified transcoder-bank manifest")
    if bank.get("model_key") != model_key or bank.get("checkpoint_hash") != checkpoint_hash:
        raise RepresentationError("Transcoder bank belongs to another initial model checkpoint")
    if (not isinstance(bank.get("policy_runtime"), dict) or not bank["policy_runtime"]
            or not isinstance(bank.get("policy_configuration"), dict) or not bank["policy_configuration"]
            or not isinstance(bank.get("policy_configuration_fingerprint"), str) or not bank["policy_configuration_fingerprint"]):
        raise RepresentationError("Transcoder bank lacks policy runtime and configuration provenance")
    if bank["policy_configuration_fingerprint"] != _configuration_hash(bank["policy_configuration"]):
        raise RepresentationError("Transcoder bank policy configuration fingerprint does not verify")
    if bank["policy_configuration"].get("policy_runtime") != bank["policy_runtime"]:
        raise RepresentationError("Transcoder bank runtime differs from its verified full configuration")
    if ((policy_runtime is not None and bank["policy_runtime"] != policy_runtime)
            or (policy_configuration_fingerprint is not None and bank["policy_configuration_fingerprint"] != policy_configuration_fingerprint)):
        raise RepresentationError("Transcoder bank policy runtime or configuration differs from collection")
    if bank.get("status") != "succeeded" or bank.get("ready_for_graphs") is not True or bank.get("complete") is not True:
        raise RepresentationError("Transcoder bank is incomplete or failed a fidelity gate")
    if any(bank.get(name) != 32 for name in ("expected_layers", "completed_layers", "passed_layers")):
        raise RepresentationError("Transcoder bank counts must cover all 32 verified layers")
    if bank.get("expected_layer_paths") != list(QWEN_MLP_PATHS) or [r["layer_path"] for r in bank["layers"]] != list(QWEN_MLP_PATHS):
        raise RepresentationError("Transcoder bank must cover all 32 canonical layers in order")
    for layer in bank["layers"]:
        checkpoint = _resolve(layer["checkpoint"], path.parent)
        if layer["status"] != "succeeded" or file_sha256(checkpoint) != layer["checkpoint_sha256"]:
            raise RepresentationError("Missing, failed, or changed transcoder checkpoint")
        metadata = layer["metadata"]
        threshold = metadata.get("max_dev_fvu")
        measured = metadata.get("dev", {}).get("output_fvu")
        if (metadata.get("fidelity_gate_passed") is not True or threshold is None
                or threshold != bank["configuration"]["max_dev_fvu"]
                or type(measured) not in (int, float) or not math.isfinite(measured)
                or metadata.get("dev", {}).get("fvu_undefined") != 0
                or measured > threshold):
            raise RepresentationError("Transcoder does not have a passed fidelity gate")
    return bank


def _verify_reused_bank_binding(bank, collection, proof, *, corpus_contract, model_key,
                               transcoder_manifest, source_selection_rule):
    """Check the admission result at its consuming graph/online boundary.

    The registered recovery helper owns scientific corpus equivalence. These
    checks bind its returned proof to this exact call and retain its immutable
    source files; they do not relabel old weights as a newly trained bank.
    The directly invoked helper has already hashed its full source inventory;
    do not rescan hundreds of GB of activation shards at this second boundary.
    """
    if (proof.get("target_core_fingerprint") != corpus_contract.get("core_fingerprint")
            or proof.get("target_source_selection_rule") != source_selection_rule):
        raise RepresentationError("Reused bank target core or source-selection rule differs from this graph call")
    if (bank.get("model_key") != model_key or collection.get("model_key") != model_key
            or any(bank.get(key) != collection.get(key) for key in
                   ("model_id", "checkpoint_hash", "policy_runtime", "policy_configuration_fingerprint"))):
        raise RepresentationError("Reused bank checkpoint/runtime/configuration differs from the target corpus")
    manifest = Path(transcoder_manifest).resolve()
    expected = {"path": str(manifest), "sha256": file_sha256(manifest)}
    if proof.get("source_bank_manifest") != expected:
        raise RepresentationError("Reused bank proof refers to another manifest or changed bytes")
    evidence = [proof["source_bank_manifest"], proof["contract"], *proof["source_evidence_files"]]
    paths = {}
    for index, item in enumerate(evidence):
        path = Path(item["path"]).resolve()
        if (not path.is_file() or (index < 2 and file_sha256(path) != item["sha256"])
                or (path in paths and paths[path] != item["sha256"])):
            raise RepresentationError("Reused bank admission evidence changed")
        paths[path] = item["sha256"]
    return sorted(paths)


def verify_graph_record(record: Mapping, source: Mapping, *, bank_fingerprint: str, pipeline_fingerprint: str) -> None:
    """A terminal success or unavailable attempt can resume only with matching evidence."""
    if record.get("schema") != GRAPH_SCHEMA:
        raise RepresentationError("Old graph record schema cannot be mixed into this run")
    if record.get("record_fingerprint") != fingerprint({k: v for k, v in record.items() if k != "record_fingerprint"}):
        raise RepresentationError("Graph record fingerprint mismatch")
    expected = {
        "decision_id": source["decision_id"], "source_fingerprint": source["source_fingerprint"],
        "source_policy_state_id": source["policy_stamp"]["state_id"],
        "checkpoint_hash": source["policy_stamp"]["checkpoint_hash"], "prefix_hash": source["prefix_hash"],
        "transcoder_bank_fingerprint": bank_fingerprint, "graph_pipeline_fingerprint": pipeline_fingerprint,
        "policy_runtime": source["policy_runtime"],
        "source_policy_configuration_fingerprint": source["source_policy_configuration_fingerprint"],
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RepresentationError("Graph resume identity, prefix, checkpoint, or transcoder hashes changed")
    if record.get("complete") is not True or record.get("graph_status") not in {"succeeded", "unavailable"}:
        raise RepresentationError("Only complete graph-attempt records may be skipped")
    if record.get("target_contract") != GRAPH_TARGET_CONTRACT:
        raise RepresentationError("Legacy after-step graph targets cannot resume as action-logprob graphs")
    stamp = _initial_stamp(record.get("replay_policy_stamp"), "Recorded graph replay")
    if stamp["checkpoint_hash"] != source["policy_stamp"]["checkpoint_hash"] or stamp["model_id"] != source["policy_stamp"]["model_id"]:
        raise RepresentationError("Recorded graph replay used different weights")
    if record.get("replay_policy_state_id") != stamp["state_id"]:
        raise RepresentationError("Recorded graph replay stamp is inconsistent")
    features = record.get("features")
    if not isinstance(features, dict) or not features or any(not k.startswith("graph.") for k in features):
        raise RepresentationError("Graph resume record has invalid feature schema")
    for value in features.values():
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise RepresentationError("Graph record contains nonfinite/invalid feature values")
    if record["graph_status"] == "succeeded":
        if features.get("graph.graph_missing") != 0 or record.get("backend_validation_passed") is not True:
            raise RepresentationError("Successful graph lacks validation or has missing graph flag")
        targets = record.get("action_targets", {})
        if (not record.get("target_spec_hash") or record["target_spec_hash"] != targets.get("target_spec_hash")
                or targets.get("source_prefix_hash") != source["prefix_hash"]):
            raise RepresentationError("Successful graph lacks an independently fingerprinted action target")
    elif features.get("graph.graph_missing") != 1 or not record.get("error"):
        raise RepresentationError("Unavailable graph needs an explicit failure reason and missing flag")
    for artifact in record.get("artifacts", []):
        if file_sha256(artifact["path"]) != artifact["sha256"]:
            raise RepresentationError("Graph sparse artifact changed")


def _load_resume(records: Sequence[Mapping], bank_fingerprint: str, pipeline_fingerprint: str, resume: bool) -> dict:
    wanted = {r["decision_id"]: r for r in records}
    files = {Path(r["decision_file"]).parent / "graph_features.jsonl" for r in records}
    existing = {}
    for path in sorted(files):
        if not path.exists():
            continue
        if not resume:
            raise RepresentationError(f"Graph output already exists; use --resume to verify it: {path}")
        raw = path.read_bytes()
        if not raw or not raw.endswith(b"\n"):
            raise RepresentationError("Empty/truncated graph journal requires explicit reconciliation")
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise RepresentationError("Truncated/invalid graph journal requires explicit reconciliation") from exc
            identity = row.get("decision_id") if isinstance(row, dict) else None
            if identity not in wanted or identity in existing:
                raise RepresentationError("Graph journal contains duplicate or undeclared decisions")
            if Path(wanted[identity]["decision_file"]).parent != path.parent:
                raise RepresentationError("Graph record is stored beside a different source collection")
            verify_graph_record(row, wanted[identity], bank_fingerprint=bank_fingerprint, pipeline_fingerprint=pipeline_fingerprint)
            existing[identity] = row
    return existing


def _failure_kind(exc: Exception, phase: str) -> str:
    message = str(exc).lower()
    if phase == "action_targets":
        return "action_target_mapping_failed"
    if "fidelity" in message or "fvu" in message:
        return "prefix_fidelity_failed"
    if "out of memory" in message:
        return "out_of_memory"
    if "prefix" in message and ("hash" in message or "differ" in message):
        return "prefix_integrity_failed"
    if "parity" in message or "forward logits" in message:
        return "forward_parity_failed"
    return "backend_validation_failed" if phase == "validation" else "backward_or_graph_failed"


def _token_inputs(source: Mapping):
    import torch
    from .esopt import tensor_state_hash
    if file_sha256(source["input_ids_file"]) != source["input_ids_sha256"]:
        raise RepresentationError("Captured prefix token file hash changed")
    saved = torch.load(source["input_ids_file"], map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("input_ids"), torch.Tensor):
        raise RepresentationError("Captured prefix file must contain native input_ids tensor")
    ids = saved["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or ids.dtype not in {torch.int32, torch.int64}:
        raise RepresentationError("Captured input_ids must be one complete integer-token sequence")
    if saved.get("policy_stamp") != source["policy_stamp"]:
        raise RepresentationError("Captured token file policy stamp differs from source decision")
    mask = saved.get("attention_mask", torch.ones_like(ids))
    if not isinstance(mask, torch.Tensor) or mask.shape != ids.shape or not bool((mask == 1).all()):
        raise RepresentationError("Replay needs the complete unpadded captured prefix mask")
    if tensor_state_hash({"input_ids": ids, "attention_mask": mask}) != source["prefix_hash"]:
        raise RepresentationError("Captured prefix hash differs from source decision")
    return ids, mask


def _replay_records(records, policy, attributor, *, config, bank_fingerprint, pipeline_fingerprint,
                    existing, report_path: Path, sparse_artifact_dir: Path | None = None,
                    runtime: Mapping | None = None):
    """Internal replay engine; tests use explicitly synthetic adapters and artifacts."""
    import numpy as np
    from .esopt import tensor_state_hash
    from .graph_features import extract_graph_features
    from .action_targets import build_action_targets
    current = asdict(policy.model_stamp)
    _initial_stamp(current, "Current graph replay")
    completed, validations = dict(existing), []
    validated = False
    for source in records:
        if source["decision_id"] in completed:
            continue
        started, phase = time.monotonic(), "inputs"
        row = {
            "schema": GRAPH_SCHEMA, "decision_id": source["decision_id"],
            "source_fingerprint": source["source_fingerprint"],
            "source_policy_state_id": source["policy_stamp"]["state_id"],
            "replay_policy_state_id": current["state_id"], "replay_policy_stamp": current,
            "checkpoint_hash": current["checkpoint_hash"], "model_id": current["model_id"],
            "prefix_hash": source["prefix_hash"], "transcoder_bank_fingerprint": bank_fingerprint,
            "graph_pipeline_fingerprint": pipeline_fingerprint,
            "target_contract": GRAPH_TARGET_CONTRACT,
            "runtime": dict(runtime or {}),
            "policy_runtime": source["policy_runtime"],
            "source_policy_configuration_fingerprint": source.get("source_policy_configuration_fingerprint"),
            "complete": True, "graph_status": "unavailable", "artifacts": [],
        }
        try:
            if any(current[k] != source["policy_stamp"][k] for k in ("checkpoint_hash", "model_id")):
                raise RepresentationError("Current replay checkpoint differs from source decision")
            if policy.runtime_precision_record() != source["policy_runtime"]:
                raise RepresentationError("Replay runtime precision/backend/placement differs from source decision")
            ids, mask = _token_inputs(source)
            phase = "action_targets"
            generation = source.get("action_generation", {})
            action_targets = build_action_targets(
                ids, prompt_token_count=generation.get("prompt_token_count"), tokenizer=policy.tokenizer,
                benchmark=source["benchmark"], parsed_action=generation.get("parsed_action"), attention_mask=mask,
            )
            row.update(action_targets=action_targets.to_dict(), target_spec_hash=action_targets.target_spec_hash)
            phase = "inputs"
            prepared, kwargs = policy.prepare_trace_inputs(ids, stamp=policy.model_stamp, attention_mask=mask)
            actual_prefix = tensor_state_hash({"input_ids": prepared, "attention_mask": kwargs["attention_mask"]})
            if actual_prefix != source["prefix_hash"] or prepared.shape != ids.shape:
                raise RepresentationError("Prepared full prefix differs; truncation is forbidden")
            if kwargs.get("use_cache") is not False or kwargs.get("logits_to_keep") != 1 or "attention_mask" not in kwargs:
                raise RepresentationError("Native trace preparation must preserve mask, disable cache and keep final logits")
            if not validated:
                phase = "validation"
                validation = attributor.validate_backend(
                    prepared, policy.model_stamp, model_kwargs=kwargs,
                    epsilon=config.validation_epsilon, rtol=config.validation_rtol,
                    atol=config.validation_atol, max_edges=config.validation_max_edges,
                    action_targets=action_targets,
                )
                if validation.get("passed") is not True:
                    raise RepresentationError("Native backend validation did not pass")
                validations.append(validation)
                validated = True
            phase = "attribution"
            attributed = attributor.attribute(prepared, policy.model_stamp, model_kwargs=kwargs, action_targets=action_targets)
            if attributed.metadata.get("full_prefix_hash") != source["prefix_hash"]:
                raise RepresentationError("Attribution graph prefix hash differs from its source")
            features = extract_graph_features(attributed.graph, n_layers=32,
                                              max_path_sources=config.max_path_sources,
                                              max_path_edge_visits=config.max_path_edge_visits)
            row.update(graph_status="succeeded", features={f"graph.{k}": float(v) for k, v in features.values.items()},
                       feature_metadata=features.metadata, attribution_metadata=attributed.metadata,
                       backend_validation_passed=True, full_prefix_tokens=int(ids.shape[1]))
            if sparse_artifact_dir is not None:
                sparse_artifact_dir.mkdir(parents=True, exist_ok=True)
                artifact = sparse_artifact_dir / (hashlib.sha256(source["decision_id"].encode()).hexdigest() + ".npz")
                graph = attributed.graph
                np.savez_compressed(artifact, sources=graph.sources, targets=graph.targets, weights=graph.weights,
                                    node_mask=graph.node_mask, edge_mask=graph.edge_mask, node_types=graph.node_types)
                row["artifacts"].append({"path": str(artifact.resolve()), "sha256": file_sha256(artifact), "kind": "sparse_edges_no_dense_adjacency"})
            del attributed
        except (RepresentationError, ValueError, RuntimeError, OSError, TypeError) as exc:
            row.update(graph_status="unavailable", features={"graph.graph_missing": 1.0},
                       failure_kind=_failure_kind(exc, phase), error_type=type(exc).__name__, error=str(exc),
                       backend_validation_passed=validated)
        row["elapsed_seconds"] = time.monotonic() - started
        row["record_fingerprint"] = fingerprint(row)
        verify_graph_record(row, source, bank_fingerprint=bank_fingerprint, pipeline_fingerprint=pipeline_fingerprint)
        journal = Path(source["decision_file"]).parent / "graph_features.jsonl"
        with journal.open("a") as handle:
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        completed[source["decision_id"]] = row
        report = _graph_report(records, completed, validations, len(existing), bank_fingerprint, pipeline_fingerprint)
        write_json_atomic(report_path, report)
    report = _graph_report(records, completed, validations, len(existing), bank_fingerprint, pipeline_fingerprint)
    write_json_atomic(report_path, report)
    return report


def _graph_report(records, completed, validations, resumed, bank_fingerprint, pipeline_fingerprint):
    statuses = Counter(row["graph_status"] for row in completed.values())
    successful = statuses["succeeded"]
    complete = len(completed) == len(records)
    return {
        "schema": GRAPH_SCHEMA, "classification": "training_development_graph_replay_not_test_results",
        "status": "succeeded" if complete and successful == len(records) else "partially_failed" if successful else "failed",
        "complete": complete, "expected_decisions": len(records), "completed_decisions": len(completed),
        "successful_graphs": successful, "unavailable_graphs": statuses["unavailable"],
        "graph_coverage": successful / len(records) if records else 0.0,
        "failures_by_kind": dict(Counter(row["failure_kind"] for row in completed.values() if row["graph_status"] == "unavailable")),
        "resumed_verified_records": resumed, "new_backend_validations": validations,
        "transcoder_bank_fingerprint": bank_fingerprint, "graph_pipeline_fingerprint": pipeline_fingerprint,
        "test_used": False, "future_labels_used": False, "prefix_truncation": False,
        "sample_cap": None, "dense_graph_artifacts_written": False,
    }


def generate_graph_features(
    manifest_paths: Sequence[str | Path], *, model_key: str, model_manifest: str | Path,
    checkpoint_dir: str | Path, transcoder_manifest: str | Path, report_path: str | Path,
    config: GraphStageConfig | None = None, resume: bool = False,
    sparse_artifact_dir: str | Path | None = None,
    corpus_contract: Mapping | None = None,
) -> dict:
    """Load one verified Qwen checkpoint and replay every supplied train/dev decision."""
    config = config or GraphStageConfig()
    config.validate()
    reused = None
    if corpus_contract is not None:
        from .graph_recovery import bank_for_corpus
        reused = bank_for_corpus(corpus_contract, manifest_paths=manifest_paths,
                                model_key=model_key, transcoder_manifest=transcoder_manifest)
    if reused is None:
        collection = read_collections(manifest_paths, model_key=model_key)
        bank = validate_transcoder_bank(transcoder_manifest, model_key=model_key, checkpoint_hash=collection["checkpoint_hash"],
            policy_runtime=collection["policy_runtime"], policy_configuration_fingerprint=collection["policy_configuration_fingerprint"])
        reuse_proof = None
    else:
        bank, collection, reuse_proof = reused
        _verify_reused_bank_binding(bank, collection, reuse_proof, corpus_contract=corpus_contract,
            model_key=model_key, transcoder_manifest=transcoder_manifest, source_selection_rule=config.source_selection_rule)
    import torch
    from .graph_features import SCHEMA_VERSION
    from .native_attribution import CONTRACT, NativeAttributor, TranscoderBinding
    from .policy import QwenPolicyAdapter
    from .transcoders import load_transcoder
    source_threads = collection["policy_runtime"].get("torch_cpu_threads")
    replay_threads = config.torch_cpu_threads if config.torch_cpu_threads is not None else source_threads
    if type(replay_threads) is not int or replay_threads < 1:
        raise RepresentationError("Source runtime must declare its positive torch_cpu_threads setting")
    torch.set_num_threads(replay_threads)
    policy = QwenPolicyAdapter.from_verified_checkpoint(model_manifest, model_key, checkpoint_dir,
        device=config.device, max_input_tokens=None, dtype=config.dtype,
        attn_implementation=config.attn_implementation,
        cpu_embedding_and_lm_head=config.cpu_embedding_and_lm_head, sdpa_backend=config.sdpa_backend,
        attention_checkpointing=config.attention_checkpointing)
    stamp = _initial_stamp(asdict(policy.model_stamp), "Verified replay policy")
    if stamp["checkpoint_hash"] != collection["checkpoint_hash"] or stamp["model_id"] != collection["model_id"]:
        raise RepresentationError("Loaded Qwen checkpoint differs from collection checkpoint")
    if policy.runtime_precision_record() != collection["policy_runtime"]:
        raise RepresentationError("Graph replay runtime precision/backend/placement must exactly match the collection")
    if tuple(policy.mlp_paths) != QWEN_MLP_PATHS:
        raise RepresentationError("Loaded policy must expose every canonical Qwen MLP boundary")
    bindings = []
    for layer in bank["layers"]:
        path = _resolve(layer["checkpoint"], Path(transcoder_manifest).resolve().parent)
        transcoder, metadata = load_transcoder(path, expected_policy_fingerprint=policy.checkpoint_hash, expected_layer_path=layer["layer_path"])
        if metadata != layer["metadata"] or transcoder.checkpoint_hash() != layer["transcoder_hash"]:
            raise RepresentationError("Loaded transcoder metadata differs from its full-bank manifest")
        # The bank stays resident on the selected device. Native capture copies
        # inputs to the TC; source directions return to each policy cut device.
        bindings.append(TranscoderBinding(layer["layer_path"], transcoder.to(config.transcoder_device or config.device), metadata))
    attributor = NativeAttributor(
        policy.model, bindings, state_id_getter=policy.get_state_id,
        architecture_review=policy.architecture_review, max_nodes=config.max_nodes,
        max_feature_nodes=config.max_feature_nodes, max_backward_targets=config.max_backward_targets,
        max_logits=config.max_logits,
        constant_storage_device=config.constant_storage_device,
        source_selection_rule=config.source_selection_rule,
    )
    runtime = attributor.runtime_metadata()
    pipeline = {
        "schema": GRAPH_SCHEMA, "graph_feature_schema": SCHEMA_VERSION, "contract": CONTRACT,
        "config": asdict(config), "torch_version": str(torch.__version__),
        "architecture_review": policy.architecture_review, "checkpoint_hash": policy.checkpoint_hash,
        "policy_configuration_fingerprint": policy.configuration_fingerprint,
        "source_policy_configuration_fingerprint": collection["policy_configuration_fingerprint"],
        "source_policy_configuration": collection["policy_configuration"],
        "policy_runtime": collection["policy_runtime"],
        "runtime": runtime,
        "transcoder_bank_fingerprint": bank["bank_fingerprint"],
        "source_code_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in (
            "representation_training.py", "action_targets.py", "native_attribution.py", "graph_features.py", "policy.py", "attention_checkpointing.py")},
    }
    if reuse_proof is not None:
        pipeline["passed_bank_reuse"] = reuse_proof
    pipeline_fingerprint = fingerprint(pipeline)
    existing = _load_resume(collection["records"], bank["bank_fingerprint"], pipeline_fingerprint, resume)
    report_path = Path(report_path).resolve()
    write_json_atomic(report_path.with_suffix(".provenance.json"), {
        "pipeline": pipeline, "graph_pipeline_fingerprint": pipeline_fingerprint,
        "collections": collection["collections"], "source_policy_stamps": collection["source_policy_stamps"],
    })
    return _replay_records(collection["records"], policy, attributor, config=config,
                           bank_fingerprint=bank["bank_fingerprint"], pipeline_fingerprint=pipeline_fingerprint,
                           existing=existing, report_path=report_path,
                           runtime=runtime,
                           sparse_artifact_dir=Path(sparse_artifact_dir).resolve() if sparse_artifact_dir else None)
