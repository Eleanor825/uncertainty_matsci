"""Registered core corpus import and one new development B50 trajectory.

Imports are a read-only derivation of verified original complete jobs. Small raw
files are copied, token paths in decisions/events are explicitly rebased, and
activation tensors remain hash-bound external read-only inputs. Original files
are never modified. A partial import or physical collection requires explicit
reconciliation; it is not silently regenerated or replayed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil

from .accounting import canonical_json, file_sha256, fingerprint, write_json_atomic
from .collection_provenance import (CollectionLineage, load_collection_lineage,
    read_json, verify_collection_completion, _mace_compatibility, FAILURE_CONTROL_003_SHA256)
from .core_protocol import (artifact, collection_jobs, collection_manifest_paths,
    read_core, require, verify_artifact, NORMALIZED_REGISTRATION, CAUSAL_GRAPH_REGISTRATIONS)


def _reuse_module(core):
    if core.get("registration") in CAUSAL_GRAPH_REGISTRATIONS:
        from . import graph_recovery
        return graph_recovery
    if core.get("registration") == NORMALIZED_REGISTRATION:
        from . import normalized_recovery
        return normalized_recovery
    from . import tc64_recovery
    return tc64_recovery


def stage_receipt_path(core, stage):
    return Path(core["workspace"]) / "experiments/core_stage_receipts" / (stage + ".json")


def verify_stage_receipt(core, stage):
    path = stage_receipt_path(core, stage)
    receipt = read_json(path)
    require(receipt.get("complete") is True and receipt.get("core_fingerprint") == core["fingerprint"]
            and receipt.get("stage") == stage, "Core stage receipt identity/closure differs")
    require(receipt.get("fingerprint") == fingerprint({k: v for k, v in receipt.items() if k != "fingerprint"}),
            "Modified core stage receipt")
    require(bool(receipt.get("artifacts")), "Core stage lacks evidence")
    for item in receipt["artifacts"]:
        verify_artifact(item)
    return receipt


def publish_stage_receipt(core, stage, files, **details):
    """Call only after the stage's domain checks and final core source check."""
    require(read_core(core["workspace"]) == core, "Core changed during stage")
    items = [artifact(path) for path in sorted({Path(path).resolve() for path in files})]
    receipt = {"schema": "deadline_core_stage_receipt_v1", "complete": True,
        "stage": stage, "core_fingerprint": core["fingerprint"], "artifacts": items,
        "original_full_study_complete": False, **details}
    receipt["fingerprint"] = fingerprint(receipt)
    path = stage_receipt_path(core, stage)
    if path.exists():
        require(verify_stage_receipt(core, stage) == receipt, "Existing core stage receipt differs; reconciliation required")
    else:
        write_json_atomic(path, receipt)
    return receipt


def _verify_original(parent, item):
    base = Path(parent) / "experiments/collection/qwen35_4b/made"
    plan = item["source_plan"]
    context = load_collection_lineage(parent, base, plan, plan["jobs"])
    proof = verify_collection_completion(base / item["job"]["job_id"], item["job"], plan, context)
    require(proof["source"]["receipt"] == item["receipt"]
            and artifact(base / item["job"]["job_id"] / "collection_manifest.json") == item["manifest"],
            "Original imported completion was changed")
    # Old serial corpus is compatible only on actually observed, exception-free
    # MACE trajectories; no assertion about unobserved parallel failure paths.
    source = Path(proof["source"]["source_manifest"]["path"]).parent
    if read_json(source / "configs/main_protocol.json").get("made_execution", {}).get("mace_num_workers", 1) == 1:
        proof["source"]["mace_compatibility"] = _mace_compatibility(
            base / item["job"]["job_id"], {Path(x["path"]).resolve() for x in proof["receipt"]["artifacts"]}, proof["episodes"])
    return proof


def inspect_train_imports(parent):
    """Verify fixed seeds 1/2 and include seed 3 iff its full receipt exists.

    The source plan remains the complete original 210-job plan. It is never
    rewritten as a smaller full-study plan. Incomplete seed 3 is never data.
    """
    from .experiment_plan import collection_jobs as original_jobs
    parent = Path(parent).resolve()
    base = parent / "experiments/collection/qwen35_4b/made"
    plan = read_json(base / "plan.json")
    expected = [j for j in original_jobs(parent) if j["model_key"] == "qwen35_4b" and j["benchmark"] == "made"]
    require(plan["jobs"] == expected, "Original full collection jobs differ from their canonical registration")
    context = load_collection_lineage(parent, base, plan, expected)
    selected = {j["seed"]: j for j in expected if j["task_id"] == "Al-Au-Hf" and j["split"] == "train"}
    result = []
    for seed in (1, 2, 3):
        job = selected[seed]
        directory = base / job["job_id"]
        if not (directory / "completion.json").is_file():
            require(seed == 3, "Required complete original training seed is absent")
            continue
        item = {"job": job, "directory": str(directory), "source_plan": plan,
            "receipt": artifact(directory / "completion.json"), "manifest": artifact(directory / "collection_manifest.json"),
            "source_evidence": list(context.provenance_files) + [artifact(base / "policy_configuration.json")],
            "usage": "training_only", "new_physical_calls": 0}
        proof = _verify_original(parent, item)
        item["source"] = proof["source"]
        result.append(item)
    return result


def _destination(core, item):
    return Path(core["workspace"]) / "experiments/collection/qwen35_4b/made" / item["job"]["job_id"]


def _derived_bytes(path, original, destination):
    """Only path-bearing decision/event files are rewritten; outcomes unchanged."""
    path, original, destination = Path(path).resolve(), Path(original).resolve(), Path(destination).resolve()
    raw = path.read_bytes()
    if path.name not in {"decisions.jsonl", "decision_events.jsonl"}:
        return raw
    require(bool(raw) and raw.endswith(b"\n"), "Truncated source decision/event file")
    rows = []
    for line in raw.splitlines():
        row = json.loads(line)
        if "input_ids_file" in row:
            source = Path(row["input_ids_file"])
            source = (source if source.is_absolute() else original / source).resolve()
            require(original in source.parents, "Imported prefix tensor lies outside original receipt directory")
            row["input_ids_file"] = str(destination / source.relative_to(original))
        rows.append(canonical_json(row))
    return ("\n".join(rows) + "\n").encode()


def _import_spec(core, item, proof):
    original, destination = Path(item["directory"]).resolve(), _destination(core, item).resolve()
    manifest = copy.deepcopy(proof["collection_manifest"])
    shards = {Path(entry["path"]).resolve() for entry in manifest["activation_shards"]}
    # Frozen-source experiments/ is a symlink to the original condition. The
    # immutable receipt may retain that execution-time alias while manifest
    # paths are canonical. Compare and relativize filesystem identities only.
    receipt_paths = [Path(entry["path"]).resolve() for entry in proof["receipt"]["artifacts"]]
    raw_files = [path for path in receipt_paths if path not in shards and path.name != "collection_manifest.json"]
    manifest["activation_shards"] = [{**entry, "path": str(Path(entry["path"]).resolve())}
                                     for entry in manifest["activation_shards"]]
    mapping = []
    for path in sorted(raw_files):
        target = destination / path.relative_to(original)
        import hashlib
        data = _derived_bytes(path, original, destination)
        mapping.append({"original": artifact(path), "derived": {"path": str(target), "sha256": hashlib.sha256(data).hexdigest()},
                        "transformation": "rebase_input_ids_file_only" if path.name in {"decisions.jsonl", "decision_events.jsonl"} else "byte_copy"})
    mapped = {x["original"]["path"]: x["derived"] for x in mapping}
    manifest["decision_files"] = [mapped[str(Path(x["path"]).resolve())] for x in manifest["decision_files"]]
    manifest["core_import"] = {"schema": "verified_original_collection_derivation_v1", "core_fingerprint": core["fingerprint"],
        "original_manifest": item["manifest"], "original_receipt": item["receipt"], "usage": "training_only",
        "transformations": "copy_raw_bytes_except_input_ids_paths_in_decisions_and_events; activation_read_only_references",
        "mapping_fingerprint": fingerprint(mapping), "activation_references": [artifact(path) for path in sorted(shards)]}
    return manifest, mapping


def validate_import_compatibility(core, item):
    """Stage-specific compatibility; source-3 inactive 003 and executor-004 only."""
    require(Path(item["directory"]).resolve() == Path(item["receipt"]["path"]).resolve().parent == Path(item["manifest"]["path"]).resolve().parent, "Original import paths disagree")
    original = Path(item["source"]["source_manifest"]["path"]).parent
    before = read_json(original / "configs/main_protocol.json")
    after = read_json(Path(core["workspace"]) / "configs/full_study_main_protocol.json")
    ignored = {"schema_version", "amendments", "made_execution", "failure_control"}
    require({k: v for k, v in before.items() if k not in ignored} == {k: v for k, v in after.items() if k not in ignored},
            "Imported baseline corpus changes scientific/model/runtime settings")
    require(fingerprint(after.get("failure_control")) == FAILURE_CONTROL_003_SHA256
            and (before.get("failure_control") is None or before["failure_control"] == after["failure_control"]),
            "Unregistered failure-control difference in original corpus")
    for name in ("configs/model_manifest.json", "configs/benchmark_tasks.json", "configs/made_splits.json",
                 "configs/policy_runtime_gates.json", "configs/assets.runtime.json", "data/raw/materials_project/index.json"):
        require(file_sha256(original / name) == file_sha256(Path(core["workspace"]) / name), "Imported fixed model/evaluator input changed: " + name)


def verify_import(core, item):
    if core.get("imported_development") is not None:
        return _reuse_module(core).verify_reused_collection(core, item["job"]["job_id"])
    validate_import_compatibility(core, item)
    proof = _verify_original(core["parent_project"], item)
    destination = _destination(core, item)
    receipt = read_json(destination / "import_receipt.json")
    manifest, mapping = _import_spec(core, item, proof)
    require(receipt.get("complete") is True and receipt.get("core_fingerprint") == core["fingerprint"]
            and receipt.get("original_receipt") == item["receipt"] and receipt.get("mapping") == mapping,
            "Imported derivation receipt changed/incomplete")
    require(receipt.get("fingerprint") == fingerprint({k: v for k, v in receipt.items() if k != "fingerprint"}), "Modified import receipt")
    require(read_json(destination / "collection_manifest.json") == manifest
            and receipt["manifest"] == artifact(destination / "collection_manifest.json"), "Derived manifest changed")
    for row in mapping:
        verify_artifact(row["derived"])
    expected = {Path(row["derived"]["path"]) for row in mapping} | {destination / "collection_manifest.json", destination / "import_receipt.json"}
    actual = {path for path in destination.rglob("*") if path.is_file() and path != destination / "graph_features.jsonl"}
    require(actual == expected, "Partial/unknown imported raw files require reconciliation")
    proof["evidence_files"] += [artifact(path) for path in sorted(expected)]
    proof["derived_manifest"] = artifact(destination / "collection_manifest.json")
    return proof


def import_training(core):
    if core.get("imported_development") is not None:
        return _reuse_module(core).import_reused_collections(core, split="train")
    for item in core["imported_train"]:
        destination = _destination(core, item)
        if destination.exists():
            require((destination / "import_receipt.json").is_file(), "Partial core import requires reconciliation; never overwrite it")
            verify_import(core, item)
            continue
        proof = _verify_original(core["parent_project"], item)
        manifest, mapping = _import_spec(core, item, proof)
        destination.mkdir(parents=True)
        for row in mapping:
            target = Path(row["derived"]["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_derived_bytes(Path(row["original"]["path"]), Path(item["directory"]), destination))
        write_json_atomic(destination / "collection_manifest.json", manifest)
        receipt = {"schema": "deadline_core_import_receipt_v1", "complete": True, "core_fingerprint": core["fingerprint"],
            "original_receipt": item["receipt"], "manifest": artifact(destination / "collection_manifest.json"),
            "mapping": mapping, "new_physical_calls": 0, "usage": "training_only"}
        receipt["fingerprint"] = fingerprint(receipt)
        write_json_atomic(destination / "import_receipt.json", receipt)
        verify_import(core, item)
    files = [_destination(core, item) / "import_receipt.json" for item in core["imported_train"]]
    return publish_stage_receipt(core, "import", files, imported_jobs=len(files), new_physical_calls=0)


def _new_context(core):
    # This is a distinct exact core matrix, not a waiver of the original 210
    # full-study plan validator. read_core binds the sole new development job.
    return CollectionLineage(core["fingerprint"], artifact(Path(core["workspace"]) / "source_manifest.json"),
                             None, {}, None, ())


def verify_development(core):
    if core.get("imported_development") is not None:
        return _reuse_module(core).verify_reused_collection(core, core["imported_development"]["job"]["job_id"])
    job = collection_jobs(core, include_imported=False)[0]
    directory = collection_manifest_paths(core)[-1].parent
    return verify_collection_completion(directory, job, {"fingerprint": core["fingerprint"]}, _new_context(core))


def completed_core_collections(core):
    for item in core["imported_train"]:
        verify_import(core, item)
    verify_development(core)
    return collection_manifest_paths(core)


def _load_policy(core):
    import torch
    from .policy import QwenPolicyAdapter, DecodingConfig
    root = Path(core["workspace"]); runtime, decode = core["policy_runtime"], core["decoding"]
    torch.set_num_threads(runtime["torch_cpu_threads"])
    torch.cuda.set_per_process_memory_fraction(runtime["cuda_memory_fraction"])
    policy = QwenPolicyAdapter.from_verified_checkpoint(root / "configs/model_manifest.json", core["model_key"], root / "data/models" / core["model_key"],
        **{key: runtime[key] for key in ("device", "dtype", "attn_implementation", "cpu_embedding_and_lm_head", "sdpa_backend", "attention_checkpointing")},
        decoding=DecodingConfig(**{key: decode[key] for key in ("max_new_tokens", "temperature", "top_p", "top_k")}),
        enable_thinking=decode["enable_thinking"], max_input_tokens=decode["max_input_tokens"])
    gates = read_json(root / "configs/policy_runtime_gates.json")
    require(gates["fingerprint"] == fingerprint({k: v for k, v in gates.items() if k != "fingerprint"}), "Modified policy runtime registry")
    gate = gates["models"][core["model_key"]]
    require(gate["assessment"]["primary_gate_passed"] and file_sha256(gate["source_report"]) == gate["source_report_sha256"]
            and gate["policy_runtime"] == policy.runtime_precision_record() and gate["checkpoint_hash"] == policy.checkpoint_hash
            and gate["configuration_fingerprint"] == policy.configuration_fingerprint, "Core policy runtime did not pass the unchanged production gate")
    return policy


def collect_development(core, *, policy_factory=None, rollout_factory=None):
    if core.get("imported_development") is not None:
        return _reuse_module(core).import_reused_collections(core, split="dev")
    verify_stage_receipt(core, "import")
    for item in core["imported_train"]:
        verify_import(core, item)
    job = collection_jobs(core, include_imported=False)[0]
    directory = collection_manifest_paths(core)[-1].parent
    started = stage_receipt_path(core, "collect").with_suffix(".started.json")
    if (directory / "completion.json").exists():
        proof = verify_development(core)
    else:
        require(not directory.exists() and not started.exists(), "Interrupted development collection requires reconciliation, never blind physical replay")
        write_json_atomic(started, {"core_fingerprint": core["fingerprint"], "job": job, "complete": False})
        policy = (policy_factory or _load_policy)(core)
        identity = {"configuration_fingerprint": policy.configuration_fingerprint, "checkpoint_hash": policy.checkpoint_hash,
                    "policy_runtime": policy.runtime_precision_record(), "policy_configuration": policy._configuration()}
        original_identity = read_json(Path(core["imported_train"][0]["directory"]).parent / "policy_configuration.json")
        require(identity == original_identity, "Core development and original training policies/runtime differ")
        write_json_atomic(directory.parent / "policy_configuration.json", identity)
        if rollout_factory is None:
            from .rollouts import DiscoveryRollout
            rollout_factory = DiscoveryRollout
        rollout = rollout_factory(Path(core["workspace"]), policy, risk_model=None, attributor=None)
        require(rollout.risk_model is None and rollout.attributor is None and job["method"] == "baseline", "Core corpus must be baseline collection")
        rollout.run(job, directory, collection=True)
        files = [artifact(path) for path in sorted(directory.rglob("*")) if path.is_file()]
        write_json_atomic(directory / "completion.json", {"complete": True, "job_fingerprint": fingerprint(job),
            "plan_fingerprint": core["fingerprint"], "policy_configuration_fingerprint": policy.configuration_fingerprint,
            "artifacts": files, "collection_execution": {"method": "baseline", "risk_model": None, "attributor": None,
                "source_manifest": _new_context(core).source_manifest, "made_execution": core["made_execution"]}})
        proof = verify_development(core)
    completed_core_collections(core)
    return publish_stage_receipt(core, "collect", [directory / "completion.json"], collection_jobs=1,
        candidate_oracle_attempts=sum(x["costs"]["candidate_oracle_attempts"] for x in proof["episodes"]))


def audit_core_corpus_costs(core):
    groups, evidence = {}, {}
    for name, proofs in (("imported_train", [verify_import(core, x) for x in core["imported_train"]]),
                         ("new_development", [verify_development(core)])):
        costs = Counter()
        for proof in proofs:
            for episode in proof["episodes"]:
                costs.update(episode["costs"])
            for item in proof["evidence_files"]:
                evidence[item["path"]] = item
        groups[name] = {"jobs": len(proofs), "episodes": sum(len(x["episodes"]) for x in proofs), "costs": dict(costs),
                        "receipts": [x["source"]["receipt"] for x in proofs]}
    total = Counter(groups["imported_train"]["costs"]); total.update(groups["new_development"]["costs"])
    if core.get("imported_development") is not None:
        groups["reused_development"] = {**groups["new_development"], "execution_origin": "historical_core_v1_completed_development"}
        groups["new_development"] = {"jobs": 0, "episodes": 0, "costs": {key: 0 for key in groups["reused_development"]["costs"]}, "receipts": []}
        groups["incremental_physical_costs"] = {key: 0 for key in ("candidate_oracle_attempts", "initialization_oracle_attempts", "surrogate_oracle_attempts", "dft_episode_attempts")}
    return {"complete": True, **groups, "total": {"jobs": len(core["imported_train"]) + 1,
        "episodes": len(core["imported_train"]) + 1, "costs": dict(total)}, "evidence_files": list(evidence.values()),
        "original_full_study_complete": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--stage", choices=("import", "collect"), required=True)
    args = parser.parse_args(); core = read_core(args.project)
    result = import_training(core) if args.stage == "import" else collect_development(core)
    print(json.dumps({"stage": args.stage, "complete": result["complete"], "core_fingerprint": core["fingerprint"]}), flush=True)


if __name__ == "__main__":
    main()
