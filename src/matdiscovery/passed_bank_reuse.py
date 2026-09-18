"""Read-only admission of an unchanged, passed normalized bank into a new core.

The source core's frozen Python package owns source admission. This module does
not retrain, copy checkpoints, edit bank metadata, or grant a general exception
to the normal corpus fingerprint check. A sealed contract binds one source bank
to four future derived manifests; their CORE1 anchors and complete scientific
contents must subsequently match, including every token tensor and shard.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Mapping

SCHEMA = "passed_normalized_bank_reuse_v1"
TARGET_REUSE_KEY = "causal_graph_corpus_reuse"
SOURCE_REUSE_KEY = "normalized_corpus_reuse"
RECIPE = "train_centered_scalar_rms_fp32_export_v1"


class BankReuseError(ValueError):
    """Missing, changed, or scientifically incompatible reuse evidence."""


def _require(condition, message):
    if not condition:
        raise BankReuseError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise BankReuseError("Nonfinite JSON constant: " + value)


def _parse(data):
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=_reject_constant)


def _read(path):
    return _parse(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute(path):
    value = Path(path)
    _require(value.is_absolute(), "Provenance paths must be absolute: " + str(value))
    return value.resolve()


def _stat(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


class _Evidence:
    def __init__(self):
        self.files = {}
        self.stats = {}

    def add(self, item):
        if isinstance(item, (str, Path)):
            path, expected = Path(item).resolve(), None
        else:
            path, expected = _absolute(item["path"]), item["sha256"]
        _require(path.is_file(), "Missing source artifact: " + str(path))
        key, before = str(path), _stat(path)
        if key not in self.files:
            digest = _sha(path)
            _require(before == _stat(path), "Artifact changed while hashing: " + key)
            self.files[key] = {"path": key, "sha256": digest}
            self.stats[key] = before
        _require(self.stats[key] == before, "Artifact changed during verification: " + key)
        result = self.files[key]
        _require(expected is None or expected == result["sha256"], "Artifact SHA mismatch: " + key)
        return dict(result)

    def finish(self):
        for path, state in self.stats.items():
            _require(_stat(Path(path)) == state, "Artifact changed during verification: " + path)
        return [self.files[key] for key in sorted(self.files)]


def _frozen_modules(workspace, core, evidence):
    """Check source bytes before executing the source-owned validation code."""
    source_path = workspace / "source_manifest.json"
    source = _read(source_path)
    expected = source.get("files", {})
    _require(expected and source.get("source_fingerprint") == _fingerprint(expected)
             and source.get("original_project") == str(workspace), "Invalid frozen source manifest")
    declared = {}
    for item in core["snapshot_files"]:
        path = _absolute(item["path"])
        _require(path == source_path or workspace in path.parents, "Source snapshot escapes its workspace")
        evidence.add(item)
        if path != source_path:
            declared[str(path.relative_to(workspace))] = item["sha256"]
    _require(declared == expected, "Snapshot and source manifest disagree")
    for name, digest in expected.items():
        path = (workspace / name).resolve()
        _require(workspace in path.parents, "Frozen source path escapes workspace")
        evidence.add({"path": str(path), "sha256": digest})
    namespace = "_passed_bank_source_" + _fingerprint({"workspace": str(workspace), "files": expected})[:24]
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        if namespace not in sys.modules:
            package = workspace / "src/matdiscovery/__init__.py"
            _require(str(package.relative_to(workspace)) in expected, "Frozen package is not source-bound")
            spec = importlib.util.spec_from_file_location(namespace, package,
                submodule_search_locations=[str(package.parent)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = module
            spec.loader.exec_module(module)
        names = ("core_protocol", "core_collection", "core_representation", "representation_training", "parallel_normalized_bank", "transcoders")
        return SimpleNamespace(**{name: importlib.import_module(namespace + "." + name) for name in names}, namespace=namespace)
    finally:
        sys.dont_write_bytecode = old


def _no_later_work(workspace, pipeline):
    for name in ("risk_models", "core_es", "core_final", "core_reports"):
        path = workspace / "experiments" / name
        _require(not path.exists() or not any(path.rglob("*")), "Source has NN/ES/test/report work: " + name)
    for stage in ("risk", "esopt", "final", "report"):
        _require(not (workspace / "experiments/core_stage_receipts" / (stage + ".json")).exists(),
                 "Source has a later stage receipt: " + stage)
    _require(not any(stage in pipeline.get("stages", {}) for stage in ("core-risk", "core-esopt", "core-final", "core-report")),
             "Source pipeline started NN/ES/test/report")


def _pause(path, workspace, protocol, evidence):
    closure_artifact = evidence.add(path)
    closure = _read(path)
    before_path = Path(path).parent / "before.json"
    before = _read(before_path)
    evidence.add(before_path)
    _require(closure.get("closed") is True and closure.get("processes_exited")
             and all(value is True for value in closure["processes_exited"].values())
             and closure.get("new_policy_or_material_oracle_calls") == 0, "Graph pause is not closed with zero new physical calls")
    _require(evidence.add(before["source_core"]) == protocol, "Pause belongs to another source core")
    declared_pids = {str(pid) for pid in before.get("pids", {}).values()}
    _require(declared_pids and set(closure["processes_exited"]) == declared_pids,
             "Pause omitted a registered graph process")
    pipeline = closure.get("pipeline", {})
    _require(pipeline.get("state") == "halted_requires_reconciliation" and pipeline.get("study_complete") is False
             and pipeline.get("stages", {}).get("core-graphs", {}).get("status") == "failed",
             "Paused source pipeline is not terminal failed graph work")
    status = workspace / "logs/pipeline_status.json"
    _require(_read(status) == pipeline, "Source pipeline changed after closure")
    evidence.add(status)
    artifacts = closure.get("artifacts", [])
    _require(artifacts, "Graph pause omitted preserved evidence")
    paths = set()
    for item in artifacts:
        source = evidence.add(item)
        _require(source["path"] not in paths and workspace in Path(source["path"]).parents,
                 "Duplicate/foreign graph closure artifact")
        paths.add(source["path"])
        _require(evidence.add({"path": item["preserved"], "sha256": item["sha256"]})["sha256"] == source["sha256"],
                 "Graph preserved bytes differ")
    _require(str(status) in paths, "Pause did not preserve final pipeline status")
    _no_later_work(workspace, pipeline)
    return closure_artifact


def _mapping(receipt, manifest_path, core_fp, reuse_key, evidence):
    root = manifest_path.parent
    _require(receipt.get("schema") == "closed_core_v1_corpus_reuse_v1" and receipt.get("complete") is True
             and receipt.get("core_fingerprint") == core_fp and receipt.get("new_physical_calls") == 0
             and receipt.get("fingerprint") == _fingerprint({k: v for k, v in receipt.items() if k != "fingerprint"}),
             "Incomplete/changed closed-corpus reuse receipt")
    _require(evidence.add(receipt["manifest"]) == evidence.add(manifest_path), "Reuse receipt refers to another manifest")
    manifest = _read(manifest_path)
    marker = manifest.get(reuse_key, {})
    _require(marker.get("core_fingerprint") == core_fp and marker.get("new_physical_calls") == 0
             and marker.get("physical_job_identity_unchanged") is True
             and marker.get("activation_files_remain_read_only_original_references") is True
             and marker.get("mapping_fingerprint") == _fingerprint(receipt["mapping"]), "Corpus derivation marker differs")
    source_root = _absolute(receipt["source"]["predecessor_manifest"]["path"]).parent
    _require(marker.get("predecessor_manifest") == receipt["source"]["predecessor_manifest"]
             and marker.get("predecessor_receipt") == receipt["source"]["predecessor_receipt"], "Corpus anchor differs")
    for key in ("predecessor_manifest", "predecessor_receipt"):
        evidence.add(receipt["source"][key])
    result, source_paths = {}, set()
    for row in receipt["mapping"]:
        original, derived = evidence.add(row["source"]), evidence.add(row["derived"])
        old, new = Path(original["path"]), Path(derived["path"])
        _require(source_root in old.parents and root in new.parents, "Corpus mapping escapes its job")
        relative = str(new.relative_to(root))
        _require(old.relative_to(source_root) == new.relative_to(root) and relative not in result
                 and str(old) not in source_paths, "Duplicate or relocated corpus mapping")
        result[relative] = {"source": original, "derived": derived}
        source_paths.add(str(old))
    _require(result and "decisions.jsonl" in result and "decision_events.jsonl" in result, "Corpus mapping omitted decision records/events")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p != root / "graph_features.jsonl"}
    _require(actual == set(result) | {"collection_manifest.json", "reuse_receipt.json"}, "Unknown/partial corpus files")
    return manifest, result


def _source_state(source_core, target_workspace, pause_closure_path):
    workspace = Path(source_core["workspace"] if isinstance(source_core, Mapping) else source_core).resolve()
    target = Path(target_workspace).resolve()
    _require(target != workspace and workspace not in target.parents and target not in workspace.parents,
             "Reuse target must be a separate workspace")
    evidence = _Evidence()
    protocol = evidence.add(workspace / "configs/deadline_core_protocol.json")
    core = _read(protocol["path"])
    if isinstance(source_core, Mapping):
        _require(dict(source_core) == core, "Supplied source core differs from its sealed protocol")
    modules = _frozen_modules(workspace, core, evidence)
    cp, cc, representation = modules.core_protocol, modules.core_collection, modules.representation_training
    _require(cp.read_core(workspace) == core and core["registration"] == cp.NORMALIZED_REGISTRATION
             and core.get("fit_recipe") == RECIPE, "Source is not a registered normalized V3 core")
    for item in core["source_evidence"]:
        evidence.add(item)
    closure = _pause(pause_closure_path, workspace, protocol, evidence)
    # audit_core_corpus_costs below invokes the complete raw/cost/derivation
    # validators. Do not perform a second whole-corpus hash pass here.
    manifests = tuple(Path(p).resolve() for p in cp.collection_manifest_paths(core))
    jobs = cp.collection_jobs(core)
    _require(len(jobs) == len(manifests) == 4 and len({j["job_id"] for j in jobs}) == 4
             and Counter(j["split"] for j in jobs) == {"train": 3, "dev": 1}
             and all(j["benchmark"] == "made" and j["method"] == "baseline" and j["model_key"] == "qwen35_4b"
                     and j["budget"] == 50 and j["expected_counts"]["candidate_oracle_attempts"] == 50 for j in jobs),
             "Source must preserve all three train and one dev B50 jobs")
    audit = cc.audit_core_corpus_costs(core)
    _require(audit.get("complete") is True and audit["total"].get("jobs") == audit["total"].get("episodes") == 4
             and audit["total"]["costs"]["candidate_oracle_attempts"] == 200,
             "Source four-job physical evidence is incomplete")
    for item in audit["evidence_files"]:
        evidence.add(item)
    source_receipts = []
    for path in manifests:
        receipt_path = path.parent / "reuse_receipt.json"
        item = {"job_id": _read(path)["job_id"], "source_manifest": evidence.add(path), "source_receipt": evidence.add(receipt_path)}
        _mapping(_read(receipt_path), path, core["fingerprint"], SOURCE_REUSE_KEY, evidence)
        source_receipts.append(item)
    collection = representation.read_collections(manifests, model_key=core["model_key"])
    _require({r["collection_job_id"] for r in collection["records"]} == {j["job_id"] for j in jobs}, "Source has an empty collection")
    for row in collection["records"]:
        evidence.add({"path": row["input_ids_file"], "sha256": row["input_ids_sha256"]})
        representation._token_inputs(row)
    paths = modules.core_representation.paths_for(core)
    bank_path = Path(paths["bank"]) / "transcoder_manifest.json"
    bank, checked_collection = modules.core_representation._bank(core, manifests, paths)
    _require(collection == checked_collection, "Source bank validator changed collection identity")
    bank_artifact = evidence.add(bank_path)
    _require(bank == _read(bank_path), "Source bank validator changed original bytes/object")
    # _bank invokes the source validate_transcoder_bank and real normalized worker validators.
    run_path = Path(core["normalization_amendment"]["normalized_precompute_result_path"])
    run = _read(run_path)
    _require(Path(core["normalization_amendment"]["normalized_bank_path"]).resolve() == bank_path.parent
             and run.get("schema") == "full_parallel_normalized_bank_v1" and run.get("complete") is True
             and run.get("core_fingerprint") == core["fingerprint"] and run.get("layers") == 32
             and run.get("epochs_per_layer") == 64 and run.get("fit_recipe") == RECIPE
             and run.get("new_policy_or_material_oracle_calls") == 0
             and run.get("original_resume_verified") is True
             and run.get("all_bank_hashes_and_mtimes_unchanged_on_resume") is True
             and evidence.add(run["bank"]) == bank_artifact, "Incomplete normalized all-layer/original-resume proof")
    tc_path = cc.stage_receipt_path(core, "transcoders")
    tc = cc.verify_stage_receipt(core, "transcoders")
    _require(bank_artifact in [{"path": x["path"], "sha256": x["sha256"]} for x in tc["artifacts"]], "TC receipt omitted source bank")
    for item in tc["artifacts"]:
        evidence.add(item)
    workers = [layer["metadata"]["execution_provenance"]["worker_receipt"] for layer in bank["layers"]]
    _require(run.get("layer_receipts") == workers and len({x["path"] for x in workers}) == 32,
             "Main result omitted, duplicated or reordered worker proofs")
    fit_shards = []
    original_shards = {(x["job_id"], x["layer_path"]): x for x in collection["activation_shards"]}
    _require(len(bank["activation_shards"]) == len(original_shards) == 128, "Bank omitted source activation shards")
    for entry in bank["activation_shards"]:
        original = original_shards.get((entry["job_id"], entry["layer_path"]))
        _require(original is not None and all(entry.get(k) == v for k, v in original.items()), "Bank changed original shard identity")
        evidence.add({"path": entry["path"], "sha256": entry["sha256"]})
        _require(entry.get("fit_shard_path"), "Source bank has an omitted/empty fit shard")
        item = evidence.add(entry["fit_shard_path"])
        fit_shards.append({**item, "job_id": entry["job_id"], "layer_path": entry["layer_path"], "split": entry["split"],
                           "tensor_hash": entry["fit_shard_tensor_hash"], "original_shard_path": entry["path"]})
    _require(len({x["path"] for x in fit_shards}) == 128, "Bank reused a fit shard for multiple jobs/layers")
    by_layer = {name: [x for x in fit_shards if x["layer_path"] == name] for name in representation.QWEN_MLP_PATHS}
    for layer, worker in zip(bank["layers"], workers):
        evidence.add({"path": layer["checkpoint"], "sha256": layer["checkpoint_sha256"]})
        sidecar = evidence.add({"path": str(layer["checkpoint"]) + ".json", "sha256": layer["metadata_sha256"]})
        _require(_read(sidecar["path"]) == layer["metadata"], "Bank layer sidecar differs")
        evidence.add(worker)
        receipt = _read(worker["path"])
        task = receipt["request"]
        declared = by_layer[layer["layer_path"]]
        _require(task["train_paths"] == [x["path"] for x in declared if x["split"] == "train"]
                 and task["dev_paths"] == [x["path"] for x in declared if x["split"] == "dev"]
                 and task["input_files"] == [{"path": x["path"], "sha256": x["sha256"]} for x in declared if x["split"] == "train"]
                    + [{"path": x["path"], "sha256": x["sha256"]} for x in declared if x["split"] == "dev"],
                 "Normalized worker did not consume the exact full fit-shard set")
        for item in task["input_files"] + task["source_files"] + [receipt["checkpoint"], receipt["metadata"]]:
            evidence.add(item)
    contract = {"schema": SCHEMA, "source_core_fp": core["fingerprint"], "source_workspace": str(workspace),
        "target_workspace": str(target), "source_core_protocol": protocol,
        "source_source_manifest": evidence.add(workspace / "source_manifest.json"),
        "source_bank_manifest": bank_artifact, "source_run_result": evidence.add(run_path),
        "source_tc_receipt": evidence.add(tc_path), "pause_closure": closure,
        "source_manifest_paths": [str(p) for p in manifests],
        "expected_target_manifest_paths": [str(target / "experiments/collection/qwen35_4b/made" / item["job_id"] / "collection_manifest.json") for item in source_receipts],
        "source_derivations": source_receipts, "source_counts": {"collection_jobs": 4,
            "decisions": len(collection["records"]), "activation_shards": 128, "fit_shards": 128,
            "candidate_oracle_attempts": 200},
        "source_collection_fingerprint": collection["collection_fingerprint"],
        "source_bank_fingerprint": bank["bank_fingerprint"], "fit_shards": fit_shards,
        "source_validator_namespace": modules.namespace, "target_reuse_key": TARGET_REUSE_KEY,
        "new_physical_calls": 0, "source_artifacts": evidence.finish()}
    contract["fingerprint"] = _fingerprint(contract)
    return contract, bank, collection, core, modules


def build_reuse_contract(source_core, target_workspace, pause_closure_path):
    """Return a static source-verified contract without writing or requiring target files."""
    return _source_state(source_core, target_workspace, pause_closure_path)[0]


def _decision_rows(path, mapping):
    data = path.read_bytes()
    _require(data and data.endswith(b"\n"), "Incomplete decision JSONL")
    anchors = {row["derived"]["path"]: row["source"]["path"] for row in mapping.values()}
    rows = []
    for line in data.splitlines():
        row = _parse(line)
        if "input_ids_file" in row:
            value = Path(row["input_ids_file"])
            value = (path.parent / value).resolve() if not value.is_absolute() else value.resolve()
            _require(str(value) in anchors, "Decision token path lacks an exact source mapping")
            row["input_ids_file"] = anchors[str(value)]
        rows.append(row)
    return rows


def _target_admission(contract, source_core, source_collection, modules, paths, target_core):
    _require(target_core is not None, "Target corpus admission requires its sealed core")
    target = Path(contract["target_workspace"])
    from .core_protocol import (CAUSAL_GRAPH_REGISTRATION, CAUSAL_GRAPH_REGISTRATIONS,
                               FAST_CAUSAL_GRAPH_REGISTRATION, core_execution_budget)
    _require(target_core.get("registration") in CAUSAL_GRAPH_REGISTRATIONS
             and target_core.get("workspace") == str(target)
             and target_core.get("fingerprint") == target_core.get("core_fingerprint")
             == _fingerprint({k: v for k, v in target_core.items() if k not in {"fingerprint", "core_fingerprint"}}),
             "Target core is not sealed under the causal graph registration")
    _require(_read(target / "configs/deadline_core_protocol.json") == target_core, "Target core protocol bytes differ")
    reference = target_core.get("passed_bank_reuse_contract", {})
    evidence = _Evidence()
    _require(_absolute(reference["path"]) == target / "configs/passed_bank_reuse_contract.json"
             and _read(reference["path"]) == contract, "Target does not reference this exact reuse contract")
    evidence.add(reference)
    _require(target_core.get("reused_collections") == source_core["reused_collections"], "Target changed the original four closed-job anchors")
    for key in ("model", "model_key", "train_tasks", "dev_tasks", "test_tasks", "training_seed", "budget", "methods",
                "transcoder", "risk_network", "failure_control", "policy_runtime", "decoding", "memory",
                "made_execution", "deadline_utc", "parent_project", "imported_train", "imported_development",
                "tc64_amendment", "normalization_amendment"):
        _require(target_core.get(key) == source_core.get(key), "Target changed frozen scientific setting: " + key)
    if target_core.get("registration") == FAST_CAUSAL_GRAPH_REGISTRATION:
        expected_es = dict(source_core.get("esopt", {}))
        expected_es["full_training_and_development_candidate_oracle_attempts"] = 6 * core_execution_budget(target_core)
        _require(target_core.get("esopt") == expected_es, "FAST reuse changed ES beyond the new-rollout budget")
    else:
        _require(target_core.get("esopt") == source_core.get("esopt"), "Target changed frozen ES settings")
    _require(target_core.get("graph") == {**source_core.get("graph", {}), **CAUSAL_GRAPH_REGISTRATION["graph"]},
             "Target changed graph settings beyond its registered source-selection rule")
    expected = [Path(p) for p in contract["expected_target_manifest_paths"]]
    supplied = [Path(p).resolve() for p in paths]
    _require(len(supplied) == len(set(supplied)) == 4 and set(supplied) == set(expected), "Target manifests are missing, duplicated or foreign")
    for item, path in zip(contract["source_derivations"], expected):
        source_path = Path(item["source_manifest"]["path"])
        source_receipt = _read(item["source_receipt"]["path"])
        old_manifest, old_mapping = _mapping(source_receipt, source_path, source_core["fingerprint"], SOURCE_REUSE_KEY, evidence)
        receipt_path = path.parent / "reuse_receipt.json"
        evidence.add(receipt_path)
        receipt = _read(receipt_path)
        manifest, mapping = _mapping(receipt, path, target_core["fingerprint"], TARGET_REUSE_KEY, evidence)
        _require(receipt["source"] == source_receipt["source"] and set(mapping) == set(old_mapping), "Target changed/omitted common CORE1 anchors")
        _require({k: v for k, v in old_manifest.items() if k not in {SOURCE_REUSE_KEY, "decision_files"}}
                 == {k: v for k, v in manifest.items() if k not in {TARGET_REUSE_KEY, "decision_files"}},
                 "Target manifest changed scientific fields or original activation paths")
        old_marker, marker = old_manifest[SOURCE_REUSE_KEY], manifest[TARGET_REUSE_KEY]
        _require({k: v for k, v in old_marker.items() if k not in {"core_fingerprint", "mapping_fingerprint"}}
                 == {k: v for k, v in marker.items() if k not in {"core_fingerprint", "mapping_fingerprint"}},
                 "Target corpus wrapper changed non-path scientific identity")
        for name in mapping:
            old, new = old_mapping[name], mapping[name]
            _require(new["source"] == old["source"], "Target copied a different physical source file")
            if name in {"decisions.jsonl", "decision_events.jsonl"}:
                original_root = Path(receipt["source"]["predecessor_manifest"]["path"]).parent
                expected_bytes = modules.core_collection._derived_bytes(Path(new["source"]["path"]), original_root, path.parent)
                _require(hashlib.sha256(expected_bytes).hexdigest() == new["derived"]["sha256"],
                         "Target decision/event bytes are not the exact registered path-only transformation")
                _require(_decision_rows(Path(old["derived"]["path"]), old_mapping)
                         == _decision_rows(Path(new["derived"]["path"]), mapping),
                         "Target changed decision/event scientific content")
            else:
                _require(new["derived"]["sha256"] == old["derived"]["sha256"] == new["source"]["sha256"],
                         "Target token/physical-data bytes differ")
        expected_files = [mapping[str(Path(x["path"]).relative_to(source_path.parent))]["derived"] for x in old_manifest["decision_files"]]
        _require(manifest["decision_files"] == expected_files, "Target decision-file inventory changed")
    collection = modules.representation_training.read_collections(expected, model_key=source_core["model_key"])
    _require(collection["activation_shards"] == source_collection["activation_shards"], "Target activation paths/content differ")
    projected = lambda rows: [{k: v for k, v in row.items() if k not in {"decision_file", "input_ids_file"}} for row in rows]
    _require(projected(collection["records"]) == projected(source_collection["records"])
             and len(collection["records"]) == contract["source_counts"]["decisions"], "Target omitted, added or changed complete prefixes")
    for row in collection["records"]:
        modules.representation_training._token_inputs(row)
    return collection, evidence.finish()


def verify_reused_bank(contract, *, target_manifest_paths=None, target_core=None):
    """Revalidate source and optionally admit the complete sealed target corpus.

    Without target_manifest_paths the returned collection is None. Consumers
    that fit or replay the new corpus must supply all four manifests and its
    sealed target_core; the resulting proof never changes the source bank.
    """
    contract = _read(contract) if isinstance(contract, (str, Path)) else dict(contract)
    _require(contract.get("schema") == SCHEMA and contract.get("fingerprint")
             == _fingerprint({k: v for k, v in contract.items() if k != "fingerprint"}), "Unsealed/modified bank reuse contract")
    expected, bank, source_collection, source_core, modules = _source_state(
        contract["source_workspace"], contract["target_workspace"], contract["pause_closure"]["path"])
    _require(contract == expected, "Reuse contract source inventory/counts/proofs changed")
    collection, target_artifacts = None, []
    if target_manifest_paths is not None:
        collection, target_artifacts = _target_admission(contract, source_core, source_collection, modules,
                                                       target_manifest_paths, target_core)
    elif target_core is not None:
        # Do not imply corpus admission without checking every manifest.
        raise BankReuseError("Target core requires complete target_manifest_paths for admission")
    proof = {"schema": "verified_passed_normalized_bank_reuse_v1", "complete": True,
        "contract_fingerprint": contract["fingerprint"], "source_core_fp": contract["source_core_fp"],
        "source_bank_manifest": contract["source_bank_manifest"], "source_run_result": contract["source_run_result"],
        "source_bank_fingerprint": bank["bank_fingerprint"], "source_artifacts": contract["source_artifacts"],
        "target_corpus_verified": collection is not None, "target_core_fp": None if target_core is None else target_core["fingerprint"],
        "target_manifest_paths": None if collection is None else contract["expected_target_manifest_paths"],
        "target_collection_fingerprint": None if collection is None else collection["collection_fingerprint"],
        "target_artifacts": target_artifacts, "new_physical_calls": 0,
        "bank_retrained": False, "bank_bytes_or_metadata_rewritten": False}
    proof["fingerprint"] = _fingerprint(proof)
    return bank, collection, proof
