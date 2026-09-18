"""CPU-only reuse contract tests, never policy/material experiment results.

The independently tested ancestral core/physical-receipt admission is isolated
at the frozen core reader and cost-auditor boundary. All four derived corpora,
128 real activation tensors, 32 real 64-epoch normalized tiny fits, worker
receipts, frozen bank validators and complete token/byte equivalence are real.
"""
from contextlib import contextmanager
from dataclasses import asdict
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.core_protocol import NORMALIZED_REGISTRATION, CAUSAL_GRAPH_REGISTRATION
from matdiscovery.core_collection import _derived_bytes
from matdiscovery.esopt import tensor_state_hash
from matdiscovery.parallel_normalized_bank import build_bank, inventory
from matdiscovery.passed_bank_reuse import build_reuse_contract, verify_reused_bank, BankReuseError
from matdiscovery import passed_bank_reuse as reuse
from matdiscovery.representation_training import QWEN_MLP_PATHS, TranscoderStageConfig
from matdiscovery.tc64_recovery import _reuse_spec
from matdiscovery.transcoders import write_activation_shard
from torch_runtime_fixture import restore_torch_runtime


def artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def seal(value):
    value["fingerprint"] = fingerprint({k: v for k, v in value.items() if k not in {"fingerprint", "core_fingerprint"}})
    if "core_fingerprint" in value:
        value["core_fingerprint"] = value["fingerprint"]
    return value


def publish_copy(core, item, key):
    root, manifest, mapping = _reuse_spec(core, item, reuse_key=key, predecessor_fingerprint="unit-CORE1-anchor")
    root.mkdir(parents=True)
    original = Path(item["predecessor_manifest"]["path"]).parent
    for row in mapping:
        target = Path(row["derived"]["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_derived_bytes(Path(row["source"]["path"]), original, root))
    write_json_atomic(root / "collection_manifest.json", manifest)
    receipt = seal({"schema": "closed_core_v1_corpus_reuse_v1", "complete": True,
        "core_fingerprint": core["fingerprint"], "source": item, "mapping": mapping,
        "manifest": artifact(root / "collection_manifest.json"), "new_physical_calls": 0})
    # Receipt fingerprint includes core_fingerprint (unlike a core protocol).
    receipt["core_fingerprint"] = core["fingerprint"]
    receipt["fingerprint"] = fingerprint({k: v for k, v in receipt.items() if k != "fingerprint"})
    write_json_atomic(root / "reuse_receipt.json", receipt)
    return root / "collection_manifest.json"


@pytest.fixture(scope="module")
def source_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("passed-bank-unit-only")
    workspace, target = root / "source-v3", root / "target-v4"
    workspace.mkdir(); (workspace / "configs").mkdir()
    shutil.copytree(Path(reuse.__file__).parent, workspace / "src/matdiscovery",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    snapshot = [artifact(p) for p in sorted((workspace / "src").rglob("*.py"))]
    source_files = {str(Path(item["path"]).relative_to(workspace)): item["sha256"] for item in snapshot}
    write_json_atomic(workspace / "source_manifest.json", {"original_project": str(workspace),
        "files": source_files, "source_fingerprint": fingerprint(source_files)})
    snapshot.append(artifact(workspace / "source_manifest.json"))
    runtime = {"classification": "unit_test_only", "device": "cpu", "dtype": "float32"}
    configuration = {"classification": "unit_test_only", "policy_runtime": runtime}
    jobs, items = [], []
    for index in range(4):
        split = "train" if index < 3 else "dev"
        job_id = f"unit-only-{split}-{index}"
        original = root / "CORE1-anchors" / job_id
        original.mkdir(parents=True)
        stamp = {"model_id": "Qwen/Qwen3.5-4B", "checkpoint_hash": "a" * 64,
            "state_id": f"unit-session-{index}", "generation": 0, "perturbation_seed": None, "perturbation_sigma": None}
        ids = torch.tensor([[index + 1, 7, 9, 11]])
        tokens = original / "tokens/complete.pt"; tokens.parent.mkdir()
        torch.save({"input_ids": ids, "policy_stamp": stamp}, tokens)
        prefix = tensor_state_hash({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        row = {"decision_id": job_id + "-decision", "local_decision_id": "step0-proposal0", "benchmark": "made",
            "model_key": "qwen35_4b", "split": split, "task_id": "Al-Au-Hf" if split == "train" else "Al-Pd-Sm",
            "group_id": "training-group" if split == "train" else "dev-group", "episode_id": "0",
            "input_ids_file": str(tokens), "input_ids_sha256": file_sha256(tokens), "prefix_hash": prefix,
            "policy_stamp": stamp, "label_future_failure": 1, "label_immediate_error": 0,
            "features": {"hidden.mean": index + .1},
            "generation": {"policy_runtime": runtime, "configuration_fingerprint": fingerprint(configuration),
                "prompt_token_count": 1, "completion_count": 3, "parsed_action": {"tool": "unit_test_only"}}}
        decisions = original / "decisions.jsonl"
        decisions.write_text(json.dumps(row) + "\n")
        (original / "decision_events.jsonl").write_text(json.dumps({**row, "event": "unit_test_only_executed"}) + "\n")
        write_json_atomic(original / "physical_evidence.json", {"classification": "unit_fixture_not_materials", "budget_shape": 50})
        job = {"job_id": job_id, "benchmark": "made", "method": "baseline", "model_key": "qwen35_4b",
            "split": split, "task_id": row["task_id"], "seed": index + 1 if split == "train" else 1, "budget": 50,
            "expected_counts": {"candidate_oracle_attempts": 50, "episodes": 1}}
        write_json_atomic(original / "job.json", job)
        shards = []
        for layer_index, layer in enumerate(QWEN_MLP_PATHS):
            generator = torch.Generator().manual_seed(42 + index)
            x = torch.rand(12, 3, generator=generator); y = x[:, :2] * .6 + .1
            path = root / "original-activation-shards" / job_id / f"layer_{layer_index:02d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            metadata = write_activation_shard(path, x, y, group_ids=[row["group_id"]] * 12,
                prefix_hashes=[prefix] * 12, split=split, policy_fingerprint=stamp["checkpoint_hash"], layer_path=layer)
            shards.append({"path": str(path), "layer_path": layer, "split": split, "tensor_hash": metadata["tensor_hash"]})
        manifest = {"job_id": job_id, "complete": True, "policy_runtime": runtime,
            "policy_configuration": configuration, "policy_configuration_fingerprint": fingerprint(configuration),
            "decision_files": [artifact(decisions)], "activation_shards": shards}
        write_json_atomic(original / "collection_manifest.json", manifest)
        receipt = root / "ancestral-receipts" / (job_id + ".json")
        write_json_atomic(receipt, {"classification": "unit_test_only_ancestral_admission_fixture", "complete": True, "job": job})
        items.append({"job": job, "predecessor_manifest": artifact(original / "collection_manifest.json"), "predecessor_receipt": artifact(receipt)})
        jobs.append(job)
    config = TranscoderStageConfig(feature_multiplier=3, top_k=4, epochs=64, batch_size=8,
                                  learning_rate=.03, max_dev_fvu=.5, device="cpu")
    runroot = root / "normalized-run"
    bankdir = workspace / "experiments/transcoders/qwen35_4b/made"
    core = {"classification": "unit_test_only_not_registered_research", "workspace": str(workspace),
        "registration": NORMALIZED_REGISTRATION, "model_key": "qwen35_4b", "fit_recipe": reuse.RECIPE,
        "core_fingerprint": None, "snapshot_files": snapshot, "source_evidence": [],
        "reused_collections": items, "imported_train": [{"job": j} for j in jobs[:3]],
        "imported_development": {"job": jobs[3]},
        "normalization_amendment": {"normalized_precompute_result_path": str(runroot / "result.json"),
                                    "normalized_bank_path": str(bankdir)}}
    seal(core); write_json_atomic(workspace / "configs/deadline_core_protocol.json", core)
    manifests = [publish_copy(core, item, reuse.SOURCE_REUSE_KEY) for item in items]
    torch.set_num_threads(2)
    bank = build_bank(manifests, bankdir, runroot, model_key="qwen35_4b", config=config,
                      registration={"core_fingerprint": core["fingerprint"], "fit_recipe": reuse.RECIPE})
    tc = {"schema": "deadline_core_stage_receipt_v1", "complete": True, "stage": "transcoders",
          "core_fingerprint": core["fingerprint"], "artifacts": [artifact(bankdir / "transcoder_manifest.json")]}
    tc["fingerprint"] = fingerprint(tc)
    write_json_atomic(workspace / "experiments/core_stage_receipts/transcoders.json", tc)
    pipeline = {"state": "halted_requires_reconciliation", "study_complete": False,
                "stages": {"core-graphs": {"status": "failed"}}}
    status = workspace / "logs/pipeline_status.json"; write_json_atomic(status, pipeline)
    pause = root / "pause"; pause.mkdir()
    preserved = pause / "preserved/pipeline_status.json"; preserved.parent.mkdir()
    shutil.copyfile(status, preserved)
    write_json_atomic(pause / "before.json", {"source_core": artifact(workspace / "configs/deadline_core_protocol.json"),
        "pids": {"domain": 10001, "wrapper": 10002, "supervisor": 10003}})
    closure = {"closed": True, "processes_exited": {"10001": True, "10002": True, "10003": True},
        "new_policy_or_material_oracle_calls": 0, "pipeline": pipeline,
        "artifacts": [{**artifact(status), "preserved": str(preserved)}]}
    write_json_atomic(pause / "closure.json", closure)
    modules = reuse._frozen_modules(workspace, core, reuse._Evidence())
    # These two upstream ancestry gates have separate full protocol/RPC tests.
    # Here they are isolated so no tiny fixture claims to be actual materials.
    modules.core_protocol.read_core = lambda path: copy.deepcopy(core)
    modules.core_collection.audit_core_corpus_costs = lambda value: {
        "complete": True, "total": {"jobs": 4, "episodes": 4, "costs": {"candidate_oracle_attempts": 200}}, "evidence_files": []}
    # Only the test's execution device differs: all production validation and
    # normalized worker/checkpoint validators themselves remain unmodified.
    modules.core_representation.transcoder_config = lambda value: config
    case = SimpleNamespace(root=root, workspace=workspace, target=target, core=core, modules=modules,
        manifests=manifests, items=items, config=config, bank=bank, bankdir=bankdir, runroot=runroot,
        closure=pause / "closure.json")
    case.contract = build_reuse_contract(core, target, case.closure)
    return case


@pytest.fixture
def target_case(source_case):
    case = source_case
    if case.target.exists(): shutil.rmtree(case.target)
    (case.target / "configs").mkdir(parents=True)
    reference = case.target / "configs/passed_bank_reuse_contract.json"
    write_json_atomic(reference, case.contract)
    core = {k: copy.deepcopy(case.core[k]) for k in ("model_key", "imported_train", "imported_development", "normalization_amendment")}
    core.update({"classification": "unit_test_only", "registration": CAUSAL_GRAPH_REGISTRATION,
            "workspace": str(case.target), "passed_bank_reuse_contract": artifact(reference),
            "reused_collections": case.core["reused_collections"], "core_fingerprint": None,
            "graph": CAUSAL_GRAPH_REGISTRATION["graph"]})
    seal(core); write_json_atomic(case.target / "configs/deadline_core_protocol.json", core)
    manifests = [publish_copy(core, item, reuse.TARGET_REUSE_KEY) for item in case.items]
    return SimpleNamespace(**vars(case), target_core=core, target_manifests=manifests)


@contextmanager
def changed(path, data):
    path = Path(path); original = path.read_bytes()
    try:
        path.write_bytes(data)
        yield
    finally:
        path.write_bytes(original)


def refresh_target(case, manifest_path, *, rewrite_mapping=True):
    manifest = json.loads(manifest_path.read_text())
    receipt_path = manifest_path.parent / "reuse_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if rewrite_mapping:
        for row in receipt["mapping"]:
            row["derived"] = artifact(row["derived"]["path"])
        manifest[reuse.TARGET_REUSE_KEY]["mapping_fingerprint"] = fingerprint(receipt["mapping"])
        manifest["decision_files"] = [artifact(x["path"]) for x in manifest["decision_files"]]
        write_json_atomic(manifest_path, manifest)
    receipt["manifest"] = artifact(manifest_path)
    receipt["fingerprint"] = fingerprint({k: v for k, v in receipt.items() if k != "fingerprint"})
    write_json_atomic(receipt_path, receipt)


def test_static_contract_future_targets_and_real_cpu_bank_unchanged(source_case):
    case = source_case
    assert case.contract["source_counts"] == {"collection_jobs": 4, "decisions": 4,
        "activation_shards": 128, "fit_shards": 128, "candidate_oracle_attempts": 200}
    assert all(not Path(p).exists() for p in case.contract["expected_target_manifest_paths"])
    before = inventory(case.bankdir)
    bank, collection, proof = verify_reused_bank(case.contract)
    assert bank == case.bank and collection is None and proof["target_corpus_verified"] is False
    assert proof["new_physical_calls"] == 0 and inventory(case.bankdir) == before
    assert bank["passed_layers"] == 32
    assert len({x["path"] for x in case.contract["source_artifacts"]}) == len(case.contract["source_artifacts"])


def test_double_core1_mapping_full_tokens_and_source_metadata_preserved(target_case):
    case = target_case
    before = inventory(case.bankdir)
    bank, collection, proof = verify_reused_bank(case.contract,
        target_manifest_paths=reversed(case.target_manifests), target_core=case.target_core)
    assert bank == case.bank and bank["collections"] != collection["collections"]
    assert len(collection["records"]) == 4 and proof["target_corpus_verified"]
    assert proof["source_bank_manifest"]["path"] == str(case.bankdir / "transcoder_manifest.json")
    assert all(layer["metadata"]["execution_provenance"]["registration"]["core_fingerprint"] == case.core["fingerprint"] for layer in bank["layers"])
    assert inventory(case.bankdir) == before


@pytest.mark.parametrize("kind", ["bank", "weight", "sidecar", "worker", "fit_shard", "source_code", "token", "closure"])
def test_source_tampering_cannot_be_hidden_by_unchanged_contract(source_case, kind):
    case = source_case
    paths = {"bank": case.bankdir / "transcoder_manifest.json", "weight": case.bank["layers"][0]["checkpoint"],
        "sidecar": str(case.bank["layers"][0]["checkpoint"]) + ".json",
        "worker": case.bank["layers"][0]["metadata"]["execution_provenance"]["worker_receipt"]["path"],
        "fit_shard": case.contract["fit_shards"][0]["path"], "source_code": case.workspace / "src/matdiscovery/transcoders.py",
        "token": case.manifests[0].parent / "tokens/complete.pt", "closure": case.closure}
    with changed(paths[kind], b"{}"):
        with pytest.raises((ValueError, KeyError, EOFError, RuntimeError)):
            verify_reused_bank(case.contract)


@pytest.mark.parametrize("kind", ["missing_job", "duplicate_job", "foreign_job", "unknown_file", "label", "feature", "prefix", "format", "physical", "token", "activation_path", "anchor", "wrapper"])
def test_target_missing_mixed_or_changed_science_rejected_even_after_resealing(target_case, kind):
    case = target_case
    paths = list(case.target_manifests)
    manifest = paths[0]
    if kind == "missing_job": paths.pop()
    elif kind == "duplicate_job": paths[-1] = paths[0]
    elif kind == "foreign_job": paths[-1] = case.manifests[-1]
    elif kind == "unknown_file": (manifest.parent / "partial.tmp").write_text("unknown")
    elif kind in {"label", "feature", "prefix"}:
        path = manifest.parent / "decisions.jsonl"; row = json.loads(path.read_text())
        if kind == "label": row["label_future_failure"] = 0
        elif kind == "feature": row["features"]["hidden.mean"] += 1
        else: row["prefix_hash"] = "f" * 64
        path.write_text(json.dumps(row) + "\n"); refresh_target(case, manifest)
    elif kind == "format":
        path = manifest.parent / "decisions.jsonl"
        path.write_text(json.dumps(json.loads(path.read_text())) + "\n")
        refresh_target(case, manifest)
    elif kind in {"physical", "token"}:
        path = manifest.parent / ("physical_evidence.json" if kind == "physical" else "tokens/complete.pt")
        path.write_bytes(b"changed"); refresh_target(case, manifest)
    elif kind == "activation_path":
        value = json.loads(manifest.read_text()); original = Path(value["activation_shards"][0]["path"])
        copied = case.target / "same-bytes-foreign-shard.pt"; shutil.copyfile(original, copied)
        value["activation_shards"][0]["path"] = str(copied)
        write_json_atomic(manifest, value); refresh_target(case, manifest)
    elif kind == "anchor":
        path = manifest.parent / "reuse_receipt.json"; receipt = json.loads(path.read_text())
        receipt["source"]["job"]["seed"] = 999
        receipt["fingerprint"] = fingerprint({k: v for k, v in receipt.items() if k != "fingerprint"})
        write_json_atomic(path, receipt)
    else:
        value = json.loads(manifest.read_text()); value[reuse.TARGET_REUSE_KEY]["physical_job_identity_unchanged"] = False
        write_json_atomic(manifest, value); refresh_target(case, manifest)
    with pytest.raises((BankReuseError, ValueError)):
        verify_reused_bank(case.contract, target_manifest_paths=paths, target_core=case.target_core)


def test_unclosed_pause_and_later_nn_work_fail_closed(source_case):
    case = source_case
    value = json.loads(case.closure.read_text()); value["closed"] = False
    with changed(case.closure, json.dumps(value).encode()):
        with pytest.raises(BankReuseError, match="not closed"):
            build_reuse_contract(case.core, case.target, case.closure)
    path = case.workspace / "experiments/risk_models/partial.json"
    path.parent.mkdir(); path.write_text("{}")
    try:
        with pytest.raises(BankReuseError, match="NN/ES/test/report"):
            build_reuse_contract(case.core, case.target, case.closure)
    finally:
        shutil.rmtree(path.parent)


def test_contract_count_reduction_and_missing_core_binding_are_rejected(target_case):
    case = target_case
    contract = copy.deepcopy(case.contract); contract["source_counts"]["decisions"] = 3
    contract["fingerprint"] = fingerprint({k: v for k, v in contract.items() if k != "fingerprint"})
    with pytest.raises(BankReuseError, match="inventory/counts"):
        verify_reused_bank(contract)
    with pytest.raises(BankReuseError, match="sealed core"):
        verify_reused_bank(case.contract, target_manifest_paths=case.target_manifests)
    core = copy.deepcopy(case.target_core); core["passed_bank_reuse_contract"]["sha256"] = "0" * 64
    seal(core); write_json_atomic(case.target / "configs/deadline_core_protocol.json", core)
    with pytest.raises(BankReuseError, match="SHA mismatch"):
        verify_reused_bank(case.contract, target_manifest_paths=case.target_manifests, target_core=core)


def test_strict_json_rejects_duplicate_and_nonfinite():
    for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
        with pytest.raises(BankReuseError): reuse._parse(raw)
