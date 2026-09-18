"""Synthetic raw recorder/receipt fixture; no policy, material or oracle model."""
import json
from pathlib import Path
import shutil

from matdiscovery.accounting import fingerprint, write_json_atomic
from matdiscovery.collection_provenance import build_collection_plan
from matdiscovery.core_protocol import prepare_core_workspace, collection_jobs, collection_manifest_paths
from matdiscovery.experiment_plan import collection_jobs as full_jobs
from test_collection_provenance import artifact, complete_job, freeze, rewrite_receipt, rows

PROJECT = Path(__file__).resolve().parents[1]


def make_core_fixture(tmp_path, *, completed=3, legacy=False, frozen_alias=False, alias_manifest=False, representation=False):
    parent = tmp_path / "original"
    (parent / "configs").mkdir(parents=True)
    for name in ("main_protocol.json", "model_manifest.json", "benchmark_tasks.json", "made_splits.json", "policy_runtime_gates.json", "deadline_core_protocol.json"):
        shutil.copyfile(PROJECT / "configs" / name, parent / "configs" / name)
    for name in ("configs/assets.runtime.json", "data/raw/materials_project/index.json"):
        write_json_atomic(parent / name, {"classification": "unit_test_only_not_a_real_dataset"})
    for name in ("scripts/run_collection.py", "src/matdiscovery/fixture.py"):
        path = parent / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# unit source snapshot; never executed as research\n")
    for name in ("vendor", "environments"):
        (parent / name).mkdir()
    current = json.loads((parent / "configs/main_protocol.json").read_text())
    if legacy:
        before = dict(current)
        for key in ("made_execution", "failure_control"):
            before.pop(key)
        before["schema_version"] = 3
        write_json_atomic(parent / "configs/main_protocol.json", before)
    jobs = [j for j in full_jobs(parent) if j["model_key"] == "qwen35_4b" and j["benchmark"] == "made"]
    base = parent / "experiments/collection/qwen35_4b/made"
    base.mkdir(parents=True)
    policy = {"classification": "unit_test_only", "policy_runtime": {"dtype": "float32", "classification": "unit_test_only"}}
    identity = {"policy_configuration": policy, "configuration_fingerprint": fingerprint(policy),
        "policy_runtime": policy["policy_runtime"], "checkpoint_hash": fingerprint("unit checkpoint, no model")}
    write_json_atomic(base / "policy_configuration.json", identity)
    plan = build_collection_plan(parent, jobs); frozen = freeze(parent, plan)
    if frozen_alias:
        (frozen / "experiments").symlink_to(parent / "experiments", target_is_directory=True)
    f = {"root": parent, "base": base, "jobs": jobs, "policy_identity": identity}
    directories = []
    for seed in range(1, completed + 1):
        index = next(i for i, job in enumerate(jobs) if job["task_id"] == "Al-Au-Hf" and job["seed"] == seed and job["split"] == "train")
        directory = complete_job(f, index, plan)
        token = directory / "prefix_tokens.pt"; token.write_bytes(b"unit-only token bytes, never a model input")
        token_name = frozen / token.relative_to(parent) if frozen_alias else token
        row = {"classification": "unit_test_only", "input_ids_file": str(token_name), "label_future_failure": None,
               "generation": {"success": False}}
        rows(directory / "decisions.jsonl", [row]); rows(directory / "decision_events.jsonl", [{"input_ids_file": str(token_name), "classification": "unit_test_only"}])
        manifest = json.loads((directory / "collection_manifest.json").read_text())
        manifest["decision_files"] = [artifact(directory / "decisions.jsonl")]
        if alias_manifest:
            for entry in manifest["decision_files"] + manifest["activation_shards"]:
                entry["path"] = str(frozen / Path(entry["path"]).relative_to(parent))
        write_json_atomic(directory / "collection_manifest.json", manifest)
        rewrite_receipt(directory, jobs[index], plan, identity)
        if frozen_alias:
            receipt = json.loads((directory / "completion.json").read_text())
            for entry in receipt["artifacts"]:
                entry["path"] = str(frozen / Path(entry["path"]).relative_to(parent))
            write_json_atomic(directory / "completion.json", receipt)
        if representation:
            add_representation_data(directory, jobs[index], identity, plan)
        directories.append(directory)
    write_json_atomic(base / "plan.json", plan)
    write_json_atomic(parent / "configs/main_protocol.json", current)
    archived = parent / "experiments/interrupted/unit-only/closed.json"
    write_json_atomic(archived, {"classification": "unit_test_only", "unknown_outcomes": 0})
    reconciliation = archived.parent / "reconciliation.json"
    write_json_atomic(reconciliation, {"schema": "interruption_reconciliation_v1", "all_requests_resolved": True,
        "previous_processes_stopped": [999999], "used_for_training": False, "used_for_final_evaluation": False,
        "additional_incurred_costs_not_subtracted_from_main_budgets": True, "observed_physical_costs": {"candidate_oracle_attempts": 0},
        "artifacts": [{"path_after": str(archived), "sha256": artifact(archived)["sha256"]}]})
    core = prepare_core_workspace(parent, tmp_path / "core", reconciliation_path=reconciliation)
    return {"parent": parent, "core": core, "directories": directories, "identity": identity, "reconciliation": reconciliation, "representation": representation}


def complete_core_dev(fixture):
    core = fixture["core"]; workspace = Path(core["workspace"])
    job = collection_jobs(core, include_imported=False)[0]
    base = collection_manifest_paths(core)[-1].parent.parent
    write_json_atomic(base / "policy_configuration.json", fixture["identity"])
    source = workspace / "frozen_sources" / ("collection-" + core["fingerprint"][:20]) / "configs/main_protocol.json"
    source.parent.mkdir(parents=True)
    shutil.copyfile(workspace / "configs/main_protocol.json", source)
    directory = complete_job({"root": workspace, "base": base, "jobs": [job], "policy_identity": fixture["identity"]}, 0,
                        {"fingerprint": core["fingerprint"]})
    if fixture.get("representation"):
        add_representation_data(directory, job, fixture["identity"], {"fingerprint": core["fingerprint"]})
    return directory


def add_representation_data(directory, job, identity, plan):
    """Small actual tensor pairs inside a clearly synthetic raw-cost fixture."""
    import torch
    from matdiscovery.esopt import tensor_state_hash
    from matdiscovery.transcoders import write_activation_shard
    from matdiscovery.representation_training import QWEN_MLP_PATHS
    ids = torch.tensor([[sum(map(ord, job["task_id"])), job["seed"], 1]])
    stamp = {"model_id": job["model_id"], "checkpoint_hash": identity["checkpoint_hash"],
        "state_id": "unit-session-" + job["job_id"], "generation": 0, "perturbation_seed": None, "perturbation_sigma": None}
    prefix = tensor_state_hash({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    token = directory / "prefix_tokens.pt"; torch.save({"input_ids": ids, "policy_stamp": stamp}, token)
    row = {"decision_id": job["job_id"] + ":d0000000-c0", "local_decision_id": "d0000000-c0", "benchmark": "made",
        "model_key": job["model_key"], "split": job["split"], "task_id": job["task_id"], "group_id": job["group_id"],
        "episode_id": "0", "episode_index": 0, "input_ids_file": str(token), "prefix_hash": prefix, "policy_stamp": stamp,
        "generation": {"policy_runtime": identity["policy_runtime"], "configuration_fingerprint": identity["configuration_fingerprint"], "success": False},
        "label_future_failure": None, "classification": "synthetic_unit_test_not_policy_observation"}
    rows(directory / "decisions.jsonl", [row]); rows(directory / "decision_events.jsonl", [{"input_ids_file": str(token), "event": "proposed", "decision_id": row["decision_id"], "prefix_hash": prefix, "policy_stamp": stamp}])
    manifest = json.loads((directory / "collection_manifest.json").read_text()); manifest["policy_configuration"] = identity["policy_configuration"]
    manifest["decision_files"] = [artifact(directory / "decisions.jsonl")]
    shards = []
    for index, layer in enumerate(QWEN_MLP_PATHS):
        generator = torch.Generator().manual_seed(index + job["seed"] + (1000 if job["split"] == "dev" else 0))
        x = torch.randn(5, 64, generator=generator); y = torch.randn(5, 64, generator=generator)
        path = directory / "activation_shards" / (str(index) + ".pt")
        meta = write_activation_shard(path, x, y, group_ids=[job["group_id"]] * 5, prefix_hashes=[prefix] * 5,
            split=job["split"], policy_fingerprint=identity["checkpoint_hash"], layer_path=layer)
        shards.append({"path": str(path), "layer_path": layer, "split": job["split"], "tensor_hash": meta["tensor_hash"]})
    manifest["activation_shards"] = shards
    write_json_atomic(directory / "collection_manifest.json", manifest)
    rewrite_receipt(directory, job, plan, identity)
