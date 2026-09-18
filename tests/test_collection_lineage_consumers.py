"""Real shared-validator integration on synthetic files, never scientific data.

The complete 210/420 matrices and B50-shaped transcripts are CPU fixtures. They
contain no model tensor, real oracle result, or production approval; no validator
is mocked. Only the canonical job provider is substituted for this synthetic
test project, whose model/config files deliberately are not real checkpoints.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.collection_provenance import CollectionProvenanceError, build_collection_plan
from matdiscovery.experiment_plan import collection_jobs
from matdiscovery import study_costs
from test_collection_provenance import (make_lineage_fixture, complete_job, freeze,
                                       rewrite_receipt, write_unit_oracle_journal)


PROJECT = Path(__file__).resolve().parents[1]


def finished_progress(f, plan):
    write_json_atomic(f["base"] / "plan.json", plan)
    value = {"expected_jobs": len(f["jobs"]), "completed_jobs": len(f["jobs"]), "complete": True,
             "collection_manifests": [str(f["base"] / job["job_id"] / "collection_manifest.json") for job in f["jobs"]]}
    write_json_atomic(f["base"] / "progress.json", value)
    return value


def full_condition(tmp_path):
    f = make_lineage_fixture(tmp_path, old_complete=2)
    for index in range(2, 210):
        complete_job(f, index, f["target_plan"])
    f["progress"] = finished_progress(f, f["target_plan"])
    return f


def test_representation_accepts_exact_full_lineage_without_rewriting_old_receipts(tmp_path, monkeypatch):
    f = full_condition(tmp_path)
    spec = importlib.util.spec_from_file_location("unit_lineage_representation", PROJECT / "scripts/run_representation_condition.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "collection_jobs", lambda root: f["jobs"])
    original = {str(p / "completion.json"): (p / "completion.json").read_bytes() for p in f["old_jobs"]}
    manifests = module.completed_collections(f["root"], "qwen35_4b", "made")
    assert len(manifests) == len(set(manifests)) == 210
    assert original == {path: Path(path).read_bytes() for path in original}

    # Rehashing a receipt cannot promote a missing production journal to valid.
    new_directory = f["base"] / f["jobs"][2]["job_id"]
    journal, receipt_path = new_directory / "environment/oracle_attempts.jsonl", new_directory / "completion.json"
    journal_bytes, receipt_bytes = journal.read_bytes(), receipt_path.read_bytes()
    journal.unlink()
    rewrite_receipt(new_directory, f["jobs"][2], f["target_plan"], f["policy_identity"])
    with pytest.raises(CollectionProvenanceError, match="receipt-bound oracle-attempt journal"):
        module.completed_collections(f["root"], "qwen35_4b", "made")
    journal.write_bytes(journal_bytes)
    receipt_path.write_bytes(receipt_bytes)

    # The exact same old receipts are not accepted without their registration.
    write_json_atomic(f["base"] / "plan.json", {k: v for k, v in f["target_plan"].items() if k != "lineage"})
    with pytest.raises(CollectionProvenanceError, match="Unregistered"):
        module.completed_collections(f["root"], "qwen35_4b", "made")
    write_json_atomic(f["base"] / "plan.json", f["target_plan"])

    duplicate = deepcopy(f["progress"])
    duplicate["collection_manifests"][-1] = duplicate["collection_manifests"][0]
    write_json_atomic(f["base"] / "progress.json", duplicate)
    with pytest.raises(RuntimeError, match="omitted or duplicated"):
        module.completed_collections(f["root"], "qwen35_4b", "made")
    write_json_atomic(f["base"] / "progress.json", f["progress"])

    orphan = f["base"] / "collection-unregistered-unit-fixture"
    orphan.mkdir()
    with pytest.raises(CollectionProvenanceError, match="Unknown"):
        module.completed_collections(f["root"], "qwen35_4b", "made")
    orphan.rmdir()

    receipt = f["old_jobs"][0] / "completion.json"
    receipt.write_bytes(receipt.read_bytes() + b"\n")
    with pytest.raises(CollectionProvenanceError, match="original receipt changed"):
        module.completed_collections(f["root"], "qwen35_4b", "made")


def test_cost_report_keeps_all_420_b50_and_hash_bound_actual_source_inventory(tmp_path, monkeypatch):
    first = full_condition(tmp_path)
    second_jobs = []
    for job in first["jobs"]:
        value = {k: v for k, v in deepcopy(job).items() if k != "job_id"}
        value.update(model_key="qwen35_9b", model_id="unit/Qwen9B-fixture")
        value["job_id"] = "collection-made-" + fingerprint(value)[:24]
        second_jobs.append(value)
    second = {"root": first["root"], "jobs": second_jobs, "policy_identity": first["policy_identity"],
              "base": first["root"] / "experiments/collection/qwen35_9b/made"}
    second["base"].mkdir(parents=True)
    write_json_atomic(second["base"] / "policy_configuration.json", second["policy_identity"])
    plan = build_collection_plan(first["root"], second_jobs)
    freeze(first["root"], plan)
    for index in range(210):
        complete_job(second, index, plan)
    finished_progress(second, plan)
    # A faithfully recorded failed MACE call is accepted and counted, not dropped.
    failed_directory = second["base"] / second_jobs[0]["job_id"]
    write_unit_oracle_journal(failed_directory, failed_role="mace")
    rewrite_receipt(failed_directory, second_jobs[0], plan, second["policy_identity"])
    other_benchmark = [job for job in collection_jobs(PROJECT) if job["benchmark"] == "crystalgym"]
    all_jobs = first["jobs"] + second_jobs + other_benchmark
    assert len(all_jobs) == len({j["job_id"] for j in all_jobs}) == 720
    monkeypatch.setattr(study_costs, "collection_jobs", lambda root: all_jobs)
    old_hashes = {str(p / "completion.json"): file_sha256(p / "completion.json") for p in first["old_jobs"]}
    report = study_costs.audit_collection_costs(first["root"], benchmark="made")
    assert report["expected_jobs"] == report["completed_jobs"] == 420
    assert report["costs"]["candidate_oracle_attempts"] == 21000
    assert report["costs"]["completed_episodes"] == 420
    assert report["costs"]["surrogate_oracle_attempts"] == 420
    assert report["costs"]["initialization_oracle_attempts"] == 840
    assert report["jobs_accepted_via_source_lineage"] == 2
    assert report["source_plan_job_counts"] == {first["old_plan"]["fingerprint"]: 2,
        first["target_plan"]["fingerprint"]: 208, plan["fingerprint"]: 210}
    evidence = {item["path"]: item["sha256"] for item in report["evidence_files"]}
    assert evidence[str(first["lineage_path"])] == file_sha256(first["lineage_path"])
    assert old_hashes == {path: file_sha256(path) for path in old_hashes}
    assert all(evidence[path] == digest for path, digest in old_hashes.items())
    assert all(row["source_provenance"]["mace_compatibility"]["mace_exceptions"] == 0
               for row in report["jobs"] if row["source_provenance"]["accepted_via_lineage"])
    current = [row["source_provenance"]["oracle_attempt_journal"] for row in report["jobs"]
               if not row["source_provenance"]["accepted_via_lineage"]]
    assert len(current) == 418 and all(j["valid"] and j["closed"] for j in current)
    assert sum(j["counts"]["candidate_oracle_attempts"] for j in current) == 418 * 50
    assert sum(j["counts"]["initialization_oracle_attempts"] for j in current) == 418 * 2
    assert sum(j["counts"]["surrogate_oracle_attempts"] for j in current) == 418
    assert sum(not j["successful"] for j in current) == 1
    assert sum(j["oracle_exceptions"] for j in current) == 1
    assert all(evidence[j["journal"]["path"]] == j["journal"]["sha256"] for j in current)
    assert all(row["source_provenance"]["oracle_attempt_journal"] is None
               for row in report["jobs"] if row["source_provenance"]["accepted_via_lineage"])

    # No relaxed global gate: all 300 CG collection jobs remain required there.
    with pytest.raises(FileNotFoundError):
        study_costs.audit_collection_costs(first["root"])
    incomplete = deepcopy(first["progress"])
    incomplete.update(completed_jobs=209, complete=False)
    write_json_atomic(first["base"] / "progress.json", incomplete)
    with pytest.raises(RuntimeError, match="every unique completed manifest"):
        study_costs.audit_collection_costs(first["root"], benchmark="made")
