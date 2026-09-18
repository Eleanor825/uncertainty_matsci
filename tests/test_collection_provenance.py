"""CPU-only synthetic lineage fixtures; never model/oracle/main study evidence."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import threading
from unittest.mock import patch

import pytest

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.collection_provenance import (
    CollectionProvenanceError, EXECUTION_V1, REVIEW_SCHEMA, SCHEMA,
    build_collection_plan, load_collection_lineage, preflight_collection,
    prepare_collection_resume, scientific_contract, verify_collection_completion,
)


def artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in values))


def canonical_jobs():
    result = []
    for split, tasks in (("train", 30), ("dev", 12)):
        for task in range(tasks):
            for seed in range(1, 6):
                name = "unit-" + split + "-" + str(task)
                job = {"stage": "collection", "model_key": "qwen35_4b", "model_id": "unit/Qwen-fixture",
                    "model_revision": "unit-fixture-not-checkpoint", "benchmark": "made", "method": "baseline", "split": split,
                    "task_id": name, "group_id": name, "task": {"id": name, "elements": ["Al", "Au", "Hf"]},
                    "seed": seed, "environment_seeds": [seed], "episode_ids": ["0"], "budget": 50,
                    "budget_unit": "candidate_oracle_attempts_per_episode",
                    "expected_counts": {"episodes": 1, "candidate_oracle_attempts": 50, "dft_episode_attempts": 0}}
                job["job_id"] = "collection-made-" + fingerprint(job)[:24]
                result.append(job)
    return result


def write_unit_oracle_journal(directory, *, failed_role=None, episode_index=0,
                              initialization_attempts=2, candidate_attempts=50, surrogate_attempts=1):
    """Use the production recorder/executor with non-scientific unit functions.

    Single-item MACE batches require one private object even with workers=4.
    No calculator, model, chemical structure, or physical oracle is constructed.
    Only fsync is elided for these disposable test fixtures; verifier is real.
    """
    from matdiscovery.benchmark_adapters import BaseAdapter
    from matdiscovery.mace_parallel import AuditedOracleExecutor

    environment = directory / "environment"
    environment.mkdir(parents=True, exist_ok=True)
    vendor = directory / "unit_vendor"
    vendor.mkdir(exist_ok=True)
    adapter = BaseAdapter({"vendor_root": str(vendor), "work_dir": str(environment), "seed": 1, "budget": 50})
    adapter.episode_index = episode_index
    adapter.oracle_attempt_file = environment / "oracle_attempts.jsonl"
    adapter.event_file = environment / "env_events.jsonl"
    adapter.oracle_attempt_file.write_text("")
    adapter.event_file.write_text("")

    class UnitGuard:
        def check(self, calculator=None):
            pass  # Unit object only; this is never a numerical/FX fidelity proof.

    class UnitOracle:
        def __init__(self, workers, role):
            self.num_workers, self.role = workers, role
            self._calculator_factory = object
            self.calculator = object()
            self._thread_local = threading.local()

        def evaluate(self, value):
            if self.num_workers == 4 and not hasattr(self._thread_local, "calculator"):
                self._thread_local.calculator = self._calculator_factory()
            if self.role == failed_role and value == "failure-unit":
                raise ValueError("recorded synthetic unit failure, no scientific evaluation")
            return {"energy": -1.0, "energy_per_atom": -1.0, "natoms": 1, "classification": "unit_test_only"}

        def batch_evaluate(self, values):
            return [self.evaluate(value) for value in values]

    orb, mace = UnitOracle(1, "orb"), UnitOracle(4, "mace")
    for oracle, role in ((orb, "orb"), (mace, "mace")):
        AuditedOracleExecutor(oracle, role=role, candidate_hash=str, invoke=adapter.invoke_oracle,
                              record=adapter.oracle_audit, guard=UnitGuard() if role == "mace" else None)
    with patch("matdiscovery.benchmark_adapters.os.fsync", return_value=None):
        for i in range(initialization_attempts):
            orb.evaluate("unit-init-" + str(i))
        adapter._phase = "candidate"
        for oracle, count in ((mace, surrogate_attempts), (orb, candidate_attempts)):
            for i in range(count):
                value = "failure-unit" if oracle.role == failed_role and i == 0 else "unit-" + str(i)
                try:
                    oracle.evaluate(value)
                except ValueError:
                    pass  # Preserve the actual recorded terminal, without retry.
    return adapter.oracle_attempt_file


def freeze(root, plan):
    destination = root / "frozen_sources" / ("collection-" + plan["fingerprint"][:20])
    for name in plan["inputs"]:
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, path)
    write_json_atomic(destination / "source_manifest.json", {"original_project": str(root),
        "plan_fingerprint": plan["fingerprint"], "inputs": plan["inputs"]})
    return destination


def complete_job(fixture, index, plan):
    """Write one small B50-shaped unit transcript, not real scientific results."""
    job = fixture["jobs"][index]
    directory = fixture["base"] / job["job_id"]
    identity = fixture["policy_identity"]
    write_json_atomic(directory / "job.json", {**job, "policy_configuration_fingerprint": identity["configuration_fingerprint"]})
    summary = {k: job[k] for k in ("benchmark", "model_key", "method", "task_id", "seed")}
    summary.update(complete=True, status="succeeded", environment_seed=job["seed"], episode_id="0",
                   costs={"initialization_oracle_attempts": 2, "candidate_oracle_attempts": 50, "surrogate_oracle_attempts": 1})
    write_json_atomic(directory / "episodes.json", [summary])
    rows(directory / "decisions.jsonl", [{"classification": "unit_test_only"}])
    rows(directory / "decision_events.jsonl", [{"classification": "unit_test_only"}])
    rpc = []
    def call(op, args, result):
        identifier = len(rpc) // 2
        rpc.extend([{"direction": "request", "payload": {"id": identifier, "op": op, "args": args}},
                    {"direction": "response", "payload": {"id": identifier, "ok": True, "result": result}}])
    init = {"benchmark": "made", "seed": job["seed"], "budget": 50, "elements": job["task"]["elements"]}
    source = fixture["root"] / "frozen_sources" / ("collection-" + plan["fingerprint"][:20])
    execution = json.loads((source / "configs/main_protocol.json").read_text()).get("made_execution")
    if execution:
        init["mace_num_workers"] = execution["mace_num_workers"]
    call("init", init, {"initialized": True})
    for index in range(50):
        call("step", {}, {"observation": {"counts": {"candidate_oracle_attempts": index + 1}}})
    call("close", {}, {"closed": True, "counts": {"initialization_oracle_attempts": 2, "candidate_oracle_attempts": 50, "surrogate_oracle_attempts": 1}})
    rows(directory / "rpc/rpc.jsonl", rpc)
    rows(directory / "environment/env_events.jsonl", [{"sequence": 0, "episode_index": 0, "kind": "oracle_evaluation", "role": "mace", "result": {"energy": -1., "energy_per_atom": -1., "natoms": 1, "unit_test_only": True}}])
    shards = []
    for layer in range(32):
        path = directory / "activation_shards" / (str(layer) + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unit fixture, not an activation tensor")
        shards.append({"path": str(path), "layer_path": "model.language_model.layers." + str(layer) + ".mlp", "tensor_hash": fingerprint(layer)})
    write_json_atomic(directory / "collection_manifest.json", {"complete": True, "job_id": job["job_id"],
        "policy_configuration_fingerprint": identity["configuration_fingerprint"], "policy_runtime": identity["policy_runtime"],
        "decision_files": [artifact(directory / "decisions.jsonl")], "activation_shards": shards})
    if execution and execution["mace_num_workers"] == 4:
        write_unit_oracle_journal(directory)
    rewrite_receipt(directory, job, plan, identity)
    return directory


def rewrite_receipt(directory, job, plan, identity):
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "completion.json")
    write_json_atomic(directory / "completion.json", {"complete": True, "job_fingerprint": fingerprint(job),
        "plan_fingerprint": plan["fingerprint"], "policy_configuration_fingerprint": identity["configuration_fingerprint"],
        "artifacts": [artifact(p) for p in paths]})


def seal_lineage(f):
    record = f["lineage"]
    record["fingerprint"] = fingerprint({k: v for k, v in record.items() if k != "fingerprint"})
    write_json_atomic(f["lineage_path"], record)
    f["target_plan"]["lineage"] = {**artifact(f["lineage_path"]), "fingerprint": record["fingerprint"]}


def make_lineage_fixture(tmp_path, old_complete=2, source_schema=4):
    """Public helper for downstream CPU consumer integration tests."""
    root = Path(tmp_path).resolve()
    base = root / "experiments/collection/qwen35_4b/made"
    base.mkdir(parents=True)
    jobs = canonical_jobs()
    failure_control = json.loads((Path(__file__).parents[1] / "configs/main_protocol.json").read_text())["failure_control"]
    protocol = {"schema_version": source_schema, "amendments": ["unit-amendment-001", "unit-amendment-002"], "classification": "unit_test_only",
        "decoding": {"temperature": .6}, "memory": {"reset": True}, "collection": {"made": {"budget": 50}, "activation_token_rows_per_complete_decision_prefix": 16},
        "policy_runtime": {"dtype": "float32"}}
    if source_schema == 4:
        protocol["failure_control"] = failure_control
        protocol["amendments"].append("protocol_amendments/003_MADE_failure_type_control.md")
    for name in ("configs/model_manifest.json", "configs/benchmark_tasks.json", "configs/made_splits.json",
                 "configs/policy_runtime_gates.json", "configs/assets.runtime.json", "data/raw/materials_project/index.json"):
        write_json_atomic(root / name, {"classification": "unit_test_only", "name": name})
    write_json_atomic(root / "configs/main_protocol.json", protocol)
    for name in ("scripts/run_collection.py", "src/matdiscovery/fixture.py", "src/matdiscovery/rollouts.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# unit test source, never executed\n")
    old_plan = build_collection_plan(root, jobs)
    old_source = freeze(root, old_plan)
    identity = {"configuration_fingerprint": fingerprint("unit-config"), "checkpoint_hash": fingerprint("unit-checkpoint"),
                "policy_runtime": {"dtype": "torch.float32", "classification": "unit_test_only"}, "policy_configuration": {"unit": True}}
    write_json_atomic(base / "policy_configuration.json", identity)
    f = {"root": root, "base": base, "jobs": jobs, "old_plan": old_plan, "old_source": old_source, "policy_identity": identity}
    old_jobs = [complete_job(f, index, old_plan) for index in range(old_complete)]
    write_json_atomic(base / "plan.json", old_plan)
    progress = {"expected_jobs": 210, "completed_jobs": old_complete, "complete": False,
                "collection_manifests": [str(p / "collection_manifest.json") for p in old_jobs]}
    write_json_atomic(base / "progress.json", progress)
    history = base / "provenance/original"
    write_json_atomic(history / "plan.json", old_plan)
    write_json_atomic(history / "progress.json", progress)
    if source_schema == 3:
        protocol["amendments"].append("protocol_amendments/003_MADE_failure_type_control.md")
        protocol["failure_control"] = failure_control
    protocol.update(schema_version=5, amendments=protocol["amendments"] + ["protocol_amendments/004_MADE_surrogate_executor.md"], made_execution=EXECUTION_V1)
    write_json_atomic(root / "configs/main_protocol.json", protocol)
    (root / "src/matdiscovery/fixture.py").write_text("# unit test optimized execution source\n")
    target = build_collection_plan(root, jobs)
    new_source = freeze(root, target)
    archive = root / "experiments/interrupted/unit-transition"
    proof = archive / "partial/rpc.jsonl"
    rows(proof, [{"classification": "unit_test_only_stopped_rpc_fixture"}])
    reconciliation = {"schema": "interruption_reconciliation_v1", "all_requests_resolved": True,
        "previous_processes_stopped": [999999], "used_for_training": False, "used_for_final_evaluation": False,
        "additional_incurred_costs_not_subtracted_from_main_budgets": True, "observed_physical_costs": {"candidate_oracle_attempts": 0},
        "artifacts": [{"path_after": str(proof), "sha256": file_sha256(proof)}]}
    write_json_atomic(archive / "reconciliation.json", reconciliation)
    validation = history / "unit-validation.json"
    write_json_atomic(validation, {"classification": "unit_test_only", "scientific_oracle_calls": 0})
    review = {"schema": REVIEW_SCHEMA, "registered": True, "source_plan_fingerprint": old_plan["fingerprint"],
        "target_plan_fingerprint": target["fingerprint"], "matrix_fingerprint": fingerprint(jobs),
        "scientific_contract_fingerprint": fingerprint(scientific_contract(new_source, jobs)),
        "input_changes": [{"path": name, "before_sha256": old_plan["inputs"].get(name), "after_sha256": target["inputs"].get(name)} for name in sorted(set(old_plan["inputs"]) | set(target["inputs"])) if old_plan["inputs"].get(name) != target["inputs"].get(name)],
        "execution_before": None, "execution_after": EXECUTION_V1,
        "baseline_collection_binding": {"method": "baseline", "risk_model": None, "attributor": None,
            "source_inputs": {"before": {name: old_plan["inputs"][name] for name in ("scripts/run_collection.py", "src/matdiscovery/rollouts.py")},
                              "after": {name: target["inputs"][name] for name in ("scripts/run_collection.py", "src/matdiscovery/rollouts.py")}}},
        "mace_acceptance_scope": "prior_completed_jobs_without_mace_exceptions_or_unknown_calls", "validation_evidence": [artifact(validation)]}
    if source_schema == 3:
        prior = root / "logs/transitions/005_MADE_failure_method_enabled/transition.json"
        write_json_atomic(prior, {"classification": "unit_test_only_prior_corpus_registration"})
        review["prior_baseline_corpus_registration"] = artifact(prior)
    write_json_atomic(history / "review.json", review)
    lineage = {"schema": SCHEMA, "registration": "operator_recorded_execution_only_transition", "model_key": "qwen35_4b", "benchmark": "made",
        "expected_jobs": 210, "matrix_fingerprint": fingerprint(jobs), "target_plan_fingerprint": target["fingerprint"],
        "policy_identity": artifact(base / "policy_configuration.json"), "reconciliation": artifact(archive / "reconciliation.json"),
        "origins": [{"plan": artifact(history / "plan.json"), "progress": artifact(history / "progress.json"),
            "source_manifest": artifact(old_source / "source_manifest.json"), "compatibility_review": artifact(history / "review.json"),
            "accepted_jobs": [{"job_id": directory.name, "receipt_sha256": file_sha256(directory / "completion.json")} for directory in old_jobs]}]}
    f.update(target_plan=target, new_source=new_source, lineage=lineage, lineage_path=base / "provenance/lineage.json", old_jobs=old_jobs)
    seal_lineage(f)
    return f


def load(f):
    return load_collection_lineage(f["root"], f["base"], f["target_plan"], f["jobs"])


def test_two_old_complete_jobs_then_new_source_job_and_resume(tmp_path):
    f = make_lineage_fixture(tmp_path)
    old_bytes = [(p / "completion.json").read_bytes() for p in f["old_jobs"]]
    context = load(f)
    first = preflight_collection(f["base"], f["target_plan"], context)
    assert len(first) == 2 and all(x["source"]["accepted_via_lineage"] for x in first.values())
    assert all(x["source"]["mace_compatibility"]["unknown_mace_attempts"] == 0 for x in first.values())
    new = complete_job(f, 2, f["target_plan"])
    resumed = preflight_collection(f["base"], f["target_plan"], load(f))
    assert len(resumed) == 3 and not resumed[new.name]["source"]["accepted_via_lineage"]
    assert [(p / "completion.json").read_bytes() for p in f["old_jobs"]] == old_bytes
    assert len(f["target_plan"]["jobs"]) == 210


@pytest.mark.parametrize("mutation", ["partial", "artifact_tamper", "duplicate_artifact", "omitted_artifact", "unregistered_receipt", "cross_model", "rng", "budget", "runtime", "missing_layer", "mace_exception", "mace_unknown", "rpc_unclosed"])
def test_refuses_partial_tampered_or_scientifically_incompatible_jobs(tmp_path, mutation):
    f = make_lineage_fixture(tmp_path)
    directory, job = f["old_jobs"][0], f["jobs"][0]
    if mutation == "partial":
        (f["base"] / f["jobs"][3]["job_id"]).mkdir()
    elif mutation == "artifact_tamper":
        (directory / "decisions.jsonl").write_text("tampered\n")
    elif mutation == "unregistered_receipt":
        del f["target_plan"]["lineage"]
    else:
        if mutation in {"duplicate_artifact", "omitted_artifact"}:
            receipt = json.loads((directory / "completion.json").read_text())
            if mutation == "duplicate_artifact":
                receipt["artifacts"].append(receipt["artifacts"][0])
            else:
                receipt["artifacts"].pop(0)
            write_json_atomic(directory / "completion.json", receipt)
        elif mutation in {"cross_model", "rng"}:
            actual = json.loads((directory / "job.json").read_text())
            actual["model_key" if mutation == "cross_model" else "environment_seeds"] = "qwen35_9b" if mutation == "cross_model" else [5]
            write_json_atomic(directory / "job.json", actual)
        elif mutation == "budget":
            summaries = json.loads((directory / "episodes.json").read_text())
            summaries[0]["costs"]["candidate_oracle_attempts"] = 49
            write_json_atomic(directory / "episodes.json", summaries)
        elif mutation in {"runtime", "missing_layer"}:
            manifest = json.loads((directory / "collection_manifest.json").read_text())
            if mutation == "runtime":
                manifest["policy_runtime"]["dtype"] = "torch.bfloat16"
            else:
                manifest["activation_shards"].pop()
            write_json_atomic(directory / "collection_manifest.json", manifest)
        elif mutation == "mace_exception":
            rows(directory / "environment/env_events.jsonl", [{"sequence": 0, "role": "mace", "kind": "oracle_exception"}])
        elif mutation == "mace_unknown":
            rows(directory / "environment/env_events.jsonl", [{"sequence": 0, "kind": "initialized"}])
        elif mutation == "rpc_unclosed":
            lines = (directory / "rpc/rpc.jsonl").read_text().splitlines()
            (directory / "rpc/rpc.jsonl").write_text("\n".join(lines[:-1]) + "\n")
        if mutation not in {"duplicate_artifact", "omitted_artifact"}:
            rewrite_receipt(directory, job, f["old_plan"], f["policy_identity"])
        # Even explicitly re-registering rehashed data cannot bypass semantics.
        f["lineage"]["origins"][0]["accepted_jobs"][0]["receipt_sha256"] = file_sha256(directory / "completion.json")
        seal_lineage(f)
    with pytest.raises((CollectionProvenanceError, FileNotFoundError)):
        preflight_collection(f["base"], f["target_plan"], load(f))


@pytest.mark.parametrize("mutation", ["unsigned", "wrong_sha", "duplicate_job", "matrix_shrunk", "unclosed_reconciliation", "source_tamper", "review_omits_change", "policy_identity_tamper"])
def test_lineage_registration_is_not_a_plan_hash_bypass(tmp_path, mutation):
    f = make_lineage_fixture(tmp_path)
    if mutation == "unsigned":
        f["lineage"]["registration"] = "draft"
    elif mutation == "wrong_sha":
        f["target_plan"]["lineage"]["sha256"] = "0" * 64
    elif mutation == "duplicate_job":
        f["lineage"]["origins"][0]["accepted_jobs"].append(f["lineage"]["origins"][0]["accepted_jobs"][0])
    elif mutation == "matrix_shrunk":
        f["target_plan"]["jobs"] = f["jobs"][:-1]
    elif mutation == "unclosed_reconciliation":
        path = Path(f["lineage"]["reconciliation"]["path"])
        value = json.loads(path.read_text()); value["all_requests_resolved"] = False
        write_json_atomic(path, value); f["lineage"]["reconciliation"] = artifact(path)
    elif mutation == "source_tamper":
        (f["old_source"] / "src/matdiscovery/fixture.py").write_text("changed frozen source")
    elif mutation == "review_omits_change":
        path = Path(f["lineage"]["origins"][0]["compatibility_review"]["path"])
        value = json.loads(path.read_text()); value["input_changes"].pop()
        write_json_atomic(path, value); f["lineage"]["origins"][0]["compatibility_review"] = artifact(path)
    elif mutation == "policy_identity_tamper":
        write_json_atomic(f["base"] / "policy_configuration.json", {"changed": True})
    if mutation not in {"wrong_sha", "matrix_shrunk"}:
        seal_lineage(f)
    with pytest.raises(CollectionProvenanceError):
        load(f)


def test_cli_registration_requires_exact_sha_and_preserves_pointer(tmp_path):
    f = make_lineage_fixture(tmp_path, old_complete=1)
    plan = {k: v for k, v in f["target_plan"].items() if k != "lineage"}
    with pytest.raises(CollectionProvenanceError):
        load_collection_lineage(f["root"], f["base"], plan, f["jobs"], lineage_path=f["lineage_path"])
    context = load_collection_lineage(f["root"], f["base"], plan, f["jobs"], lineage_path=f["lineage_path"], expected_lineage_sha256=file_sha256(f["lineage_path"]))
    assert context.lineage == f["target_plan"]["lineage"]
    assert len(preflight_collection(f["base"], plan, context)) == 1


def test_no_lineage_current_source_still_accepts_normal_full_job(tmp_path):
    f = make_lineage_fixture(tmp_path)
    for directory in f["old_jobs"]:
        shutil.rmtree(directory)
    plan = {k: v for k, v in f["target_plan"].items() if k != "lineage"}
    complete_job(f, 0, plan)
    context = load_collection_lineage(f["root"], f["base"], plan, f["jobs"])
    assert len(preflight_collection(f["base"], plan, context)) == 1
    assert context.lineage is None


def test_registered_schema3_baseline_corpus_requires_transition005(tmp_path):
    f = make_lineage_fixture(tmp_path, source_schema=3)
    assert len(preflight_collection(f["base"], f["target_plan"], load(f))) == 2
    (f["root"] / "logs/transitions/005_MADE_failure_method_enabled/transition.json").write_text("changed registration")
    with pytest.raises(CollectionProvenanceError):
        load(f)


def test_schema3_registration_rejects_flat_or_unrelated_005_evidence(tmp_path):
    f = make_lineage_fixture(tmp_path, source_schema=3)
    review_path = Path(f["lineage"]["origins"][0]["compatibility_review"]["path"])
    review = json.loads(review_path.read_text())
    wrong_path = f["root"] / "logs/transitions/005_wrong_flat_path.json"
    shutil.copyfile(review["prior_baseline_corpus_registration"]["path"], wrong_path)
    review["prior_baseline_corpus_registration"] = artifact(wrong_path)
    write_json_atomic(review_path, review)
    f["lineage"]["origins"][0]["compatibility_review"] = artifact(review_path)
    seal_lineage(f)
    with pytest.raises(CollectionProvenanceError, match="exact registered transition005"):
        load(f)


def test_exact_003_and_global_collection_fields_are_locked(tmp_path):
    from matdiscovery.collection_provenance import _protocol_transition
    f = make_lineage_fixture(tmp_path, source_schema=3)
    before = json.loads((f["old_source"] / "configs/main_protocol.json").read_text())
    after = json.loads((f["new_source"] / "configs/main_protocol.json").read_text())
    after["failure_control"]["threshold"] = .2
    with pytest.raises(CollectionProvenanceError):
        _protocol_transition(before, after)
    path = f["new_source"] / "configs/main_protocol.json"
    changed = json.loads(path.read_text())
    changed["collection"]["activation_token_rows_per_complete_decision_prefix"] = 8
    write_json_atomic(path, changed)
    assert scientific_contract(f["old_source"], f["jobs"]) != scientific_contract(f["new_source"], f["jobs"])


def test_new_source_receipt_cannot_hide_legacy_executor(tmp_path):
    f = make_lineage_fixture(tmp_path)
    directory = complete_job(f, 2, f["target_plan"])
    records = [json.loads(line) for line in (directory / "rpc/rpc.jsonl").read_text().splitlines()]
    records[0]["payload"]["args"].pop("mace_num_workers")
    rows(directory / "rpc/rpc.jsonl", records)
    rewrite_receipt(directory, f["jobs"][2], f["target_plan"], f["policy_identity"])
    with pytest.raises(CollectionProvenanceError):
        preflight_collection(f["base"], f["target_plan"], load(f))


def test_runner_admission_preserves_history_before_new_source_publication(tmp_path):
    f = make_lineage_fixture(tmp_path, source_schema=3)
    original = {name: (f["base"] / name).read_bytes() for name in ("plan.json", "progress.json")}
    target = {k: v for k, v in f["target_plan"].items() if k != "lineage"}
    with pytest.raises(CollectionProvenanceError):
        prepare_collection_resume(f["root"], f["base"], target, f["jobs"])
    admission = prepare_collection_resume(f["root"], f["base"], target, f["jobs"],
        lineage_path=f["lineage_path"], expected_lineage_sha256=file_sha256(f["lineage_path"]))
    assert admission["publish_plan"] and len(admission["verified"]) == 2
    assert all((f["base"] / name).read_bytes() == value for name, value in original.items())
    write_json_atomic(f["base"] / "plan.json", admission["plan"])
    complete_job(f, 2, admission["plan"])
    resumed = prepare_collection_resume(f["root"], f["base"], target, f["jobs"])
    assert not resumed["publish_plan"] and len(resumed["verified"]) == 3
    assert (f["base"] / "provenance/original/plan.json").read_bytes() == original["plan.json"]


@pytest.mark.parametrize("name", ["plan.json", "progress.json"])
def test_runner_refuses_to_replace_unpreserved_current_history(tmp_path, name):
    f = make_lineage_fixture(tmp_path)
    path = f["base"] / name
    # Even semantically equal JSON cannot be called preserved original bytes.
    path.write_text(path.read_text() + "\n")
    with pytest.raises(CollectionProvenanceError):
        prepare_collection_resume(f["root"], f["base"], f["target_plan"], f["jobs"])


def test_parallel_complete_job_uses_real_journal_auditor_and_exact_counters(tmp_path):
    f = make_lineage_fixture(tmp_path)
    directory = complete_job(f, 2, f["target_plan"])
    result = verify_collection_completion(directory, f["jobs"][2], f["target_plan"], load(f))
    audit = result["source"]["oracle_attempt_journal"]
    assert audit["valid"] and audit["closed"] and audit["successful"]
    assert audit["counts"] == {"initialization_oracle_attempts": 2, "candidate_oracle_attempts": 50, "surrogate_oracle_attempts": 1}
    assert audit["episode_indices"] == [0]
    assert audit["journal"] == artifact(directory / "environment/oracle_attempts.jsonl")
    old = verify_collection_completion(f["old_jobs"][0], f["jobs"][0], f["target_plan"], load(f))
    assert old["source"]["oracle_attempt_journal"] is None
    assert old["source"]["mace_compatibility"]["mace_evaluations"] == 1


@pytest.mark.parametrize("failed_role", ["mace", "orb"])
def test_complete_recorded_failures_are_not_filtered_by_journal_gate(tmp_path, failed_role):
    f = make_lineage_fixture(tmp_path)
    directory = complete_job(f, 2, f["target_plan"])
    write_unit_oracle_journal(directory, failed_role=failed_role)
    if failed_role == "orb":
        rpc_path = directory / "rpc/rpc.jsonl"
        rpc_rows = [json.loads(line) for line in rpc_path.read_text().splitlines()]
        rpc_rows[3]["payload"] = {"id": 1, "ok": False, "error": {"code": "synthetic_unit_scientific_failure",
            "details": {"counts": {"candidate_oracle_attempts": 1}}}}
        rows(rpc_path, rpc_rows)
    rewrite_receipt(directory, f["jobs"][2], f["target_plan"], f["policy_identity"])
    result = verify_collection_completion(directory, f["jobs"][2], f["target_plan"], load(f))
    audit = result["source"]["oracle_attempt_journal"]
    assert audit["valid"] and audit["closed"] and not audit["successful"]
    assert audit["oracle_exceptions"] == audit["batch_exceptions"] == 1
    assert audit["counts"]["candidate_oracle_attempts"] == 50


@pytest.mark.parametrize("mutation", ["missing", "unclosed_attempt", "unclosed_batch", "duplicate", "partial_line",
    "wrong_episode", "constructor_wrong_episode", "initialization_count", "candidate_count", "surrogate_count", "not_receipt_bound"])
def test_parallel_journal_integrity_closure_episode_and_counter_gate(tmp_path, mutation):
    f = make_lineage_fixture(tmp_path)
    directory = complete_job(f, 2, f["target_plan"])
    path = directory / "environment/oracle_attempts.jsonl"
    original = [json.loads(line) for line in path.read_text().splitlines()]
    changed = deepcopy(original)
    if mutation == "missing":
        path.unlink()
    elif mutation in {"initialization_count", "candidate_count", "surrogate_count"}:
        write_unit_oracle_journal(directory, **{mutation.replace("_count", "_attempts"): 3 if mutation == "initialization_count" else 49 if mutation == "candidate_count" else 2})
    else:
        if mutation == "unclosed_attempt":
            changed = [row for row in changed if not (row["kind"] == "oracle_attempt_returned" and row["attempt_id"] == original[1]["attempt_id"])]
        elif mutation == "unclosed_batch":
            changed = changed[:-1]
        elif mutation == "duplicate":
            changed.insert(2, deepcopy(changed[1]))
        elif mutation == "wrong_episode":
            for row in changed:
                row["episode_index"] = 1
        elif mutation == "constructor_wrong_episode":
            for row in changed:
                if row["kind"].startswith("calculator_constructor_"):
                    row["episode_index"] = 1
        # Preserve contiguous sequence so missing-terminal tests exercise closure.
        for index, row in enumerate(changed):
            row["sequence"] = index
        rows(path, changed)
        if mutation == "partial_line":
            with path.open("a") as stream:
                stream.write('{"schema":')
    rewrite_receipt(directory, f["jobs"][2], f["target_plan"], f["policy_identity"])
    if mutation == "not_receipt_bound":
        receipt_path = directory / "completion.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["artifacts"] = [item for item in receipt["artifacts"] if Path(item["path"]) != path]
        write_json_atomic(receipt_path, receipt)
    with pytest.raises(CollectionProvenanceError):
        verify_collection_completion(directory, f["jobs"][2], f["target_plan"], load(f))
