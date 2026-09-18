"""Strict, read-only acceptance of explicitly registered collection source lineage.

This module never creates a registration, archives a job, changes a receipt, or
dispatches work. A source transition is accepted only after an operator has
published the complete hash-bound lineage/reconciliation described below.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Mapping

from .accounting import file_sha256, fingerprint

SCHEMA = "collection_source_lineage_v1"
REVIEW_SCHEMA = "collection_execution_compatibility_review_v1"
SHA = re.compile(r"[0-9a-f]{64}")
FAILURE_CONTROL_003_SHA256 = "25e6f97c42ffaddacb102af65cd77d3825f2fc92a3c9908b8f3ef4c5232bd019"
EXECUTION_V1 = {"schema": "fixed_mace_surrogate_executor_v1", "mace_num_workers": 4, "orb_num_workers": 1,
    "calculator_construction": "serial_per_composition_batch_before_parallel_evaluation", "scientific_parameters_unchanged": True,
    "result_order": "official_input_order", "exception_policy": "original_parallel_exception_semantics_all_started_attempts_accounted",
    "old_complete_collection": "source_bound_fixed_historical_train_dev_only",
    "final_paired_execution": "same_mace_executor_for_all_models_methods_and_seeds"}


class CollectionProvenanceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CollectionProvenanceError(message)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise CollectionProvenanceError("Bare nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def read_json(path):
    return _json(Path(path).read_text())


def _artifact(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def _verify_artifact(record):
    require(isinstance(record, Mapping) and isinstance(record.get("path"), str)
            and isinstance(record.get("sha256"), str) and SHA.fullmatch(record["sha256"]), "Malformed provenance artifact")
    actual = _artifact(record["path"])
    require(actual["sha256"] == record["sha256"], "Provenance file changed: " + record["path"])
    return actual


def _rows(path):
    raw = Path(path).read_bytes()
    require(bool(raw) and raw.endswith(b"\n"), "Missing/truncated JSONL evidence: " + str(path))
    rows = [_json(line) for line in raw.splitlines()]
    require(all(isinstance(row, dict) for row in rows), "Non-object JSONL record")
    return rows


def validate_collection_plan(plan, expected_jobs):
    """The complete canonical matrix is identical across all source revisions."""
    require(isinstance(plan, Mapping) and isinstance(plan.get("inputs"), dict), "Missing collection plan")
    require(plan.get("jobs") == expected_jobs and plan.get("expected_jobs") == len(expected_jobs), "Collection matrix changed or shrank")
    require(plan.get("fingerprint") == fingerprint({"jobs": plan["jobs"], "inputs": plan["inputs"]}), "Invalid plan fingerprint")
    require(bool(expected_jobs), "Empty collection matrix")
    identifiers = [j["job_id"] for j in expected_jobs]
    require(len(identifiers) == len(set(identifiers)), "Duplicate canonical collection job")
    for key in ("model_key", "model_id", "model_revision", "benchmark", "method"):
        require(len({j[key] for j in expected_jobs}) == 1, "Mixed collection condition: " + key)
    require(expected_jobs[0]["model_key"] in {"qwen35_4b", "qwen35_9b"} and expected_jobs[0]["method"] == "baseline", "Not the registered baseline condition")
    benchmark = expected_jobs[0]["benchmark"]
    require(len(expected_jobs) == (210 if benchmark == "made" else 150 if benchmark == "crystalgym" else -1), "Incomplete model/benchmark collection matrix")
    if benchmark == "made":
        for partition, size in (("train", 30), ("dev", 12)):
            subset = [j for j in expected_jobs if j["split"] == partition]
            require(len(subset) == size * 5 and len({j["task_id"] for j in subset}) == size, "Incomplete MADE task partition")
            for task in {j["task_id"] for j in subset}:
                these = [j for j in subset if j["task_id"] == task]
                require(sorted(j["seed"] for j in these) == [1, 2, 3, 4, 5], "MADE seed matrix changed")
        for job in expected_jobs:
            require(job["budget"] == 50 and job["environment_seeds"] == [job["seed"]] and job["episode_ids"] == ["0"]
                    and job["expected_counts"] == {"episodes": 1, "candidate_oracle_attempts": 50, "dft_episode_attempts": 0}, "MADE B50/RNG contract changed")
    return plan


def build_collection_plan(project, expected_jobs):
    """Prepare the exact target identity without loading any model or evaluator."""
    root = Path(project)
    require(bool(expected_jobs), "Cannot build an empty collection plan")
    names = ["configs/main_protocol.json", "configs/model_manifest.json", "configs/benchmark_tasks.json",
             "configs/made_splits.json", "configs/policy_runtime_gates.json", "scripts/run_collection.py"]
    names += [str(path.relative_to(root)) for path in sorted((root / "src/matdiscovery").rglob("*.py"))]
    names += ["configs/assets.runtime.json", "data/raw/materials_project/index.json"] if expected_jobs[0]["benchmark"] == "made" else ["configs/qe.runtime.json"]
    inputs = {name: file_sha256(root / name) for name in names}
    return validate_collection_plan({"jobs": expected_jobs, "inputs": inputs, "expected_jobs": len(expected_jobs),
        "fingerprint": fingerprint({"jobs": expected_jobs, "inputs": inputs})}, expected_jobs)


def scientific_contract(source_root, jobs):
    """Collection semantics; made_execution is separately reviewed, never hidden.

    The exact registered 003 type-control block is excluded only for baseline
    collection; it is separately checked by _protocol_transition and a source
    review binding both runners to absent controllers. No arbitrary later
    method/config additions are ignored.
    """
    source_root = Path(source_root)
    protocol = read_json(source_root / "configs/main_protocol.json")
    benchmark = jobs[0]["benchmark"]
    fixed_files = ["configs/model_manifest.json", "configs/benchmark_tasks.json", "configs/made_splits.json",
                   "configs/policy_runtime_gates.json"]
    fixed_files += ["configs/assets.runtime.json", "data/raw/materials_project/index.json"] if benchmark == "made" else ["configs/qe.runtime.json"]
    return {"schema": "baseline_collection_scientific_contract_v1", "matrix_fingerprint": fingerprint(jobs),
            "protocol": {k: v for k, v in protocol.items() if k not in {"schema_version", "amendments", "made_execution", "failure_control"}},
            "fixed_inputs": {name: file_sha256(source_root / name) for name in fixed_files}}


def _protocol_transition(before, after):
    require(before.get("schema_version") in {3, 4} and after.get("schema_version") == 5, "Only the registered protocol 3/4-to-5 transitions are supported")
    require("made_execution" not in before and after.get("made_execution") == EXECUTION_V1, "MACE executor transition differs from the fixed registered semantics")
    require(fingerprint(after.get("failure_control")) == FAILURE_CONTROL_003_SHA256, "Failure-control block differs from the exact registered amendment 003")
    additions = ["protocol_amendments/004_MADE_surrogate_executor.md"]
    if before["schema_version"] == 3:
        require("failure_control" not in before, "Original schema3 unexpectedly contains type control")
        additions.insert(0, "protocol_amendments/003_MADE_failure_type_control.md")
    else:
        require(before.get("failure_control") == after["failure_control"], "Existing amendment003 method configuration changed")
    require(isinstance(before.get("amendments"), list)
            and after.get("amendments") == before["amendments"] + additions,
            "Protocol amendment history was changed instead of appended")


def _source(record, plan, real_root):
    artifact = _verify_artifact(record)
    manifest = read_json(artifact["path"])
    require(manifest.get("plan_fingerprint") == plan["fingerprint"] and manifest.get("inputs") == plan["inputs"], "Source manifest differs from original plan")
    require(Path(manifest["original_project"]).resolve() == real_root, "Source belongs to another project")
    source_root = Path(artifact["path"]).parent
    require(source_root == real_root / "frozen_sources" / ("collection-" + plan["fingerprint"][:20]), "Source is not the preserved canonical frozen tree")
    for name, digest in plan["inputs"].items():
        require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts,
                "Unsafe source input path")
        require(file_sha256(source_root / name) == digest, "Frozen source/input changed: " + name)
    return artifact, source_root


def _input_changes(before, after):
    return [{"path": name, "before_sha256": before.get(name), "after_sha256": after.get(name)}
            for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name)]


@dataclass(frozen=True)
class CollectionLineage:
    active_plan_fingerprint: str
    source_manifest: dict
    lineage: dict | None
    accepted_jobs: dict
    policy_identity: dict | None
    provenance_files: tuple


def load_collection_lineage(project, base, active_plan, expected_jobs, *, lineage_path=None, expected_lineage_sha256=None):
    """Validate an existing explicit registration, including every prior source.

    ``active_plan.lineage`` is the downstream discovery pointer. The runner may
    instead supply --lineage and its exact --lineage-sha256 before publishing the
    new active plan. No supplied flag implies permission to invent a lineage.
    """
    validate_collection_plan(active_plan, expected_jobs)
    base = Path(base).resolve()
    real_root = (Path(project).resolve() / "experiments").resolve().parent
    expected_base = real_root / "experiments/collection" / expected_jobs[0]["model_key"] / expected_jobs[0]["benchmark"]
    require(base == expected_base, "Collection directory differs from canonical condition")
    source_path = real_root / "frozen_sources" / ("collection-" + active_plan["fingerprint"][:20]) / "source_manifest.json"
    active_source, active_source_root = _source(_artifact(source_path), active_plan, real_root)
    pointer = active_plan.get("lineage")
    if lineage_path is not None:
        require(expected_lineage_sha256 is not None, "Explicit lineage requires its registered SHA256")
        supplied = {"path": str(Path(lineage_path).resolve()), "sha256": expected_lineage_sha256}
        require(pointer is None or (Path(pointer["path"]).resolve() == Path(supplied["path"]) and pointer["sha256"] == supplied["sha256"]), "CLI lineage differs from plan registration")
        pointer = supplied
    else:
        require(expected_lineage_sha256 is None, "Lineage SHA supplied without a path")
    if pointer is None:
        return CollectionLineage(active_plan["fingerprint"], active_source, None, {}, None, (active_source,))
    lineage_artifact = _verify_artifact(pointer)
    require(base / "provenance" in Path(lineage_artifact["path"]).parents, "Lineage must be explicitly registered under condition/provenance")
    record = read_json(lineage_artifact["path"])
    require(record.get("schema") == SCHEMA and record.get("fingerprint") == fingerprint({k: v for k, v in record.items() if k != "fingerprint"}), "Invalid lineage schema/fingerprint")
    require(pointer.get("fingerprint", record["fingerprint"]) == record["fingerprint"], "Plan lineage fingerprint mismatch")
    require(record.get("registration") == "operator_recorded_execution_only_transition", "Lineage lacks the explicit operator execution record")
    require(record.get("model_key") == expected_jobs[0]["model_key"] and record.get("benchmark") == expected_jobs[0]["benchmark"] == "made", "Cross-model/benchmark lineage")
    require(record.get("target_plan_fingerprint") == active_plan["fingerprint"] and record.get("matrix_fingerprint") == fingerprint(expected_jobs)
            and record.get("expected_jobs") == len(expected_jobs), "Lineage target/matrix differs")
    policy_file = _verify_artifact(record["policy_identity"])
    policy_identity = read_json(policy_file["path"])
    require(policy_file["path"] == str((base / "policy_configuration.json").resolve()), "Lineage policy identity is not the collection identity")
    require(policy_identity.get("policy_runtime") and policy_identity.get("configuration_fingerprint") and policy_identity.get("checkpoint_hash"), "Missing policy/runtime identity")
    reconciliation = _verify_artifact(record["reconciliation"])
    stopped = read_json(reconciliation["path"])
    require(stopped.get("schema") == "interruption_reconciliation_v1" and stopped.get("all_requests_resolved") is True
            and bool(stopped.get("previous_processes_stopped")) and stopped.get("used_for_training") is False
            and stopped.get("used_for_final_evaluation") is False and stopped.get("additional_incurred_costs_not_subtracted_from_main_budgets") is True,
            "Source switch requires closed/stopped and cost-preserving reconciliation")
    archived = stopped.get("artifacts", [])
    require(bool(archived) and isinstance(stopped.get("observed_physical_costs"), dict), "Reconciliation omits raw evidence/costs")
    archived_paths = set()
    evidence = [active_source, lineage_artifact, policy_file, reconciliation]
    for item in archived:
        artifact = _verify_artifact({"path": item["path_after"], "sha256": item["sha256"]})
        require(artifact["path"] not in archived_paths and Path(reconciliation["path"]).parent in Path(artifact["path"]).parents, "Duplicate/outside archived evidence")
        archived_paths.add(artifact["path"])
        evidence.append(artifact)
    contract = scientific_contract(active_source_root, expected_jobs)
    accepted, plans = {}, set()
    origins = record.get("origins")
    require(isinstance(origins, list) and origins, "Lineage must enumerate prior plans and accepted jobs")
    for origin in origins:
        plan_file, progress_file = _verify_artifact(origin["plan"]), _verify_artifact(origin["progress"])
        require(base / "provenance" in Path(plan_file["path"]).parents and base / "provenance" in Path(progress_file["path"]).parents,
                "Original plan/progress history must be preserved under provenance")
        old = validate_collection_plan(read_json(plan_file["path"]), expected_jobs)
        old_fp = old["fingerprint"]
        require(old_fp not in plans and old_fp != active_plan["fingerprint"], "Duplicate/self-referential origin plan")
        plans.add(old_fp)
        source, source_root = _source(origin["source_manifest"], old, real_root)
        require(scientific_contract(source_root, expected_jobs) == contract, "Collection scientific/model/data/runtime contract changed")
        review_file = _verify_artifact(origin["compatibility_review"])
        review = read_json(review_file["path"])
        changes = _input_changes(old["inputs"], active_plan["inputs"])
        require(review.get("schema") == REVIEW_SCHEMA and review.get("registered") is True
                and review.get("source_plan_fingerprint") == old_fp and review.get("target_plan_fingerprint") == active_plan["fingerprint"]
                and review.get("matrix_fingerprint") == fingerprint(expected_jobs)
                and review.get("scientific_contract_fingerprint") == fingerprint(contract)
                and review.get("input_changes") == changes,
                "Compatibility review does not bind every exact source/input change")
        old_protocol, new_protocol = read_json(source_root / "configs/main_protocol.json"), read_json(active_source_root / "configs/main_protocol.json")
        old_execution, new_execution = old_protocol.get("made_execution"), new_protocol.get("made_execution")
        _protocol_transition(old_protocol, new_protocol)
        require(review.get("execution_before") == old_execution and review.get("execution_after") == new_execution,
                "Execution optimization was omitted/misrepresented in review")
        support = review.get("validation_evidence")
        require(isinstance(support, list) and support and review.get("mace_acceptance_scope") == "prior_completed_jobs_without_mace_exceptions_or_unknown_calls", "Missing bounded MACE compatibility validation")
        baseline = {"method": "baseline", "risk_model": None, "attributor": None,
            "source_inputs": {"before": {name: old["inputs"][name] for name in ("scripts/run_collection.py", "src/matdiscovery/rollouts.py")},
                              "after": {name: active_plan["inputs"][name] for name in ("scripts/run_collection.py", "src/matdiscovery/rollouts.py")}}}
        require(review.get("baseline_collection_binding") == baseline, "Review must bind original and new baseline runners with no risk model/attributor")
        if old_protocol["schema_version"] == 3:
            registration = _verify_artifact(review["prior_baseline_corpus_registration"])
            require(Path(registration["path"]) == real_root / "logs/transitions/005_MADE_failure_method_enabled/transition.json",
                    "Schema3 corpus requires the exact registered transition005 evidence")
            evidence.append(registration)
        evidence.extend(_verify_artifact(item) for item in support)
        progress = read_json(progress_file["path"])
        jobs = origin.get("accepted_jobs")
        require(isinstance(jobs, list) and jobs, "Origin lacks accepted job inventory")
        origin_ids = []
        for item in jobs:
            job_id = item["job_id"]
            require(job_id not in accepted and job_id in {j["job_id"] for j in expected_jobs}, "Duplicated/unknown accepted job")
            require(isinstance(item.get("receipt_sha256"), str) and SHA.fullmatch(item["receipt_sha256"]), "Accepted job lacks original receipt hash")
            origin_ids.append(job_id)
            accepted[job_id] = {"receipt_sha256": item["receipt_sha256"], "plan_fingerprint": old_fp,
                "source_manifest": source, "original_plan": plan_file, "original_progress": progress_file, "compatibility_review": review_file}
        names = progress.get("collection_manifests")
        require(isinstance(names, list) and progress.get("expected_jobs") == len(expected_jobs)
                and progress.get("completed_jobs") == len(names) and len(names) == len(set(names)), "Original progress history is inconsistent")
        progressed_ids = [Path(name).parent.name for name in names]
        require(len(progressed_ids) == len(set(progressed_ids))
                and all(Path(name).resolve() == base / Path(name).parent.name / "collection_manifest.json" for name in names), "Historical progress aliases or points outside the original condition")
        require(set(progressed_ids) <= set(origin_ids), "Historical progress includes unregistered complete jobs")
        require(sorted(origin.get("progress_unacknowledged_job_ids", [])) == sorted(set(origin_ids) - set(progressed_ids)), "Receipt/progress gap was not explicitly reconciled")
        evidence.extend([plan_file, progress_file, source, review_file])
    return CollectionLineage(active_plan["fingerprint"], active_source,
        {**lineage_artifact, "fingerprint": record["fingerprint"]}, accepted, policy_identity, tuple(evidence))


def _rpc_completion(path, job, *, expected_mace_workers=None):
    rows = _rows(path)
    require(len(rows) % 2 == 0, "Completed job contains an unresolved RPC")
    pairs = []
    for i in range(0, len(rows), 2):
        request, response = rows[i], rows[i + 1]
        require(request.get("direction") == "request" and response.get("direction") == "response", "Non-paired completed RPC log")
        a, b = request["payload"], response["payload"]
        require(a.get("id") == b.get("id") == i // 2 and type(b.get("ok")) is bool, "Duplicate/missing completed RPC ID")
        pairs.append((a, b))
    require(pairs and pairs[0][0]["op"] == "init" and pairs[0][1]["ok"] is True
            and pairs[-1][0]["op"] == "close" and pairs[-1][1]["ok"] is True, "Completed collection RPC was not initialized/closed")
    init = pairs[0][0]["args"]
    require(init.get("benchmark") == job["benchmark"] and init.get("seed") == job["environment_seeds"][0] and init.get("budget") == job["budget"], "Raw RPC init changes benchmark/RNG/budget")
    if job["benchmark"] == "made":
        require(type(init.get("mace_num_workers", 1)) is int and init.get("mace_num_workers", 1) == expected_mace_workers, "Actual RPC MACE executor differs from its source protocol")
        require(init.get("elements") == job["task"]["elements"] and sum(a["op"] == "step" for a, b in pairs) == 50, "Completed MADE RPC task/B50 differs")
        counts = pairs[-1][1].get("result", {}).get("counts", {})
        require(counts.get("candidate_oracle_attempts") == 50, "Closed RPC lacks complete actual B50 counter")
    return pairs


def _mace_compatibility(directory, artifact_paths, episodes):
    path = (directory / "environment/env_events.jsonl").resolve()
    require(path in artifact_paths, "Old job MACE evidence is absent from receipt")
    rows = _rows(path)
    require([row.get("sequence") for row in rows] == list(range(len(rows))), "Incomplete/duplicated environment event sequence")
    mace = [row for row in rows if row.get("role") == "mace"]
    require(all(row.get("kind") == "oracle_evaluation" for row in mace), "Old job contains a MACE exception/unknown event")
    for row in mace:
        value = row.get("result")
        require(isinstance(value, dict) and type(value.get("natoms")) is int and value["natoms"] > 0
                and all(type(value.get(key)) in (int, float) and math.isfinite(value[key]) for key in ("energy", "energy_per_atom")),
                "Old job MACE return is missing/nonfinite; outcome is not a supported successful evaluation")
    expected = sum(row["costs"].get("surrogate_oracle_attempts", 0) for row in episodes)
    require(len(mace) == expected, "Old job has unmatched/unknown MACE attempts")
    return {"env_events": _artifact(path), "mace_evaluations": len(mace), "mace_exceptions": 0,
            "unknown_mace_attempts": 0, "scope": "this completed source job only; no claim of equivalent failure paths"}


def _parallel_attempt_journal(directory, artifact_paths, episodes, rpc_counts):
    """Admit complete scientific records, including faithfully recorded failures."""
    from .mace_parallel import audit_oracle_attempt_journal

    path = (directory / "environment/oracle_attempts.jsonl").resolve()
    require(path in artifact_paths, "Parallel collection lacks a receipt-bound oracle-attempt journal")
    audited = audit_oracle_attempt_journal(path)
    require(audited.get("valid") is True and audited.get("closed") is True
            and audited.get("counts_reliable") is True, "Oracle-attempt journal is invalid or has an unresolved lifecycle")
    records = _rows(path)
    require(all(type(row.get("episode_index")) is int and row["episode_index"] == 0 for row in records),
            "MADE B50 journal mixes episode indices")
    require(all(row.get("num_workers") == (4 if row.get("role") == "mace" else 1)
                for row in audited["starts"]), "Journal execution workers differ from the source protocol")
    counters = ("initialization_oracle_attempts", "candidate_oracle_attempts", "surrogate_oracle_attempts")
    for key in counters:
        costs = [row["costs"].get(key, 0) for row in episodes]
        require(all(type(value) is int and value >= 0 for value in costs), "Malformed episode physical count: " + key)
        require(audited["counts"].get(key, 0) == sum(costs) == rpc_counts.get(key, 0),
                "Oracle journal/episode/closed-RPC physical counts disagree: " + key)
    artifact = _artifact(path)
    require(artifact["sha256"] == audited["sha256"], "Oracle journal changed during its audit")
    return {"journal": artifact, "audit_schema": audited["schema"], "valid": True, "closed": True,
            "successful": audited["successful"], "counts": audited["counts"],
            "by_role_phase_episode": audited["by_role_phase_episode"], "episode_indices": [0],
            "oracle_exceptions": len(audited["oracle_errors"]), "constructor_exceptions": len(audited["constructor_errors"]),
            "batch_exceptions": len(audited["batch_errors"]),
            "acceptance_rule": "valid_and_closed_with_matched_actual_counts; recorded_failures_are_retained"}


def verify_collection_completion(directory, job, active_plan, lineage_context, *, policy_configuration_fingerprint=None):
    """Verify a complete current-source or explicitly registered original job.

    Return receipt, episodes, collection_manifest, source and evidence_files.
    Does not modify old bytes or silently promote a partial job to complete.
    """
    directory = Path(directory).resolve()
    context = lineage_context
    require(context.active_plan_fingerprint == active_plan["fingerprint"] and directory.name == job["job_id"], "Completion/context identity mismatch")
    receipt_path = directory / "completion.json"
    receipt = read_json(receipt_path)
    receipt_file = _artifact(receipt_path)
    require(receipt.get("complete") is True and receipt.get("job_fingerprint") == fingerprint(job), "Receipt is incomplete or belongs to another canonical job")
    origin = context.accepted_jobs.get(job["job_id"])
    actual_plan = receipt.get("plan_fingerprint")
    if origin:
        require(actual_plan == origin["plan_fingerprint"] and receipt_file["sha256"] == origin["receipt_sha256"], "Registered original receipt changed or was replaced")
    else:
        require(actual_plan == active_plan["fingerprint"], "Unregistered old source/plan completion")
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, list) and artifacts, "Completion lacks original artifacts")
    paths, evidence = set(), [receipt_file]
    for item in artifacts:
        artifact = _verify_artifact(item)
        path = Path(artifact["path"])
        require(path not in paths and directory in path.parents and path != receipt_path, "Duplicated/outside/self-referential raw completion artifact")
        paths.add(path)
        evidence.append(artifact)
    required = ["job.json", "episodes.json", "collection_manifest.json", "decisions.jsonl", "decision_events.jsonl", "rpc/rpc.jsonl"]
    require(all((directory / name).resolve() in paths for name in required), "Completion omitted required raw evidence")
    # The one predefined offline sidecar is added only after collection. All
    # other files are original raw outputs and must remain in the receipt.
    raw_inventory = {p.resolve() for p in directory.rglob("*") if p.is_file()
                     and p != receipt_path and p != directory / "graph_features.jsonl"}
    require(paths == raw_inventory, "Completion receipt does not cover the complete original raw file inventory")
    actual_job = read_json(directory / "job.json")
    require(all(actual_job.get(k) == value for k, value in job.items()), "Recorded job task/model/RNG differs from canonical job")
    policy_identity = read_json(directory.parent / "policy_configuration.json")
    if context.policy_identity is not None:
        require(policy_identity == context.policy_identity, "Collection policy identity changed after lineage registration")
    configured = policy_identity.get("configuration_fingerprint")
    require(configured and policy_identity.get("policy_runtime") and policy_identity.get("checkpoint_hash"), "Missing full runtime/checkpoint provenance")
    require(receipt.get("policy_configuration_fingerprint") == configured == actual_job.get("policy_configuration_fingerprint"), "Job/receipt policy configuration mismatch")
    require(policy_configuration_fingerprint is None or configured == policy_configuration_fingerprint, "Current loaded policy differs from original collection")
    manifest = read_json(directory / "collection_manifest.json")
    require(manifest.get("complete") is True and manifest.get("job_id") == job["job_id"]
            and manifest.get("policy_configuration_fingerprint") == configured
            and manifest.get("policy_runtime") == policy_identity["policy_runtime"], "Manifest is incomplete or changed policy runtime")
    decisions = manifest.get("decision_files")
    require(isinstance(decisions, list) and decisions and len({Path(x["path"]).resolve() for x in decisions}) == len(decisions), "Invalid decision-file inventory")
    for item in decisions:
        verified = _verify_artifact(item)
        require(Path(verified["path"]) in paths, "Manifest decision file omitted from receipt")
    shards = manifest.get("activation_shards")
    require(isinstance(shards, list) and len(shards) == 32 and len({x["layer_path"] for x in shards}) == 32,
            "Collection must preserve all 32 activation shards")
    require(all(Path(x["path"]).resolve() in paths for x in shards), "Activation shard omitted from raw receipt")
    episodes = read_json(directory / "episodes.json")
    require(isinstance(episodes, list) and len(episodes) == job["expected_counts"]["episodes"]
            and {x["episode_id"] for x in episodes} == set(job["episode_ids"]), "Incomplete/duplicate episode matrix")
    for episode in episodes:
        ordinal = job["episode_ids"].index(episode["episode_id"])
        require(episode.get("complete") is True and all(episode.get(k) == job[k] for k in ("benchmark", "model_key", "method", "task_id", "seed"))
                and episode.get("environment_seed") == job["environment_seeds"][ordinal], "Episode identity/RNG mismatch")
    for key in ("candidate_oracle_attempts", "dft_episode_attempts"):
        require(sum(x["costs"].get(key, 0) for x in episodes) == job["expected_counts"][key], "Completed physical budget differs: " + key)
    actual_source = origin["source_manifest"] if origin else context.source_manifest
    source_protocol = read_json(Path(actual_source["path"]).parent / "configs/main_protocol.json")
    expected_workers = source_protocol.get("made_execution", {}).get("mace_num_workers", 1)
    pairs = _rpc_completion(directory / "rpc/rpc.jsonl", job, expected_mace_workers=expected_workers)
    journal = None
    if job["benchmark"] == "made":
        final_counts = pairs[-1][1]["result"]["counts"]
        require(sum(row["costs"].get("surrogate_oracle_attempts", 0) for row in episodes) == final_counts.get("surrogate_oracle_attempts", 0), "Episode and closed RPC MACE counts disagree")
        if expected_workers == 4:
            journal = _parallel_attempt_journal(directory, paths, episodes, final_counts)
    mace = _mace_compatibility(directory, paths, episodes) if origin else None
    require(file_sha256(receipt_path) == receipt_file["sha256"], "Receipt changed during verification")
    return {"receipt": receipt, "episodes": episodes, "collection_manifest": manifest,
            "source": {"plan_fingerprint": actual_plan, "source_manifest": origin["source_manifest"] if origin else context.source_manifest,
                       "lineage": context.lineage, "accepted_via_lineage": origin is not None, "receipt": receipt_file,
                       "mace_compatibility": mace, "oracle_attempt_journal": journal},
            "evidence_files": evidence + list(context.provenance_files)}


def preflight_collection(base, plan, context):
    """Validate all extant jobs before model loading or any new physical action."""
    base = Path(base).resolve()
    expected = {job["job_id"]: job for job in plan["jobs"]}
    complete = {}
    for directory in sorted(base.glob("collection-*")):
        require(not directory.is_symlink() and directory.is_dir() and directory.name in expected, "Unknown/aliased collection output")
        require((directory / "completion.json").is_file(), "Partial job requires stopped-RPC archive reconciliation; no automatic resume: " + str(directory))
        complete[directory.name] = verify_collection_completion(directory, expected[directory.name], plan, context)
    require(set(context.accepted_jobs) <= set(complete), "A registered original completed job is missing")
    return complete


def prepare_collection_resume(project, base, target_plan, expected_jobs, *, lineage_path=None, expected_lineage_sha256=None):
    """Read-only runner preflight; publication occurs only after this succeeds.

    Returns plan/context/verified/publish_plan. Original plan/progress must have
    already been preserved byte-for-byte by the operator; this function never
    manufactures history or registration for replacing them.
    """
    base = Path(base).resolve()
    plan_path = base / "plan.json"
    previous = read_json(plan_path) if plan_path.exists() else None
    plan = previous if previous is not None and previous.get("fingerprint") == target_plan["fingerprint"] else target_plan
    context = load_collection_lineage(project, base, plan, expected_jobs,
        lineage_path=lineage_path, expected_lineage_sha256=expected_lineage_sha256)
    if context.lineage is not None:
        plan = {**plan, "lineage": context.lineage}
    if previous is not None and previous != plan:
        origins = list(context.accepted_jobs.values())
        require(origins and file_sha256(plan_path) in {row["original_plan"]["sha256"] for row in origins},
                "Cannot replace an unregistered/unpreserved original collection plan")
        progress_path = base / "progress.json"
        require(progress_path.is_file() and file_sha256(progress_path) in {row["original_progress"]["sha256"] for row in origins},
                "Cannot replace unpreserved original progress history")
    verified = preflight_collection(base, plan, context)
    return {"plan": plan, "context": context, "verified": verified, "publish_plan": previous != plan}
