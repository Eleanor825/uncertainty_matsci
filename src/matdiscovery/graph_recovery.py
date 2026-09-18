"""Pre-test causal feature selection with explicit unchanged-bank reuse.

V3's failed graph journals stay immutable. V4 derives the same four closed
physical trajectories into new graph directories, verifies the successful V3
bank under its original frozen implementation, and never trains it again.
"""
from __future__ import annotations

import copy
import math
import numbers
from pathlib import Path
import shutil

import numpy as np

from .accounting import file_sha256, fingerprint, write_json_atomic
from .collection_provenance import read_json
from .core_protocol import (SCHEMA, NORMALIZED_REGISTRATION, NORMALIZED_RECIPE,
    CAUSAL_GRAPH_REGISTRATION, CAUSAL_GRAPH_REGISTRATIONS, FAST_CAUSAL_GRAPH_REGISTRATION,
    CAUSAL_GRAPH_RULE, core_execution_budget, _core_hash, _derived_main,
    artifact, collection_jobs, collection_manifest_paths, final_jobs, read_core,
    require, verify_artifact)
from .tc64_recovery import _pretest, _verify_reused_collection, _import_reused_collections


AMENDMENT_SPEC = {
    "schema": "pretest_qwen_causal_feature_eligibility_v1",
    "from_source_selection_rule": "global_activation",
    "to_source_selection_rule": CAUSAL_GRAPH_RULE,
    "target": "exclude_only_provably_unreachable_final_MLP_positions_before_existing_activation_ranking",
    "full_prefix_preserved": True, "all_decisions_replayed_in_new_namespace": True,
    "original_coverage_denominators_preserved": True,
    "model_runtime_evaluator_chemistries_seeds_B50_ES_and_tests_unchanged": True,
    "native_epsilon": .001, "native_rtol": .05, "native_atol": .0001,
    "transcoder_training_calls": 0, "reuse_completed_normalized_bank": True,
    "test_or_ES_outcomes_used": False,
}
FAST_AMENDMENT_SPEC = {key: value for key, value in AMENDMENT_SPEC.items()
    if key != "model_runtime_evaluator_chemistries_seeds_B50_ES_and_tests_unchanged"}
FAST_AMENDMENT_SPEC.update(schema="pretest_qwen_causal_feature_eligibility_B10_subset_v1",
    model_runtime_evaluator_chemistries_seeds_ES_formula_unchanged=True,
    historical_collection_budget=50, new_ES_and_test_episode_budget=10,
    historical_candidate_calls=200, new_ES_candidate_calls=60, new_test_candidate_calls=40,
    development_generations_observed=[1, 2], generation_zero_full_method_evaluation_available=False,
    monotonic_improvement_required_or_claimed=False,
    zero_update_outcome="complete_with_no_evolution_signal_not_evidence_of_effectiveness")


def _amendment_spec(registration):
    return FAST_AMENDMENT_SPEC if registration == FAST_CAUSAL_GRAPH_REGISTRATION else AMENDMENT_SPEC


def _probe(path, *, source_project, source_core, bank_reference):
    value = read_json(path)
    require(value.get("fingerprint") == fingerprint({k: v for k, v in value.items() if k != "fingerprint"})
        and value.get("schema") == "causal_native_single_prefix_validation_v1"
        and value.get("complete") is True and value.get("passed") is True
        and value.get("source_selection_rule") == CAUSAL_GRAPH_RULE
        and value.get("source_core_fingerprint") == source_core["fingerprint"]
        and value.get("source_bank_manifest") == bank_reference
        and value.get("new_policy_or_oracle_calls") == 0,
        "Causal eligibility requires a real, source-bound, successful unchanged-bank probe")
    native = Path(source_project) / "src/matdiscovery/native_attribution.py"
    require(value["source_sha256"]["native_attribution.py"] == file_sha256(native),
            "The real native probe validated another implementation")
    if "workspace" in source_core:
        # The real read_core admission always has a workspace. Tiny unit fixtures
        # isolate this ancestry; they cannot be used by prepare_causal_workspace.
        reference = value.get("staged_source_manifest")
        require(isinstance(reference, dict), "Real probe omitted its staged numerical source manifest")
        verify_artifact(reference)
        delta = read_json(reference["path"])
        require(delta["source_core"] == source_core["workspace"]
            and delta["all_other_source_bytes_equal"] is True
            and set(delta["delta"]) == {"src/matdiscovery/native_attribution.py"},
            "The successful probe changed more than the declared native selection source")
        for name in ("native_attribution.py", "policy.py", "action_targets.py", "transcoders.py", "parallel_normalized_bank.py"):
            relative = "src/matdiscovery/" + name
            require(value["source_sha256"][name] == delta["new_files"][relative]
                == file_sha256(Path(source_project) / relative), "Probe calculation source differs: " + name)
        for name in ("policy.py", "attention_checkpointing.py", "action_targets.py", "transcoders.py", "parallel_normalized_bank.py"):
            relative = "src/matdiscovery/" + name
            require(delta["old_files"][relative] == delta["new_files"][relative]
                == file_sha256(Path(source_core["workspace"]) / relative)
                == file_sha256(Path(source_project) / relative), "Unregistered numerical source change: " + name)
    require(value["validator_kwargs"] == {"epsilon": .001, "rtol": .05, "atol": .0001, "max_edges": 4},
            "Native probe relaxed the original finite-difference settings")
    validation = value["validation"]
    require(validation["passed"] is True and len(validation["checks"]) == 4
        and all(row["passed"] is True for row in validation["checks"])
        and validation["epsilon"] == .001 and validation["rtol"] == .05 and validation["atol"] == .0001
        and validation["required_direct_action_sink_source_kinds"] == ["token", "feature"],
        "Causal probe must pass both actual token and feature action-score finite differences")
    require(value.get("nonzero_feature_to_score_count", 0) > 0
        and value.get("reachable_feature_count", 0) > 0,
        "New research graph still contains no feature connected to its action score")
    metadata = value["attribution_metadata"]
    require(metadata["selected_features"] == CAUSAL_GRAPH_REGISTRATION["graph"]["max_feature_nodes"]
        and metadata["sink_count"] == 1
        and metadata["backward_targets"] == metadata["selected_features"] + metadata["sink_count"]
        <= CAUSAL_GRAPH_REGISTRATION["graph"]["max_backward_targets"]
        and metadata["backend_validation"] == validation,
        "Real probe omitted the registered production graph budget/backend validation")
    targets = value["targets"]
    require(metadata["full_prefix_hash"] == validation["prefix_hash"] == value["source_prefix_hash"] == targets["source_prefix_hash"]
        and metadata["full_prefix_tokens"] == targets["source_length"]
        and metadata["target_spec_hash"] == validation["target_spec_hash"] == targets["target_spec_hash"]
        and metadata["causal_horizon"] == targets["causal_horizon"] == max(targets["prediction_positions"])
        and metadata["source_selection_rule"] == validation["source_selection_rule"] == CAUSAL_GRAPH_RULE
        and metadata["backend_signature"] == validation["backend_signature"]
        and metadata["runtime"] == validation["runtime"] == value["actual_attribution_runtime"]
        and metadata["source_selection_proof"] == validation["source_selection_proof"] == value["actual_source_selection_proof"],
        "Probe FD, production graph, and registered action use different targets/runtime/selection")
    def finite(number):
        return isinstance(number, numbers.Real) and not isinstance(number, bool) and math.isfinite(number)
    required_sources = set()
    for row in validation["checks"]:
        analytical, numerical = row["analytical"], row["finite_difference"]
        require(finite(analytical) and finite(numerical)
            and bool(np.isclose(numerical, analytical, rtol=.05, atol=.0001)),
            "Claimed FD success fails the original numerical comparison")
        if row["target"].get("kind") == "score":
            require(row["target"].get("target_spec_hash") == targets["target_spec_hash"], "FD edge points to another action score")
            if analytical != 0:
                required_sources.add(row["source"]["kind"])
    require({"token", "feature"} <= required_sources, "Missing actual nonzero token/feature edge to this action score")
    require(finite(metadata["future_source_gradient_max_abs"]) and 0 <= metadata["future_source_gradient_max_abs"] <= 1e-7,
            "Future action tokens have a noncausal gradient")
    require(all(finite(item["forward_parity_max_abs"]) and item["forward_parity_max_abs"] >= 0 for item in (metadata, validation))
        and metadata["forward_parity_max_abs"] == validation["forward_parity_max_abs"], "Invalid/inconsistent forward parity evidence")
    require(len(metadata["fidelity"]) == 32 and all(finite(item["output_fvu"]) and item["output_fvu"] <= .5
        and not item["fvu_undefined"] for item in metadata["fidelity"].values()), "The real full-prefix TC fidelity gate did not pass")
    require(0 < value["nonzero_feature_to_score_count"] <= 32 and 0 < value["reachable_feature_count"] <= 32
        and value["completed_validate_backend_calls"] == value["completed_attribute_calls"] == 1,
        "Invalid production feature/call accounting")
    return value


def _implementation_validation(path, source_project):
    require(path is not None, "The causal core requires its complete source-bound CPU regression")
    value = read_json(path)
    counts = value.get("counts", {})
    require(value.get("schema") == "MADE_method_implementation_validation_v1" and value.get("passed") is True
        and counts.get("tests", 0) > 0 and counts.get("failed") == counts.get("errors") == 0
        and value.get("source_identity_policy") == "reviewed_single_native_source_delta_with_unchanged_cut_vjp_fd_ast"
        and value.get("source_bound_real_gpu_fd_required") is True
        and value.get("source_bound_real_gpu_fd_passed") is False,
        "Incomplete CPU regression or missing explicit causal source review")
    verify_artifact(value["causal_source_selection_proof"])
    project = Path(source_project)
    actual = {str(p.relative_to(project)): file_sha256(p) for directory in ("src", "scripts")
        for p in (project / directory).rglob("*.py") if "__pycache__" not in p.parts}
    tested = {p: digest for p, digest in value["source_sha256"].items() if p.startswith(("src/", "scripts/"))}
    require(tested == actual, "The causal implementation differs from its full CPU regression source")
    return value


def validate_causal_graph_registration(core):
    require(core["registration"] in CAUSAL_GRAPH_REGISTRATIONS and core["fit_recipe"] == NORMALIZED_RECIPE,
            "Unknown causal graph/reused-bank registration")
    amendment = core["causal_graph_amendment"]
    require(amendment["spec"] == _amendment_spec(core["registration"])
        and read_json(Path(core["workspace"]) / "configs/causal_graph_amendment.json") == amendment,
        "Causal feature-selection amendment changed")
    reference = core["passed_bank_reuse_contract"]; verify_artifact(reference)
    contract = read_json(reference["path"])
    require(contract["fingerprint"] == fingerprint({k: v for k, v in contract.items() if k != "fingerprint"})
        and contract["schema"] == "passed_normalized_bank_reuse_v1"
        and contract["target_workspace"] == core["workspace"], "Unsealed/different bank reuse contract")
    verify_artifact(contract["source_core_protocol"])
    previous = read_core(Path(contract["source_core_protocol"]["path"]).parent.parent)
    require(previous["registration"] == NORMALIZED_REGISTRATION
        and previous["fingerprint"] == contract["source_core_fp"], "Causal graph source is not the completed V3 bank core")
    _pretest(previous)
    from .core_collection import verify_stage_receipt
    for name in ("import", "collect", "transcoders"): verify_stage_receipt(previous, name)
    require(not (Path(previous["workspace"]) / "experiments/core_stage_receipts/risk.json").exists(),
            "This pre-test amendment was not registered before uncertainty fitting")
    for key in ("model", "model_key", "train_tasks", "dev_tasks", "test_tasks", "training_seed", "budget", "methods",
                "transcoder", "risk_network", "failure_control", "policy_runtime", "decoding", "memory",
                "made_execution", "deadline_utc", "parent_project", "imported_train", "imported_development",
                "reused_collections", "tc64_amendment", "normalization_amendment"):
        require(core[key] == previous[key], "Causal graph amendment changed an unregistered setting: " + key)
    expected_es = dict(previous["esopt"])
    if core["registration"] == FAST_CAUSAL_GRAPH_REGISTRATION:
        expected_es["full_training_and_development_candidate_oracle_attempts"] = 6 * core_execution_budget(core)
    require(core["esopt"] == expected_es, "Causal subset changed ES beyond its declared new-rollout budget")
    require(core["graph"] == {**previous["graph"], **CAUSAL_GRAPH_REGISTRATION["graph"]},
            "Causal graph changed an undeclared budget/numerical setting")
    require(contract["expected_target_manifest_paths"] == [str(p) for p in collection_manifest_paths(core)],
            "Bank reuse points to another target corpus")
    for key in ("source_core_protocol", "source_source_manifest", "source_bank_manifest", "source_run_result",
                "source_tc_receipt", "pause_closure"):
        verify_artifact(contract[key])
    closure = read_json(contract["pause_closure"]["path"])
    require(closure["closed"] is True and all(closure["processes_exited"].values())
        and closure["new_policy_or_material_oracle_calls"] == 0
        and closure["pipeline"]["state"] == "halted_requires_reconciliation"
        and not any(key in closure["pipeline"]["stages"] for key in ("core-risk", "core-esopt", "core-final", "core-report")),
        "Old graph failure is not closed before new graph execution")
    require(amendment["source_core_protocol"] == contract["source_core_protocol"]
        and amendment["pause_closure"] == contract["pause_closure"], "Causal graph history lost its original source/closure")
    for item in amendment["diagnostic_history"]: verify_artifact(item)
    verify_artifact(amendment["implementation_validation"])
    _implementation_validation(amendment["implementation_validation"]["path"], core["workspace"])
    verify_artifact(amendment["validated_probe"])
    _probe(amendment["validated_probe"]["path"], source_project=core["workspace"],
           source_core=previous, bank_reference=contract["source_bank_manifest"])
    original = read_core(Path(previous["normalization_amendment"]["corpus_predecessor_protocol"]["path"]).parent.parent)
    return previous, original


def prepare_causal_workspace(source_project, destination, *, predecessor_workspace,
                             pause_closure_path, validated_probe_path, diagnostic_history_paths,
                             implementation_validation_path=None, fast_subset=False):
    from .passed_bank_reuse import build_reuse_contract
    source, target = Path(source_project).resolve(), Path(destination).resolve()
    require(type(fast_subset) is bool, "Subset selection must be explicit")
    registration = FAST_CAUSAL_GRAPH_REGISTRATION if fast_subset else CAUSAL_GRAPH_REGISTRATION
    existing = read_core(target) if target.exists() else None
    previous = read_core(predecessor_workspace)
    require(previous["registration"] == NORMALIZED_REGISTRATION, "Causal graph must reuse the completed V3 core")
    _pretest(previous)
    contract = build_reuse_contract(previous, target, pause_closure_path)
    probe = artifact(validated_probe_path)
    _probe(probe["path"], source_project=source, source_core=previous, bank_reference=contract["source_bank_manifest"])
    _implementation_validation(implementation_validation_path, source)
    implementation = artifact(implementation_validation_path)
    amendment = {"spec": _amendment_spec(registration), "source_core_protocol": contract["source_core_protocol"],
        "pause_closure": contract["pause_closure"], "validated_probe": probe,
        "implementation_validation": implementation,
        "diagnostic_history": [artifact(path) for path in diagnostic_history_paths]}
    require(amendment["diagnostic_history"], "All failed/original selection diagnostics must remain in the amendment")
    if existing is not None:
        require(existing["registration"] == registration and existing["source_project"] == str(source) and existing["causal_graph_amendment"] == amendment
            and read_json(existing["passed_bank_reuse_contract"]["path"]) == contract,
            "Existing/partial causal workspace needs reconciliation; no refresh or overwrite")
        return existing
    require(target != source and target != Path(previous["workspace"]) and target not in source.parents,
            "Invalid new causal graph workspace")
    core = copy.deepcopy(previous)
    core.update(registration=copy.deepcopy(registration), study_id=registration["study_id"], scope=registration["scope"],
        workspace=str(target), source_project=str(source), causal_graph_amendment=amendment)
    if fast_subset:
        core["execution_budget"] = 10
    target.mkdir(parents=True); copied = {}
    for name in ("src", "scripts"):
        originals = {str(p.relative_to(source)): file_sha256(p) for p in (source / name).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        shutil.copytree(source / name, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        require(all(file_sha256(source / p) == digest == file_sha256(target / p) for p, digest in originals.items()),
                "Source changed while freezing causal graph workspace")
        copied.update(originals)
    prior = Path(previous["workspace"])
    shutil.copytree(prior / "configs", target / "configs", ignore=shutil.ignore_patterns("stage_queue.json", "deadline_core_protocol.json"))
    shutil.copyfile(source / "configs/method_alignment_gate.json", target / "configs/method_alignment_gate.json")
    for name in ("data", "vendor", "environments"):
        (target / name).symlink_to((prior / name).resolve(), target_is_directory=True)
    for name in ("experiments", "logs"): (target / name).mkdir()
    write_json_atomic(target / "configs/deadline_core_registration.json", registration)
    write_json_atomic(target / "configs/causal_graph_amendment.json", amendment)
    write_json_atomic(target / "configs/passed_bank_reuse_contract.json", contract)
    core["passed_bank_reuse_contract"] = artifact(target / "configs/passed_bank_reuse_contract.json")
    main = _derived_main(read_json(target / "configs/full_study_main_protocol.json"), core)
    core["graph"] = main["graph"]
    core["esopt"] = main["esopt"]
    write_json_atomic(target / "configs/main_protocol.json", main)
    core["derived_main_protocol"] = artifact(target / "configs/main_protocol.json")
    core["collection_jobs"], core["final_jobs"] = collection_jobs(core), final_jobs(core)
    frozen = sorted(p for name in ("src", "scripts", "configs") for p in (target / name).rglob("*") if p.is_file())
    files = {str(p.relative_to(target)): file_sha256(p) for p in frozen}
    write_json_atomic(target / "source_manifest.json", {"schema": "deadline_core_source_snapshot_v1",
        "original_project": str(target), "parent_project": core["parent_project"], "source_project": str(source),
        "predecessor_project": str(prior), "files": files, "source_fingerprint": fingerprint(files), "copied_source_inputs": copied})
    core["snapshot_files"] = [artifact(p) for p in frozen] + [artifact(target / "source_manifest.json")]
    evidence = list(previous["source_evidence"]) + [contract[key] for key in ("source_core_protocol", "source_source_manifest",
        "source_bank_manifest", "source_run_result", "source_tc_receipt", "pause_closure")]
    evidence += amendment["diagnostic_history"] + [probe, implementation]
    core["source_evidence"] = list({x["path"]: x for x in evidence}.values())
    core["fingerprint"] = core["core_fingerprint"] = _core_hash(core)
    write_json_atomic(target / "configs/deadline_core_protocol.json", core)
    return read_core(target)


def verify_reused_collection(core, job_id):
    _, original = validate_causal_graph_registration(core)
    return _verify_reused_collection(core, job_id, original, reuse_key="causal_graph_corpus_reuse")


def import_reused_collections(core, *, split):
    _, original = validate_causal_graph_registration(core)
    return _import_reused_collections(core, split=split, previous=original, reuse_key="causal_graph_corpus_reuse")


def reused_bank_for_core(core, manifests):
    from .passed_bank_reuse import verify_reused_bank
    require(core["registration"] in CAUSAL_GRAPH_REGISTRATIONS, "Only the registered causal graph core may reuse this bank")
    reference = core["passed_bank_reuse_contract"]; verify_artifact(reference)
    contract = read_json(reference["path"])
    bank, collection, underlying = verify_reused_bank(contract, target_manifest_paths=manifests, target_core=core)
    proof = {"schema": "causal_core_passed_bank_adoption_v1", "complete": True,
        "source_bank_manifest": contract["source_bank_manifest"], "source_core_fingerprint": contract["source_core_fp"],
        "target_core_fingerprint": core["fingerprint"], "target_source_selection_rule": core["graph"]["source_selection_rule"],
        "contract": reference, "source_evidence_files": contract["source_artifacts"], "validation": underlying,
        "new_transcoder_training_calls": 0, "new_policy_or_material_oracle_calls": 0}
    return bank, collection, proof


def bank_for_corpus(contract, *, manifest_paths=None, model_key, transcoder_manifest):
    from .core_protocol import validate_corpus_contract
    validated = validate_corpus_contract(contract, manifest_paths=manifest_paths, model_key=model_key)
    core = read_core(Path(validated["protocol_path"]).parent.parent)
    if core["registration"] not in CAUSAL_GRAPH_REGISTRATIONS:
        return None
    paths = [Path(item["path"]) for item in validated["manifests"]]
    bank, collection, proof = reused_bank_for_core(core, paths)
    require(Path(transcoder_manifest).resolve() == Path(proof["source_bank_manifest"]["path"]).resolve(),
            "The controller requested a bank outside its explicit reuse contract")
    return bank, collection, proof
