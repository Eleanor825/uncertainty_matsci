"""Separate, complete G2/P2 MADE ES condition for the declared deadline core.

This module never weakens the original 16-generation/full-matrix loaders. It
reuses the same full-parameter AgenticESOpt and whole-trajectory callbacks, with
a separately source-bound case catalog, selected-weight proof and acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import torch

from .accounting import file_sha256, fingerprint, write_json_atomic
from .esopt import AgenticESOpt
from .es_training import ESTrainingConfig, ESTrainingDriver, EvaluationRequest, TrainingCase
from .training_jobs import (
    TrainingJobCallbacks, TrainingContractError, _read, _require, make_training_job,
    reconstruct_scientific_evidence, cpu_clean_reload_validator, load_frozen_controllers,
)


CORE_ES_SCHEMA = "deadline_core_full_parameter_es_v1"


def _read_core(project_or_core):
    from .core_protocol import read_core
    if isinstance(project_or_core, Mapping):
        core = read_core(Path(project_or_core["workspace"]))
        _require(core == dict(project_or_core), "Core protocol mapping differs from its source-bound workspace")
        return core
    return read_core(Path(project_or_core))


@dataclass(frozen=True)
class CoreTrainingCondition:
    project: Path
    model_key: str = "qwen35_4b"
    benchmark: str = "made"
    method: str = "esopt_graph_risk"
    seed: int = 1
    property_name: str | None = None

    def __post_init__(self):
        core = _read_core(self.project)
        object.__setattr__(self, "project", Path(core["workspace"]).resolve())
        _require(self.model_key == core["model_key"] == "qwen35_4b" and self.benchmark == "made"
                 and self.method == "esopt_graph_risk" and self.seed == core["training_seed"] == 1
                 and self.property_name is None, "Core ES is exactly one 4B MADE full-method seed")
        self.protocol()

    def output_directory(self, root: Path | None = None):
        return (Path(root) if root is not None else self.project / "experiments/core_es").resolve()

    def protocol(self):
        core = _read_core(self.project)
        from .core_protocol import registered_transcoder_epochs
        # The source-bound reader admits only the original16 or the explicitly
        # pre-test amended64 contract, never an arbitrary caller-supplied value.
        tc_epochs = registered_transcoder_epochs(core)
        source = core["derived_main_protocol"]
        path = Path(source["path"])
        _require(path.resolve() == self.project / "configs/main_protocol.json"
                 and file_sha256(path) == source["sha256"], "Core ES runtime protocol changed")
        protocol = _read(path)
        fixed = {"generations": 2, "population": 2, "cases_per_generation": 1,
                 "alpha": .0005, "sigma_start": .001, "sigma_end": .0002,
                 "validation_every_generations": 1, "schedule": "cosine",
                 "parameter_scope": "full_HF_policy_including_retained_unused_visual_parameters",
                 "standardization": "population_z_score_ddof0", "extra_inverse_sigma": False}
        _require(all(protocol["esopt"].get(k) == v and core["esopt"].get(k) == v for k, v in fixed.items()),
                 "Core ES requires both complete generations and the unchanged full-parameter formula")
        _require(protocol["models"] == [self.model_key] and protocol["training_seeds"] == [self.seed]
                 and protocol["graph"]["max_feature_nodes"] == 32
                 and protocol["graph"]["max_backward_targets"] == 42
                 and protocol["transcoder"]["epochs"] == tc_epochs
                 and protocol["transcoder"]["max_development_output_fvu"] == .5
                 and protocol["graph"]["validation_epsilon"] == .001
                 and protocol["graph"]["validation_rtol"] == .05
                 and protocol["graph"]["validation_atol"] == .0001,
                 "Core model/seed, graph budget, TC fidelity or native numerical gate changed")
        from .execution_contract import declared_mace_workers
        _require(declared_mace_workers(protocol) == 4, "Core requires the registered CPU MACE4 / ORB1 executor")
        return protocol

    def config(self):
        es = self.protocol()["esopt"]
        return ESTrainingConfig(benchmark="made", method=self.method, seed=self.seed,
            dev_every=es["validation_every_generations"],
            **{key: es[key] for key in ("generations", "population", "cases_per_generation", "alpha", "sigma_start", "sigma_end")})

    def cases(self):
        core = _read_core(self.project)
        groups = {name: [tuple(sorted(task["elements"])) for task in core[name + "_tasks"]]
                  for name in ("train", "dev", "test")}
        _require([len(groups[name]) for name in ("train", "dev", "test")] == [1, 1, 2]
                 and len(set(groups["train"] + groups["dev"] + groups["test"])) == 4,
                 "Core needs one train, one different dev and two distinct held-out chemistries")
        result = []
        for split in ("train", "dev"):
            task = core[split + "_tasks"][0]
            result.append((TrainingCase(f"{split}:made:{task['id']}", split, "made",
                payload={"task_id": task["id"], "task": {"id": task["id"], "elements": list(task["elements"])}}),))
        return tuple(result)


def _core_job(condition, request, model_entry, configuration_fingerprint):
    core = _read_core(condition.project)
    from .core_protocol import core_execution_budget
    job = make_training_job(condition, request, model_entry=model_entry,
                            policy_configuration_fingerprint=configuration_fingerprint,
                            expected_made_budget=core_execution_budget(core),
                            core_protocol=core if core_execution_budget(core) != 50 else None)
    return {**job, "source": "deadline_core_full_parameter_es_callback",
            "core_fingerprint": core["fingerprint"], "core_workspace": str(condition.project)}


def _verify_current_graphs(output, request):
    from .native_attribution import CONTRACT
    decisions = [json.loads(line) for line in (Path(output) / "decisions.jsonl").read_text().splitlines() if line.strip()]
    succeeded = 0
    generation = request.generation - 1 if request.phase == "train" else request.generation
    for row in decisions:
        _require(row.get("graph_status") in {"succeeded", "unavailable"}, "Core full-method decision omitted graph accounting")
        if row["graph_status"] == "unavailable":
            _require(bool(row.get("graph_error")), "Unavailable core graph has no recorded reason")
            continue
        metadata = row["graph_metadata"]
        stamp = metadata["policy"]
        _require(row["graph_contract"] == CONTRACT and stamp["state_id"] == request.policy_state_id
                 and stamp["generation"] == generation
                 and metadata["full_prefix_hash"] == row["prefix_hash"],
                 "Core graph is stale or not attributed under the actual evaluated weights/prefix")
        succeeded += 1
    _require(succeeded > 0, "Core full-method acceptance requires real current-policy graphs, not only missing proxies")
    return {"decisions": len(decisions), "succeeded": succeeded, "unavailable": len(decisions) - succeeded}


class CoreTrainingJobCallbacks(TrainingJobCallbacks):
    def __init__(self, condition, policy, output_dir, *, corpus_contract, **kwargs):
        from .core_protocol import validate_corpus_contract, core_execution_budget
        contract = validate_corpus_contract(corpus_contract, model_key=condition.model_key)
        self.core = _read_core(condition.project)
        budget = core_execution_budget(self.core)
        super().__init__(condition, policy, output_dir, expected_made_budget=budget,
                         core_protocol=self.core if budget != 50 else None, **kwargs)
        self.inputs[str(Path(contract["protocol_path"]).resolve())] = file_sha256(contract["protocol_path"])
        for item in contract["manifests"]:
            self.inputs[str(Path(item["path"]).resolve())] = item["sha256"]
        self.callback_identity.update(core_schema=CORE_ES_SCHEMA,
            core_fingerprint=self.core["fingerprint"], corpus_contract=contract)
        self.fingerprint = fingerprint(self.callback_identity)

    def job_factory(self, request):
        self._unchanged()
        return _core_job(self.condition, request, self.model_entry, self.policy.configuration_fingerprint)

    def result_verifier(self, request, result):
        proof = super().result_verifier(request, result)
        _verify_current_graphs(self._directory(request), request)
        return proof


def _descriptor_path(directory):
    return directory.parent / (directory.name + ".core_execution.json")


def _audit_core_run(directory, core_protocol):
    """Read-only reconstruction of every core training/dev call and checkpoint."""
    core = _read_core(core_protocol)
    from .core_protocol import core_execution_budget, FAST_CAUSAL_GRAPH_REGISTRATION
    budget = core_execution_budget(core)
    fast = core["registration"] == FAST_CAUSAL_GRAPH_REGISTRATION
    condition = CoreTrainingCondition(Path(core["workspace"]))
    directory = Path(directory).resolve()
    _require(directory == condition.output_directory(), "Core ES directory is outside the declared condition namespace")
    summary_path, manifest_path = directory / "training_summary.json", directory / "run_manifest.json"
    descriptor_path = _descriptor_path(directory)
    summary, manifest, descriptor = map(_read, (summary_path, manifest_path, descriptor_path))
    identity, config = manifest["identity"], condition.config()
    train, dev = condition.cases()
    _require(summary["status"] == "complete" and summary["completed_generations"] == summary["required_generations"] == 2
             and summary["expected_evaluations"] == summary["completed_evaluations"] == 6
             and summary["test_data_used"] is False and summary["clean_reload_checked_generation"] == 2,
             "Core selected weights require the complete G2/P2 train/dev schedule")
    _require(identity["config"] == asdict(config) and identity["train_cases"] == [asdict(c) for c in train]
             and identity["dev_cases"] == [asdict(c) for c in dev] and identity["parameter_scope"] == "full"
             and identity["population_design"] == "one_sided_gaussian"
             and identity["metric_name"] == "AUDC" and identity["reward_source"] == "made_official_orb_audc",
             "Core ES checkpoint belongs to another algorithm, case catalog or reward")
    _require(summary["run_fingerprint"] == manifest["run_fingerprint"] == fingerprint(identity), "Core ES run fingerprint mismatch")
    callback = descriptor["callback_identity"]
    _require(descriptor["core_fingerprint"] == core["fingerprint"]
             and callback["core_schema"] == CORE_ES_SCHEMA and callback["core_fingerprint"] == core["fingerprint"]
             and fingerprint(callback) == descriptor["callback_fingerprint"] == identity["callback_fingerprint"],
             "Core ES controller/source lineage differs from its registered callback")
    from .core_protocol import validate_corpus_contract
    validate_corpus_contract(callback["corpus_contract"], model_key=condition.model_key)
    files = {str(path): file_sha256(path) for path in (summary_path, manifest_path, descriptor_path)}
    for name, checksum in callback["inputs"].items():
        _require(file_sha256(name) == checksum, "A source or frozen controller artifact changed after core ES")
        files[name] = checksum
    entry = next(item for item in _read(condition.project / "configs/model_manifest.json")["models"] if item["key"] == condition.model_key)
    _require(identity["model_id"] == entry["model_id"] and identity["model_revision"] == entry["revision"], "Core ES base model/revision changed")
    states, histories, markers = {}, {}, {}
    for generation in (0, 1, 2):
        folder = directory / "checkpoints" / f"generation_{generation:04d}"
        marker_path = folder / "complete.json"
        marker = _read(marker_path)
        _require(marker["status"] == "complete" and marker["generation"] == generation
                 and marker["run_fingerprint"] == summary["run_fingerprint"]
                 and set(marker["files"]) == {"policy.pt", "policy.pt.json", "driver_state.json", "es_history.json"},
                 "Core checkpoint markers must contain all three complete generations including the base")
        files[str(marker_path)] = file_sha256(marker_path)
        for name, checksum in marker["files"].items():
            path = folder / name
            _require(path.is_file() and not path.is_symlink() and file_sha256(path) == checksum,
                     "Core checkpoint/metadata/history missing or changed; no retired/partial fallback")
            files[str(path)] = checksum
        states[generation], histories[generation], markers[generation] = _read(folder / "driver_state.json"), _read(folder / "es_history.json"), marker
        state = states[generation]
        _require(state["generation"] == generation and len(state["generations"]) == generation
                 and state["run_fingerprint"] == summary["run_fingerprint"]
                 and state["actual_model_state_hash"] == marker["actual_model_state_hash"], "Core checkpoint state disagrees with commit marker")
    records, plan = states[2]["generations"], identity["plan"]
    _require([r["generation"] for r in records] == [1, 2] and [p["generation"] for p in plan] == [1, 2]
             and states[1]["generations"] == records[:1], "Core history must preserve both complete generations")
    initial = manifest["initial_actual_model_state_hash"]
    _require(initial == markers[0]["actual_model_state_hash"] == histories[2]["base_state_hash"], "Core actual initial tensor hash is inconsistent")
    planner = object.__new__(ESTrainingDriver)
    planner.config, planner.train_cases, planner.dev_cases = config, train, dev
    _require(plan == planner._make_plan(), "Core independent population/case/environment random streams changed")
    for generation in (0, 1, 2):
        history = histories[generation]
        _require(history["history"] == histories[2]["history"][:generation]
                 and len(history["history"]) == generation and history["base_state_hash"] == initial
                 and history["current_state_hash"] == markers[generation]["actual_model_state_hash"]
                 and history["parameter_scope"] == "full"
                 and history["parameter_manifest"] == manifest["full_parameter_manifest"],
                 "Core checkpoints disagree about complete full-parameter update history")
    jobs, phase_costs, graphs = {}, {"train": Counter(), "dev": Counter()}, {}
    tasks = _read(condition.project / "configs/benchmark_tasks.json")
    for generation, record in enumerate(records, 1):
        schedule, step = plan[generation - 1], histories[2]["history"][generation - 1]
        _require(len(schedule["population_seeds"]) == 2 and len(set(schedule["population_seeds"])) == 2
                 and schedule["train_case_indices"] == [0] and len(schedule["environment_seeds"]) == 1
                 and schedule["run_dev"] is True and len(schedule["dev_environment_seeds"]) == 1,
                 "Core generation lost its full paired population/case/dev schedule")
        _require(record["actual_model_state_hash_before"] == markers[generation - 1]["actual_model_state_hash"] == step["state_hash_before"]
                 and record["actual_model_state_hash_after"] == markers[generation]["actual_model_state_hash"] == step["state_hash_after"]
                 and step["generation"] == generation - 1 and step["seeds"] == schedule["population_seeds"]
                 and step["rewards"] == record["fitness"] and step["metadata"]["population"] == record["population"]
                 and step["alpha"] == config.alpha and step["sigma"] == schedule["sigma"] == record["sigma"]
                 and step["ddof"] == 0 and step["parameter_delta_l2"] == record["actual_parameter_delta_l2"],
                 "Core actual update history differs from full-parameter AgenticESOpt")
        _require(len(record["population"]) == 2 and len(record["dev"]) == 1, "Core generation omitted population or dev results")
        rows = []
        for index, candidate in enumerate(record["population"]):
            _require(candidate["candidate_index"] == index and len(candidate["results"]) == 1
                     and candidate["perturbation_seed"] == schedule["population_seeds"][index]
                     and candidate["fitness"] == candidate["results"][0]["metric"] == record["fitness"][index],
                     "Core candidate identity or official reward used for ES changed")
            rows.append(("train", candidate["results"][0], index, candidate))
        rows.append(("dev", record["dev"][0], None, None))
        for phase, result_row, index, candidate in rows:
            request_id = result_row["request_id"]
            _require(request_id not in jobs, "Core evaluation was counted more than once")
            path = directory / "jobs" / (request_id + ".json")
            saved = _read(path)
            request = EvaluationRequest.from_dict(saved["request"])
            _require(saved["status"] == "complete" and request.request_id == request_id == fingerprint(request.slot_identity())
                     and request.run_fingerprint == summary["run_fingerprint"] and request.generation == generation
                     and request.phase == phase and request.candidate_index == index
                     and request.environment_seed == (schedule["environment_seeds"][0] if phase == "train" else schedule["dev_environment_seeds"][0])
                     and request.actual_model_state_hash == (candidate["actual_perturbed_model_state_hash"] if candidate else record["actual_model_state_hash_after"])
                     and request.perturbation_seed == (candidate["perturbation_seed"] if candidate else None)
                     and request.perturbation_sigma == (schedule["sigma"] if candidate else None),
                     "Core evaluation request is incomplete, unpaired, stale or from another state")
            job = _core_job(condition, request, entry, identity["policy_configuration_fingerprint"])
            output = directory / "rollouts" / request_id
            evidence = reconstruct_scientific_evidence(output, job, tasks=tasks, expected_mace_num_workers=4,
                                                      expected_made_budget=budget, core_protocol=core if fast else None)
            receipt_path = output / "completion_receipt.json"
            receipt = _read(receipt_path)
            _require(receipt["complete"] is True and receipt["callback_fingerprint"] == identity["callback_fingerprint"]
                     and receipt["request_identity"] == request.semantic_identity() and receipt["job_fingerprint"] == fingerprint(job)
                     and receipt["evidence"] == evidence and receipt["artifacts"] == TrainingJobCallbacks._inventory(output),
                     "Core official trajectory receipt/raw artifacts changed")
            result = saved["result"]
            _require(result["complete"] is True and result["request_id"] == request_id
                     and result["reward_source"] == request.reward_source and result["metric_name"] == evidence["metric_name"] == "AUDC"
                     and result["metric_value"] == evidence["metric_value"] == result_row["metric"]
                     and result["costs"] == evidence["costs"], "Core ES reward/cost must match complete official RPC evidence")
            expected_artifacts = [{"path": str(Path(name).resolve()), "sha256": file_sha256(name), "size_bytes": Path(name).stat().st_size}
                                  for name in result["result_paths"]]
            expected_paths = [item["path"] for item in receipt["artifacts"]] + [str(receipt_path)]
            _require(result["result_paths"] == expected_paths and saved["artifacts"] == expected_artifacts,
                     "Core driver ledger omitted or altered scientific evidence paths")
            phase_costs[phase].update(evidence["costs"])
            graphs[request_id] = _verify_current_graphs(output, request)
            for item in expected_artifacts: files[item["path"]] = item["sha256"]
            files[str(path)] = file_sha256(path)
            jobs[request_id] = saved
        _require(record["dev_mean"] == record["dev"][0]["metric"], "Core dev selection changed the observed AUDC")
    _require({p.stem for p in (directory / "jobs").glob("*.json")} == set(jobs), "Core run contains unknown/extra evaluation records")
    # Counter addition drops zero keys; costs preserve the complete measured schema.
    total = {key: phase_costs["train"].get(key, 0) + phase_costs["dev"].get(key, 0) for key in phase_costs["train"].keys() | phase_costs["dev"].keys()}
    reported = summary["actual_evaluator_costs"]
    # Regrouping seconds by train/dev changes floating-point summation order.
    # Raw per-job values were checked exactly above; integer call/token budgets
    # remain exact. This is accounting roundoff, never an oracle/FD tolerance.
    same_costs = (total.keys() == reported.keys() and all(
        value == reported[key] if not key.endswith("seconds") else
        math.isclose(value, reported[key], rel_tol=1e-12, abs_tol=1e-9)
        for key, value in total.items()))
    _require(same_costs and phase_costs["train"]["candidate_oracle_attempts"] == 4 * budget
             and phase_costs["dev"]["candidate_oracle_attempts"] == 2 * budget, "Core ES scientific cost ledger is incomplete or double-counted")
    best = max(records, key=lambda row: (row["dev_mean"], -row["generation"]))["generation"]
    _require(summary["best_generation"] == best and summary["best_dev_metric"] == records[best - 1]["dev_mean"]
             and summary["best_actual_model_state_hash"] == markers[best]["actual_model_state_hash"]
             and summary["final_actual_model_state_hash"] == markers[2]["actual_model_state_hash"], "Core best-of-two selection is not strictly dev-only")
    if not fast:
        _require(summary["best_actual_model_state_hash"] != initial and any(
            value > 0 for record in records[:best] for name, value in record["actual_parameter_delta_l2"].items()
            if ".visual." not in name and not name.startswith("visual.")),
            "Core selected policy has no measured nonzero text-parameter update; cannot claim trained full method")
    proof = summary["clean_reload"]
    _verify_reload_proof(proof, markers[2], summary["final_actual_model_state_hash"])
    costs = {"complete": True, "training": {"costs": dict(phase_costs["train"]), "job_count": 4},
             "development": {"costs": dict(phase_costs["dev"]), "job_count": 2},
             "evidence_files": [{"path": name, "sha256": checksum} for name, checksum in sorted(files.items())],
             "run_fingerprint": summary["run_fingerprint"], "graph_coverage": graphs}
    if fast:
        costs.update(_fast_evolution_evidence(records, histories[2]["history"], initial=initial,
                                             selected_generation=best, config=config))
    return core, condition, summary, manifest, descriptor, markers, costs


def _fast_evolution_evidence(records, optimizer_history, *, initial, selected_generation, config):
    """Separate observed selected-policy change from complete B10 execution.

    Only a verified z-score tie can explain an unchanged generation. Reward
    jitter, arbitrary extra noise, absent updates and claimed zero norms with
    different hashes are refused. The unchanged AgenticESOpt formula is used.
    """
    from .esopt import population_zscores
    _require([r["generation"] for r in records] == [1, 2] and len(optimizer_history) == 2,
             "FAST evolution needs both complete generations")
    curve = []
    for record, step in zip(records, optimizer_history, strict=True):
        rewards = record["fitness"]
        _require(len(rewards) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in rewards),
                 "FAST fitness must be the complete finite P2 result")
        standardized = population_zscores(rewards, config.normalization_epsilon)
        _require(step["normalized_rewards"] == standardized
                 and step["normalization_epsilon"] == config.normalization_epsilon,
                 "FAST history changed the original population z-score formula")
        norms = record["actual_parameter_delta_l2"]
        _require(norms and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in norms.values()),
                 "Invalid measured FAST parameter delta")
        changed = record["actual_model_state_hash_before"] != record["actual_model_state_hash_after"]
        nonzero = any(v > 0 for v in norms.values())
        tied = all(v == rewards[0] for v in rewards)
        _require(changed == nonzero, "FAST actual state hashes and parameter delta norms disagree")
        if not changed:
            _require(tied and all(v == 0 for v in standardized), "Unchanged FAST weights are not explained by a true fitness tie")
        if tied:
            _require(not changed and all(v == 0 for v in standardized), "Tied fitness cannot produce a fabricated parameter update")
        text_changed = any(v > 0 for name, v in norms.items() if ".visual." not in name and not name.startswith("visual."))
        curve.append({"generation": record["generation"], "fitness": list(rewards), "normalized_rewards": list(standardized),
            "population_fitness_tied": tied, "parameter_update_observed": changed,
            "text_parameter_update_observed": text_changed, "all_parameter_deltas_zero": not nonzero,
            "actual_model_state_hash_before": record["actual_model_state_hash_before"],
            "actual_model_state_hash_after": record["actual_model_state_hash_after"], "dev_mean": record["dev_mean"],
            "dev_environment_seeds": [item["environment_seed"] for item in record["dev"]]})
    selected_matches = records[selected_generation - 1]["actual_model_state_hash_after"] == initial
    selected_signal = not selected_matches and any(row["text_parameter_update_observed"] for row in curve[:selected_generation])
    _require(selected_signal or selected_matches, "FAST changed selected weights without a measured text-parameter update")
    proof = {"schema": "verified_fast_es_zero_selected_update_v1", "complete": True,
        "selected_generation": selected_generation, "selected_matches_initial": selected_matches,
        "initial_actual_model_state_hash": initial, "generations": curve,
        "any_generation_updated": any(row["parameter_update_observed"] for row in curve),
        "original_formula_verified": True, "initial_dev_evaluated": False,
        "interpretation": "No observed selected-policy evolution; complete execution is not evidence of improvement."}
    return {"evolution_signal_observed": selected_signal,
        "evolution_status": "nonzero_updated_policy" if selected_signal else "no_evolution_signal",
        "nonzero_text_parameter_update_verified": selected_signal,
        "zero_update_proof": None if selected_signal else proof, "generation_curve": curve,
        "initial_dev_evaluated": False}


def _evolution_fields(core, costs):
    from .core_protocol import FAST_CAUSAL_GRAPH_REGISTRATION
    if core["registration"] != FAST_CAUSAL_GRAPH_REGISTRATION:
        return {"nonzero_text_parameter_update_verified": True}
    return {key: costs[key] for key in ("evolution_signal_observed", "evolution_status",
        "nonzero_text_parameter_update_verified", "zero_update_proof", "generation_curve", "initial_dev_evaluated")}


def _verify_reload_proof(proof, marker, state_hash):
    _require(proof.get("verified") is True and proof.get("fresh_instance") is True and proof.get("allclose") is True
             and proof.get("expected_state_hash") == proof.get("loaded_state_hash") == state_hash
             and proof.get("checkpoint_file_sha256") == marker["files"]["policy.pt"]
             and proof.get("atol") == 1e-6 and proof.get("rtol") == 1e-5
             and type(proof.get("max_abs_logit_difference")) in (int, float)
             and math.isfinite(proof["max_abs_logit_difference"]), "Core checkpoint lacks fixed-tolerance fresh-instance forward proof")


def audit_core_es_costs(condition_dir, core_protocol):
    """Audit six actual official trajectories once; summary is only a cross-check."""
    return _audit_core_run(condition_dir, core_protocol)[-1]


def _load_optimizer(policy, manifest, checkpoint, config):
    identity = manifest["identity"]
    _require(policy.model_id == identity["model_id"] and policy.revision == identity["model_revision"]
             and policy.checkpoint_hash == identity["initial_checkpoint_manifest_hash"]
             and policy.configuration_fingerprint == identity["policy_configuration_fingerprint"],
             "Core checkpoint is incompatible with the loaded base policy/runtime")
    optimizer = AgenticESOpt(policy.model, policy_model_id=f"{policy.model_id}@{policy.revision}",
        parameter_scope="full", noise_chunk_size=config.noise_chunk_size,
        normalization_epsilon=config.normalization_epsilon)
    _require(optimizer.parameter_manifest() == manifest["full_parameter_manifest"],
             "Core optimizer must retain the complete policy parameter inventory including vision")
    optimizer.load_checkpoint(checkpoint)
    return optimizer


def _core_receipt_path(directory):
    return Path(directory) / "core_training_receipt.json"


def finalize_core_training(policy, condition_dir, *, core_protocol,
                           clean_reload_validator=cpu_clean_reload_validator):
    """Publish only after all six trajectories and selected/final reloads verify.

    Keeps all three small-core checkpoints. Loading the selected checkpoint is a
    weight operation, not physical trajectory replay. Existing receipts are only
    accepted through the independent loader; they are never silently replaced.
    """
    directory = Path(condition_dir).resolve()
    if _core_receipt_path(directory).exists():
        core = _read_core(core_protocol)
        profile = load_core_selected_es(policy, directory, core_protocol=core)
        _publish_stage_receipt(core, directory, profile)
        return profile
    core, condition, summary, manifest, descriptor, markers, costs = _audit_core_run(directory, core_protocol)
    best = summary["best_generation"]
    checkpoint = directory / "checkpoints" / f"generation_{best:04d}" / "policy.pt"
    optimizer = _load_optimizer(policy, manifest, checkpoint, condition.config())
    policy.mark_state("reload", generation=best)
    actual = optimizer.model_state_hash()
    _require(actual == summary["best_actual_model_state_hash"], "Selected core checkpoint did not load the actual updated weights")
    selected_proof = clean_reload_validator(policy, checkpoint, actual)
    _verify_reload_proof(selected_proof, markers[best], actual)
    _require(optimizer.model_state_hash() == actual, "Selected clean reload changed live core weights")
    receipt = {"schema": CORE_ES_SCHEMA, "complete": True,
        "core_fingerprint": core["fingerprint"], "run_fingerprint": summary["run_fingerprint"],
        "condition": asdict(condition) | {"project": str(condition.project)},
        "initial_actual_model_state_hash": manifest["initial_actual_model_state_hash"],
        "selected_generation": best, "selected_actual_model_state_hash": actual,
        "final_actual_model_state_hash": summary["final_actual_model_state_hash"],
        "selected_checkpoint": str(checkpoint), "selected_checkpoint_sha256": file_sha256(checkpoint),
        "selected_clean_reload": selected_proof, "final_clean_reload": summary["clean_reload"],
        "parameter_scope": "full_including_retained_unused_visual", **_evolution_fields(core, costs),
        "selection": "best_of_generations_1_and_2_on_the_only_registered_dev_chemistry; earliest_generation_breaks_ties",
        "original_full_study_complete": False, "cost_audit": costs,
        "controller_provenance": descriptor["callback_identity"],
        "artifacts": {item["path"]: item["sha256"] for item in costs["evidence_files"]}}
    receipt["fingerprint"] = fingerprint(receipt)
    write_json_atomic(_core_receipt_path(directory), receipt)
    profile = load_core_selected_es(policy, directory, core_protocol=core)
    _publish_stage_receipt(core, directory, profile)
    return profile


def _publish_stage_receipt(core, directory, profile):
    stage_path = Path(core["workspace"]) / "experiments/core_stage_receipts/esopt.json"
    stage = {"schema": "deadline_core_stage_receipt_v1", "stage": "esopt", "complete": True,
             "core_fingerprint": core["fingerprint"], "selected_profile": profile,
             "training_receipt": {"path": str(_core_receipt_path(directory)), "sha256": file_sha256(_core_receipt_path(directory))},
             "artifacts": profile["artifacts"], "original_full_study_complete": False}
    stage["fingerprint"] = fingerprint(stage)
    if stage_path.exists():
        _require(_read(stage_path) == stage, "Completed core ES stage receipt changed")
    else:
        write_json_atomic(stage_path, stage)


def load_core_selected_es(policy, condition_dir, *, core_protocol, model_key="qwen35_4b",
                          method="esopt_graph_risk", seed=1):
    """Independently verify the complete core evidence, then load selected weights."""
    directory = Path(condition_dir).resolve()
    core, condition, summary, manifest, descriptor, markers, costs = _audit_core_run(directory, core_protocol)
    _require((model_key, method, seed) == (condition.model_key, condition.method, condition.seed),
             "Core selected checkpoint requested for another model/method/seed")
    receipt_path = _core_receipt_path(directory)
    receipt = _read(receipt_path)
    _require(receipt.get("schema") == CORE_ES_SCHEMA and receipt.get("complete") is True
             and receipt.get("fingerprint") == fingerprint({k: v for k, v in receipt.items() if k != "fingerprint"})
             and receipt["core_fingerprint"] == core["fingerprint"] and receipt["run_fingerprint"] == summary["run_fingerprint"]
             and receipt["cost_audit"] == costs and receipt["controller_provenance"] == descriptor["callback_identity"],
             "Core trained-policy receipt or scientific/controller lineage changed")
    best = summary["best_generation"]
    checkpoint = directory / "checkpoints" / f"generation_{best:04d}" / "policy.pt"
    _require(receipt["selected_generation"] == best and receipt["selected_checkpoint"] == str(checkpoint)
             and receipt["selected_actual_model_state_hash"] == summary["best_actual_model_state_hash"]
             and receipt["initial_actual_model_state_hash"] == manifest["initial_actual_model_state_hash"]
             and receipt["final_actual_model_state_hash"] == summary["final_actual_model_state_hash"]
             and receipt["selected_checkpoint_sha256"] == file_sha256(checkpoint)
             and receipt["artifacts"] == {item["path"]: item["sha256"] for item in costs["evidence_files"]},
             "Core selected/final weight proof differs from actual complete generations")
    _verify_reload_proof(receipt["selected_clean_reload"], markers[best], summary["best_actual_model_state_hash"])
    _require(receipt["final_clean_reload"] == summary["clean_reload"], "Core final clean-reload proof changed")
    _require(all(receipt.get(key) == value for key, value in _evolution_fields(core, costs).items()),
             "Core selected evolution/zero-update proof changed")
    optimizer = _load_optimizer(policy, manifest, checkpoint, condition.config())
    policy.mark_state("reload", generation=best)
    actual = optimizer.model_state_hash()
    _require(actual == summary["best_actual_model_state_hash"], "Loaded core weights are not the actual selected checkpoint")
    from .core_protocol import FAST_CAUSAL_GRAPH_REGISTRATION
    if core["registration"] != FAST_CAUSAL_GRAPH_REGISTRATION:
        _require(actual != manifest["initial_actual_model_state_hash"],
                 "Loaded core weights are not the actual nonzero updated selected checkpoint")
    artifacts = dict(receipt["artifacts"])
    artifacts[str(receipt_path)] = file_sha256(receipt_path)
    return {"training_directory": str(directory), "training_run_fingerprint": summary["run_fingerprint"],
            "run_fingerprint": summary["run_fingerprint"], "core_fingerprint": core["fingerprint"],
            "selected_generation": best, "selected_checkpoint": str(checkpoint), "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint), "actual_model_state_hash": actual,
            "initial_actual_model_state_hash": manifest["initial_actual_model_state_hash"],
            "initial_checkpoint_manifest_hash": policy.checkpoint_hash,
            "parameter_scope": "full_including_retained_unused_visual", **_evolution_fields(core, costs),
            "training_costs": summary["actual_evaluator_costs"], "cost_audit": costs,
            "checkpoint_selection": "development_only_best_of_two_complete_generations",
            "controller_provenance": descriptor["callback_identity"], "selected_clean_reload": receipt["selected_clean_reload"],
            "condition_lineage": receipt["condition"], "artifacts": artifacts}


def run_core_es(project, *, resume=False, plan_only=False):
    core = _read_core(project)
    from .core_protocol import core_execution_budget
    budget = core_execution_budget(core)
    condition = CoreTrainingCondition(Path(core["workspace"]))
    protocol, config = condition.protocol(), condition.config()
    train, dev = condition.cases()
    directory = condition.output_directory()
    plan = {"schema": CORE_ES_SCHEMA, "core_fingerprint": core["fingerprint"], "condition": asdict(condition) | {"project": str(condition.project)},
            "config": asdict(config), "train_cases": [asdict(c) for c in train], "dev_cases": [asdict(c) for c in dev],
            "training_evaluations": 4, "development_evaluations": 2, "candidate_oracle_attempts": 6 * budget,
            "output_directory": str(directory), "stage_receipt": str(condition.project / "experiments/core_stage_receipts/esopt.json")}
    if plan_only:
        return {"status": "plan_only_no_physical_execution", **plan}
    from .core_protocol import collection_manifest_paths, corpus_contract
    from .policy import DecodingConfig, QwenPolicyAdapter, declared_runtime_options
    from .rollouts import RolloutSettings
    runtime = declared_runtime_options(protocol)
    torch.set_num_threads(runtime["torch_cpu_threads"])
    if torch.device(runtime["device"]).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(runtime["cuda_memory_fraction"], device=runtime["device"])
    decode = protocol["decoding"]
    policy = QwenPolicyAdapter.from_verified_checkpoint(condition.project / "configs/model_manifest.json", condition.model_key,
        condition.project / "data/models" / condition.model_key,
        **{key: runtime[key] for key in ("device", "dtype", "attn_implementation", "sdpa_backend", "cpu_embedding_and_lm_head", "attention_checkpointing")},
        decoding=DecodingConfig(**{key: decode[key] for key in ("max_new_tokens", "temperature", "top_p", "top_k")}),
        enable_thinking=decode["enable_thinking"], max_input_tokens=decode["max_input_tokens"])
    corpus = corpus_contract(core, collection_manifest_paths(core))
    risk_path = condition.project / "experiments/risk_models/qwen35_4b/made/graph_risk.pt"
    from .core_representation import paths_for
    tc_path = paths_for(core)["bank"] / "transcoder_manifest.json"
    risk, attributor, auxiliary = load_frozen_controllers(policy, model_key=condition.model_key, benchmark="made", method=condition.method,
        risk_checkpoint=risk_path, transcoder_manifest=tc_path, graph_config=protocol["graph"], device=runtime["device"],
        failure_control=protocol["failure_control"], corpus_contract=corpus)
    settings = RolloutSettings(max_generation_retries=protocol["risk_network"]["maximum_candidate_generations_per_decision"],
        risk_threshold=protocol["risk_network"]["threshold"], history_results=protocol["memory"]["scientific_history_per_episode"],
        recent_tools=protocol["memory"]["recent_tool_responses"],
        activation_tokens_per_decision=protocol["collection"]["activation_token_rows_per_complete_decision_prefix"],
        failure_aware_control="made" in protocol["failure_control"]["enabled_benchmarks"])
    callbacks = CoreTrainingJobCallbacks(condition, policy, directory, corpus_contract=corpus,
        settings=settings, risk_model=risk, attributor=attributor, auxiliary_files=auxiliary)
    descriptor = {"plan": plan, "core_fingerprint": core["fingerprint"], "callback_identity": callbacks.callback_identity,
                  "callback_fingerprint": callbacks.fingerprint, "policy_configuration_fingerprint": policy.configuration_fingerprint}
    path = _descriptor_path(directory)
    if path.exists():
        _require(_read(path) == descriptor, "Core ES execution sources differ from the existing run")
    else:
        write_json_atomic(path, descriptor)
    driver = ESTrainingDriver(policy, config, train_cases=train, dev_cases=dev,
        job_factory=callbacks.job_factory, evaluate=callbacks.evaluate, result_verifier=callbacks.result_verifier,
        recover_result=callbacks.recover_result, clean_reload_validator=cpu_clean_reload_validator,
        output_dir=directory, callback_fingerprint=callbacks.fingerprint)
    if (directory / "training_summary.json").exists():
        _require(resume, "Complete core ES requires explicit resume for read-only verification")
        # Avoid rerunning the completed driver's finalization, which would replace
        # measured session timing and invalidate an already bound acceptance proof.
    else:
        driver.run(resume=resume)
    profile = finalize_core_training(policy, directory, core_protocol=core)
    return {"status": "complete", "complete": True, "stage_receipt": plan["stage_receipt"], "selected_profile": profile}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run_core_es(args.project, resume=args.resume, plan_only=args.plan_only), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
