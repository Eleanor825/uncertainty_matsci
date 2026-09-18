"""Four registered core final jobs, with their own conservative durable ledger.

This module never constructs or shrinks the 2160-job full-study manifest. A
scientific failure with all registered attempts accounted is an observation; interrupted
or unknown physical outcomes are quarantined and never automatically replayed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import gc
import json
import math
from pathlib import Path
import time
import uuid

from .accounting import file_sha256, fingerprint, write_json_atomic, validate_made_budget_scope

SCOPE = "registered_deadline_core_not_full_study"
SCHEMA = "core_final_matrix_v1"
COUNTS = {"jobs": 4, "episodes": 4, "candidate_oracle_attempts": 200, "dft_episode_attempts": 0}
METHODS = ("baseline", "esopt_graph_risk")
STATES = {"pending", "running", "succeeded", "failed", "orphaned"}


class CoreFinalError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CoreFinalError(message)


def read(path):
    return json.loads(Path(path).read_text())


def publish_once(path, value):
    path = Path(path)
    if path.exists():
        require(read(path) == value, "Completed core receipt changed; reconciliation is required")
    else:
        write_json_atomic(path, value)


def build_core_final_manifest(core_fingerprint, jobs, *, core_protocol=None):
    result = {"schema": SCHEMA, "scope": SCOPE, "core_protocol_fingerprint": core_fingerprint,
              "global_study_complete": False, "expected_counts": dict(COUNTS), "jobs": deepcopy(list(jobs))}
    if core_protocol is not None:
        from .core_protocol import core_execution_budget
        budget = core_execution_budget(core_protocol)
        if budget != 50:
            verified = validate_made_budget_scope(budget, core_protocol)
            require(verified["fingerprint"] == core_fingerprint, "FAST final matrix belongs to another core")
            path = Path(verified["workspace"]) / "configs/deadline_core_protocol.json"
            result.update(execution_budget=budget, registered_execution_scope=verified["scope"],
                core_protocol={"path": str(path.resolve()), "sha256": file_sha256(path), "fingerprint": core_fingerprint},
                expected_counts={**COUNTS, "candidate_oracle_attempts": 4 * budget})
    result["fingerprint"] = fingerprint(result)
    validate_core_final_manifest(result)
    return result


def validate_core_final_manifest(manifest):
    require(manifest.get("schema") == SCHEMA and manifest.get("scope") == SCOPE
            and manifest.get("global_study_complete") is False, "Not the independent core-final namespace")
    require(manifest.get("fingerprint") == fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}), "Core final manifest changed")
    budget, core = core_manifest_budget(manifest)
    require(manifest.get("expected_counts") == {**COUNTS, "candidate_oracle_attempts": 4 * budget}, "Core final denominator cannot shrink")
    jobs = manifest.get("jobs", [])
    require(len(jobs) == 4 and len({j["job_id"] for j in jobs}) == 4, "Core requires four distinct final jobs")
    tasks = {j["task_id"] for j in jobs}
    require(len(tasks) == 2 and {(j["task_id"], j["method"]) for j in jobs} == {(task, method) for task in tasks for method in METHODS}, "Core requires the complete two-system paired method matrix")
    require(len({(j["model_id"], j["model_revision"]) for j in jobs}) == 1, "Core methods must share one fixed initial model")
    for job in jobs:
        require(job["stage"] == "final_eval" and job["benchmark"] == "made" and job["model_key"] == "qwen35_4b"
                and job["seed"] == 1 and job["environment_seeds"] == [1] and job["episode_ids"] == ["0"]
                and job["budget"] == budget and job["budget_unit"] == "candidate_oracle_attempts_per_episode"
                and job["expected_counts"] == {"episodes": 1, "candidate_oracle_attempts": budget, "dft_episode_attempts": 0},
                "Core final job changes the fixed model/seed/registered budget contract")
        require(job["task_id"] == "-".join(sorted(job["task"]["elements"])), "Core task name/elements mismatch")
    if core is not None:
        from .core_protocol import final_jobs
        require(jobs == final_jobs(core), "FAST final matrix differs from the sealed core jobs")


def core_manifest_budget(manifest):
    """Return (budget, verified FAST core), never trusting a job's own budget."""
    reference = manifest.get("core_protocol")
    if reference is None:
        require("execution_budget" not in manifest and "registered_execution_scope" not in manifest,
                "An unbound core matrix cannot change its B50 budget")
        return 50, None
    from .core_protocol import read_core
    require(isinstance(reference, dict) and file_sha256(reference["path"]) == reference.get("sha256"),
            "FAST final protocol source changed")
    core = read_core(Path(reference["path"]).parent.parent)
    validate_made_budget_scope(manifest.get("execution_budget"), core)
    require(manifest.get("execution_budget") == 10
            and reference.get("fingerprint") == core["fingerprint"] == manifest["core_protocol_fingerprint"]
            and manifest.get("registered_execution_scope") == core["scope"], "FAST final budget/scope binding differs")
    return 10, core


def verify_selected_evolution(selected, initial_hash, actual_hash, *, fast=False):
    """A complete FAST tie is an explicit null outcome, not an invented update."""
    require(isinstance(selected, dict) and selected.get("selected_generation") in {1, 2}
            and selected.get("actual_model_state_hash") == actual_hash
            and selected.get("initial_actual_model_state_hash") == initial_hash,
            "Full core arm lacks the actual selected ES checkpoint identity")
    if not fast:
        require(actual_hash != initial_hash, "Full core arm lacks genuinely updated selected ES weights")
        return
    changed = actual_hash != initial_hash
    require(selected.get("evolution_signal_observed") is changed
            and selected.get("nonzero_text_parameter_update_verified") is changed
            and selected.get("evolution_status") == ("nonzero_updated_policy" if changed else "no_evolution_signal")
            and selected.get("initial_dev_evaluated") is False,
            "FAST selected evolution status differs from actual weights")
    curve = selected.get("generation_curve")
    require(isinstance(curve, list) and [row.get("generation") for row in curve] == [1, 2],
            "FAST selected policy lacks both completed generation records")
    previous = initial_hash
    for row in curve:
        rewards, standardized = row.get("fitness", []), row.get("normalized_rewards", [])
        require(len(rewards) == len(standardized) == 2
                and all(type(v) in (int, float) and math.isfinite(v) for v in rewards + standardized)
                and type(row.get("dev_mean")) in (int, float) and math.isfinite(row["dev_mean"])
                and row.get("actual_model_state_hash_before") == previous,
                "FAST generation curve is incomplete or breaks the actual weight chain")
        update = row["actual_model_state_hash_before"] != row.get("actual_model_state_hash_after")
        tied = rewards[0] == rewards[1]
        require(row.get("population_fitness_tied") is tied and row.get("parameter_update_observed") is update
                and row.get("all_parameter_deltas_zero") is (not update)
                and (update or (tied and all(v == 0 for v in standardized)
                                and row.get("text_parameter_update_observed") is False))
                and (not tied or not update), "FAST zero update is not explained by the original fitness tie")
        previous = row["actual_model_state_hash_after"]
    require(curve[selected["selected_generation"] - 1]["actual_model_state_hash_after"] == actual_hash
            and len(curve[0].get("dev_environment_seeds", [])) == 1
            and curve[0]["dev_environment_seeds"] == curve[1].get("dev_environment_seeds"),
            "FAST selected state/development seed differs from its two-generation curve")
    if changed:
        require(selected.get("zero_update_proof") is None, "Updated FAST policy cannot claim a zero-update proof")
    else:
        proof = selected.get("zero_update_proof")
        require(isinstance(proof, dict) and proof.get("schema") == "verified_fast_es_zero_selected_update_v1"
                and proof.get("complete") is True and proof.get("selected_matches_initial") is True
                and proof.get("selected_generation") == selected["selected_generation"]
                and proof.get("initial_actual_model_state_hash") == initial_hash
                and proof.get("original_formula_verified") is True and proof.get("initial_dev_evaluated") is False
                and proof.get("generations") == selected.get("generation_curve")
                and [row.get("generation") for row in proof.get("generations", [])] == [1, 2],
                "Unchanged FAST checkpoint lacks its independently verified tie/update evidence")


class CoreFinalLedger:
    """Atomic JSON ledger; caller must hold the exclusive runner lock.

    Only one physical attempt is permitted per job. This deliberately provides
    no failed->pending transition. A complete envelope left before a commit
    interruption can be verified and reconciled without another physical call.
    """
    def __init__(self, path, manifest, *, readonly=False):
        self.path, self.manifest = Path(path), manifest
        self.readonly = readonly
        validate_core_final_manifest(manifest)
        self.fingerprint = manifest["fingerprint"]
        if self.path.exists():
            self.data = read(self.path)
            self.validate()
        else:
            require(not readonly, "Core final ledger does not exist")
            self.data = {"schema": "core_final_ledger_v1", "manifest_fingerprint": self.fingerprint,
                "scope": SCOPE, "global_study_complete": False,
                "jobs": {job["job_id"]: {"state": "pending", "attempt_number": 0, "attempt_id": None,
                    "result_path": None, "result_sha256": None} for job in manifest["jobs"]}, "events": []}
            self.save()

    def validate(self):
        require(self.data.get("schema") == "core_final_ledger_v1" and self.data.get("manifest_fingerprint") == self.fingerprint
                and self.data.get("scope") == SCOPE and self.data.get("global_study_complete") is False,
                "Core ledger belongs to another protocol or namespace")
        require(set(self.data.get("jobs", {})) == {j["job_id"] for j in self.manifest["jobs"]}, "Core ledger omitted/added jobs")
        claims = {key: 0 for key in self.data["jobs"]}
        for index, event in enumerate(self.data["events"]):
            require(event["sequence"] == index and event["job_id"] in claims, "Core ledger event history changed")
            if event["event"] == "claimed":
                claims[event["job_id"]] += 1
        for job_id, state in self.data["jobs"].items():
            require(state["state"] in STATES and state["attempt_number"] == claims[job_id] and claims[job_id] <= 1,
                    "Core ledger contains an unregistered/repeated physical attempt")
            require((state["state"] == "pending") == (state["attempt_number"] == 0), "Core pending/attempt history mismatch")
            if state["state"] == "succeeded":
                require(state["result_path"] and state["result_sha256"], "Completed core job lacks a verified result")

    def save(self):
        require(not self.readonly, "A read-only core ledger cannot be changed")
        self.validate()
        write_json_atomic(self.path, self.data)

    def event(self, job_id, event, **details):
        self.data["events"].append({"sequence": len(self.data["events"]), "time": time.time(),
                                    "job_id": job_id, "event": event, "details": details})

    def claim(self, job_id):
        state = self.data["jobs"][job_id]
        require(state["state"] == "pending" and state["attempt_number"] == 0, "Unknown/failed core job cannot be blindly replayed")
        attempt = uuid.uuid4().hex
        state.update(state="running", attempt_number=1, attempt_id=attempt)
        self.event(job_id, "claimed", attempt_id=attempt)
        self.save()
        return attempt

    def quarantine_running(self):
        for job_id, state in self.data["jobs"].items():
            if state["state"] == "running":
                state["state"] = "orphaned"
                self.event(job_id, "orphaned", attempt_id=state["attempt_id"], automatic_replay=False)
        self.save()

    def finish(self, job_id, result_path, result_sha256, *, reconciled=False):
        state = self.data["jobs"][job_id]
        require(state["state"] in {"running", "orphaned"}, "Unexpected core completion transition")
        require(file_sha256(result_path) == result_sha256, "Core result changed before ledger commit")
        state.update(state="succeeded", result_path=str(Path(result_path).resolve()), result_sha256=result_sha256)
        self.event(job_id, "reconciled_complete" if reconciled else "succeeded", attempt_id=state["attempt_id"],
                   result_path=state["result_path"], result_sha256=result_sha256, physical_replay=False)
        self.save()

    def halt(self, job_id, error, *, closed=False, evidence_files=(), observed_costs=None):
        state = self.data["jobs"][job_id]
        require(state["state"] in {"running", "orphaned"}, "Unexpected core halt transition")
        state["state"] = "failed" if closed else "orphaned"
        details = {"attempt_id": state["attempt_id"], "error_type": type(error).__name__, "error": str(error),
                   "execution_closed": closed, "observed_costs": observed_costs, "automatic_replay": False,
                   "artifacts": [{"path": str(p), "sha256": file_sha256(p)} for p in evidence_files]}
        state["halt"] = details
        self.event(job_id, state["state"], **details)
        self.save()

    def completion(self):
        counts = {name: sum(s["state"] == name for s in self.data["jobs"].values()) for name in sorted(STATES)}
        return {"scope": SCOPE, "expected_jobs": 4, "completed_jobs": counts["succeeded"],
                "core_final_complete": counts["succeeded"] == 4, "global_study_complete": False,
                "states": counts, "expected_counts": deepcopy(self.manifest["expected_counts"]), "ledger_fingerprint": fingerprint(self.data)}


def verify_core_envelope(path, job, manifest, *, tasks, expected_profile=None):
    """Reuse the full physical verifier without invoking the full-study ledger."""
    from .final_evaluation import verify_final_envelope
    validate_core_final_manifest(manifest)
    budget, core = core_manifest_budget(manifest)
    envelope = read(path)
    require(envelope.get("scope") == SCOPE and envelope.get("core_protocol_fingerprint") == manifest["core_protocol_fingerprint"]
            and envelope.get("global_study_complete") is False, "Result does not belong to the registered independent core")
    profile = envelope["execution_profile"]
    require(profile.get("core_protocol_fingerprint") == manifest["core_protocol_fingerprint"]
            and profile.get("mace_num_workers") == 4 and profile.get("orb_num_workers") == 1,
            "Core final policy/evaluator contract changed")
    selected = profile.get("selected_es")
    if job["method"] == "baseline":
        require(selected is None and profile["actual_model_state_hash"] == profile["initial_actual_model_state_hash"], "Baseline was not the clean fixed initial weights")
    else:
        verify_selected_evolution(selected, profile["initial_actual_model_state_hash"],
                                  profile["actual_model_state_hash"], fast=budget == 10)
    require(envelope["execution_job"].get("source") == job.get("source", "registered_core_final_evaluation"), "Core final mislabeled as full study/training")
    verified = verify_final_envelope(path, job, manifest["fingerprint"], tasks=tasks,
                                     expected_profile=expected_profile, expected_mace_num_workers=4,
                                     expected_made_budget=budget, core_protocol=core)
    for episode in envelope["episodes"]:
        # The shared verifier has already reconstructed this complete curve
        # from official RPC, so mSUN is linked to the same physical evidence.
        require(episode["metrics"].get("mSUN") == episode["discovery_curve"][-1][1] / budget,
                "Core mSUN differs from the verified official discovery curve")
    return verified


class CoreFinalRunner:
    def __init__(self, project, *, output=None, policy_factory=None, rollout_factory=None,
                 controller_loader=None, selected_loader=None):
        from .core_protocol import read_core, final_jobs, manifest_input_files, collection_manifest_paths, corpus_contract
        self.core = read_core(project)
        self.project = Path(self.core["workspace"]).resolve()
        self.output = Path(output).resolve() if output else self.project / "experiments/core_final"
        require(self.output.is_relative_to(self.project / "experiments") and self.output.name != "final_evaluation",
                "Core final output must stay in its independent workspace namespace")
        self.manifest = build_core_final_manifest(self.core["fingerprint"], final_jobs(self.core), core_protocol=self.core)
        self.protocol = read(self.project / "configs/main_protocol.json")
        self.tasks = read(self.project / "configs/benchmark_tasks.json")
        from .policy import declared_runtime_options
        self.runtime = declared_runtime_options(self.protocol)
        critical = [self.project / name for name in ("configs/main_protocol.json", "configs/model_manifest.json",
            "configs/benchmark_tasks.json", "configs/made_splits.json", "configs/assets.runtime.json",
            "configs/policy_runtime_gates.json", "data/raw/materials_project/index.json")]
        self.inputs = {str(Path(p).resolve()): file_sha256(p) for p in (*manifest_input_files(self.core), *critical)}
        self.corpus = corpus_contract(self.core, collection_manifest_paths(self.core))
        self.policy_factory, self.rollout_factory = policy_factory, rollout_factory
        self.controller_loader, self.selected_loader = controller_loader, selected_loader
        self.risk_root = self.project / "experiments/risk_models/qwen35_4b/made"
        from .core_representation import paths_for
        self.transcoder_manifest = paths_for(self.core)["bank"] / "transcoder_manifest.json"
        self.execution = {"scope": SCOPE, "core_protocol_fingerprint": self.core["fingerprint"], "inputs": self.inputs,
                          "runtime": self.runtime, "corpus_contract_fingerprint": fingerprint(self.corpus),
                          "risk_root": str(self.risk_root), "transcoder_manifest": str(self.transcoder_manifest)}

    def unchanged(self):
        require(all(file_sha256(path) == digest for path, digest in self.inputs.items()), "Core execution input/source changed")

    @contextmanager
    def lock(self):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / ".runner.lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CoreFinalError("Another core final runner owns the physical trajectories") from exc
            yield

    def load_policy(self):
        if self.policy_factory:
            return self.policy_factory("qwen35_4b")
        from .policy import QwenPolicyAdapter, DecodingConfig
        decode = self.protocol["decoding"]
        return QwenPolicyAdapter.from_verified_checkpoint(self.project / "configs/model_manifest.json", "qwen35_4b",
            self.project / "data/models/qwen35_4b", **{key: self.runtime[key] for key in (
                "device", "dtype", "attn_implementation", "sdpa_backend", "cpu_embedding_and_lm_head", "attention_checkpointing")},
            decoding=DecodingConfig(**{k: decode[k] for k in ("max_new_tokens", "temperature", "top_p", "top_k")}),
            enable_thinking=decode["enable_thinking"], max_input_tokens=decode["max_input_tokens"])

    def profile(self, policy, job, base_state, base_hash):
        from .esopt import tensor_state_hash
        from .training_jobs import load_frozen_controllers
        from .core_es import load_core_selected_es
        policy.model.load_state_dict(base_state, strict=True)
        policy.mark_state("reload", generation=0)
        require(tensor_state_hash(dict(policy.model.state_dict())) == base_hash, "Core initial weight restore failed")
        selected, files = None, {}
        if job["method"] == "esopt_graph_risk":
            loader = self.selected_loader or load_core_selected_es
            directory = self.project / "experiments/core_es"
            selected = loader(policy, directory, core_protocol=self.core)
            files.update(selected["artifacts"])
            verify_selected_evolution(selected, base_hash, selected["actual_model_state_hash"],
                                      fast=self.manifest.get("execution_budget") == 10)
        loader = self.controller_loader or load_frozen_controllers
        method = "graph_risk" if job["method"] == "esopt_graph_risk" else "entropy_risk"
        risk, graph, auxiliary = loader(policy, model_key="qwen35_4b", benchmark="made", method=job["method"],
            risk_checkpoint=self.risk_root / (method + ".pt"),
            transcoder_manifest=self.transcoder_manifest if job["method"] == "esopt_graph_risk" else None,
            graph_config=self.protocol["graph"], device=self.runtime["device"],
            failure_control=self.protocol.get("failure_control"), corpus_contract=self.corpus)
        files.update({str(Path(path).resolve()): file_sha256(path) for path in auxiliary})
        actual_hash = tensor_state_hash(dict(policy.model.state_dict()))
        require(actual_hash == (selected["actual_model_state_hash"] if selected else base_hash), "Unexpected core policy weights")
        profile = {"scope": SCOPE, "core_protocol_fingerprint": self.core["fingerprint"], "model_key": "qwen35_4b",
            "benchmark": "made", "method": job["method"], "seed": 1 if selected else None, "property": None,
            "actual_model_state_hash": actual_hash, "initial_actual_model_state_hash": base_hash,
            "initial_checkpoint_manifest_hash": policy.checkpoint_hash, "policy_configuration_fingerprint": policy.configuration_fingerprint,
            "execution_inputs_fingerprint": fingerprint(self.execution), "corpus_contract_fingerprint": fingerprint(self.corpus),
            "artifacts": files, "mace_num_workers": 4, "orb_num_workers": 1,
            "selected_es": selected, "auxiliary_modules_attached_to_policy": False}
        return profile, risk, graph

    def run(self):
        import torch
        from .esopt import tensor_state_hash
        from .final_evaluation import execution_job
        from .training_jobs import TrainingJobCallbacks, reconstruct_scientific_evidence, _rpc_pairs
        from .rollouts import DiscoveryRollout, RolloutSettings
        rollout_factory = self.rollout_factory or DiscoveryRollout
        budget, budget_core = core_manifest_budget(self.manifest)
        torch.set_num_threads(self.runtime["torch_cpu_threads"])
        if torch.device(self.runtime["device"]).type == "cuda":
            torch.cuda.set_per_process_memory_fraction(self.runtime["cuda_memory_fraction"], device=self.runtime["device"])
        with self.lock():
            self.unchanged()
            for name, value in (("manifest.json", self.manifest), ("execution_inputs.json", self.execution)):
                path = self.output / name
                if path.exists():
                    require(read(path) == value, "Existing core execution matrix/provenance differs")
                else:
                    write_json_atomic(path, value)
            jobs_root = self.output / "jobs"
            require((self.output / "ledger.json").exists() or not jobs_root.exists() or not any(jobs_root.iterdir()),
                    "Untracked core physical output exists without its durable ledger")
            ledger = CoreFinalLedger(self.output / "ledger.json", self.manifest)
            ledger.quarantine_running()
            if jobs_root.exists():
                for folder in jobs_root.iterdir():
                    require(folder.name in ledger.data["jobs"] and folder.is_dir() and not folder.is_symlink(), "Unknown/aliased core final job output")
                    tracked = ledger.data["jobs"][folder.name]
                    require(tracked["state"] != "pending" and {p.name for p in folder.iterdir()} == {tracked["attempt_id"]},
                            "Unregistered or duplicated core final physical attempt")
            # Unknown physical work is refused before loading another model.
            for job_id, state in ledger.data["jobs"].items():
                if state["state"] == "failed":
                    raise CoreFinalError("Failed core job requires explicit reconciliation; no replay: " + job_id)
                if state["state"] == "orphaned":
                    require((self.output / "jobs" / job_id / state["attempt_id"] / "result.json").is_file(),
                            "Unknown/partial core outcome requires reconciliation; no physical replay: " + job_id)
            policy = self.load_policy()
            base_state = {name: value.detach().cpu().clone() for name, value in policy.model.state_dict().items()}
            base_hash = tensor_state_hash(base_state)
            current_method, profile, risk, graph = None, None, None, None
            verified_profiles = {}
            for job in self.manifest["jobs"]:
                if current_method != job["method"]:
                    risk = graph = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    profile, risk, graph = self.profile(policy, job, base_state, base_hash)
                    verified_profiles[job["method"]] = profile
                    current_method = job["method"]
                self.unchanged()
                state = ledger.data["jobs"][job["job_id"]]
                if state["state"] in {"succeeded", "orphaned"}:
                    path = self.output / "jobs" / job["job_id"] / state["attempt_id"] / "result.json"
                    proof = verify_core_envelope(path, job, self.manifest, tasks=self.tasks, expected_profile=profile)
                    if state["state"] == "succeeded":
                        require(state["result_path"] == str(path.resolve()) and state["result_sha256"] == proof["result_sha256"], "Core committed result changed")
                    else:
                        ledger.finish(job["job_id"], path, proof["result_sha256"], reconciled=True)
                    continue
                attempt = ledger.claim(job["job_id"])
                output = self.output / "jobs" / job["job_id"] / attempt
                require(not output.exists(), "Core final attempt directory already exists")
                actual_job = execution_job(job, policy, profile)
                actual_job["source"] = job.get("source", "registered_core_final_evaluation")
                actual_job["group_id"] = job.get("group_id", actual_job["group_id"])
                stamp = policy.get_state_id()
                settings = RolloutSettings(max_generation_retries=self.protocol["risk_network"]["maximum_candidate_generations_per_decision"],
                    risk_threshold=self.protocol["risk_network"]["threshold"], history_results=self.protocol["memory"]["scientific_history_per_episode"],
                    recent_tools=self.protocol["memory"]["recent_tool_responses"], failure_aware_control=True)
                try:
                    rollout_factory(self.project, policy, settings=settings, risk_model=risk, attributor=graph).run(actual_job, output, collection=False)
                    require(policy.get_state_id() == stamp and tensor_state_hash(dict(policy.model.state_dict())) == profile["actual_model_state_hash"], "Final trajectory changed LLM weights")
                    evidence = reconstruct_scientific_evidence(output, actual_job, tasks=self.tasks, partition="final_test", expected_mace_num_workers=4,
                                                              expected_made_budget=budget, core_protocol=budget_core)
                    envelope = {"scope": SCOPE, "core_protocol_fingerprint": self.core["fingerprint"], "global_study_complete": False,
                        "job_id": job["job_id"], "manifest_fingerprint": self.manifest["fingerprint"], "model_revision": job["model_revision"],
                        "complete": True, "episodes": read(output / "episodes.json"), "artifacts": TrainingJobCallbacks._inventory(output),
                        "evidence_schema": "final_official_rpc_v1", "scientific_evidence": evidence, "execution_job": actual_job,
                        "execution_profile": profile, "execution_profile_fingerprint": fingerprint(profile)}
                    path = output / "result.json"
                    write_json_atomic(path, envelope)
                    self.unchanged()
                    proof = verify_core_envelope(path, job, self.manifest, tasks=self.tasks, expected_profile=profile)
                    ledger.finish(job["job_id"], path, proof["result_sha256"])
                except BaseException as exc:
                    closed = False
                    try:
                        _rpc_pairs(output / "rpc/rpc.jsonl")
                        closed = not (output / "result.json").exists()
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                    ledger.halt(job["job_id"], exc, closed=closed,
                        evidence_files=[p for p in output.rglob("*") if p.is_file()] if output.exists() else ())
                    raise
                write_json_atomic(self.output / "progress.json", ledger.completion())
            completion = ledger.completion()
            require(completion["core_final_complete"], "Core final still lacks one or more registered-budget jobs")
            self.unchanged()
            for job in self.manifest["jobs"]:
                state = ledger.data["jobs"][job["job_id"]]
                proof = verify_core_envelope(state["result_path"], job, self.manifest, tasks=self.tasks,
                                             expected_profile=verified_profiles[job["method"]])
                require(proof["result_sha256"] == state["result_sha256"], "Earlier core result changed before final acceptance")
            completion.update(complete=True, core_protocol_fingerprint=self.core["fingerprint"],
                manifest_fingerprint=self.manifest["fingerprint"], result_files=[{
                    "path": ledger.data["jobs"][job["job_id"]]["result_path"],
                    "sha256": ledger.data["jobs"][job["job_id"]]["result_sha256"]} for job in self.manifest["jobs"]])
            if budget == 10:
                selected = verified_profiles["esopt_graph_risk"]["selected_es"]
                completion.update(execution_budget=10, registered_execution_scope=self.core["scope"],
                    original_b50_core_complete=False, evolution_status=selected["evolution_status"],
                    evolution_signal_observed=selected["evolution_signal_observed"], scientific_improvement_assumed=False)
            write_json_atomic(self.output / "progress.json", completion)
            publish_once(self.output / "completion.json", completion)
            stage = {"schema": "deadline_core_stage_receipt_v1", "stage": "final", "core_fingerprint": self.core["fingerprint"],
                "complete": True, "scope": SCOPE, "core_final_complete": True, "global_study_complete": False,
                "original_full_study_complete": False,
                "core_protocol_fingerprint": self.core["fingerprint"], "completion": {
                    "path": str(self.output / "completion.json"), "sha256": file_sha256(self.output / "completion.json")},
                "artifacts": [{"path": str(self.output / name), "sha256": file_sha256(self.output / name)}
                              for name in ("completion.json", "manifest.json", "ledger.json", "execution_inputs.json")]}
            stage["fingerprint"] = fingerprint(stage)
            publish_once(self.project / "experiments/core_stage_receipts/final.json", stage)
            return completion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = CoreFinalRunner(args.project, output=args.output).run()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
