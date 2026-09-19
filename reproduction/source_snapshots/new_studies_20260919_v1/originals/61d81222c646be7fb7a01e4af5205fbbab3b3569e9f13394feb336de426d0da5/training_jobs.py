"""Protocol-locked ES jobs and read-only reconstruction of scientific fitness.

This module never supplies a surrogate reward and never retries an interrupted
physical trajectory. A complete receipt is written only after raw RPC evidence,
episode summaries, fixed evaluators, token records and budgets agree.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from .accounting import file_sha256, fingerprint, write_json_atomic
from .es_training import (
    ESTrainingConfig, EvaluationRequest, EvaluationResult, TrainingCase,
    TrainingContractError, TrainingHalted, verify_clean_reload_checkpoint,
)
from .esopt import tensor_state_hash


METHODS = ("esopt", "esopt_graph_risk")
MODELS = ("qwen35_4b", "qwen35_9b")
PROPERTIES = ("bm", "density", "band_gap")
TRAIN_PROTOTYPES = {"C2": 630, "C3": 2271, "C4": 8666, "C5": 8906, "C6": 8354}
SEED_RULE = "4*(request.environment_seed mod 2**30) + (0 for train; 2+episode_index for dev)"
EVIDENCE_SCHEMA = "es_official_trajectory_evidence_v1"


def _read(path: Path):
    def invalid(value):
        raise TrainingContractError(f"Nonfinite JSON in {path}: {value}")
    return json.loads(path.read_text(), parse_constant=invalid)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise TrainingContractError(message)


def _number(value, name, *, nonnegative=False):
    _require(type(value) in (int, float) and math.isfinite(value)
             and (not nonnegative or value >= 0), f"Invalid measured {name}")
    return value


def _registered_made_budget(expected_made_budget, core_protocol=None):
    """B50 remains the default; B10 needs the independently sealed FAST core."""
    _require(type(expected_made_budget) is int and expected_made_budget in {10, 50}, "Unsupported explicit MADE execution budget")
    if expected_made_budget == 50 and core_protocol is None:
        return 50
    from .core_protocol import FAST_CAUSAL_GRAPH_REGISTRATION, core_execution_budget, read_core
    _require(isinstance(core_protocol, Mapping), "Nondefault MADE budget requires its sealed core protocol")
    verified = read_core(core_protocol["workspace"])
    _require(verified == dict(core_protocol) and core_execution_budget(verified) == expected_made_budget,
             "MADE execution budget differs from its sealed core")
    _require(expected_made_budget == 50 or verified["registration"] == FAST_CAUSAL_GRAPH_REGISTRATION,
             "Only the explicit FAST registration admits B10 execution")
    return expected_made_budget


@dataclass(frozen=True)
class TrainingCondition:
    project: Path
    model_key: str
    benchmark: str
    method: str
    seed: int
    property_name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "project", Path(self.project).resolve())
        _require(self.model_key in MODELS and self.method in METHODS, "Only the two locked Qwen sizes and independent ES arms are supported")
        _require(type(self.seed) is int and self.seed in range(1, 6), "The full study uses training seeds 1 through 5")
        _require(self.benchmark in {"made", "crystalgym"}, "Unknown benchmark")
        _require((self.benchmark == "made" and self.property_name is None)
                 or (self.benchmark == "crystalgym" and self.property_name in PROPERTIES),
                 "CrystalGym requires exactly one property; MADE uses AUDC")

    def output_directory(self, root: Path) -> Path:
        return Path(root).resolve() / self.model_key / self.benchmark / (self.property_name or "audc") / self.method / f"seed_{self.seed}"

    def protocol(self) -> dict:
        protocol = _read(self.project / "configs/main_protocol.json")
        _require(protocol["models"] == list(MODELS) and protocol["training_seeds"] == [1, 2, 3, 4, 5], "The declared complete model/seed matrix changed")
        fixed = {"generations": 16, "population": 8, "cases_per_generation": 1,
                 "alpha": .0005, "sigma_start": .001, "sigma_end": .0002,
                 "validation_every_generations": 4, "schedule": "cosine",
                 "parameter_scope": "full_HF_policy_including_retained_unused_visual_parameters",
                 "standardization": "population_z_score_ddof0", "extra_inverse_sigma": False,
                 "independent_training_arms": list(METHODS)}
        _require(all(protocol["esopt"].get(k) == v for k, v in fixed.items()), "ES protocol must retain all 16 generations, eight candidates and full parameter scope")
        _require(protocol["collection"]["made"] == {"train_systems": 30, "development_systems": 12,
                 "episodes_per_system": 5, "oracle_attempts_per_episode": 50}, "MADE collection/split protocol changed")
        crystal = protocol["collection"]["crystalgym"]
        _require(crystal["training_prototypes"] == 5 and crystal["development_rollouts_per_prototype_and_seed"] == 2,
                 "CrystalGym must retain all five training prototypes and two dev episodes")
        return protocol

    def config(self) -> ESTrainingConfig:
        es = self.protocol()["esopt"]
        return ESTrainingConfig(benchmark=self.benchmark, method=self.method, seed=self.seed,
                               property_name=self.property_name, dev_every=es["validation_every_generations"],
                               **{k: es[k] for k in ("generations", "population", "cases_per_generation", "alpha", "sigma_start", "sigma_end")})

    def cases(self) -> tuple[tuple[TrainingCase, ...], tuple[TrainingCase, ...]]:
        self.protocol()
        tasks = _read(self.project / "configs/benchmark_tasks.json")
        result = {"train": [], "dev": []}
        if self.benchmark == "made":
            splits = _read(self.project / "configs/made_splits.json")["splits"]
            normalized = {split: [tuple(sorted(elements)) for elements in splits[split]] for split in ("train", "dev", "test")}
            _require(all(len(normalized[s]) == n and len(set(normalized[s])) == n for s, n in (("train", 30), ("dev", 12), ("test", 30))), "Full 30/12/30 MADE partitions are required")
            _require(not (set(normalized["train"]) & set(normalized["dev"]) or set(normalized["train"]) & set(normalized["test"]) or set(normalized["dev"]) & set(normalized["test"])), "Held-out chemical systems cannot enter ES training/development")
            official = {tuple(sorted(row["elements"])) for row in tasks["made"]["systems"]}
            _require(set(normalized["test"]) == official, "All official MADE main systems must remain held out")
            for split in result:
                for elements in sorted(normalized[split]):
                    task_id = "-".join(elements)
                    result[split].append(TrainingCase(f"{split}:made:{task_id}", split, "made",
                        payload={"task_id": task_id, "task": {"id": task_id, "elements": list(elements)}}))
        else:
            crystal = tasks["crystalgym"]
            prototypes = [p for p in crystal["prototypes"] if p["split"] == "train"]
            _require(len(prototypes) == 5 and {p["id"]: p["index"] for p in prototypes} == TRAIN_PROTOTYPES,
                     "Use exactly C2..C6; C1/C7 and held-out indices are forbidden")
            _require(set(crystal["train_indices"]) == set(TRAIN_PROTOTYPES.values())
                     and set(crystal["heldout_indices"]) == {3403, 2190}, "CrystalGym split indices changed")
            prop = next(p for p in crystal["properties"] if p["id"] == self.property_name)
            fixed = {"bm": (500., "GPa"), "density": (5., "g/cm^3"), "band_gap": (2., "eV")}[self.property_name]
            _require((prop["target"], prop["unit"]) == fixed, "The official property target or units changed")
            for split in result:
                for prototype in sorted(prototypes, key=lambda p: p["id"]):
                    task_id = self.property_name + ":" + prototype["id"]
                    result[split].append(TrainingCase(f"{split}:crystalgym:{task_id}", split, "crystalgym", self.property_name,
                        {"task_id": task_id, "task": {"property": {k: prop[k] for k in ("id", "target", "unit")}, "prototype": prototype},
                         "development_kind": "same_prototype_independent_seed" if split == "dev" else None}))
        return tuple(result["train"]), tuple(result["dev"])


def make_training_job(condition: TrainingCondition, request: EvaluationRequest, *, model_entry: Mapping,
                      policy_configuration_fingerprint: str, expected_made_budget: int = 50, core_protocol=None) -> dict:
    """Pure job construction; population index NEVER changes random seeds."""
    _require(request.phase in {"train", "dev"} and request.case.split == request.phase, "No test/held-out training jobs")
    _require(request.case.benchmark == condition.benchmark and request.case.property_name == condition.property_name
             and request.method == condition.method, "Request belongs to another benchmark, property or ES arm")
    train, dev = condition.cases()
    catalog = {case.case_id: case for case in train + dev}
    _require(request.case.case_id in catalog and asdict(request.case) == asdict(catalog[request.case.case_id]), "Request is outside the complete frozen train/dev case catalog")
    _require(type(request.environment_seed) is int and 0 <= request.environment_seed < 2**32, "Invalid driver environment seed")
    episodes = 2 if condition.benchmark == "crystalgym" and request.phase == "dev" else 1
    # The native adapter increments the initial seed on reset. Reserve disjoint
    # blocks so dev never reuses a train seed, including its second episode.
    first_seed = 4 * (request.environment_seed % 2**30) + (2 if request.phase == "dev" else 0)
    seeds = [first_seed + episode for episode in range(episodes)]
    _registered_made_budget(expected_made_budget, core_protocol)
    budget = expected_made_budget if condition.benchmark == "made" else episodes
    return {"stage": "es_train" if request.phase == "train" else "es_dev", "split": request.phase,
            "benchmark": condition.benchmark, "method": condition.method, "model_key": condition.model_key,
            "model_id": model_entry["model_id"], "model_revision": model_entry["revision"],
            "task_id": request.case.payload["task_id"], "task": copy.deepcopy(request.case.payload["task"]),
            "job_id": "es-" + request.request_id, "group_id": f"{request.case.case_id}:training_seed:{condition.seed}:generation:{request.generation}",
            "seed": request.environment_seed, "training_seed": condition.seed, "environment_seeds": seeds,
            "environment_seed_derivation": SEED_RULE, "episode_ids": [str(i) for i in range(episodes)],
            "budget": budget, "budget_unit": "candidate_oracle_attempts_per_episode" if condition.benchmark == "made" else "dft_episode_attempts",
            "expected_counts": {"episodes": episodes, "candidate_oracle_attempts": budget if condition.benchmark == "made" else 0,
                                "dft_episode_attempts": episodes if condition.benchmark == "crystalgym" else 0},
            "training_request": request.to_dict(), "source": "full_agentic_es_scientific_training_callback",
            "policy_configuration_fingerprint": policy_configuration_fingerprint,
            "actual_model_state_hash": request.actual_model_state_hash,
            "initial_checkpoint_manifest_hash": request.initial_checkpoint_manifest_hash,
            "policy_state_id": request.policy_state_id}


def _rpc_pairs(path: Path) -> list[tuple[dict, dict, dict]]:
    pairs, pending = [], None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        _require("worker_exception" not in row, "RPC trajectory has an unresolved worker exception")
        if row.get("direction") == "request":
            _require(pending is None, "RPC request has an unknown outcome")
            pending = row["payload"]
        elif row.get("direction") == "response":
            response = row["payload"]
            _require(pending is not None and pending["id"] == response["id"], "RPC response correlation mismatch")
            _number(row["elapsed_seconds"], "RPC duration", nonnegative=True)
            pairs.append((pending, response, row))
            pending = None
        else:
            raise TrainingContractError("Unknown RPC evidence row")
    _require(pending is None and pairs and pairs[0][0]["op"] == "init" and pairs[-1][0]["op"] == "close"
             and pairs[-1][1].get("ok") is True, "Only cleanly closed complete RPC trajectories can provide fitness")
    return pairs


def reconstruct_scientific_evidence(output: Path, job: Mapping, *, tasks: Mapping, partition: str = "training",
                                    expected_mace_num_workers: int | None = None, expected_made_budget: int = 50,
                                    core_protocol=None, made_all30_extension=None, made_budget_sweep=None, component_registration=None) -> dict:
    """Recompute official fitness from raw RPC, then audit summary/cost claims."""
    output = Path(output)
    if component_registration is not None:
        _require(core_protocol is None and made_all30_extension is None and made_budget_sweep is None, "Component scope must be exclusive")
        from .component_scope import validate_component_reconstruction
        validate_component_reconstruction(component_registration, job, expected_made_budget, partition)
    elif made_budget_sweep is not None:
        from .made_budget_sweep import validate_job_scope
        _require(made_all30_extension is None and partition == "final_test", "Sweep admission is mutually exclusive and final evaluation only")
        validate_job_scope(made_budget_sweep, job, expected_made_budget=expected_made_budget,
                           core_protocol=core_protocol, execution=True)
    elif made_all30_extension is not None:
        from .made_all30_extension import validate_job_scope
        _require(partition == "final_test", "All30 admission is final evaluation only")
        validate_job_scope(made_all30_extension, job, expected_made_budget=expected_made_budget,
                           core_protocol=core_protocol, execution=True)
    else:
        _registered_made_budget(expected_made_budget, core_protocol)
    if job["benchmark"] == "made":
        _require(job["budget"] == expected_made_budget and job["expected_counts"]["candidate_oracle_attempts"] == expected_made_budget,
                 "MADE job differs from the independently expected execution budget")
    _require(partition in {"training", "final_test"}, "Unknown scientific-evidence partition")
    _require((job["stage"] == "final_eval" and job["split"] == "test") if partition == "final_test"
             else job["stage"] in {"es_train", "es_dev"} and job["split"] in {"train", "dev"},
             "Scientific evidence cannot cross the training/final-test boundary")
    recorded = _read(output / "job.json")
    _require(all(recorded.get(key) == value for key, value in job.items()), "Saved rollout job differs from its ES request")
    summaries = _read(output / "episodes.json")
    _require(isinstance(summaries, list) and len(summaries) == job["expected_counts"]["episodes"], "Incomplete episode matrix")
    pairs = _rpc_pairs(output / "rpc/rpc.jsonl")
    init_request, init_response, _ = pairs[0]
    _require(init_response.get("ok") is True, "Scientific evaluator did not initialize")
    arguments, initialized = init_request["args"], init_response["result"]
    metadata = _read(output / "environment_metadata.json")
    _require(metadata == initialized["metadata"] and metadata["vendor"]["commit"] == tasks[job["benchmark"]]["official_commit"], "Evaluator metadata is not the locked official benchmark")
    _require(arguments["seed"] == job["environment_seeds"][0] and arguments["budget"] == job["budget"]
             and arguments["benchmark"] == job["benchmark"], "RPC task budget or seed differs from the job")
    if job["benchmark"] == "made":
        _require(arguments["elements"] == job["task"]["elements"] and metadata["oracle"] == tasks["made"]["oracle"]
                 and metadata["stability_tolerance"] == .1, "MADE fitness requires the fixed official ORB oracle and threshold")
        if expected_mace_num_workers is not None:
            from .execution_contract import verify_mace_workers
            verify_mace_workers(arguments, metadata, expected_mace_num_workers)
    else:
        prop, proto = job["task"]["property"], job["task"]["prototype"]
        prototypes = {"C1": 3403, "C7": 2190} if partition == "final_test" else TRAIN_PROTOTYPES
        _require(proto["id"] in prototypes and prototypes[proto["id"]] == proto["index"]
                 and metadata["prototype_index"] == proto["index"] and metadata["split"] == ("heldout" if partition == "final_test" else "train")
                 and metadata["property"] == prop["id"] and metadata["target"] == prop["target"], "CrystalGym evaluation has wrong property, target, or held-out prototype")
        _require(metadata["qe"]["calculation"] == ("vc-relax" if prop["id"] == "density" else "scf")
                 and metadata["qe"]["occupations"] == ("fixed" if prop["id"] == "band_gap" else "smearing")
                 and metadata["qe"]["ecutwfc"] == 50 and metadata["qe"]["ecutrho"] == 400,
                 "CrystalGym fitness requires the fixed Quantum ESPRESSO settings")
    observations = {0: initialized["observation"]}
    before = dict(initialized["observation"].get("counts", {}))
    initial_counts = dict(before)
    terminals, curve, last_stable, attempt = [], [[0, 0]], 0, 0
    for rpc_request, response, _ in pairs[1:]:
        operation = rpc_request["op"]
        if operation == "reset":
            _require(response.get("ok") is True, "An independent dev episode did not reset successfully")
            observation = response["result"]
            observations[observation["episode_index"]] = observation
            before = dict(observation["counts"])
        if operation != "step":
            continue
        if job["benchmark"] == "made":
            attempt += 1
            if response.get("ok") is True:
                result = response["result"]
                observation = result["observation"]
                last_stable = result["official_metrics"].get("num_newly_discovered_stable", 0)
            else:
                # Failures count only when the official attempt counter advanced.
                observation = response["error"]["details"]
            _require(observation["counts"].get("candidate_oracle_attempts", 0) == attempt,
                     "Every MADE step must account for exactly one real candidate oracle attempt")
            _require(type(last_stable) in (int, float) and math.isfinite(last_stable) and float(last_stable).is_integer() and 0 <= last_stable <= attempt, "Invalid official discovery count")
            curve.append([attempt, last_stable])
            terminals = [{"observation": observation, "before": initial_counts}]
        else:
            _require(response.get("ok") is True, "DFT step has an unresolved physical outcome")
            result = response["result"]
            if result["episode_done"]:
                scientific = result["scientific_result"]
                _require(scientific["terminal"] is True and scientific["property"] == job["task"]["property"]["id"], "Invalid official terminal DFT result")
                _number(scientific["reward"], "official DFT reward")
                terminals.append({"observation": result["observation"], "before": dict(before), "scientific": scientific})
    _require(len(terminals) == len(summaries), "RPC terminal results do not cover the declared episodes")
    decisions_path = output / "decisions.jsonl"
    _require(decisions_path.is_file(), "Complete policy decision evidence is missing")
    decisions = [json.loads(line) for line in decisions_path.read_text().splitlines() if line.strip()]
    _require(bool(decisions) and len({row["decision_id"] for row in decisions}) == len(decisions), "Decision evidence is empty or duplicated")
    for row in decisions:
        stamp = row["policy_stamp"]
        _require(row["split"] == job["split"] and row["model_key"] == job["model_key"] and row["task_id"] == job["task_id"]
                 and row["group_id"] == job["group_id"] and stamp["state_id"] == job["policy_state_id"]
                 and stamp["checkpoint_hash"] == job["initial_checkpoint_manifest_hash"], "Decision evidence has a wrong split, group, or policy stamp")
        tokens = Path(row["input_ids_file"]).resolve()
        _require(tokens.is_file() and output.resolve() in tokens.parents, "Actual generated token evidence is missing or outside its trajectory")
    values, costs = [], {}
    for episode, (summary, terminal) in enumerate(zip(summaries, terminals, strict=True)):
        observation = terminal["observation"]
        _require(summary.get("complete") is True and summary["episode_id"] == str(episode)
                 and observation["episode_index"] == episode and summary["environment_seed"] == job["environment_seeds"][episode]
                 and observation["episode_seed"] == job["environment_seeds"][episode], "Episode completion or actual environment seed mismatch")
        _require(all(summary.get(k) == job[k] for k in ("benchmark", "model_key", "method", "task_id", "seed")), "Episode identity mismatch")
        deltas = {key: value - terminal["before"].get(key, 0) for key, value in observation["counts"].items()}
        for key, value in deltas.items():
            # Rollout accounting adds initialization ORB calls to the first
            # episode separately; they are never candidate-query budget units.
            expected = value + initial_counts.get(key, 0) if episode == 0 and key == "initialization_oracle_attempts" else value
            _require(summary["costs"].get(key, 0) == expected, "Actual physical counter disagrees with episode summary: " + key)
        if job["benchmark"] == "made":
            _require(attempt == expected_made_budget and deltas.get("candidate_oracle_attempts") == expected_made_budget
                     and summary["discovery_curve"] == curve, "MADE ES requires its complete registered attempt-axis discovery curve")
            area = sum((x1-x0)*(y1+y0)/2 for (x0,y0),(x1,y1) in zip(curve, curve[1:]))
            metric = 2 * area / (expected_made_budget * expected_made_budget)
            _require(summary["metrics"]["AUDC"] == metric, "Summary AUDC differs from raw official attempt-axis evidence")
        else:
            _require(deltas.get("dft_episode_attempts") == 1, "Each CrystalGym ES episode must include one complete real DFT attempt")
            scientific = terminal["scientific"]
            metric = scientific["reward"]
            _require(summary["metrics"]["reward"] == metric and summary["metrics"]["property_value"] == scientific["property_value"], "Summary reward differs from the official DFT return")
        values.append(metric)
        these = [row for row in decisions if row["episode_index"] == episode]
        measured = {"llm_calls": len(these), "completion_tokens": sum(row["generation"]["completion_count"] for row in these),
                    "prompt_tokens": sum(row["generation"]["prompt_token_count"] for row in these),
                    "graph_seconds": sum(row.get("graph_seconds", 0) for row in these)}
        _require(all(summary["costs"].get(k) == v for k, v in measured.items()), "Policy/graph costs do not match raw decision records")
        for key, value in summary["costs"].items():
            costs[key] = costs.get(key, 0) + _number(value, key, nonnegative=True)
    _require(costs.get("initialization_oracle_attempts", 0) == initial_counts.get("initialization_oracle_attempts", 0), "Initialization ORB costs must be counted separately")
    for key in ("candidate_oracle_attempts", "dft_episode_attempts"):
        _require(costs.get(key, 0) == job["expected_counts"][key], "Prescribed physical budget is incomplete: " + key)
    if job["benchmark"] == "made" and expected_mace_num_workers == 4:
        from .mace_parallel import audit_oracle_attempt_journal
        journal = audit_oracle_attempt_journal(output / "environment/oracle_attempts.jsonl")
        _require(journal["valid"] and journal["closed"], "Parallel MADE oracle attempts lack a valid closed journal")
        for key in ("candidate_oracle_attempts", "initialization_oracle_attempts", "surrogate_oracle_attempts"):
            _require(journal["counts"].get(key, 0) == costs.get(key, 0), "Parallel oracle journal and physical costs disagree: " + key)
        _require(all(type(group["episode_index"]) is int and 0 <= group["episode_index"] < len(summaries)
                     for group in journal["by_role_phase_episode"]), "Oracle journal references an undeclared episode")
    costs["rpc_calls"] = len(pairs)
    costs["rpc_elapsed_seconds"] = sum(row["elapsed_seconds"] for _, _, row in pairs)
    costs["completed_episodes"] = len(summaries)
    return {"metric_name": "AUDC" if job["benchmark"] == "made" else "reward",
            "metric_value": sum(values) / len(values), "costs": costs,
            "episode_metric_values": values, "official_commit": metadata["vendor"]["commit"],
            "environment_seeds": job["environment_seeds"], "reward_is_llm_self_score": False}


class TrainingJobCallbacks:
    """Independent whole-trajectory callbacks; persisted results are read-only."""
    def __init__(self, condition: TrainingCondition, policy, output_dir: Path, *, settings,
                 risk_model=None, attributor=None, auxiliary_files=(), rollout_factory=None, expected_made_budget: int = 50,
                 core_protocol=None):
        from .rollouts import DiscoveryRollout
        self.condition, self.policy = condition, policy
        _registered_made_budget(expected_made_budget, core_protocol)
        self.expected_made_budget = expected_made_budget
        self.core_protocol = core_protocol
        self.output = Path(output_dir).resolve()
        self.settings, self.risk_model, self.attributor = settings, risk_model, attributor
        self.rollout_factory = rollout_factory or DiscoveryRollout
        _require((condition.method == "esopt_graph_risk") == (attributor is not None), "Graph ES requires a validated transcoder bank and native attributor")
        _require(condition.method != "esopt_graph_risk" or risk_model is not None, "Graph ES requires a calibrated risk checkpoint")
        manifest = _read(condition.project / "configs/model_manifest.json")
        self.model_entry = next(model for model in manifest["models"] if model["key"] == condition.model_key)
        _require(self.model_entry["model_id"] == policy.model_id and self.model_entry["revision"] == policy.revision, "Loaded policy differs from selected manifest checkpoint")
        self.tasks = _read(condition.project / "configs/benchmark_tasks.json")
        from .execution_contract import declared_mace_workers
        self.expected_mace_num_workers = (declared_mace_workers(_read(condition.project / "configs/main_protocol.json"))
                                         if condition.benchmark == "made" else None)
        files = [condition.project / name for name in ("configs/main_protocol.json", "configs/benchmark_tasks.json", "configs/made_splits.json", "configs/model_manifest.json")]
        files += sorted((condition.project / "src/matdiscovery").glob("*.py"))
        files += [condition.project / "scripts/train_es_condition.py"]
        files += [condition.project / name for name in (["configs/assets.runtime.json", "data/raw/materials_project/index.json"] if condition.benchmark == "made" else ["configs/qe.runtime.json"])]
        files += [Path(path).resolve() for path in auxiliary_files]
        self.inputs = {str(path.resolve()): file_sha256(path) for path in files}
        self.callback_identity = {"schema": EVIDENCE_SCHEMA, "inputs": self.inputs, "settings": asdict(settings),
                                  "environment_seed_rule": SEED_RULE, "policy_seed_rule": "request.environment_seed; same across a generation's candidates",
                                  "graph_artifacts": "frozen_initial_policy_transcoders_and_risk; never fitted on ES/dev/test trajectories",
                                  "reward_sources": "official RPC only; mean across fixed-budget episodes"}
        if expected_made_budget != 50:
            self.callback_identity["made_episode_budget"] = expected_made_budget
        self.fingerprint = fingerprint(self.callback_identity)

    def _unchanged(self):
        _require(all(file_sha256(path) == digest for path, digest in self.inputs.items()), "A source, protocol, evaluator input or frozen auxiliary artifact changed")

    def job_factory(self, request):
        self._unchanged()
        return make_training_job(self.condition, request, model_entry=self.model_entry,
                                 policy_configuration_fingerprint=self.policy.configuration_fingerprint,
                                 expected_made_budget=self.expected_made_budget, core_protocol=self.core_protocol)

    def _directory(self, request):
        return self.output / "rollouts" / request.request_id

    @staticmethod
    def _inventory(output):
        files = []
        for path in sorted(output.rglob("*")):
            if path.name == "completion_receipt.json":
                continue
            _require(not path.is_symlink(), "Raw result evidence cannot be a mutable external symlink")
            if path.is_file():
                _require(not path.name.endswith(".partial"), "A physical trajectory contains unfinished artifacts")
                files.append({"path": str(path.resolve()), "sha256": file_sha256(path), "size_bytes": path.stat().st_size})
        return files

    def _result(self, request, receipt):
        return EvaluationResult(request.request_id, receipt["evidence"]["metric_name"], receipt["evidence"]["metric_value"],
            request.reward_source, receipt["evidence"]["costs"],
            tuple(row["path"] for row in receipt["artifacts"]) + (str(self._directory(request) / "completion_receipt.json"),))

    def evaluate(self, policy, job, request):
        _require(policy is self.policy, "Callback received another policy instance")
        self._unchanged()
        output = self._directory(request)
        if output.exists():
            raise TrainingHalted("Existing physical trajectory requires read-only reconciliation, never an automatic retry")
        rollout = self.rollout_factory(self.condition.project, policy, settings=self.settings,
                                       risk_model=self.risk_model, attributor=self.attributor)
        rollout.run(job, output, collection=False)
        evidence = reconstruct_scientific_evidence(output, job, tasks=self.tasks,
            expected_mace_num_workers=self.expected_mace_num_workers, expected_made_budget=self.expected_made_budget,
            core_protocol=self.core_protocol)
        self._unchanged()
        receipt = {"schema": EVIDENCE_SCHEMA, "complete": True, "request_identity": request.semantic_identity(),
                   "job_fingerprint": fingerprint(job), "callback_fingerprint": self.fingerprint,
                   "evidence": evidence, "artifacts": self._inventory(output)}
        write_json_atomic(output / "completion_receipt.json", receipt)
        return self._result(request, receipt)

    def result_verifier(self, request, result):
        self._unchanged()
        output = self._directory(request)
        receipt = _read(output / "completion_receipt.json")
        job = self.job_factory(request)
        _require(receipt.get("complete") is True and receipt["schema"] == EVIDENCE_SCHEMA
                 and receipt["request_identity"] == request.semantic_identity()
                 and receipt["job_fingerprint"] == fingerprint(job) and receipt["callback_fingerprint"] == self.fingerprint,
                 "Complete trajectory receipt differs from the frozen request/callback")
        _require(self._inventory(output) == receipt["artifacts"], "Completed raw artifacts changed; refusing reward reuse")
        evidence = reconstruct_scientific_evidence(output, job, tasks=self.tasks,
            expected_mace_num_workers=self.expected_mace_num_workers, expected_made_budget=self.expected_made_budget,
            core_protocol=self.core_protocol)
        _require(evidence == receipt["evidence"] and fingerprint(asdict(result)) == fingerprint(asdict(self._result(request, receipt))), "Reported ES fitness/costs differ from verified scientific evidence")
        return {"verified": True, "evidence": "Closed raw official RPC + complete attempt/DFT budgets + episode/token records + SHA256 inventory",
                "metric_name": evidence["metric_name"], "metric_value": evidence["metric_value"], "costs": evidence["costs"]}

    def recover_result(self, request, durable_record):
        """Only reuse an existing completion receipt; never create or run a job."""
        self._unchanged()
        original = EvaluationRequest.from_dict(durable_record["request"])
        _require(original.semantic_identity() == request.semantic_identity(), "Cannot recover a trajectory evaluated under different weights")
        path = self._directory(original) / "completion_receipt.json"
        if not path.is_file():
            return None
        result = self._result(original, _read(path))
        self.result_verifier(original, result)
        return result


def load_frozen_controllers(policy, *, model_key: str, benchmark: str, method: str, risk_checkpoint: Path | None,
                            transcoder_manifest: Path | None, graph_config: Mapping, device: str,
                            failure_control: Mapping | None = None, corpus_contract: Mapping | None = None):
    """Load external auxiliaries only after provenance and all bank gates pass."""
    from .uncertainty import CalibratedRiskModel
    from .representation_training import validate_transcoder_bank, _verify_reused_bank_binding
    from .native_attribution import NativeAttributor, TranscoderBinding
    from .transcoders import load_transcoder
    _require(method in (*METHODS, "baseline", "entropy_risk", "hidden_risk", "graph_risk"), "Unknown policy method")
    needs_graph = method in {"graph_risk", "esopt_graph_risk"}
    typed_enabled = benchmark in (failure_control or {}).get("enabled_benchmarks", [])
    _require(not typed_enabled or benchmark == "made", "Failure-type control is currently validated for MADE only")
    _require(not typed_enabled or risk_checkpoint is not None, "Failure-aware evaluation requires fitted primary/type confidence even for the logging-only baseline")
    _require(not needs_graph or (risk_checkpoint is not None and transcoder_manifest is not None), "Graph methods need both risk and transcoder checkpoints")
    _require(needs_graph or transcoder_manifest is None, "This method does not use a transcoder bank")
    files, risk, attributor = [], None, None
    expected_corpus = None
    if corpus_contract is not None:
        from .core_protocol import validate_corpus_contract
        expected_corpus = validate_corpus_contract(corpus_contract, model_key=model_key)
        _require(benchmark == "made" and expected_corpus["benchmark"] == benchmark,
                 "Core controller contract must identify this MADE corpus")
        files += [Path(expected_corpus["protocol_path"])]
        files += [Path(item["path"]) for item in expected_corpus["manifests"]]
    if risk_checkpoint is not None:
        checkpoint = Path(risk_checkpoint).resolve()
        fit_path, schema_path = checkpoint.with_suffix(".fit.json"), checkpoint.with_suffix(".schema.json")
        fit, schema = _read(fit_path), _read(schema_path)
        expected_method = "graph_risk" if needs_graph else "hidden_risk" if method == "hidden_risk" else "entropy_risk"
        _require(fit["status"] == "succeeded" and fit["method"] == expected_method
                 and fit["checkpoint_sha256"] == file_sha256(checkpoint), "Risk checkpoint lacks a successful matching SHA256 fit receipt")
        risk = CalibratedRiskModel.load(checkpoint)
        provenance = risk.provenance
        collection = provenance["collection_provenance"]
        if expected_corpus is not None:
            paths = [Path(item["path"]).resolve() for item in collection["collection_manifests"]]
            validate_corpus_contract(expected_corpus, manifest_paths=paths, model_key=model_key)
            _require(len(paths) == expected_corpus["expected_collection_jobs"],
                     "Core primary risk corpus omitted or duplicated collections")
            for item in collection["collection_manifests"]:
                _require(item["content"] == _read(Path(item["path"])),
                         "Core primary risk embeds a different collection manifest")
            sources = collection.get("source_files", [])
            _require(bool(sources) and len({str(Path(item["path"]).resolve()) for item in sources}) == len(sources),
                     "Core primary risk lacks a unique full source-file inventory")
            for item in sources:
                path = Path(item["path"]).resolve()
                _require(file_sha256(path) == item["sha256"], "Core primary risk source data changed")
                files.append(path)
        _require(provenance["method"] == expected_method and collection["model_key"] == model_key
                 and collection["benchmarks"] == [benchmark]
                 and collection.get("policy_runtime") == policy.runtime_precision_record()
                 and collection.get("policy_configuration_fingerprint") == policy.configuration_fingerprint
                 and collection["checkpoint_hash"] == policy.checkpoint_hash and provenance["test_used_for_fit"] is False
                 and collection["test_used_for_fit"] is False and collection["test_used_for_threshold_selection"] is False
                 and collection["label_kind"] == "future_failure" and provenance["feature_schema"] == schema,
                 "Risk fit must use the matching initial Qwen train/dev data and future-failure labels, never test")
        _require(schema["feature_names"] == list(risk.features.names), "Risk schema differs from loaded preprocessing")
        risk.model.eval()
        risk.model.requires_grad_(False)
        files += [checkpoint, fit_path, schema_path]
        if typed_enabled:
            from .failure_risk import FailureAwareRiskModel, HEADS
            typed_path = checkpoint.with_name(checkpoint.stem + ".failure_heads.pt")
            typed_fit_path = checkpoint.with_name(checkpoint.stem + ".failure_heads.fit.json")
            typed_fit = _read(typed_fit_path)
            _require(typed_fit["status"] == "succeeded" and typed_fit["method"] == expected_method
                     and typed_fit["heads"] == list(HEADS)
                     and typed_fit["source_provenance"] == collection
                     and typed_fit["primary_checkpoint_sha256"] == file_sha256(checkpoint),
                     "Failure-type predictor lacks matching successful source/primary evidence")
            inventory_path = Path(typed_fit["label_inventory"]["path"])
            _require(file_sha256(inventory_path) == typed_fit["label_inventory"]["sha256"], "Typed label inventory changed")
            inventory = _read(inventory_path)
            _require(inventory["fingerprint"] == fingerprint({k: v for k, v in inventory.items() if k != "fingerprint"})
                     and inventory["model_key"] == model_key and inventory["heads"] == list(HEADS)
                     and inventory["dataset_fingerprint"] == collection["dataset_fingerprint"],
                     "Typed label inventory does not match the complete initial-policy corpus")
            label_files = [Path(item["path"]).resolve() for item in inventory["sources"]]
            expected_count = expected_corpus["expected_collection_jobs"] if expected_corpus is not None else 210
            _require(len(label_files) == expected_count and len(set(label_files)) == expected_count
                     and inventory["expected_collection_jobs"] == expected_count
                     and all(file_sha256(item["path"]) == item["sha256"] for item in inventory["sources"]),
                     "Typed source sidecars are incomplete, duplicated, missing or modified")
            if expected_corpus is not None:
                _require(inventory.get("corpus_contract") == expected_corpus,
                         "Typed core predictor was fitted under another corpus contract")
                expected_sources = {str(Path(item["path"]).resolve()): item["sha256"] for item in expected_corpus["manifests"]}
                actual_sources = [item["source_manifest"] for item in inventory["sources"]]
                _require(len(actual_sources) == len(expected_sources)
                         and {str(Path(item["path"]).resolve()): item["sha256"] for item in actual_sources} == expected_sources,
                         "Typed core labels do not cover exactly the registered train/dev manifests")
            else:
                _require(inventory.get("corpus_contract") is None,
                         "A deadline core predictor cannot satisfy the full-study controller gate")
            risk = FailureAwareRiskModel.load(typed_path, primary=risk, primary_checkpoint=checkpoint,
                expected_sha256=typed_fit["checkpoint_sha256"], expected_primary_sha256=file_sha256(checkpoint),
                expected_policy_runtime=policy.runtime_precision_record(), expected_provenance=collection,
                require_control_ready=True)
            _require(risk.failure_provenance.get("label_inventory") == inventory,
                     "Typed weights and verified observed-label inventory disagree")
            files += [typed_path, typed_fit_path, inventory_path, *label_files]
    if needs_graph:
        manifest = Path(transcoder_manifest).resolve()
        reused = None
        if expected_corpus is not None:
            from .graph_recovery import bank_for_corpus
            reused = bank_for_corpus(expected_corpus,
                manifest_paths=[Path(item["path"]).resolve() for item in expected_corpus["manifests"]],
                model_key=model_key, transcoder_manifest=manifest)
        if reused is None:
            bank = validate_transcoder_bank(manifest, model_key=model_key, checkpoint_hash=policy.checkpoint_hash,
                                           policy_runtime=policy.runtime_precision_record())
        else:
            bank, graph_collection, reuse_proof = reused
            files += _verify_reused_bank_binding(bank, graph_collection, reuse_proof, corpus_contract=expected_corpus,
                model_key=model_key, transcoder_manifest=manifest,
                source_selection_rule=graph_config.get("source_selection_rule", "global_activation"))
            _require(bank["checkpoint_hash"] == policy.checkpoint_hash
                     and bank["policy_runtime"] == policy.runtime_precision_record()
                     and bank["policy_configuration_fingerprint"] == policy.configuration_fingerprint,
                     "Reused transcoder bank differs from the loaded policy checkpoint/runtime/configuration")
        _require(bank.get("test_used_for_training_or_fidelity") is False and bool(bank.get("collections"))
                 and all(row["job_id"].startswith("collection-" + benchmark + "-") for row in bank["collections"]),
                 "Transcoder bank must use this benchmark's initial-policy train/dev collections only")
        if expected_corpus is not None and reused is None:
            validate_corpus_contract(expected_corpus,
                manifest_paths=[Path(item["path"]).resolve() for item in bank["collections"]], model_key=model_key)
            _require(len(bank["collections"]) == expected_corpus["expected_collection_jobs"],
                     "Core transcoder bank omitted or duplicated declared collections")
        for collection in bank["collections"]:
            path = Path(collection["path"])
            _require(file_sha256(path) == collection["sha256"], "Transcoder source collection manifest changed")
            files.append(path)
        bindings = []
        for layer in bank["layers"]:
            entry = layer["checkpoint"]
            path = Path(entry["path"] if isinstance(entry, dict) else entry)
            path = path if path.is_absolute() else manifest.parent / path
            transcoder, metadata = load_transcoder(path, expected_policy_fingerprint=policy.checkpoint_hash, expected_layer_path=layer["layer_path"])
            _require(metadata == layer["metadata"] and transcoder.checkpoint_hash() == layer["transcoder_hash"], "Loaded transcoder differs from validated bank")
            transcoder.eval().requires_grad_(False)
            bindings.append(TranscoderBinding(layer["layer_path"], transcoder.to(graph_config.get("transcoder_device", "cpu")), metadata))
            files.append(path)
        files.append(manifest)
        attributor = NativeAttributor(policy.model, bindings, state_id_getter=policy.get_state_id,
            architecture_review=policy.architecture_review,
            constant_storage_device=graph_config.get("constant_storage_device"),
            source_selection_rule=graph_config.get("source_selection_rule", "global_activation"),
            **{key: graph_config[key] for key in ("max_nodes", "max_feature_nodes", "max_backward_targets", "max_logits")})
    return risk, attributor, files


def cpu_clean_reload_validator(policy, checkpoint_path: Path, expected_state_hash: str, *, fresh_model_factory=None):
    """Compare two CPU models, without moving/mutating the live GPU policy.

    A CPU reference is loaded from the live tensors (and hash-checked), then a
    separate CPU model is loaded from disk. This verifies serialization and CPU
    forward equality, not a claim of CPU/GPU numerical equivalence.
    """
    live = policy.model
    state_id = policy.get_state_id()
    _require(tensor_state_hash(dict(live.state_dict())) == expected_state_hash, "Live final tensors differ from the committed checkpoint")
    if fresh_model_factory is None:
        from transformers import AutoModelForImageTextToText
        _require(live.get_input_embeddings().weight.dtype is torch.float32,
                 "New scientific clean-reload acceptance requires the declared FP32 policy")
        def fresh_model_factory():
            config = copy.deepcopy(live.config)
            # Construct clean CPU modules from configuration, not deepcopy(live):
            # live placement hooks close over the GPU adapter and must not leak.
            return AutoModelForImageTextToText.from_config(config, dtype=torch.float32, attn_implementation="eager").cpu().eval()
    reference = fresh_model_factory()
    _require(reference is not live and all(p.device.type == "cpu" for p in reference.parameters()), "Clean reference must be a distinct CPU model")
    _require(all(not module._forward_hooks and not module._forward_pre_hooks for module in reference.modules()),
             "Clean CPU reference must not retain live GPU placement-hook closures")
    reference.load_state_dict(live.state_dict(), strict=True)
    reference.eval()
    probe = torch.tensor([[1, 2, 3]], dtype=torch.long)
    try:
        proof = verify_clean_reload_checkpoint(reference, checkpoint_path, expected_state_hash,
            fresh_policy_factory=fresh_model_factory, input_ids=probe,
            model_kwargs={"attention_mask": torch.ones_like(probe), "logits_to_keep": 1}, atol=1e-6, rtol=1e-5)
        _require(policy.get_state_id() == state_id and tensor_state_hash(dict(live.state_dict())) == expected_state_hash,
                 "Clean reload must preserve the live policy state and weights")
        return {**proof, "backend": "CPU eager, same dtype on both instances", "reference_origin": "live final tensors copied to a separate CPU model and exact-hash checked",
                "gpu_cross_backend_equality_claimed": False}
    finally:
        del reference
        gc.collect()
