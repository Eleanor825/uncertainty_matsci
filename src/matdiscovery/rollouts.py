"""Complete benchmark rollouts with explicit failures, costs, and decision records."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from .esopt import tensor_state_hash
from .action_targets import ActionTargetError, build_action_targets
from .graph_features import extract_graph_features
from .native_attribution import CONTRACT, UnsupportedAttribution
from .policy import PolicyContextError
from .rpc import EnvironmentClient
from .transcoders import write_activation_shard
from .failure_labels import HEADS, HORIZONS, scientific_failure_labels, tool_failure_labels
from .failure_controller import plan_failure_response, rank_failure_candidates
from .execution_contract import declared_mace_workers


MADE_TOOLS = ["generate_structures", "create_structure", "score_buffer", "query_structures", "list_compositions", "get_buffer_stats", "select_for_evaluation"]
TOOL_ARGUMENTS = {
    "generate_structures": {"compositions", "generator_name", "num_candidates"},
    "create_structure": {"structure", "a", "b", "c", "alpha", "beta", "gamma", "species", "frac_coords"},
    "score_buffer": {"scorer_name", "composition"},
    "query_structures": {"composition", "k", "mode", "scorer_name", "include_structure_details"},
    "list_compositions": set(), "get_buffer_stats": set(),
    "select_for_evaluation": {"composition", "structure_hash", "structure_index", "scorer_name"},
}
MADE_INSTRUCTIONS = """You operate a closed-loop materials discovery environment. Discover stable, unique, novel crystals efficiently within the physical evaluation budget. Only the supplied chemical elements are allowed, with at most 20 atoms per cell. Output one JSON object: {"tool":NAME,"arguments":{...},"rationale":"brief reason"}.
Tools:
generate_structures: compositions (list of explicit integer formulas), generator_name (chemeleon or random), num_candidates (normally 32).
create_structure: a,b,c,alpha,beta,gamma, species (element list), frac_coords (list of coordinate triples).
score_buffer: scorer_name (oracle for MACE surrogate, diversity, or random); optionally composition.
query_structures: composition,k (normally 5),mode (all/top/bottom/random),scorer_name where appropriate.
list_compositions and get_buffer_stats: empty arguments.
select_for_evaluation: composition and structure_hash (preferred), or structure_index. This submits one crystal to the fixed ORB evaluator.
Use generation, screening and the observed feedback to choose subsequent actions. A surrogate score is not the final oracle. A nonfinite surrogate score is unavailable, not a measured stability label. Do not invent tool results. Select a buffered candidate after preparing it; unnecessary repeated inspections waste computation."""
CRYSTAL_INSTRUCTIONS = """You complete a crystal one atomic site at a time. The supplied lattice and coordinates are fixed. Select the element for focus_site, considering the target property and previously selected elements. Only legal_actions are available. Output exactly one JSON object {"action":"ELEMENT","rationale":"brief reason"}. The true Quantum ESPRESSO evaluator runs only after every site is filled. Do not claim a property has been measured before its evaluation."""


@dataclass(frozen=True)
class RolloutSettings:
    max_tools_per_query: int = 10
    max_generation_retries: int = 2
    risk_threshold: float = .6
    history_results: int = 8
    recent_tools: int = 4
    activation_tokens_per_decision: int = 16
    oracle_device: str = "cpu"
    generator_device: str = "cuda"
    failure_aware_control: bool = False


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _small(value, *, depth=0, limit=8):
    """An explicit fixed policy-visible summary, not a truncation of attribution input."""
    if depth > 5:
        return {"summary": "nested details omitted"}
    if isinstance(value, dict):
        return {k: _small(v, depth=depth + 1, limit=limit) for k, v in value.items()
                if k not in {"@module", "@class", "structure", "structures", "cif", "official_info", "policy_visible", "dataset", "ground_truth_pd", "ground_truth"}}
    if isinstance(value, list):
        return [_small(v, depth=depth + 1, limit=limit) for v in value[-limit:]] + ([{"omitted_items": len(value)-limit}] if len(value) > limit else [])
    if isinstance(value, str) and len(value) > 2000:
        return value[:2000] + " [policy-summary text limit]"
    return value


def _scientific_summary(value):
    if not isinstance(value, dict):
        return value
    keys = {"reduced_formula", "formation_energy_per_atom", "energy_per_atom", "e_above_hull", "is_stable", "is_newly_discovered", "property", "property_value", "reward", "absolute_target_error", "official_error_flag", "dft_success", "scientific_label", "oracle_exception"}
    return {k: _small(v, limit=2) for k, v in value.items() if k in keys}


def _tool_summary(value, limit):
    if not isinstance(value, dict):
        return str(value)[:500]
    if value.get("ok") is False:
        error = value.get("error", {})
        return {"error": error.get("code"), "message": str(error.get("message", ""))[:400]}
    result = value.get("result", value)
    if not isinstance(result, dict):
        return _small(result, limit=limit)
    tool, output = result.get("tool"), result.get("output", {})
    if isinstance(output, dict):
        summary = {k: output[k] for k in ["generated", "accepted", "composition", "selection_key"] if k in output}
        records = output.get("candidates", output.get("records", []))
        if isinstance(records, list):
            summary["candidates"] = [{k: row[k] for k in ["accepted", "reason", "composition", "hash", "structure_hash", "structure_index", "scores", "num_sites"] if k in row} for row in records[:limit] if isinstance(row, dict)]
            summary["omitted_candidates"] = max(0, len(records) - limit)
        if not summary:
            summary = _small(output, limit=limit)
    elif isinstance(output, list):
        summary = [{k: _small(row[k], limit=limit) for k in ["composition", "scores"] if k in row} if isinstance(row, dict) else _small(row, limit=limit) for row in output[:limit]]
    else:
        summary = str(output)[:800]
    return {"tool": tool, "output": summary}


def _messages(benchmark, observation, recent, history, *, settings, compact=False, level=None):
    level = int(compact) if level is None else level
    candidate_limit, composition_limit, known_limit = [(3, 5, 6), (1, 3, 3), (1, 1, 0)][level]
    if benchmark == "made":
        state = observation.get("state", {})
        known = state.get("phase_diagram_all_entries", [])
        known = sorted(known, key=lambda entry: json.dumps(entry.get("composition", {}), sort_keys=True) + str(entry.get("energy", "")))
        known_summary = []
        for entry in known[:known_limit]:
            composition = {k: v for k, v in entry.get("composition", {}).items() if isinstance(v, (float, int))}
            atoms = sum(composition.values()) or 1
            known_summary.append({"composition": composition, "energy_per_atom": float(entry.get("energy", 0)) / atoms})
        raw_buffer = observation.get("buffer", {})
        buffer = {comp: [{k: row[k] for k in ["composition", "hash", "full_formula", "scores", "num_sites"] if k in row} for row in entries[:candidate_limit]] for comp, entries in list(raw_buffer.items())[-composition_limit:]}
        visible = {"elements": state.get("elements"), "query_count": state.get("query_count"),
                   "budget": observation.get("budget"), "selected": observation.get("selected"),
                   "buffer": buffer, "buffer_compositions_count": len(raw_buffer),
                   "last_observation": _scientific_summary(state.get("last_observation")),
                   "known_entries": known_summary, "known_entries_count": len(known)}
        instruction = MADE_INSTRUCTIONS
    else:
        public = {"property", "target", "lattice_lengths", "lattice_angles", "fractional_coordinates", "filled_elements", "focus_site", "legal_actions", "num_sites", "filled_count"}
        visible = {k: v for k, v in observation.items() if k in public}
        instruction = CRYSTAL_INSTRUCTIONS
    recent_count = [settings.recent_tools, 1, 0][level]
    history_count = [settings.history_results, 2, 0][level]
    payload = {"observation": visible,
               "recent_tools": [_tool_summary(value, candidate_limit) for value in recent[-recent_count:]] if recent_count else [],
               "scientific_history": [_scientific_summary(value) for value in history[-history_count:]] if history_count else [],
               "memory_summary_level": level}
    return [{"role": "system", "content": instruction}, {"role": "user", "content": json.dumps(payload, separators=(",", ":"), ensure_ascii=False)}]


def make_environment_arguments(project: Path, job: dict, output: Path, settings: RolloutSettings):
    benchmark = job["benchmark"]
    args = {"benchmark": benchmark, "vendor_root": str(project / "vendor" / ("MADE" if benchmark == "made" else "crystal-gym")),
            "work_dir": str(output / "environment"), "seed": int(job["environment_seeds"][0]), "budget": job["budget"]}
    if benchmark == "made":
        elements = job["task"]["elements"]
        index = json.loads((project / "data/raw/materials_project/index.json").read_text())
        snapshot = index["snapshots"]["-".join(sorted(elements))]
        raw_assets = json.loads((project / "configs/assets.runtime.json").read_text())
        protocol_path = project / "configs/main_protocol.json"
        protocol = json.loads(protocol_path.read_text()) if protocol_path.is_file() else {}
        assets = {k: {"path": v, "sha256": raw_assets[k + "_sha256"]} if k + "_sha256" in raw_assets else v for k, v in raw_assets.items() if not k.endswith("_sha256")}
        args.update(elements=elements, mp_cache_path=snapshot["path"], mp_cache_sha256=snapshot["sha256"], assets=assets,
                    device=settings.oracle_device, generator_device=settings.generator_device, stability_tolerance=.1,
                    mace_num_workers=declared_mace_workers(protocol))
    else:
        qe = json.loads((project / "configs/qe.runtime.json").read_text())
        prop = job["task"]["property"]
        prototype = job["task"].get("prototype")
        args.update(assets=qe["assets"], budget_unit="dft_episode_attempts", property=prop["id"], target=prop["target"],
                    split="heldout" if job.get("stage") == "final_eval" else "train",
                    prototype_index=prototype["index"] if prototype else None)
    return args


class DiscoveryRollout:
    def __init__(self, project: Path, policy, *, settings=None, risk_model=None, attributor=None):
        self.project, self.policy = Path(project).resolve(), policy
        self.settings = settings or RolloutSettings()
        self.risk_model, self.attributor = risk_model, attributor

    def _features(self, generation, row, *, capture, collection, output, method):
        entropy = np.asarray(generation.entropy, dtype=float)
        logprobs = np.asarray(generation.logprobs, dtype=float)
        values = {"sampling.entropy_mean": float(entropy.mean()) if len(entropy) else None,
                  "sampling.entropy_max": float(entropy.max()) if len(entropy) else None,
                  "sampling.mean_logprob": float(logprobs.mean()) if len(logprobs) else None,
                  "sampling.completion_tokens": float(generation.completion_count),
                  "sampling.prompt_tokens": float(generation.prompt_token_count),
                  "sampling.valid_json_action": float(generation.success)}
        action_name = (generation.parsed_action or {}).get("tool", "atomic_action") if isinstance(generation.parsed_action, dict) else "invalid"
        values.update({"action." + name: float(action_name == name) for name in MADE_TOOLS + ["atomic_action", "invalid"]})
        if capture:
            observed = self.policy.capture_prefix(generation.input_ids_with_completion, stamp=generation.model_stamp)
            for layer, statistics in observed.hidden_summaries.items():
                for key, value in statistics.items():
                    if isinstance(value, (int, float)):
                        values["hidden." + layer + "." + key] = float(value)
            probabilities = observed.last_logits.float().softmax(-1)
            values["hidden.raw_next_token_entropy"] = float(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())
            values["hidden.capture_parity_max_abs"] = observed.parity_max_abs
            if collection:
                total = generation.input_ids_with_completion.shape[1]
                indices = np.linspace(0, total - 1, min(total, self.settings.activation_tokens_per_decision), dtype=int)
                selection = torch.as_tensor(np.unique(indices), dtype=torch.long)
                for path in self.policy.mlp_paths:
                    self._activation_buffer.setdefault(path, []).append({
                        "inputs": observed.mlp_inputs[path][0, selection].float(),
                        "outputs": observed.mlp_outputs[path][0, selection].float(),
                        "group_id": row["group_id"], "prefix_hash": row["prefix_hash"],
                        "split": row["split"], "policy_fingerprint": generation.model_stamp.checkpoint_hash,
                    })
        if method in {"graph_risk", "esopt_graph_risk"}:
            if self.attributor is None:
                raise RuntimeError("A graph method cannot run without its trained, validated graph extractor")
            started = time.monotonic()
            try:
                ids, kwargs = self.policy.prepare_trace_inputs(generation.input_ids_with_completion, stamp=generation.model_stamp)
                targets = build_action_targets(ids, prompt_token_count=generation.prompt_token_count,
                    tokenizer=self.policy.tokenizer, benchmark=row["benchmark"],
                    parsed_action=generation.parsed_action, attention_mask=kwargs["attention_mask"])
                if self.attributor._validation is None:
                    self.attributor.validate_backend(ids, generation.model_stamp, model_kwargs=kwargs, action_targets=targets)
                graph = self.attributor.attribute(ids, generation.model_stamp, model_kwargs=kwargs, action_targets=targets)
                extracted = extract_graph_features(graph.graph)
                values.update({"graph." + k: float(v) for k, v in extracted.values.items()})
                row["graph_status"] = "succeeded"
                row["graph_contract"] = CONTRACT
                row["graph_metadata"] = graph.metadata
            except (UnsupportedAttribution, ActionTargetError) as exc:
                row["graph_status"] = "unavailable"
                row["graph_error"] = str(exc)
                values["graph.graph_missing"] = 1.0
            row["graph_seconds"] = time.monotonic() - started
        return values

    def _generate(self, job, output, observation, recent, history, rows, sequence, *, collection):
        benchmark, method = job["benchmark"], job["method"]
        typed_enabled = self.settings.failure_aware_control and benchmark == "made"
        typed_control = typed_enabled and method in {"entropy_risk", "hidden_risk", "graph_risk", "esopt_graph_risk"}
        if typed_enabled and (self.risk_model is None or not hasattr(self.risk_model, "predict_failure_types")):
            raise RuntimeError("Failure-aware execution requires the verified fitted type heads")
        schema = {"type": "object", "required": ["tool", "arguments"], "properties": {"tool": {"type": "string", "enum": MADE_TOOLS}, "arguments": {"type": "object"}}} if benchmark == "made" else {"type": "object", "required": ["action"], "properties": {"action": {"type": "string", "enum": [a["element"] for a in observation["legal_actions"]]}}}
        candidates = []
        retry_feedback = None
        def validate_action(action):
            if benchmark == "made":
                unknown = set(action["arguments"]) - TOOL_ARGUMENTS[action["tool"]]
                if unknown:
                    return "Unknown tool arguments: " + ", ".join(sorted(unknown))
            return True
        for attempt in range(self.settings.max_generation_retries):
            sampling_base = job.get("policy_sampling_seed", job["seed"])
            seed = (sampling_base * 10_000_019 + sequence * 97 + attempt) % (2**63)
            started = time.monotonic()
            for memory_level in range(3):
                messages = _messages(benchmark, observation, recent, history, settings=self.settings, level=memory_level)
                if retry_feedback is not None:
                    payload = json.loads(messages[-1]["content"])
                    payload["decision_risk_feedback"] = retry_feedback
                    messages[-1]["content"] = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
                try:
                    generated = self.policy.generate_action(messages, seed=seed, legal_schema=schema, action_validator=validate_action)
                    break
                except PolicyContextError:
                    if memory_level == 2:
                        raise
            identifier = f"d{sequence:07d}-c{attempt}"
            token_path = output / "tokens" / (identifier + ".pt")
            token_path.parent.mkdir(exist_ok=True)
            torch.save({"input_ids": generated.input_ids_with_completion, "policy_stamp": asdict(generated.model_stamp)}, token_path)
            prefix_hash = tensor_state_hash({"input_ids": generated.input_ids_with_completion, "attention_mask": torch.ones_like(generated.input_ids_with_completion)})
            split = job.get("split", "test" if job["stage"] == "final_eval" else "train")
            generation_record = generated.to_record()
            generation_record["policy_runtime"] = self.policy.runtime_precision_record()
            row = {"decision_id": job["job_id"] + ":" + identifier, "local_decision_id": identifier,
                   "benchmark": benchmark, "model_key": job["model_key"], "split": split, "task_id": job["task_id"],
                   "group_id": job.get("group_id", job["task_id"]), "episode_id": job["job_id"] + ":episode:" + str(observation.get("episode_index", 0)),
                   "episode_index": int(observation.get("episode_index", 0)),
                   "input_ids_file": str(token_path), "prefix_hash": prefix_hash, "policy_stamp": asdict(generated.model_stamp),
                   "generation": generation_record, "label_immediate_error": int(not generated.success), "label_future_failure": None,
                   "generation_seconds": time.monotonic() - started, "disposition": "candidate", "memory_summary_level": memory_level}
            if benchmark == "made":
                row["observed_failure_types"] = {head: None for head in HEADS}
                row["observed_failure_types"]["generation_invalid"] = int(not generated.success)
                row["failure_type_horizons"] = dict(HORIZONS)
                row["failure_type_observation_rpc_ids"] = {head: [] for head in HEADS}
            # Separate durable pre-action evidence from labels that only exist later.
            with (output / "decision_events.jsonl").open("a") as f:
                f.write(json.dumps({"decision_id": row["decision_id"], "event": "proposed", "policy_stamp": row["policy_stamp"], "prefix_hash": prefix_hash, "input_ids_file": str(token_path)}) + "\n")
            row["features"] = self._features(generated, row, capture=collection or method in {"hidden_risk", "graph_risk", "esopt_graph_risk"}, collection=collection, output=output, method=method)
            risk = 1.0 if not generated.success else None
            if generated.success and method in {"entropy_risk", "hidden_risk", "graph_risk", "esopt_graph_risk"}:
                if self.risk_model is None:
                    raise RuntimeError("Risk method requires a fitted train/dev checkpoint")
                names = self.risk_model.features.names
                x = np.array([[row["features"].get(name, np.nan) for name in names]], dtype=float)
                risk = float(self.risk_model.predict_proba(x, names)[0])
            row["predicted_failure_probability"] = risk
            if generated.success and method in {"baseline", "esopt"} and self.risk_model is not None:
                # Record calibrated entropy confidence for analysis without changing
                # baseline decisions, resampling, or candidate selection.
                names = self.risk_model.features.names
                x = np.array([[row["features"].get(name, np.nan) for name in names]], dtype=float)
                row["predicted_failure_probability"] = float(self.risk_model.predict_proba(x, names)[0])
                row["confidence_used_for_control"] = False
            type_plan = None
            if typed_enabled:
                names = self.risk_model.features.names
                x = np.array([[row["features"].get(name, np.nan) for name in names]], dtype=float)
                predictions = self.risk_model.predict_failure_types(x, names)
                row["failure_type_probabilities"] = {head: None if predictions[head] is None
                    else float(predictions[head][0]) for head in HEADS}
                row["failure_types_used_for_control"] = typed_control
                if typed_control:
                    visible = json.loads(messages[-1]["content"])["observation"]
                    type_plan = plan_failure_response(benchmark, generated.parsed_action,
                        row["failure_type_probabilities"], row["predicted_failure_probability"],
                        visible, threshold=self.settings.risk_threshold)
                    row["failure_controller"] = type_plan
                    row["feedback_received_for_this_candidate"] = retry_feedback
                    if type_plan["request_retry"]:
                        retry_feedback = {"predicted_failure_type": type_plan["trigger"],
                            "instruction": type_plan["feedback_for_retry"],
                            "suggested_legal_tool": type_plan["optional_tool_preference"],
                            "source": "current_candidate_predictions_or_visible_schema_issue",
                            "not_a_measured_scientific_result": True}
            rows.append(row)
            candidates.append((generated, row, risk))
            if generated.success and (risk is None or risk < self.settings.risk_threshold) and not (type_plan and type_plan["request_retry"]):
                break
        valid = [candidate for candidate in candidates if candidate[0].success]
        if not valid:
            for _, row, _ in candidates:
                row["disposition"] = "invalid_not_executed"
            return None, candidates[-1][1]
        if typed_control:
            selection = rank_failure_candidates(benchmark, [
                {"action": candidate[0].parsed_action, "valid": candidate[0].success,
                 "risk_types": candidate[1]["failure_type_probabilities"],
                 "overall_risk": candidate[1]["predicted_failure_probability"]}
                for candidate in candidates], visible, threshold=self.settings.risk_threshold)
            for candidate in candidates:
                candidate[1]["failure_candidate_selection"] = selection
            if selection["selected_index"] is None:
                for candidate in candidates:
                    candidate[1]["disposition"] = "invalid_not_executed"
                return None, candidates[-1][1]
            chosen = candidates[selection["selected_index"]]
        else:
            chosen = min(valid, key=lambda x: x[2] if x[2] is not None else 0)
        for candidate in candidates:
            candidate[1]["disposition"] = "executed" if candidate is chosen else "rejected_not_executed"
        return chosen[0].parsed_action, chosen[1]

    @staticmethod
    def _label_rows(output, rows, failure, scientific_response=None):
        with (output / "decisions.jsonl").open("a") as f:
            for row in rows:
                if row["disposition"] == "executed":
                    row["label_future_failure"] = failure
                    if scientific_response is not None and row.get("benchmark") == "made":
                        measured = scientific_failure_labels(scientific_response)
                        row["observed_failure_types"].update(measured)
                        for head, value in measured.items():
                            row["failure_type_observation_rpc_ids"][head] = [scientific_response["id"]]
                # The risk target is a subsequent outcome, never proof of a causal step error.
                row["label_kind"] = "next_scientific_evaluation_failure"
                f.write(json.dumps(row, allow_nan=False) + "\n")
        rows.clear()

    def run(self, job, output: Path, *, collection=False):
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        self._activation_buffer = {}
        dump_json(output / "job.json", {**job, "rollout_settings": asdict(self.settings), "policy_configuration_fingerprint": self.policy.configuration_fingerprint})
        arguments = make_environment_arguments(self.project, job, output, self.settings)
        all_summaries, history, recent, rows = [], [], [], []
        sequence, tools_since_query = 0, 0
        random_recovery = random.Random(job.get("policy_sampling_seed", job["seed"]))
        curve, last_metrics, last_observation = [(0, 0)], {}, None
        started = time.monotonic()
        with EnvironmentClient(self.project, job["benchmark"], output / "rpc") as environment:
            initialized = environment.request("init", arguments)
            observation = initialized["observation"]
            initialization_seconds = time.monotonic() - started
            dump_json(output / "environment_metadata.json", initialized["metadata"])
            episode_start_counts = dict(observation.get("counts", {}))
            episode_start_time = time.monotonic()
            while not observation.get("done", False):
                if job["benchmark"] == "made" and tools_since_query >= self.settings.max_tools_per_query:
                    # A fixed, shared, logged recovery completes the physical budget;
                    # it is never represented as a successful LLM decision.
                    buffer = observation.get("buffer", {})
                    nonempty = [(comp, entries) for comp, entries in buffer.items() if entries]
                    if nonempty:
                        comp, entries = sorted(nonempty)[0]
                        action = {"tool": "select_for_evaluation", "arguments": {"composition": comp, "structure_hash": entries[0]["hash"]}}
                    else:
                        elements = arguments["elements"]
                        formula = "".join(elements) + str(1 + random_recovery.randrange(2))
                        action = {"tool": "generate_structures", "arguments": {"generator_name": "random", "compositions": [formula], "num_candidates": 32}}
                    decision = None
                    with (output / "decision_events.jsonl").open("a") as f:
                        f.write(json.dumps({"event": "fixed_failure_recovery", "sequence": sequence, "action": action}) + "\n")
                    if tools_since_query > self.settings.max_tools_per_query + 10:
                        raise RuntimeError("Recovery could not produce a legal crystal; the budget is incomplete")
                else:
                    action, decision = self._generate(job, output, observation, recent, history, rows, sequence, collection=collection)
                sequence += 1
                if action is None:
                    recent.append({"error": "invalid_action_generation"})
                    tools_since_query += 1
                    if job["benchmark"] == "crystalgym":
                        action = {"action": random_recovery.choice(observation["legal_actions"])["element"]}
                        with (output / "decision_events.jsonl").open("a") as f:
                            f.write(json.dumps({"event": "fixed_failure_recovery", "sequence": sequence, "action": action}) + "\n")
                    else:
                        continue
                if job["benchmark"] == "made":
                    response = environment.request("tool", {"name": action["tool"], "arguments": action["arguments"]}, check=False)
                    if decision is not None:
                        measured = tool_failure_labels(action, response)
                        decision["observed_failure_types"].update(measured)
                        for head, value in measured.items():
                            if value is not None or (head == "candidate_generation_failure"
                                    and action["tool"] in {"generate_structures", "create_structure"}):
                                decision["failure_type_observation_rpc_ids"][head] = [response["id"]]
                    tools_since_query += 1
                    recent.append(_small(response))
                    if not response["ok"]:
                        if decision:
                            decision["label_immediate_error"] = 1
                            decision["tool_error"] = response["error"]
                        observation = environment.request("observe")
                        continue
                    if action["tool"] != "select_for_evaluation":
                        observation = environment.request("observe")
                        continue
                    response = environment.request("step", check=False)
                    if response["ok"]:
                        result = response["result"]
                        last_metrics = result["official_metrics"]
                        last_observation = result["official_observation"]
                        failure = int(not (last_observation.get("is_stable", False) and last_observation.get("is_newly_discovered", False)))
                        history.append(_small(last_observation))
                        observation = result["observation"]
                        curve.append((int(observation["counts"].get("candidate_oracle_attempts", 0)), int(last_metrics.get("num_newly_discovered_stable", 0))))
                    else:
                        failure = 1
                        if decision:
                            decision["label_immediate_error"] = 1
                        history.append({"oracle_exception": response["error"]})
                        observation = environment.request("observe")
                        curve.append((int(observation["counts"].get("candidate_oracle_attempts", 0)), curve[-1][1]))
                    self._label_rows(output, rows, failure, scientific_response=response)
                    tools_since_query = 0
                else:
                    response = environment.request("step", {"action": action["action"]}, check=False)
                    if not response["ok"]:
                        if decision:
                            decision["label_immediate_error"] = 1
                        raise RuntimeError("CrystalGym step exception requires reconciliation: " + str(response["error"]))
                    result = response["result"]
                    observation = result["observation"]
                    if not result["episode_done"]:
                        continue
                    scientific = result["scientific_result"]
                    # This target includes numerical failure or a >50% relative
                    # target miss. It is a predictive outcome, not causal blame.
                    relative = scientific["absolute_target_error"] / arguments["target"] if scientific["absolute_target_error"] is not None else None
                    failure = int(not scientific["dft_success"] or relative is None or relative > .5)
                    self._label_rows(output, rows, failure)
                    history.append(_small(scientific))
                    summary = self._summary(job, observation, episode_start_counts, episode_start_time)
                    summary["status"] = "succeeded" if scientific["dft_success"] else "failed"
                    summary["metrics"] = {"reward": scientific["reward"], "property_value": scientific["property_value"], "property_abs_error": scientific["absolute_target_error"], "failure_rate": float(not scientific["dft_success"])}
                    all_summaries.append(summary)
                    if not observation["done"]:
                        observation = environment.request("reset")
                        history.clear()
                        recent.clear()
                        episode_start_counts, episode_start_time = dict(observation["counts"]), time.monotonic()
            if job["benchmark"] == "made":
                summary = self._summary(job, observation, episode_start_counts, episode_start_time)
                b = job["budget"]
                if curve[-1][0] < b:
                    curve.append((b, curve[-1][1]))
                area = sum((x1-x0)*(y1+y0)/2 for (x0,y0),(x1,y1) in zip(curve, curve[1:]))
                summary["metrics"] = {"mSUN": float(last_metrics.get("num_newly_discovered_stable", 0))/b, "AUDC": 2*area/(b*b), "failure_rate": 1-float(last_metrics.get("num_newly_discovered_stable", 0))/b}
                summary["official_metrics"] = last_metrics
                summary["discovery_curve"] = curve
                all_summaries.append(summary)
            if rows:
                self._label_rows(output, rows, None)
        decision_path = output / "decisions.jsonl"
        if decision_path.exists():
            decisions = [json.loads(line) for line in decision_path.read_text().splitlines()]
            for summary in all_summaries:
                these = [r for r in decisions if str(r["episode_index"]) == summary["episode_id"]]
                summary["costs"].update(llm_calls=len(these), completion_tokens=sum(r["generation"]["completion_count"] for r in these), prompt_tokens=sum(r["generation"]["prompt_token_count"] for r in these), graph_seconds=sum(r.get("graph_seconds", 0) for r in these))
        if all_summaries:
            all_summaries[0]["costs"]["initialization_wall_seconds"] = initialization_seconds
            all_summaries[0]["costs"]["initialization_oracle_attempts"] = initialized["observation"].get("counts", {}).get("initialization_oracle_attempts", 0)
        activation_shards = []
        for layer, chunks in self._activation_buffer.items():
            destination = output / "activation_shards" / (layer.replace(".", "_") + ".pt")
            meta = write_activation_shard(destination, torch.cat([c["inputs"] for c in chunks]), torch.cat([c["outputs"] for c in chunks]),
                group_ids=[c["group_id"] for c in chunks for _ in range(len(c["inputs"]))],
                prefix_hashes=[c["prefix_hash"] for c in chunks for _ in range(len(c["inputs"]))],
                split=chunks[0]["split"], policy_fingerprint=chunks[0]["policy_fingerprint"], layer_path=layer)
            activation_shards.append({"path": str(destination), "layer_path": layer, "split": chunks[0]["split"], "tensor_hash": meta["tensor_hash"]})
        self._activation_buffer.clear()
        dump_json(output / "episodes.json", all_summaries)
        dump_json(output / "collection_manifest.json", {"decision_files": [{"path": str(decision_path), "sha256": file_hash(decision_path)}] if decision_path.exists() else [], "activation_shards": activation_shards, "job_id": job["job_id"], "complete": True, "elapsed_seconds": time.monotonic()-started,
            "policy_runtime": self.policy.runtime_precision_record(), "policy_configuration": self.policy._configuration(),
            "policy_configuration_fingerprint": self.policy.configuration_fingerprint})
        return all_summaries

    @staticmethod
    def _summary(job, observation, before, started):
        counts = observation.get("counts", {})
        delta = {key: value - before.get(key, 0) for key, value in counts.items()}
        episode = int(observation.get("episode_index", 0))
        return {"benchmark": job["benchmark"], "model_key": job["model_key"], "method": job["method"], "task_id": job["task_id"], "seed": job["seed"], "environment_seed": observation["episode_seed"], "episode_id": str(episode), "complete": True, "status": "succeeded", "costs": {**delta, "wall_seconds": time.monotonic()-started}}
