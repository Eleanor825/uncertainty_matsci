"""Four-parameter SnAr decisions using the original verified Qwen adapter.

No oracle, retry, clamping, checkpoint selection, or MADE risk model is hidden
here. The caller owns query accounting, the objective, and train/dev/test splits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from typing import Mapping, Sequence

from matdiscovery.accounting import fingerprint
from matdiscovery.policy import ActionGeneration, DecodingConfig, QwenPolicyAdapter


ACTION_NAMES = ("tau", "equiv_pldn", "conc_dfnb", "temperature")
ACTION_BOUNDS = {
    "tau": (0.5, 2.0), "equiv_pldn": (1.0, 5.0),
    "conc_dfnb": (0.1, 0.5), "temperature": (30.0, 120.0),
}
ACTION_UNITS = {"tau": "min", "equiv_pldn": "pyrrolidine equivalents",
                "conc_dfnb": "mol/L inlet after mixing", "temperature": "degC"}
ACTION_SCHEMA = {"type": "object", "required": list(ACTION_NAMES),
                 "additionalProperties": False,
                 "properties": {name: {"type": "number", "minimum": lo, "maximum": hi}
                                for name, (lo, hi) in ACTION_BOUNDS.items()}}


class SnArContractError(ValueError):
    pass


def validate_action(action: Mapping) -> dict:
    if not isinstance(action, Mapping) or set(action) != set(ACTION_NAMES):
        raise SnArContractError("SnAr requires exactly tau/equiv_pldn/conc_dfnb/temperature")
    for name, (lo, hi) in ACTION_BOUNDS.items():
        value = action[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise SnArContractError(f"Out-of-domain finite numeric parameter: {name}")
    return {name: action[name] for name in ACTION_NAMES}


def parse_action_json(text: str) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise SnArContractError("Duplicate SnAr parameter")
            result[key] = value
        return result
    def nonfinite(value):
        raise SnArContractError("Nonfinite JSON number")
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (ValueError, TypeError) as exc:
        raise SnArContractError("The entire visible completion must be one four-parameter JSON object") from exc
    return validate_action(value)


def visible_completion(generation: ActionGeneration, tokenizer) -> str:
    """Decode the ORIGINAL completion IDs; remove only actual special-token IDs."""
    ids = generation.input_ids_with_completion
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise SnArContractError("One complete original generated sequence is required")
    start = generation.prompt_token_count
    if type(start) is not int or not 1 <= start < ids.shape[1]:
        raise SnArContractError("Invalid original prompt/completion boundary")
    special = set(getattr(tokenizer, "all_special_ids", ()))
    tokens = [int(t) for t in ids[0, start:] if int(t) not in special]
    return tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)


@dataclass(frozen=True)
class SnArPolicyConfig:
    context_budget: int = 2048
    max_input_tokens: int = 1920
    max_new_tokens: int = 128
    history_window: int = 6
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    def __post_init__(self):
        for name in ("context_budget", "max_input_tokens", "max_new_tokens", "history_window"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise SnArContractError(f"{name} must be a positive explicit integer")
        if self.context_budget > 2048 or self.max_new_tokens > 128 or self.max_input_tokens+self.max_new_tokens > self.context_budget:
            raise SnArContractError("Complete prefix budget is at most2048, including at most128 completion tokens")
        self.decoding()

    def decoding(self):
        return DecodingConfig(max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                              top_p=self.top_p, top_k=self.top_k)


def _observed_history(history: Sequence[Mapping]) -> list[dict]:
    result = []
    for row in history:
        if not isinstance(row, Mapping) or "parameters" not in row or "objectives" not in row:
            raise SnArContractError("History contains only previously observed parameters/objectives")
        values = row["objectives"]
        if not isinstance(values, Mapping) or set(values) != {"sty", "e_factor"}:
            raise SnArContractError("Observed objective names must be sty/e_factor")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values.values()):
            raise SnArContractError("History objectives must be finite observed values")
        # Whitelist the *observable* x/y, never labels, held-out data, or fitted scores.
        result.append({"parameters": validate_action(row["parameters"]),
                       "objectives": {k: values[k] for k in ("sty", "e_factor")}})
    return result


def build_messages(observation: Mapping, history: Sequence[Mapping], *, config: SnArPolicyConfig,
                   objective_description: str, prior_history: Sequence[Mapping] = ()) -> tuple[list[dict], dict]:
    if not isinstance(objective_description, str) or not objective_description.strip():
        raise SnArContractError("Supply the protocol's frozen objective description")
    if not isinstance(observation, Mapping) or set(observation) != {"query_count", "budget"}:
        raise SnArContractError("Observation is exactly query_count/budget; response history is separate")
    count, budget = observation["query_count"], observation["budget"]
    if any(type(v) is not int for v in (count, budget)) or not 0 <= count < budget:
        raise SnArContractError("Invalid remaining query budget")
    observed = _observed_history(history)
    if len(observed) != count:
        raise SnArContractError("History must account for every already observed successful query")
    used = observed[-config.history_window:]
    prior = _observed_history(prior_history)
    if len(prior) > 4:
        raise SnArContractError("Shared prior must be the protocol's deterministic summary of at most4 observed rows")
    payload = {"query_count": count, "budget": budget, "remaining_budget": budget-count,
               "history": used, "history_omitted_count": count-len(used), "shared_prior": prior}
    system = ("Choose the next SnAr reaction experiment. Return exactly one JSON object with four "
              "finite numeric parameters and no other keys or text: tau [0.5,2.0] min; "
              "equiv_pldn [1,5] pyrrolidine equivalents; conc_dfnb [0.1,0.5] mol/L; "
              "temperature [30,120] degC. Observed objectives are sty (kg/m^3/h, maximize) "
              "and e_factor (minimize). " + objective_description)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)}]
    provenance = {"schema": "snar_observed_history_prompt_v1", "prompt_payload": payload,
                  "objective_description": objective_description, "history_window": config.history_window,
                  "message_fingerprint": fingerprint(messages), "source_history_rows": count,
                  "shared_prior_fingerprint": fingerprint(prior), "shared_prior_rows": len(prior),
                  "context_budget": config.context_budget,
                  "max_input_tokens": config.max_input_tokens, "max_new_tokens": config.max_new_tokens}
    return messages, provenance


class SnArPolicyAdapter:
    def __init__(self, policy: QwenPolicyAdapter, *, config=SnArPolicyConfig(), objective_description: str):
        if policy.max_input_tokens != config.max_input_tokens or policy.decoding != config.decoding():
            raise SnArContractError("SnAr decoding/context must match the actual policy configuration")
        self.policy, self.config = policy, config
        self.objective_description = objective_description

    @classmethod
    def from_verified_checkpoint(cls, manifest, directory, *, objective_description,
                                 config=SnArPolicyConfig(), device="cuda:0"):
        policy = QwenPolicyAdapter.from_verified_checkpoint(
            manifest, "qwen35_4b", directory, device=device, dtype="float32",
            attn_implementation="eager", sdpa_backend="auto",
            cpu_embedding_and_lm_head=True, attention_checkpointing=True,
            max_input_tokens=config.max_input_tokens, decoding=config.decoding())
        return cls(policy, config=config, objective_description=objective_description)

    def generate_action(self, observation: Mapping, history: Sequence[Mapping], *, seed: int,
                        prior_history: Sequence[Mapping] = ()):
        messages, prompt = build_messages(observation, history, config=self.config,
                                         objective_description=self.objective_description, prior_history=prior_history)
        generation = self.policy.generate_action(messages, seed=seed, legal_schema=ACTION_SCHEMA)
        if generation.success:
            try:
                actual = parse_action_json(visible_completion(generation, self.policy.tokenizer))
                if actual != generation.parsed_action:
                    raise SnArContractError("Parsed action does not equal the exact emitted four-parameter object")
            except SnArContractError as exc:
                generation = replace(generation, success=False, failure_code="snar_strict_action_json",
                                     error=str(exc))
        # Sampling scores remain explicitly HF-warped; graph score is separate.
        return generation, prompt


def replay_base_generation(record: Mapping, policy: QwenPolicyAdapter):
    """Rebind stored base tokens to a new *verified base* process, without a query."""
    import torch
    from matdiscovery.esopt import tensor_state_hash
    from matdiscovery.native_attribution import PolicyStamp
    source = PolicyStamp(**record["model_stamp"])
    current = policy.model_stamp
    for stamp in (source, current):
        if stamp.generation != 0 or stamp.perturbation_seed is not None or stamp.perturbation_sigma is not None:
            raise SnArContractError("Only clean initial-base captures may be replayed this way")
    if (source.checkpoint_hash != current.checkpoint_hash or source.model_id != current.model_id
            or record["configuration_fingerprint"] != policy.configuration_fingerprint
            or record.get("policy_runtime") != policy.runtime_precision_record()):
        raise SnArContractError("Stored generation and current verified base/config/runtime differ")
    ids = torch.tensor(record["input_ids_with_completion"], dtype=torch.long)
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] > 2048:
        raise SnArContractError("Stored complete token sequence violates SnAr context")
    count = record["prompt_token_count"]
    if type(count) is not int or not 1 <= count < ids.shape[1] or record["completion_count"] != ids.shape[1]-count:
        raise SnArContractError("Stored completion boundary changed")
    if tensor_state_hash({"input_ids": ids}) != record["prefix_hash"]:
        raise SnArContractError("Stored original generation token hash changed")
    raw = policy.tokenizer.decode(ids[0,count:].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if raw != record["raw_text"]:
        raise SnArContractError("Stored original text/tokenizer decoding changed")
    kwargs = {k:v for k,v in record.items() if k in ActionGeneration.__dataclass_fields__}
    kwargs.update(input_ids_with_completion=ids, model_stamp=current,
                  logprobs=tuple(record["logprobs"]), entropy=tuple(record["entropy"]))
    generation = ActionGeneration(**kwargs)
    if generation.success and parse_action_json(visible_completion(generation, policy.tokenizer)) != generation.parsed_action:
        raise SnArContractError("Stored successful action no longer matches original tokens")
    return generation, {"schema": "snar_base_generation_replay_v1", "source_policy_stamp": asdict(source),
                         "replay_policy_stamp": asdict(current), "original_generation_prefix_hash": record["prefix_hash"],
                         "trace_prefix_hash": tensor_state_hash({"input_ids": ids, "attention_mask": torch.ones_like(ids)}),
                         "scientific_oracle_calls": 0, "new_generation_performed": False}
