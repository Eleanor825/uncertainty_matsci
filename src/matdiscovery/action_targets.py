"""Exact emitted-action token targets for unwarped teacher-forced log likelihood.

Targets are tokens bearing JSON *values*: CrystalGym's action value, or MADE's
tool value and scalar values under arguments. Field names, rationale values,
special tokens, and tokens containing punctuation alone are not targets. A token
may contain inseparable adjacent JSON punctuation as well as a semantic value.
Ambiguous decoding/token boundaries fail closed; text is never re-tokenized into
a different sequence and represented as the actually generated action.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Mapping

from .accounting import fingerprint


TARGET_SCHEMA = "emitted_action_value_mean_unwarped_logprob_v2"


class ActionTargetError(ValueError):
    pass


@dataclass(frozen=True)
class ActionTargets:
    benchmark: str
    prompt_token_count: int
    token_positions: tuple[int, ...]
    prediction_positions: tuple[int, ...]
    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]
    semantic_paths: tuple[str, ...]
    special_token_ids: tuple[int, ...]
    source_prefix_hash: str
    source_length: int
    target_spec_hash: str
    tokenizer_class: str
    schema: str = TARGET_SCHEMA

    @property
    def causal_horizon(self) -> int:
        return max(self.prediction_positions)

    def specification(self) -> dict:
        # Source evidence is hashed separately: appending irrelevant future text
        # changes the source hash, but not which earlier action tokens are scored.
        return {key: value for key, value in asdict(self).items()
                if key not in {"source_prefix_hash", "source_length", "target_spec_hash"}}

    def to_dict(self) -> dict:
        return {**asdict(self), "causal_horizon": self.causal_horizon,
                "token_count": len(self.token_ids),
                "score_semantics": "mean log p_theta(emitted value token | all strictly earlier tokens), before temperature/top-k/top-p",
                "score_units": "nats per selected emitted value token",
                "probability_semantics": "A mean log probability, not an action probability, sum of probabilities, or correctness confidence.",
                "weighting": "uniform over selected original token IDs",
                "source_hash_inputs": ["input_ids", "attention_mask"]}

    def validate(self, input_ids, attention_mask) -> None:
        from .esopt import tensor_state_hash
        if self.schema != TARGET_SCHEMA or fingerprint(self.specification()) != self.target_spec_hash:
            raise ActionTargetError("Unknown or modified action target specification")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] != self.source_length:
            raise ActionTargetError("Action targets belong to a different complete input sequence")
        if tensor_state_hash({"input_ids": input_ids, "attention_mask": attention_mask}) != self.source_prefix_hash:
            raise ActionTargetError("Action targets belong to a different source prefix")
        if (not self.token_positions or tuple(sorted(set(self.token_positions))) != self.token_positions
                or tuple(t - 1 for t in self.token_positions) != self.prediction_positions
                or min(self.token_positions) < self.prompt_token_count
                or max(self.token_positions) >= self.source_length):
            raise ActionTargetError("Invalid token-to-prediction-position alignment")
        if tuple(int(input_ids[0, t]) for t in self.token_positions) != self.token_ids:
            raise ActionTargetError("Target token IDs differ from the emitted sequence")
        if set(self.token_ids) & set(self.special_token_ids):
            raise ActionTargetError("Special tokens cannot be action-value targets")

    def mean_logprob(self, selected_logits):
        """Logits must be raw LM outputs at exactly prediction_positions, in order."""
        import torch
        if (selected_logits.ndim != 3 or selected_logits.shape[0] != 1
                or selected_logits.shape[1] != len(self.token_ids)):
            raise ActionTargetError("Raw logits are not aligned to the selected prediction positions")
        if not torch.isfinite(selected_logits).all():
            raise ActionTargetError("Unwarped LM logits must be finite; processed/filtered scores are forbidden")
        target = torch.tensor(self.token_ids, dtype=torch.long, device=selected_logits.device)
        if int(target.max()) >= selected_logits.shape[-1] or int(target.min()) < 0:
            raise ActionTargetError("Target token is outside the original LM vocabulary")
        values = selected_logits[0].float().log_softmax(-1).gather(-1, target[:, None]).squeeze(-1)
        return values.mean()


class _JSONSpans:
    def __init__(self, text):
        self.text = text
        self.leaves = []
        self.keys = []

    def _space(self, pos):
        while pos < len(self.text) and self.text[pos].isspace():
            pos += 1
        return pos

    def parse(self, pos, path=()):
        pos = self._space(pos)
        if pos >= len(self.text):
            raise ActionTargetError("Incomplete JSON value")
        first = self.text[pos]
        if first == "{":
            result, pos = {}, self._space(pos + 1)
            if pos < len(self.text) and self.text[pos] == "}":
                return result, pos + 1
            while True:
                if pos >= len(self.text) or self.text[pos] != '"':
                    raise ActionTargetError("JSON object key is not a string")
                key, end = json.decoder.scanstring(self.text, pos + 1, True)
                if key in result:
                    raise ActionTargetError("Duplicate JSON field name makes the executed action ambiguous")
                self.keys.append((pos + 1, end - 1))
                pos = self._space(end)
                if pos >= len(self.text) or self.text[pos] != ":":
                    raise ActionTargetError("Missing JSON object colon")
                result[key], pos = self.parse(pos + 1, (*path, key))
                pos = self._space(pos)
                if pos < len(self.text) and self.text[pos] == "}":
                    return result, pos + 1
                if pos >= len(self.text) or self.text[pos] != ",":
                    raise ActionTargetError("Missing JSON object separator")
                pos = self._space(pos + 1)
        if first == "[":
            result, pos = [], self._space(pos + 1)
            if pos < len(self.text) and self.text[pos] == "]":
                return result, pos + 1
            while True:
                value, pos = self.parse(pos, (*path, len(result)))
                result.append(value)
                pos = self._space(pos)
                if pos < len(self.text) and self.text[pos] == "]":
                    return result, pos + 1
                if pos >= len(self.text) or self.text[pos] != ",":
                    raise ActionTargetError("Missing JSON array separator")
                pos = self._space(pos + 1)
        if first == '"':
            value, end = json.decoder.scanstring(self.text, pos + 1, True)
            self.leaves.append((path, pos + 1, end - 1))
            return value, end
        value, end = json.JSONDecoder().raw_decode(self.text, pos)
        if isinstance(value, (dict, list, str)) or value not in (None, True, False) and not isinstance(value, (int, float)):
            raise ActionTargetError("Unexpected JSON primitive")
        if isinstance(value, float) and not math.isfinite(value):
            raise ActionTargetError("Nonfinite numbers are not an emitted valid JSON action")
        self.leaves.append((path, pos, end))
        return value, end


def _semantic(path, benchmark):
    if not path or "rationale" in path:
        return False
    if benchmark == "crystalgym":
        return path == ("action",)
    return path == ("tool",) or path[0] == "arguments"


def build_action_targets(
    input_ids, *, prompt_token_count: int, tokenizer, benchmark: str,
    parsed_action: Mapping[str, Any] | None = None, attention_mask=None,
) -> ActionTargets:
    """Map the actual emitted JSON values to immutable original token indices."""
    import torch
    from .esopt import tensor_state_hash
    if benchmark not in {"made", "crystalgym"}:
        raise ActionTargetError("Action target mapping requires a declared materials benchmark")
    if not isinstance(parsed_action, Mapping):
        raise ActionTargetError("The original parsed emitted action is required; invalid/ambiguous generations have no action target")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype not in {torch.int32, torch.int64}:
        raise ActionTargetError("Provide one complete original integer token sequence")
    length = input_ids.shape[1]
    if type(prompt_token_count) is not int or not 1 <= prompt_token_count < length:
        raise ActionTargetError("The original prompt token count is required for teacher-forced alignment")
    mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
    if mask.shape != input_ids.shape or not bool((mask == 1).all()):
        raise ActionTargetError("The complete unpadded sequence and its original mask are required")
    ids = input_ids[0, prompt_token_count:].detach().cpu().tolist()
    special = tuple(sorted(set(int(t) for t in getattr(tokenizer, "all_special_ids", []))))
    decode = lambda tokens: tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = decode(ids)
    boundaries = [0]
    for end in range(1, len(ids) + 1):
        prefix = decode(ids[:end])
        boundaries.append(len(prefix) if text.startswith(prefix) else None)
    spans = [(start, end) if start is not None and end is not None and end >= start else None
             for start, end in zip(boundaries, boundaries[1:])]
    visible = list(text)
    for token, span in zip(ids, spans):
        if token in special:
            if span is None:
                raise ActionTargetError("Cannot map a special token's exact decoded span")
            visible[span[0]:span[1]] = " " * (span[1] - span[0])
    scan_text = "".join(visible)
    matches = []
    for start, character in enumerate(scan_text):
        if character != "{":
            continue
        parser = _JSONSpans(scan_text)
        try:
            value, end = parser.parse(start)
        except (ValueError, IndexError, RecursionError):
            continue
        valid = isinstance(value, dict) and (
            isinstance(value.get("action"), str) if benchmark == "crystalgym" else
            isinstance(value.get("tool"), str) and isinstance(value.get("arguments"), dict)
        )
        if valid and (parsed_action is None or value == dict(parsed_action)):
            matches.append((start, end, parser))
    if len(matches) != 1:
        raise ActionTargetError("The exact emitted action JSON span is missing or ambiguous")
    start, end, parser = matches[0]
    selected = [(path, a, b) for path, a, b in parser.leaves if _semantic(path, benchmark) and a < b]
    forbidden = parser.keys + [(a, b) for path, a, b in parser.leaves if not _semantic(path, benchmark)]
    if not selected:
        raise ActionTargetError("The emitted JSON has no nonempty action-bearing scalar values")
    positions, texts, covered = [], [], set()
    for offset, (token, span) in enumerate(zip(ids, spans)):
        if token in special or span is None:
            continue
        a, b = span
        overlap = {p for _, left, right in selected for p in range(max(a, left), min(b, right))}
        if not overlap:
            continue
        if any(max(a, left) < min(b, right) for left, right in forbidden):
            raise ActionTargetError("A token inseparably mixes action values with field names or rationale")
        positions.append(prompt_token_count + offset)
        texts.append(text[a:b])
        covered.update(overlap)
    wanted = {p for _, a, b in selected for p in range(a, b)}
    if not positions or covered != wanted:
        raise ActionTargetError("Action value characters cannot be mapped completely to original token boundaries")
    values = {
        "benchmark": benchmark, "prompt_token_count": prompt_token_count,
        "token_positions": tuple(positions), "prediction_positions": tuple(t - 1 for t in positions),
        "token_ids": tuple(int(input_ids[0, t]) for t in positions), "token_texts": tuple(texts),
        "semantic_paths": tuple("/" + "/".join(map(str, path)) for path, _, _ in selected),
        "special_token_ids": special, "source_prefix_hash": tensor_state_hash({"input_ids": input_ids, "attention_mask": mask}),
        "source_length": length, "tokenizer_class": type(tokenizer).__module__ + "." + type(tokenizer).__name__,
        "schema": TARGET_SCHEMA,
    }
    specification = {key: value for key, value in values.items() if key not in {"source_prefix_hash", "source_length"}}
    targets = ActionTargets(**values, target_spec_hash=fingerprint(specification))
    targets.validate(input_ids, mask)
    return targets
