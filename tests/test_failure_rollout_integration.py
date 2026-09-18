"""Synthetic CPU integration through real rollout control, never model/oracle calls.

FakePolicy supplies fixed JSON; _features is stubbed and EnvironmentClient is
replaced only for the online-label test. These tests are not material results.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from matdiscovery.failure_controller import HEADS as CONTROLLER_HEADS, MAX_FEEDBACK_CHARS
from matdiscovery.failure_labels import HEADS, SCIENTIFIC_HEADS, scientific_failure_labels, tool_failure_labels
from matdiscovery.native_attribution import PolicyStamp
from matdiscovery.policy import PolicyContextError
from matdiscovery.rollouts import DiscoveryRollout, RolloutSettings
import matdiscovery.rollouts as rollout_module


SELECT_A = {"tool": "select_for_evaluation", "arguments": {"composition": "NaCl", "structure_hash": "fixture-a"}}
SELECT_B = {"tool": "select_for_evaluation", "arguments": {"composition": "NaCl", "structure_hash": "fixture-b"}}
GENERATE = {"tool": "generate_structures", "arguments": {"compositions": ["Na1Cl1"], "generator_name": "random", "num_candidates": 2}}
SCORE = {"tool": "score_buffer", "arguments": {"composition": "NaCl", "scorer_name": "oracle"}}
OBSERVATION = {"state": {"elements": ["Na", "Cl"], "query_count": 0,
                         "phase_diagram_all_entries": [], "ground_truth": {"future_failure": 1}},
               "buffer": {"NaCl": [{"hash": "fixture-a", "scores": {}, "num_sites": 2},
                                    {"hash": "fixture-b", "scores": {}, "num_sites": 2}]},
               "selected": False, "budget": 1, "episode_index": 0,
               "ground_truth": {"future_failure": 1}}


def head_values(**values):
    return {**dict.fromkeys(HEADS, .1), **values}


def job(method="entropy_risk", benchmark="made"):
    return {"job_id": "synthetic-integration-only", "stage": "final_eval", "benchmark": benchmark,
            "method": method, "model_key": "qwen35_4b", "seed": 1, "task_id": "Na-Cl",
            "group_id": "synthetic-group", "budget": 1, "environment_seeds": [1], "episode_ids": ["0"]}


class FakePolicy:
    configuration_fingerprint = "synthetic-policy-only"
    mlp_paths = ()

    def __init__(self, actions, successes=None, context_failures_for_second=0):
        self.actions = copy.deepcopy(actions)
        self.successes = successes or [True] * len(actions)
        self.calls = []
        self.generated_count = 0
        self.context_failures_for_second = context_failures_for_second
        self.stamp = PolicyStamp("synthetic-state", "f" * 64, "synthetic-policy-not-Qwen", 0)

    def runtime_precision_record(self):
        return {"synthetic_fixture_only": True, "dtype": "float32", "device": "cpu"}

    def _configuration(self):
        return {"synthetic_fixture_only": True}

    def generate_action(self, messages, **kwargs):
        self.calls.append({"messages": copy.deepcopy(messages), "seed": kwargs["seed"]})
        if self.generated_count == 1 and self.context_failures_for_second:
            self.context_failures_for_second -= 1
            raise PolicyContextError("synthetic pre-generation context check")
        index = self.generated_count
        if index >= len(self.actions):
            raise AssertionError("Unexpected additional proposal: candidate budget was exceeded")
        self.generated_count += 1
        action, success = self.actions[index], self.successes[index]
        if success:
            assert kwargs["action_validator"](action) is True
        return SimpleNamespace(success=success, parsed_action=copy.deepcopy(action), fixture_index=index,
            entropy=(.1,), logprobs=(-.1,), completion_count=1, prompt_token_count=2,
            input_ids_with_completion=torch.tensor([[11, 12, 20 + index]], dtype=torch.long), model_stamp=self.stamp,
            to_record=lambda: {"synthetic_fixture_only": True, "success": success, "parsed_action": copy.deepcopy(action),
                               "completion_count": 1, "prompt_token_count": 2})


class FakeTypedRisk:
    features = SimpleNamespace(names=["fixture.candidate_index"])

    def __init__(self, overall, typed):
        self.overall, self.typed = list(overall), copy.deepcopy(typed)
        self.primary_calls, self.type_calls = [], []

    def predict_proba(self, x, names):
        assert names == self.features.names
        index = int(x[0, 0]); self.primary_calls.append(index)
        return np.array([self.overall[index]])

    def predict_failure_types(self, x, names):
        assert names == self.features.names
        index = int(x[0, 0]); self.type_calls.append(index)
        return {head: None if self.typed[index][head] is None else np.array([self.typed[index][head]]) for head in HEADS}


def runner(tmp_path, monkeypatch, policy, risk, *, enabled=True):
    value = DiscoveryRollout(tmp_path, policy, settings=RolloutSettings(failure_aware_control=enabled), risk_model=risk)
    monkeypatch.setattr(value, "_features", lambda generation, row, **kwargs: {"fixture.candidate_index": generation.fixture_index})
    return value


def generate(value, tmp_path, method="entropy_risk", observation=None, benchmark="made"):
    directory = tmp_path / "synthetic_generate"
    directory.mkdir()
    rows = []
    action, selected = value._generate(job(method, benchmark), directory,
        copy.deepcopy(OBSERVATION if observation is None else observation), [], [], rows, 0, collection=False)
    return action, selected, rows, directory


@pytest.mark.parametrize("method", ["entropy_risk", "hidden_risk", "graph_risk", "esopt_graph_risk"])
def test_high_type_low_overall_retries_actual_prompt_and_changes_scalar_min_choice(tmp_path, monkeypatch, method):
    assert HEADS == CONTROLLER_HEADS
    policy = FakePolicy([SELECT_A, SELECT_B])
    risk = FakeTypedRisk([.05, .2], [head_values(unstable=.95), head_values(unstable=.2)])
    value = runner(tmp_path, monkeypatch, policy, risk)
    seen_observations = []
    original = rollout_module.plan_failure_response
    def inspect_public(*args, **kwargs):
        seen_observations.append(copy.deepcopy(args[4]))
        return original(*args, **kwargs)
    monkeypatch.setattr(rollout_module, "plan_failure_response", inspect_public)
    action, selected, rows, _ = generate(value, tmp_path, method)
    assert policy.generated_count == 2 and action == SELECT_B
    assert rows[0]["predicted_failure_probability"] < rows[1]["predicted_failure_probability"]
    assert [row["disposition"] for row in rows] == ["rejected_not_executed", "executed"]
    first = json.loads(policy.calls[0]["messages"][-1]["content"])
    second = json.loads(policy.calls[1]["messages"][-1]["content"])
    assert "decision_risk_feedback" not in first
    feedback = second["decision_risk_feedback"]
    assert feedback["predicted_failure_type"] == "unstable"
    assert feedback["instruction"] == rows[0]["failure_controller"]["feedback_for_retry"]
    assert feedback["not_a_measured_scientific_result"] is True
    assert len(feedback["instruction"]) <= MAX_FEEDBACK_CHARS
    assert rows[1]["feedback_received_for_this_candidate"] == feedback
    assert seen_observations == [first["observation"], second["observation"]]
    assert all("ground_truth" not in observation for observation in seen_observations)
    assert all(row["observed_failure_types"][head] is None for row in rows for head in SCIENTIFIC_HEADS)
    assert selected["failure_candidate_selection"]["selected_index"] == 1
    assert "unstable" in selected["failure_candidate_selection"]["common_comparison_heads"]


@pytest.mark.parametrize("method", ["baseline", "esopt"])
def test_baseline_and_esopt_log_predictions_but_never_retry_or_select_by_them(tmp_path, monkeypatch, method):
    policy = FakePolicy([SELECT_A, SELECT_B])
    risk = FakeTypedRisk([.95, .01], [head_values(unstable=.99), head_values()])
    value = runner(tmp_path, monkeypatch, policy, risk)
    action, selected, rows, _ = generate(value, tmp_path, method)
    assert action == SELECT_A and len(policy.calls) == len(rows) == 1
    assert selected["predicted_failure_probability"] == .95
    assert selected["failure_type_probabilities"]["unstable"] == .99
    assert selected["failure_types_used_for_control"] is False
    assert selected["confidence_used_for_control"] is False
    assert "failure_controller" not in selected and "failure_candidate_selection" not in selected
    assert "decision_risk_feedback" not in json.loads(policy.calls[0]["messages"][-1]["content"])


def test_type_control_requires_explicit_setting_and_stays_disabled_for_cg(tmp_path, monkeypatch):
    policy = FakePolicy([SELECT_A, SELECT_B])
    risk = FakeTypedRisk([.05, .1], [head_values(unstable=.99), head_values()])
    value = runner(tmp_path, monkeypatch, policy, risk, enabled=False)
    action, selected, rows, _ = generate(value, tmp_path)
    assert action == SELECT_A and len(rows) == 1 and risk.type_calls == []
    assert "failure_controller" not in selected
    cg = tmp_path / "cg"; cg.mkdir()
    cg_policy = FakePolicy([{"action": "Na"}])
    cg_risk = FakeTypedRisk([.05], [head_values(unstable=.99)])
    cg_value = runner(cg, monkeypatch, cg_policy, cg_risk, enabled=True)
    action, selected, rows, _ = generate(cg_value, cg, benchmark="crystalgym", observation={"legal_actions": [{"element": "Na"}], "episode_index": 0})
    assert action == {"action": "Na"} and cg_risk.type_calls == []
    assert "failure_type_probabilities" not in selected and "failure_controller" not in selected


def test_rejected_candidate_stays_unknown_and_failed_science_is_not_instability(tmp_path, monkeypatch):
    policy = FakePolicy([SELECT_A, SELECT_B])
    risk = FakeTypedRisk([.05, .2], [head_values(unstable=.9), head_values()])
    value = runner(tmp_path, monkeypatch, policy, risk)
    _, _, rows, output = generate(value, tmp_path)
    response = {"id": 42, "ok": False, "error": {"code": "synthetic_numerical_failure"}}
    value._label_rows(output, rows, 1, scientific_response=response)
    assert rows == []
    saved = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
    assert saved[0]["label_future_failure"] is None
    assert all(saved[0]["observed_failure_types"][head] is None for head in HEADS[1:])
    assert saved[1]["label_future_failure"] == 1
    assert saved[1]["observed_failure_types"]["scientific_evaluation_failure"] == 1
    assert saved[1]["observed_failure_types"]["unstable"] is None
    assert saved[1]["observed_failure_types"]["not_new"] is None
    assert saved[1]["failure_type_observation_rpc_ids"]["scientific_evaluation_failure"] == [42]


def test_successful_science_labels_are_raw_booleans_not_joint_outcome(tmp_path, monkeypatch):
    policy = FakePolicy([SELECT_A])
    risk = FakeTypedRisk([.1], [head_values()])
    value = runner(tmp_path, monkeypatch, policy, risk)
    _, _, rows, output = generate(value, tmp_path)
    response = {"id": 17, "ok": True, "result": {"official_observation": {"is_stable": False, "is_newly_discovered": True}}}
    value._label_rows(output, rows, 1, scientific_response=response)
    saved = json.loads((output / "decisions.jsonl").read_text())
    assert saved["observed_failure_types"]["scientific_evaluation_failure"] == 0
    assert saved["observed_failure_types"]["unstable"] == 1
    assert saved["observed_failure_types"]["not_new"] == 0
    assert all(saved["failure_type_observation_rpc_ids"][head] == [17] for head in SCIENTIFIC_HEADS)


def test_legal_unknown_heads_remain_none_with_explicit_fallback(tmp_path, monkeypatch):
    policy = FakePolicy([SELECT_A, SELECT_B])
    risk = FakeTypedRisk([.9, .2], [dict.fromkeys(HEADS), dict.fromkeys(HEADS)])
    value = runner(tmp_path, monkeypatch, policy, risk)
    action, selected, rows, _ = generate(value, tmp_path)
    assert action == SELECT_B
    assert all(row["failure_type_probabilities"][head] is None for row in rows for head in HEADS)
    selection = selected["failure_candidate_selection"]
    assert selection["type_signals_used"] is False
    assert selection["common_comparison_heads"] == []
    assert selection["fallback"] == "scalar_only_no_common_available_types"
    assert all(row["compared_head_probabilities"] == {} for row in selection["ranking"])


def test_retry_feedback_survives_original_three_summary_levels(tmp_path, monkeypatch):
    policy = FakePolicy([SELECT_A, SELECT_B], context_failures_for_second=2)
    risk = FakeTypedRisk([.05, .2], [head_values(unstable=.9), head_values()])
    value = runner(tmp_path, monkeypatch, policy, risk)
    _, _, rows, _ = generate(value, tmp_path)
    assert policy.generated_count == 2 and len(policy.calls) == 4
    retry_payloads = [json.loads(call["messages"][-1]["content"]) for call in policy.calls[1:]]
    assert [payload["memory_summary_level"] for payload in retry_payloads] == [0, 1, 2]
    assert all(payload["decision_risk_feedback"] == retry_payloads[0]["decision_risk_feedback"] for payload in retry_payloads)
    assert rows[1]["memory_summary_level"] == 2


class FakeEnvironment:
    """Three tools then one fake failed science call, with genuine RPC-shaped IDs."""
    def __init__(self, generation_ok=True):
        self.generation_ok = generation_ok
        self.counter = 0
        self.completed = False
        self.requests = []
        self.responses = {}

    def __enter__(self): return self
    def __exit__(self, *args): return None

    def observation(self):
        return {**copy.deepcopy(OBSERVATION), "episode_seed": 1, "done": self.completed,
                "counts": {"candidate_oracle_attempts": int(self.completed)}}

    def request(self, operation, arguments=None, check=True):
        rpc_id = self.counter; self.counter += 1
        self.requests.append((rpc_id, operation, copy.deepcopy(arguments)))
        if operation == "init": return {"metadata": {"synthetic_fixture_only": True}, "observation": self.observation()}
        if operation == "observe": return self.observation()
        if operation == "tool":
            name = arguments["name"]
            if name == "generate_structures" and not self.generation_ok:
                response = {"id": rpc_id, "ok": False, "error": {"code": "synthetic_generator_exception"}}
            else:
                output = {"generated": 2, "accepted": 0, "records": [{"accepted": False}, {"accepted": False}]} if name == "generate_structures" else [{"composition": "NaCl", "scores": [{"nonfinite": "nan"}]}] if name == "score_buffer" else {"hash": "fixture-a"}
                response = {"id": rpc_id, "ok": True, "result": {"tool": name, "output": output}}
            self.responses[rpc_id] = response
            return copy.deepcopy(response)
        assert operation == "step"
        self.completed = True
        response = {"id": rpc_id, "ok": False, "error": {"code": "synthetic_ORB_failure", "details": self.observation()}}
        self.responses[rpc_id] = response
        return copy.deepcopy(response)


@pytest.mark.parametrize("generation_ok", [True, False])
def test_run_online_labels_come_from_actual_tool_rpc_and_next_science_response(tmp_path, monkeypatch, generation_ok):
    policy = FakePolicy([GENERATE, SCORE, SELECT_A])
    risk = FakeTypedRisk([.1] * 3, [head_values()] * 3)
    value = runner(tmp_path, monkeypatch, policy, risk)
    environment = FakeEnvironment(generation_ok)
    monkeypatch.setattr(rollout_module, "EnvironmentClient", lambda *args, **kwargs: environment)
    monkeypatch.setattr(rollout_module, "make_environment_arguments", lambda *args: {"budget": 1, "seed": 1, "elements": ["Na", "Cl"]})
    output = tmp_path / "synthetic_full_run"
    result = value.run(job(), output, collection=False)
    rows = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and policy.generated_count == 3
    assert result[0]["costs"]["candidate_oracle_attempts"] == 1
    tool_calls = [(rpc_id, args) for rpc_id, operation, args in environment.requests if operation == "tool"]
    science_id = next(rpc_id for rpc_id, operation, _ in environment.requests if operation == "step")
    for row, (rpc_id, args) in zip(rows, tool_calls):
        expected = tool_failure_labels({"tool": args["name"], "arguments": args["arguments"]}, environment.responses[rpc_id])
        assert row["observed_failure_types"]["generation_invalid"] == 0
        for head, label in expected.items():
            assert row["observed_failure_types"][head] == label
            if label is not None: assert row["failure_type_observation_rpc_ids"][head] == [rpc_id]
        assert {head: row["observed_failure_types"][head] for head in SCIENTIFIC_HEADS} == scientific_failure_labels(environment.responses[science_id])
        assert row["failure_type_observation_rpc_ids"]["scientific_evaluation_failure"] == [science_id]
    assert rows[0]["observed_failure_types"]["candidate_generation_failure"] == (1 if generation_ok else None)
    assert rows[0]["observed_failure_types"]["tool_execution_failure"] == (0 if generation_ok else 1)
    assert rows[1]["observed_failure_types"]["screening_unavailable"] == 1
    assert all(row["observed_failure_types"]["unstable"] is None for row in rows)
    assert all("observed_failure_types" not in json.loads(call["messages"][-1]["content"]) for call in policy.calls)
