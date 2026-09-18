"""UNIT TEST FIXTURES ONLY: fake policy/environment, no material-evaluator results."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from matdiscovery.native_attribution import PolicyStamp
from matdiscovery.rollouts import DiscoveryRollout
import matdiscovery.rollouts as rollouts


MLPS = tuple(f"model.language_model.layers.{i}.mlp" for i in range(32))


class UnitTestPolicy:
    """Deterministic synthetic outputs; deliberately not a material-discovery model."""
    configuration_fingerprint = "unit_test_policy_configuration"
    mlp_paths = MLPS

    def __init__(self, benchmark):
        self.benchmark = benchmark
        self.calls = []
        self.model_stamp = PolicyStamp("unit_test_state", "a" * 64, "Qwen/Qwen3.5-4B", 0)

    def runtime_precision_record(self):
        return {"unit_test_fixture_only": True, "dtype": "torch.float32", "device": "cpu"}

    def _configuration(self):
        return {"unit_test_fixture_only": True, "runtime": self.runtime_precision_record()}

    def generate_action(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        number = len(self.calls)
        if self.benchmark == "made":
            action = {"tool": "list_compositions", "arguments": {}} if number % 2 else {
                "tool": "select_for_evaluation", "arguments": {"composition": "AlVZn", "structure_hash": "unit_test_structure"}}
        else:
            action = {"action": "Li"}
        ids = torch.arange(32, dtype=torch.long).reshape(1, -1)
        ids[0, -1] = 100 + number
        return SimpleNamespace(
            success=True, parsed_action=action, entropy=(0.2, 0.3), logprobs=(-0.1, -0.2),
            completion_count=2, prompt_token_count=30, input_ids_with_completion=ids,
            model_stamp=self.model_stamp, to_record=lambda: {
                "unit_test_fixture_only": True, "completion_count": 2, "prompt_token_count": 30,
            },
        )

    def capture_prefix(self, ids, **kwargs):
        values = ids.float().unsqueeze(-1).repeat(1, 1, 2)
        return SimpleNamespace(
            hidden_summaries={layer: {"norm": 1.0} for layer in MLPS},
            last_logits=torch.tensor([[0.0, 1.0]]), parity_max_abs=0.0,
            mlp_inputs={layer: values for layer in MLPS},
            mlp_outputs={layer: values + 1 for layer in MLPS},
        )


class UnitTestEnvironment:
    """Synthetic accounting state machine; it never calls ORB or QE."""

    def __init__(self, benchmark, fail_attempt=7):
        self.benchmark, self.fail_attempt = benchmark, fail_attempt
        self.requests, self.completed, self.episode, self.site, self.successes = [], 0, 0, 0, 0
        self.counts = {"initialization_oracle_attempts": 2} if benchmark == "made" else {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def observe(self):
        common = {"done": self.completed >= self.budget, "counts": dict(self.counts),
                  "episode_index": self.episode, "episode_seed": self.seed + self.episode,
                  "budget": self.budget}
        if self.benchmark == "made":
            return {**common, "state": {"elements": ["Al", "V", "Zn"], "query_count": self.successes},
                    "buffer": {"AlVZn": [{"hash": "unit_test_structure"}]}, "selected": False}
        return {**common, "property": "band_gap", "target": 2.0,
                "legal_actions": [{"element": "Li"}, {"element": "Na"}],
                "num_sites": 2, "filled_count": self.site, "focus_site": self.site,
                "filled_elements": ["Li"] * self.site,
                "lattice_lengths": [3.0, 3.0, 3.0], "lattice_angles": [90.0] * 3,
                "fractional_coordinates": [[0.0] * 3, [0.5] * 3]}

    def request(self, operation, arguments=None, *, check=True):
        self.requests.append((operation, arguments))
        correlation_id = len(self.requests)
        if operation == "init":
            self.seed, self.budget = arguments["seed"], arguments["budget"]
            return {"metadata": {"unit_test_fixture_only": True}, "observation": self.observe()}
        if operation == "observe":
            return self.observe()
        if operation == "tool":
            return {"id": correlation_id, "ok": True,
                    "result": {"tool": arguments["name"], "output": [], "unit_test_fixture_only": True}}
        if operation == "reset":
            assert self.site == 2 and self.completed < self.budget
            self.episode += 1
            self.site = 0
            return self.observe()
        assert operation == "step"
        if self.benchmark == "made":
            self.completed += 1
            self.counts["candidate_oracle_attempts"] = self.completed
            if self.completed == self.fail_attempt:
                return {"id": correlation_id, "ok": False, "error": {"code": "unit_test_oracle_exception"}}
            self.successes += 1
            return {"id": correlation_id, "ok": True, "result": {
                "observation": self.observe(), "official_metrics": {"num_newly_discovered_stable": self.successes},
                "official_observation": {"is_stable": True, "is_newly_discovered": True},
            }}
        self.site += 1
        self.counts["atomic_action_attempts"] = self.counts.get("atomic_action_attempts", 0) + 1
        terminal = self.site == 2
        if terminal:
            self.completed += 1
            self.counts["dft_episode_attempts"] = self.completed
        success = self.episode != 2
        return {"id": correlation_id, "ok": True, "result": {"observation": self.observe(), "episode_done": terminal,
                "scientific_result": {"dft_success": success if terminal else None,
                                      "absolute_target_error": 0.0 if success else None,
                                      "reward": 1.0 if success else -1.0,
                                      "property_value": 2.0 if success else None}}}


def install_environment(monkeypatch, benchmark):
    environment = UnitTestEnvironment(benchmark)
    monkeypatch.setattr(rollouts, "EnvironmentClient", lambda *args, **kwargs: environment)
    monkeypatch.setattr(rollouts, "make_environment_arguments", lambda project, job, output, settings: {
        "budget": job["budget"], "seed": job["environment_seeds"][0],
        "elements": ["Al", "V", "Zn"], "target": 2.0,
    })
    return environment


def job(benchmark):
    budget = 50 if benchmark == "made" else 5
    episodes = 1 if benchmark == "made" else 5
    return {"job_id": "unit-test-only-" + benchmark, "stage": "collection", "benchmark": benchmark,
            "method": "baseline", "model_key": "qwen35_4b", "seed": 1, "split": "train",
            "task_id": "unit-test-task", "group_id": "unit-test-group", "budget": budget,
            "environment_seeds": list(range(100, 100 + episodes)),
            "episode_ids": [str(i) for i in range(episodes)]}


def test_full_b50_counts_failed_oracle_once_and_collects_all_layer_shards(tmp_path, monkeypatch):
    environment = install_environment(monkeypatch, "made")
    policy = UnitTestPolicy("made")
    output = tmp_path / "unit-test-b50"
    summaries = DiscoveryRollout(tmp_path, policy).run(job("made"), output, collection=True)
    assert len(summaries) == 1
    assert summaries[0]["complete"] is True
    assert summaries[0]["costs"]["candidate_oracle_attempts"] == 50
    assert sum(op == "step" for op, _ in environment.requests) == 50
    assert summaries[0]["metrics"]["mSUN"] == pytest.approx(49 / 50)
    rows = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
    assert len(rows) == 100
    assert sum(row["label_future_failure"] == 1 for row in rows) == 2  # selected query plus preceding tool
    assert summaries[0]["costs"]["llm_calls"] == 100
    manifest = json.loads((output / "collection_manifest.json").read_text())
    assert len(manifest["activation_shards"]) == 32
    for shard in manifest["activation_shards"]:
        saved = torch.load(shard["path"], map_location="cpu", weights_only=True)
        assert saved["metadata"]["rows"] == 100 * 16
        assert saved["metadata"]["policy_fingerprint"] == policy.model_stamp.checkpoint_hash
        assert saved["metadata"]["tensor_hash"] == shard["tensor_hash"]
    assert all("label_future_failure" not in event for event in [json.loads(line) for line in (output / "decision_events.jsonl").read_text().splitlines()])


def test_crystal_episodes_reset_history_and_keep_failure_in_budget(tmp_path, monkeypatch):
    environment = install_environment(monkeypatch, "crystalgym")
    policy = UnitTestPolicy("crystalgym")
    output = tmp_path / "unit-test-crystal"
    summaries = DiscoveryRollout(tmp_path, policy).run(job("crystalgym"), output, collection=True)
    assert len(summaries) == 5
    assert [row["environment_seed"] for row in summaries] == list(range(100, 105))
    assert [row["costs"]["dft_episode_attempts"] for row in summaries] == [1] * 5
    assert sum(row["status"] == "failed" for row in summaries) == 1
    assert sum(op == "reset" for op, _ in environment.requests) == 4
    for call in policy.calls[::2]:
        payload = json.loads(call["messages"][1]["content"])
        assert payload["scientific_history"] == []
        assert payload["recent_tools"] == []
        assert payload["observation"]["filled_count"] == 0
        assert "seed" not in payload["observation"]
        assert "prototype_id" not in payload["observation"]
    rows = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
    assert len(rows) == 10
    assert [row["label_future_failure"] for row in rows] == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]


def test_rejected_candidate_never_receives_counterfactual_scientific_label(tmp_path):
    policy = UnitTestPolicy("crystalgym")
    probabilities = iter([0.9, 0.1])
    risk = SimpleNamespace(features=SimpleNamespace(names=["sampling.entropy_mean"]),
                           predict_proba=lambda *args: np.asarray([next(probabilities)]))
    runner = DiscoveryRollout(tmp_path, policy, risk_model=risk)
    case = {**job("crystalgym"), "method": "entropy_risk"}
    output = tmp_path / "unit-test-counterfactual"
    output.mkdir()
    rows = []
    action, selected = runner._generate(case, output,
                                        {"legal_actions": [{"element": "Li"}], "episode_index": 0},
                                        [], [], rows, 0, collection=False)
    assert action == {"action": "Li"}
    assert rows[0]["disposition"] == "rejected_not_executed"
    assert rows[1]["disposition"] == "executed"
    runner._label_rows(output, rows, 1)
    recorded = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
    assert recorded[0]["label_future_failure"] is None
    assert recorded[1]["label_future_failure"] == 1


def test_collection_uses_partitioned_policy_stream_not_only_reporting_seed(tmp_path):
    # Same visible toy state, same analysis seed; different train/dev sampling
    # streams must reach the model instead of mechanically replaying a prefix.
    observed = []
    for partition, sampling_seed in (("train", 123456), ("dev", 234567)):
        policy = UnitTestPolicy("crystalgym")
        runner = DiscoveryRollout(tmp_path, policy)
        output = tmp_path / partition
        output.mkdir()
        case = {**job("crystalgym"), "split": partition, "policy_sampling_seed": sampling_seed}
        runner._generate(case, output, {"legal_actions": [{"element": "Li"}], "episode_index": 0},
            [], [], [], 0, collection=False)
        observed.append(policy.calls[0]["seed"])
    assert observed == [123456 * 10_000_019, 234567 * 10_000_019]
