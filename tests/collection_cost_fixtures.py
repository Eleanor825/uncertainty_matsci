"""Complete synthetic cost files accepted by the REAL shared source validator.

MADE journals use the production recorder/executor with only unit return
functions. No source, journal, closure, or receipt verifier is mocked. CG rows
are explicit counter/transcript fixtures, not DFT results; activation files are
marked placeholders and are never passed to fitting. No research data or
production registration is created here.
"""
from collections import defaultdict
from pathlib import Path

from matdiscovery.accounting import fingerprint, write_json_atomic
from matdiscovery.collection_provenance import EXECUTION_V1, build_collection_plan
from test_collection_provenance import artifact, complete_job, freeze, rewrite_receipt, rows


def _complete_cg(fixture, index, plan):
    """A full registered episode-count fixture, without any physical DFT call."""
    job = fixture["jobs"][index]
    directory = fixture["base"] / job["job_id"]
    identity = fixture["policy_identity"]
    write_json_atomic(directory / "job.json", {**job, "policy_configuration_fingerprint": identity["configuration_fingerprint"]})
    episodes = [{**{key: job[key] for key in ("benchmark", "model_key", "method", "task_id", "seed")},
        "episode_id": episode, "environment_seed": seed, "complete": True,
        "costs": {"candidate_oracle_attempts": 0, "dft_episode_attempts": 1, "wall_seconds": 1.0},
        "classification": "synthetic_cost_fixture_not_DFT_result"}
        for episode, seed in zip(job["episode_ids"], job["environment_seeds"], strict=True)]
    write_json_atomic(directory / "episodes.json", episodes)
    for name in ("decisions.jsonl", "decision_events.jsonl"):
        rows(directory / name, [{"classification": "unit_test_only_not_policy_output"}])
    rpc = []
    def call(op, args, result):
        identifier = len(rpc) // 2
        rpc.extend([{"direction": "request", "payload": {"id": identifier, "op": op, "args": args}},
                    {"direction": "response", "payload": {"id": identifier, "ok": True, "result": result}}])
    call("init", {"benchmark": "crystalgym", "seed": job["environment_seeds"][0], "budget": job["budget"]},
         {"classification": "unit_test_only"})
    for index in range(len(episodes)):
        call("step", {"action": "unit-only"}, {"counts": {"dft_episode_attempts": index + 1}, "classification": "unit_test_only"})
    call("close", {}, {"closed": True, "counts": {"dft_episode_attempts": len(episodes)}})
    rows(directory / "rpc/rpc.jsonl", rpc)
    shards = []
    for layer in range(32):
        path = directory / "activation_shards" / (str(layer) + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unit fixture, not an activation tensor")
        shards.append({"path": str(path), "layer_path": f"model.language_model.layers.{layer}.mlp", "tensor_hash": fingerprint(layer)})
    write_json_atomic(directory / "collection_manifest.json", {"complete": True, "job_id": job["job_id"],
        "policy_configuration_fingerprint": identity["configuration_fingerprint"], "policy_runtime": identity["policy_runtime"],
        "decision_files": [artifact(directory / "decisions.jsonl")], "activation_shards": shards})
    rewrite_receipt(directory, job, plan, identity)
    return directory


def write_cost_collection_fixtures(root, jobs, *, benchmark=None):
    root = Path(root).resolve()
    write_json_atomic(root / "configs/main_protocol.json", {"schema_version": 5, "classification": "unit_test_only",
                     "made_execution": EXECUTION_V1})
    for name in ("configs/model_manifest.json", "configs/benchmark_tasks.json", "configs/made_splits.json",
                 "configs/policy_runtime_gates.json", "configs/assets.runtime.json", "configs/qe.runtime.json",
                 "data/raw/materials_project/index.json"):
        write_json_atomic(root / name, {"classification": "unit_test_only_not_scientific_input"})
    for name in ("scripts/run_collection.py", "src/matdiscovery/fixture.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# unit test frozen source, never executed\n")
    policy = {"classification": "unit_test_only", "policy_runtime": {"dtype": "torch.float32", "classification": "unit_test_only"}}
    identity = {"policy_configuration": policy, "configuration_fingerprint": fingerprint(policy),
                "policy_runtime": policy["policy_runtime"], "checkpoint_hash": fingerprint("unit checkpoint, no weights")}
    grouped = defaultdict(list)
    for job in jobs:
        if benchmark is None or job["benchmark"] == benchmark:
            grouped[(job["model_key"], job["benchmark"])].append(job)
    directories = []
    for key, expected in grouped.items():
        base = root / "experiments/collection" / key[0] / key[1]
        base.mkdir(parents=True, exist_ok=True)
        write_json_atomic(base / "policy_configuration.json", identity)
        plan = build_collection_plan(root, expected)
        freeze(root, plan)
        fixture = {"root": root, "base": base, "jobs": expected, "policy_identity": identity}
        for index in range(len(expected)):
            directories.append((complete_job if key[1] == "made" else _complete_cg)(fixture, index, plan))
        write_json_atomic(base / "plan.json", plan)
        write_json_atomic(base / "progress.json", {"complete": True, "expected_jobs": len(expected), "completed_jobs": len(expected),
            "collection_manifests": [str(base / job["job_id"] / "collection_manifest.json") for job in expected]})
    return directories
