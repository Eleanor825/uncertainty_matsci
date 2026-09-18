#!/usr/bin/env python3
"""One-time evidence-preserving reconciliation of the recorded prompt failure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    condition = root / "experiments/collection/qwen35_4b/made"
    job = condition / "collection-made-b5744f83a99f950e49cfafd4"
    rpc = job / "rpc/rpc.jsonl"
    regression = read(root / "technical_not_main/prompt_summary_regression.json")
    assert hashlib.sha256(rpc.read_bytes()).hexdigest() == regression["source_rpc_sha256"]
    assert regression["all_recorded_requests_have_responses"]
    assert min(regression["repaired_prompt_token_counts"]) <= 3072
    rows = [json.loads(line) for line in rpc.read_text().splitlines()]
    requests = {r["payload"]["id"] for r in rows if r.get("direction") == "request"}
    responses = {r["payload"]["id"] for r in rows if r.get("direction") == "response"}
    assert requests == responses
    status = read(root / "logs/pipeline_status.json")
    assert status["state"] == "halted_requires_reconciliation"
    failed = status["stages"]["collect-qwen35_4b-made"]
    assert failed["status"] == "failed" and failed["returncode"] == 1
    pids = {status["supervisor_pid"], failed["pid"]}
    pids.update(r["environment_pid"] for r in rows if "environment_pid" in r)
    for pid in pids:
        assert not Path(f"/proc/{pid}").exists(), f"Previous process {pid} still exists"
    assert not list(condition.rglob("completion.json")), "A complete job must not be discarded"
    costs = regression["last_scientific_counts"]
    assert costs["initialization_oracle_attempts"] == 28
    assert costs["surrogate_oracle_attempts"] == 6
    assert costs["candidate_oracle_attempts"] == 1
    archive = root / "experiments/interrupted/001_prompt_context_failure"
    assert not archive.exists()
    archive.mkdir(parents=True)
    artifacts = [{"path_before": str(p), "path_after": str(archive / "condition" / p.relative_to(condition)),
                  "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                 for p in sorted(condition.rglob("*")) if p.is_file()]
    shutil.move(str(condition), str(archive / "condition"))
    shutil.copy2(root / "logs/pipeline_status.json", archive / "pipeline_status_before.json")
    shutil.copy2(root / "configs/stage_queue.json", archive / "stage_queue_before.json")
    receipt = {"schema": "interruption_reconciliation_v1", "time": time.time(),
        "reason": "PolicyContextError after completed candidate evaluation; explicit summary repaired",
        "observed_physical_costs": costs, "all_requests_resolved": True, "previous_processes_stopped": sorted(pids),
        "completed_jobs": 0, "used_for_training": False, "used_for_final_evaluation": False,
        "additional_incurred_costs_not_subtracted_from_main_budgets": True,
        "frozen_source_preserved": str(root / "frozen_sources/collection-e08bfd6784aea8dc9418"),
        "artifacts": artifacts, "prompt_regression": regression}
    write(archive / "reconciliation.json", receipt)
    for a in artifacts:
        assert hashlib.sha256(Path(a["path_after"]).read_bytes()).hexdigest() == a["sha256"]
    queue = read(root / "configs/stage_queue.json")
    assert queue["stages"][0]["id"] == "collect-qwen35_4b-made"
    queue["stages"][0]["id"] = "collect-qwen35_4b-made-v2"
    queue["amendment"] = "protocol_amendments/001_action_target_and_memory.md"
    write(root / "configs/stage_queue.json", queue)
    write(root / "logs/pipeline_status.json", {"stages": {}, "study_complete": False,
        "state": "reconciled_ready_to_restart", "reconciliations": [str(archive / "reconciliation.json")]})
    print(json.dumps({"reconciled": True, "archive": str(archive), "costs": costs}))


if __name__ == "__main__":
    main()
