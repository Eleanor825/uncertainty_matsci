#!/usr/bin/env python3
"""Pause this study's BF16 collection only at a fully observed physical boundary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("w") as handle:
        handle.write(json.dumps(value, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def boundary(path):
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        return None
    try:
        rows = [json.loads(line) for line in data.splitlines()]
    except json.JSONDecodeError:
        return None
    requests = {r["payload"]["id"]: r for r in rows if r.get("direction") == "request"}
    responses = {r["payload"]["id"]: r for r in rows if r.get("direction") == "response"}
    if not requests or set(requests) != set(responses) or rows[-1].get("direction") != "response":
        return None
    response = rows[-1]["payload"]
    request = requests[response["id"]]["payload"]
    if request["op"] not in {"observe", "step", "init"} or not response.get("ok"):
        return None
    result = response["result"]
    observation = result.get("observation", result)
    if "counts" not in observation:
        return None
    return {"counts": observation["counts"], "last_response_id": response["id"],
            "last_operation": request["op"], "rpc_sha256": hashlib.sha256(data).hexdigest(),
            "requests": len(requests), "responses": len(responses)}


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split()[2] != "Z"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    status_path = root / "logs/pipeline_status.json"
    status = read(status_path)
    stage_id = "collect-qwen35_4b-made-v2"
    assert status["state"] == "running" and status["active_stage"] == stage_id
    record = status["stages"][stage_id]
    pid = record["pid"]
    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode()
    assert "run_collection.py" in command and str(root) in command and "qwen35_4b" in command
    condition = root / "experiments/collection/qwen35_4b/made"
    assert not list(condition.rglob("completion.json")), "Completed BF16 jobs need a separate multi-job archive accounting"
    jobs = list(condition.glob("collection-*"))
    assert len(jobs) == 1
    rpc = jobs[0] / "rpc/rpc.jsonl"
    archive = root / "experiments/interrupted/002_bf16_precision_revision"
    assert not archive.exists()
    end = time.monotonic() + 1800
    paused = False
    while time.monotonic() < end:
        if not alive(pid):
            raise RuntimeError("Collection exited before controlled boundary pause; inspect evidence")
        if boundary(rpc):
            os.kill(pid, signal.SIGSTOP)
            paused = True
            time.sleep(.1)
            proof = boundary(rpc)
            if proof:
                break
            os.kill(pid, signal.SIGCONT)
            paused = False
        time.sleep(.5)
    else:
        raise TimeoutError("No safe complete-observation boundary; collection left running")
    try:
        archive.mkdir(parents=True)
        receipt = {"schema": "interruption_reconciliation_v1", "time": time.time(),
            "reason": "Numerically validated FP32 protocol revision: real BF16 action gradients failed finite differences; not a material/evaluator failure",
            "controlled_boundary": proof, "observed_physical_costs": proof["counts"],
            "all_requests_resolved": True, "completed_jobs": 0, "used_for_training": False,
            "used_for_final_evaluation": False, "additional_incurred_costs_not_subtracted_from_main_budgets": True,
            "frozen_source_preserved": str(root / "frozen_sources/collection-f9d7d61103c1631346e0"),
            "policy_pid": pid, "state": "paused_at_verified_boundary"}
        write(archive / "reconciliation.json", receipt)
        # SIGTERM is pending while stopped; SIGCONT delivers it before the next
        # policy action. No environment/DFT process is interrupted or killed.
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        paused = False
        for _ in range(120):
            if not alive(pid) and read(status_path).get("state") == "halted_requires_reconciliation":
                break
            time.sleep(.5)
        else:
            raise RuntimeError("Controlled stop did not reach a known supervisor outcome")
        final_proof = boundary(rpc)
        assert final_proof == proof, "Physical evidence changed after the verified boundary"
        receipt["artifacts"] = [{"path_before": str(p), "path_after": str(archive / "condition" / p.relative_to(condition)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(condition.rglob("*")) if p.is_file()]
        shutil.copy2(status_path, archive / "pipeline_status_before.json")
        shutil.copy2(root / "configs/stage_queue.json", archive / "stage_queue_before.json")
        shutil.move(str(condition), str(archive / "condition"))
        receipt.update(state="archived_for_precision_revision", previous_processes_stopped=[pid],
                       raw_evidence_unchanged_after_stop=True)
        write(archive / "reconciliation.json", receipt)
        print(json.dumps({"state": receipt["state"], "archive": str(archive), "costs": proof["counts"]}), flush=True)
    finally:
        if paused and alive(pid):
            os.kill(pid, signal.SIGCONT)


if __name__ == "__main__":
    main()
