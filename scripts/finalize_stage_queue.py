#!/usr/bin/env python3
"""Mark execution complete only after a verifiable full final-study audit."""
import argparse
import json
from pathlib import Path
import sqlite3

from matdiscovery.accounting import FULL_COUNTS, build_final_manifest, file_sha256, fingerprint, write_json_atomic
from matdiscovery.final_evaluation import verify_final_envelope
from matdiscovery.study_costs import stage_time_projection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    source = root / "source_manifest.json"
    actual = Path(json.loads(source.read_text())["original_project"]) if source.exists() else root
    acceptance = json.loads(args.acceptance.read_text())
    if acceptance.get("fingerprint") != fingerprint({k: v for k, v in acceptance.items() if k != "fingerprint"}):
        raise RuntimeError("Acceptance artifact was modified")
    if not acceptance.get("all_final_experiments_complete") or acceptance["expected_counts"] != FULL_COUNTS:
        raise RuntimeError("The complete final study has not passed accounting")
    if acceptance.get("all_attempt_costs_accounted") is not True or acceptance.get("all_es_training_conditions_accounted") is not True:
        raise RuntimeError("Unknown prior physical-attempt costs or incomplete ES-condition accounting require reconciliation")
    if acceptance.get("all_end_to_end_experimental_costs_accounted") is not True:
        raise RuntimeError("Shared collection, offline stage wall time, or interrupted-work costs remain incomplete")
    for artifact in acceptance["artifacts"]:
        if file_sha256(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("A final report changed after evidence verification")
    evidence = args.acceptance.parent / "verified_final_evidence.json"
    if file_sha256(evidence) != acceptance["verified_evidence_sha256"]:
        raise RuntimeError("Final evidence inventory changed")
    manifest = build_final_manifest(root)
    if acceptance["manifest_fingerprint"] != manifest["manifest_fingerprint"]:
        raise RuntimeError("Acceptance belongs to another final-study matrix")
    inventory = json.loads(evidence.read_text())
    by_id = {row["job_id"]: row for row in inventory}
    if len(inventory) != FULL_COUNTS["jobs"] or len(by_id) != FULL_COUNTS["jobs"] or set(by_id) != {job["job_id"] for job in manifest["jobs"]}:
        raise RuntimeError("Acceptance inventory must contain exactly all 2160 unique final results")
    final = actual / "experiments/final_evaluation"
    with sqlite3.connect(f"file:{final / 'ledger.sqlite3'}?mode=ro", uri=True) as database:
        database.row_factory = sqlite3.Row
        states = {row["job_id"]: dict(row) for row in database.execute("SELECT * FROM jobs")}
        events = [{**dict(row), "details": json.loads(row["details"])} for row in database.execute("SELECT * FROM events ORDER BY event_id")]
    if set(states) != set(by_id) or fingerprint(events) != acceptance["ledger_events_fingerprint"]:
        raise RuntimeError("Final ledger membership or attempt history changed after reporting")
    tasks = json.loads((root / "configs/benchmark_tasks.json").read_text())
    protocol = json.loads((root / "configs/main_protocol.json").read_text())
    from matdiscovery.execution_contract import declared_mace_workers
    for index, job in enumerate(manifest["jobs"]):
        row, state = by_id[job["job_id"]], states[job["job_id"]]
        if (state["state"] != "succeeded" or state["result_path"] != row["path"] or state["result_sha256"] != row["sha256"]
                or file_sha256(row["path"]) != row["sha256"]):
            raise RuntimeError("A final result or ledger state changed after report verification")
        verify_final_envelope(row["path"], job, manifest["manifest_fingerprint"], tasks=tasks,
            expected_mace_num_workers=declared_mace_workers(protocol))
        if (index + 1) % 25 == 0:
            print(json.dumps({"revalidated_final_jobs": index + 1, "expected_jobs": FULL_COUNTS["jobs"]}), flush=True)
    if file_sha256(evidence) != acceptance["verified_evidence_sha256"] or any(file_sha256(a["path"]) != a["sha256"] for a in acceptance["artifacts"]):
        raise RuntimeError("Final report artifacts changed during finalization")
    queue_path = actual / "configs/stage_queue.json"
    queue = json.loads(queue_path.read_text())
    status = json.loads((actual / "logs/pipeline_status.json").read_text())
    cost_report = json.loads((args.acceptance.parent / "study_costs.json").read_text())
    if cost_report["offline_and_other_stage_times"]["projection_fingerprint"] != fingerprint(stage_time_projection(queue, status)):
        raise RuntimeError("Experimental stage timing or command history changed after the cost audit")
    for artifact in cost_report["evidence_files"]:
        if file_sha256(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("Collection, interrupted-work, or diagnostic cost evidence changed after reporting")
    if queue["stages"][-1]["id"] != "finalize-complete-study" or status["active_stage"] != "finalize-complete-study":
        raise RuntimeError("Finalization is only the last declared stage")
    if not queue["sealed"] or any(status["stages"].get(s["id"], {}).get("status") != "succeeded" for s in queue["stages"][:-1]):
        raise RuntimeError("A prerequisite full experimental stage has not completed")
    queue.update(final_acceptance_verified=True, execution_acceptance_sha256=file_sha256(args.acceptance),
                 execution_acceptance_path=str(args.acceptance.resolve()), raw_results_revalidated_at_finalization=True,
                 empirical_improvement="see_endpoint_results_no_assumption")
    write_json_atomic(queue_path, queue)
    print(json.dumps({"all_experimental_stages_verified": True, "improvement_not_assumed": True}))


if __name__ == "__main__":
    main()
