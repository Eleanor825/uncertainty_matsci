"""Auditable end-to-end study cost accounting, separate from matched test budgets.

Shared collection/representation work is counted once. Stage wall time contains
episode wall time, so the two are never added. Technical diagnostics have their
own inventory and are excluded from scientific method endpoints.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from .accounting import file_sha256, fingerprint
from .collection_provenance import load_collection_lineage, preflight_collection, read_json as read_collection_json
from .experiment_plan import collection_jobs


PHYSICAL = ("candidate_oracle_attempts", "initialization_oracle_attempts", "surrogate_oracle_attempts", "dft_episode_attempts")
EXPERIMENT_PREFIXES = ("collect-", "transcoders-", "graphs-", "risk-", "es-", "final-")


def _read(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _valid(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _costs(values, context):
    _require(isinstance(values, dict) and all(isinstance(k, str) and _valid(v) for k, v in values.items()),
             "Invalid measured costs: " + context)
    return values


def _artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def actual_project(root):
    root = Path(root).resolve()
    source = root / "source_manifest.json"
    return Path(_read(source)["original_project"]).resolve() if source.is_file() else root


def audit_collection_costs(root, *, benchmark=None):
    """Audit the full collection plan; an explicit benchmark selects a phase view."""
    root = Path(root)
    jobs = collection_jobs(root)
    _require(len(jobs) == 720 and len({j["job_id"] for j in jobs}) == 720, "Cost audit requires all 720 unique collection jobs")
    _require(benchmark in {None, "made", "crystalgym"}, "Unknown benchmark cost scope")
    if benchmark is not None:
        jobs = [job for job in jobs if job["benchmark"] == benchmark]
        _require(len(jobs) == {"made": 420, "crystalgym": 300}[benchmark], "Benchmark collection matrix was reduced")
    expected_jobs = len(jobs)
    totals, by_condition, records, inputs = Counter(), defaultdict(Counter), [], []
    conditions = defaultdict(list)
    for job in jobs:
        conditions[(job["model_key"], job["benchmark"])].append(job)
    contexts = {}
    for key, expected in conditions.items():
        base = root / "experiments/collection" / key[0] / key[1]
        plan_path, progress_path = base / "plan.json", base / "progress.json"
        plan, progress = read_collection_json(plan_path), read_collection_json(progress_path)
        lineage = load_collection_lineage(root, base, plan, expected)
        manifests = [(base / job["job_id"] / "collection_manifest.json").resolve() for job in expected]
        _require(progress.get("complete") is True and progress.get("expected_jobs") == len(expected)
                 and progress.get("completed_jobs") == len(expected)
                 and isinstance(progress.get("collection_manifests"), list)
                 and len(progress["collection_manifests"]) == len(manifests)
                 and {Path(path).resolve() for path in progress["collection_manifests"]} == set(manifests),
                 "Collection cost audit requires every unique completed manifest in progress")
        verified_jobs = preflight_collection(base, plan, lineage)
        _require(set(verified_jobs) == {job["job_id"] for job in expected},
                 "Collection cost audit requires every unique completed job")
        contexts[key] = verified_jobs
        inputs.extend([_artifact(plan_path), _artifact(progress_path), *lineage.provenance_files])
    for job in jobs:
        directory = root / "experiments/collection" / job["model_key"] / job["benchmark"] / job["job_id"]
        receipt_path, episode_path = directory / "completion.json", directory / "episodes.json"
        verified = contexts[(job["model_key"], job["benchmark"])][job["job_id"]]
        receipt = verified["receipt"]
        _require(receipt.get("complete") is True and receipt["job_fingerprint"] == fingerprint(job), "Collection cost receipt does not match its complete job")
        seen = set()
        for artifact in receipt["artifacts"]:
            path = Path(artifact["path"]).resolve()
            _require(path not in seen and directory.resolve() in path.parents and file_sha256(path) == artifact["sha256"],
                     "Collection raw-cost evidence is duplicated, outside the job, or modified")
            seen.add(path)
        _require(episode_path.resolve() in seen, "Collection receipt must checksum its episode costs")
        episodes = verified["episodes"]
        _require(len(episodes) == job["expected_counts"]["episodes"] and {e["episode_id"] for e in episodes} == set(job["episode_ids"]),
                 "Collection cost audit found missing/duplicate episodes")
        costs = Counter()
        for episode in episodes:
            ordinal = job["episode_ids"].index(episode["episode_id"])
            _require(episode.get("complete") is True and all(episode.get(k) == job[k] for k in ("benchmark", "model_key", "method", "task_id", "seed"))
                     and episode["environment_seed"] == job["environment_seeds"][ordinal], "Collection costs belong to another incomplete episode or seed")
            costs.update(_costs(episode["costs"], job["job_id"]))
        for key in ("candidate_oracle_attempts", "dft_episode_attempts"):
            _require(costs.get(key, 0) == job["expected_counts"][key], "Collection physical denominator changed: " + key)
        costs["completed_episodes"] = len(episodes)
        totals.update(costs)
        by_condition[(job["model_key"], job["benchmark"])].update(costs)
        records.append({"job_id": job["job_id"], "model_key": job["model_key"], "benchmark": job["benchmark"], "split": job["split"],
                        "episodes": len(episodes), "costs": dict(costs), "receipt": str(receipt_path.resolve()),
                        "source_provenance": verified["source"]})
        inputs.extend([_artifact(receipt_path), _artifact(episode_path), *verified["evidence_files"]])
    expected_orb = 0 if benchmark == "crystalgym" else 21000
    expected_dft = 0 if benchmark == "made" else 1800
    _require(totals["candidate_oracle_attempts"] == expected_orb and totals["dft_episode_attempts"] == expected_dft,
             "Complete collection counters do not match the full declared scope")
    unique_inputs = {}
    for item in inputs:
        path = str(Path(item["path"]).resolve())
        _require(path not in unique_inputs or unique_inputs[path]["sha256"] == item["sha256"],
                 "Collection provenance evidence changed during cost audit")
        unique_inputs[path] = {"path": path, "sha256": item["sha256"]}
    return {"complete": True, "benchmark_scope": benchmark or "full_study", "expected_jobs": expected_jobs, "completed_jobs": len(records), "costs": dict(totals),
            "conditions": [{"model_key": k[0], "benchmark": k[1], "costs": dict(v)} for k, v in sorted(by_condition.items())],
            "jobs": records, "evidence_files": [unique_inputs[path] for path in sorted(unique_inputs)],
            "source_plan_job_counts": dict(Counter(row["source_provenance"]["plan_fingerprint"] for row in records)),
            "jobs_accepted_via_source_lineage": sum(row["source_provenance"]["accepted_via_lineage"] for row in records),
            "counted_once_across_all_method_arms": True,
            "per_method_allocation": None, "allocation_policy": "Shared fixed train/dev collections; never multiply by the six methods"}


def stage_time_projection(queue, status):
    stages = queue["stages"] + queue.get("declared_next_stages", []) + queue.get("deferred_stages", [])
    _require(len({stage["id"] for stage in stages}) == len(stages), "Duplicate stage across runnable, declared, or deferred plans")
    projection = []
    for stage in stages:
        if stage["id"].startswith(EXPERIMENT_PREFIXES):
            record = status["stages"].get(stage["id"], {})
            projection.append({"id": stage["id"], "stage": stage,
                "record": {key: record.get(key) for key in ("status", "returncode", "started_at", "finished_at", "command_fingerprint", "command")}})
    return projection


def _stage_benchmark(stage):
    command = stage["command"]
    value = None
    if "--benchmark" in command:
        index = command.index("--benchmark")
        _require(index + 1 < len(command), "Stage benchmark flag has no value")
        value = command[index + 1]
        _require(value in {"made", "crystalgym"}, "Unknown experimental stage benchmark")
    names = [name for name in ("made", "crystalgym") if name in stage["id"].split("-")]
    _require(len(names) == 1 and (value is None or value == names[0]), "Stage benchmark identity is missing or contradictory")
    return names[0]


def audit_stage_times(root, *, benchmark=None):
    actual = actual_project(root)
    queue, status = _read(actual / "configs/stage_queue.json"), _read(actual / "logs/pipeline_status.json")
    projection = stage_time_projection(queue, status)
    expected = {"collect": 4, "transcoders": 4, "graphs": 4, "risk": 4, "es": 80, "final": 4}
    counts = Counter(p["id"].split("-", 1)[0] for p in projection)
    _require(counts == expected and len({p["id"] for p in projection}) == 100, "Stage wall-time audit requires all 100 experimental stages")
    _require(benchmark in {None, "made", "crystalgym"}, "Unknown benchmark stage scope")
    if benchmark is not None:
        projection = [item for item in projection if _stage_benchmark(item["stage"]) == benchmark]
        phase_expected = {"collect": 2, "transcoders": 2, "graphs": 2, "risk": 2, "es": 20 if benchmark == "made" else 60, "final": 2}
        _require(Counter(p["id"].split("-", 1)[0] for p in projection) == phase_expected,
                 "Benchmark phase stage matrix is missing or reduced")
    expected_stages = len(projection)
    rows, unknown, totals = [], [], Counter()
    for item in projection:
        stage, record = item["stage"], item["record"]
        digest = hashlib.sha256(json.dumps(stage, sort_keys=True).encode()).hexdigest()
        start, finish = record["started_at"], record["finished_at"]
        valid = (record["status"] == "succeeded" and record["returncode"] == 0 and _valid(start) and _valid(finish) and finish >= start
                 and record["command_fingerprint"] == digest and record["command"] == stage["command"])
        row = {"stage_id": item["id"], "kind": item["id"].split("-", 1)[0], "started_at": start, "finished_at": finish,
               "wall_seconds": finish - start if valid else None, "verified_completed_stage": valid}
        rows.append(row)
        if valid:
            totals[row["kind"]] += row["wall_seconds"]
        else:
            unknown.append({"stage_id": row["stage_id"], "reason": "missing/incomplete/invalid stage timestamps or command identity"})
    representation = [r for r in rows if r["kind"] in {"transcoders", "graphs", "risk"}]
    return {"complete": not unknown, "benchmark_scope": benchmark or "full_study", "expected_stages": expected_stages, "known_stage_wall_seconds": sum(totals.values()),
            "by_kind_known_wall_seconds": dict(totals), "stages": rows, "unknown": unknown,
            "representation_expected_stages": 12 if benchmark is None else 6, "representation_stages": representation,
            "representation_wall_seconds": sum(r["wall_seconds"] for r in representation) if all(r["verified_completed_stage"] for r in representation) else None,
            "representation_known_wall_seconds": sum(r["wall_seconds"] for r in representation if r["verified_completed_stage"]),
            "projection_fingerprint": fingerprint(projection), "source_project": str(actual),
            "wall_time_semantics": "Sum of process-stage elapsed seconds, including model loading, ES updates, serialization and offline fitting; not GPU-hours or CPU-seconds",
            "episode_wall_is_nested_and_not_added_again": True}


def audit_interrupted_costs(root):
    actual = actual_project(root)
    base = actual / "experiments/interrupted"
    records, unknown, totals, files, intervals = [], [], Counter(), [], []
    status_path = actual / "logs/pipeline_status.json"
    expected_receipts = _read(status_path).get("reconciliations", []) if status_path.is_file() else []
    for name in expected_receipts:
        if not Path(name).is_file():
            unknown.append({"path": str(name), "reason": "A supervisor-listed interruption receipt is missing"})
    if not base.exists():
        return {"complete": not unknown, "records": [], "costs": {}, "known_archived_stage_wall_seconds": 0,
                "intervals": [], "unknown": unknown, "evidence_files": [], "inventory": []}
    directories = sorted(p for p in base.iterdir() if p.is_dir())
    for directory in directories:
        receipt_path = directory / "reconciliation.json"
        if not receipt_path.is_file():
            unknown.append({"path": str(directory), "reason": "No recognized interruption reconciliation receipt"})
            continue
        receipt = _read(receipt_path)
        if (receipt.get("schema") != "interruption_reconciliation_v1" or receipt.get("all_requests_resolved") is not True
                or not receipt.get("previous_processes_stopped") or receipt.get("used_for_training") is not False
                or receipt.get("used_for_final_evaluation") is not False
                or receipt.get("additional_incurred_costs_not_subtracted_from_main_budgets") is not True):
            unknown.append({"path": str(receipt_path), "reason": "Unknown or incomplete interruption receipt schema"})
            files.append(_artifact(receipt_path))
            continue
        costs = _costs(receipt["observed_physical_costs"], str(receipt_path))
        artifacts = receipt.get("artifacts", [])
        _require(bool(artifacts), "Reconciled interruption requires original raw evidence")
        seen = set()
        for artifact in artifacts:
            path = Path(artifact["path_after"]).resolve()
            _require(path not in seen and directory.resolve() in path.parents and file_sha256(path) == artifact["sha256"], "Archived interruption raw evidence changed")
            seen.add(path)
        totals.update(costs)
        files.append(_artifact(receipt_path))
        stage_path = directory / "pipeline_status_before.json"
        archived_times = []
        if stage_path.is_file():
            files.append(_artifact(stage_path))
            for name, stage in _read(stage_path).get("stages", {}).items():
                # Earlier successful stages may be present in the snapshot; only
                # stopped failed stages are additional incurred work.
                if stage.get("status") != "failed":
                    continue
                start, finish = stage.get("started_at"), stage.get("finished_at")
                if _valid(start) and _valid(finish) and finish >= start:
                    interval = {"stage_id": name, "started_at": start, "finished_at": finish, "wall_seconds": finish - start}
                    archived_times.append(interval)
                    if interval not in intervals:
                        intervals.append(interval)
                else:
                    unknown.append({"path": str(stage_path), "stage_id": name, "reason": "Archived failed-stage wall time is unknown"})
        if not archived_times:
            unknown.append({"path": str(directory), "reason": "No complete failed-stage time interval in the archive"})
        records.append({"archive": str(directory), "receipt_sha256": file_sha256(receipt_path), "costs": costs,
                        "raw_artifacts": [{"path": a["path_after"], "sha256": a["sha256"]} for a in artifacts], "stage_times": archived_times,
                        "not_subtracted_from_any_matched_budget": True})
    return {"complete": not unknown, "records": records, "costs": dict(totals), "intervals": intervals,
            "known_archived_stage_wall_seconds": sum(i["wall_seconds"] for i in intervals), "unknown": unknown,
            "evidence_files": files, "inventory": [str(p) for p in directories]}


def audit_technical_costs(root):
    """Recognize explicit diagnostic schemas; retain every unknown item as unknown."""
    base = actual_project(root) / "technical_not_main"
    records, unknown, totals, evidence, seen_hashes = [], [], Counter(), [], set()
    if not base.exists():
        return {"records": [], "known_scientific_call_counts": {}, "unknown": [], "evidence_files": [], "excluded_from_main_results": True}
    candidates = list(base.glob("*.json"))
    for directory in sorted(p for p in base.iterdir() if p.is_dir()):
        if (directory / "summary.json").is_file():
            candidates.append(directory / "summary.json")
        elif (directory / "diagnostic_manifest.json").is_file():
            unknown.append({"path": str(directory), "reason": "Diagnostic has a declaration but no completed summary; no budget is assumed consumed or free"})
            evidence.append(_artifact(directory / "diagnostic_manifest.json"))
        else:
            candidates.extend(directory.glob("*.json"))
    for path in sorted(candidates):
        artifact = _artifact(path)
        evidence.append(artifact)
        if artifact["sha256"] in seen_hashes:
            records.append({"path": str(path), "duplicate_content_not_counted_again": True})
            continue
        seen_hashes.add(artifact["sha256"])
        try:
            data = _read(path)
        except (ValueError, UnicodeError):
            unknown.append({"path": str(path), "reason": "Unrecognized diagnostic JSON"})
            continue
        explicit = next((data[k] for k in ("physical_oracle_calls", "scientific_oracle_calls", "scientific_calls") if k in data), None)
        plan_only = data.get("classification") == "technical_plan_only_no_physical_calls" and data.get("plan", {}).get("status") == "plan_only_no_physical_execution"
        if type(explicit) in (int, float) and explicit == 0 or plan_only:
            # Prompt-regression reports may quote old scientific counters; their
            # explicit zero-new-call disclosure takes precedence over that context.
            records.append({"path": str(path), "known_scientific_calls": 0, "counts": {}, "status": "explicit_no_new_scientific_calls"})
            continue
        if not (data.get("classification") == "technical_not_main" and data.get("scientific_main_result") is False):
            unknown.append({"path": str(path), "reason": "No recognized diagnostic cost schema or explicit no-call declaration"})
            continue
        counter = data.get("counters", data.get("final_counters", data.get("candidate_result", {}).get("counts")))
        if not isinstance(counter, dict):
            counter = data.get("initial_counters", {})
        known = {k: v for k, v in counter.items() if k in PHYSICAL and _valid(v)}
        malformed = [k for k in PHYSICAL if k in counter and not _valid(counter[k])]
        if malformed:
            unknown.append({"path": str(path), "reason": "Invalid physical counters", "keys": malformed})
        # Record only explicit cumulative counters, never configured maxima,
        # candidate step requests, terminal budgets, or copied old observations.
        if not known:
            unknown.append({"path": str(path), "reason": "No explicit final physical counter; configured budgets are not actual costs"})
        totals.update(known)
        missing = [k for k in PHYSICAL if k not in known]
        records.append({"path": str(path), "status": data.get("status"), "counts": known,
                        "unspecified_counter_keys_not_assumed_zero": missing,
                        "reported_wall_seconds": data.get("elapsed_seconds") if _valid(data.get("elapsed_seconds")) else None})
    return {"records": records, "known_scientific_call_counts": dict(totals), "unknown": unknown, "evidence_files": evidence,
            "excluded_from_main_results": True, "included_in_experimental_matched_budget": False,
            "totals_are_known_counts_not_an_assertion_unknown_diagnostics_were_free": True}


def build_study_cost_report(root, summaries, es_conditions, attempt_accounting, *, benchmark=None):
    _require(benchmark in {None, "made", "crystalgym"}, "Unknown end-to-end cost scope")
    # Preserve the original full-study path and its 720/100/80 gates by default.
    collection = audit_collection_costs(root) if benchmark is None else audit_collection_costs(root, benchmark=benchmark)
    stages = audit_stage_times(root) if benchmark is None else audit_stage_times(root, benchmark=benchmark)
    interrupted, technical = audit_interrupted_costs(root), audit_technical_costs(root)
    if benchmark is not None:
        _require(all(row["benchmark"] == benchmark for row in summaries), "Cross-benchmark final episodes in a phase report")
        _require(all(condition.get("benchmark") == benchmark for condition in es_conditions.values()), "Cross-benchmark ES conditions in a phase report")
    final, es, matched = Counter(), Counter(), defaultdict(Counter)
    for row in summaries:
        values = _costs(row["costs"], "final episode")
        final.update(values)
        property_name = row["task_id"].split(":")[0] if row["benchmark"] == "crystalgym" else "stability"
        matched[(row["model_key"], row["benchmark"], row["method"], property_name)].update(values)
    for condition in es_conditions.values():
        es.update(_costs(condition["costs"], "independent ES condition"))
    expected_conditions = 80 if benchmark is None else {"made": 20, "crystalgym": 60}[benchmark]
    _require(len(es_conditions) == expected_conditions, f"End-to-end costs require {expected_conditions} unique ES conditions")
    if benchmark == "made":
        _require(es["candidate_oracle_attempts"] == 176000 and es["dft_episode_attempts"] == 0,
                 "MADE phase requires all 176000 ES train/development candidate ORB attempts")
        _require(final["candidate_oracle_attempts"] == 90000 and final["dft_episode_attempts"] == 0,
                 "MADE phase requires all 90000 final candidate ORB attempts")
    categories = {"shared_collection_once": collection["costs"], "independent_es_training": dict(es),
                  "matched_final_evaluation": dict(final), "previous_final_failed_attempts": attempt_accounting["known_prior_attempt_costs"],
                  "archived_interrupted_work": interrupted["costs"]}
    known = Counter()
    for values in categories.values():
        known.update(values)
    # Summing stage durations is valid for the supervisor's serial pipeline.
    # Reconciled archives must not overlap any current successful stage interval.
    overlaps = []
    for old in interrupted["intervals"]:
        for current in stages["stages"]:
            if current["verified_completed_stage"] and max(old["started_at"], current["started_at"]) < min(old["finished_at"], current["finished_at"]):
                overlaps.append([old["stage_id"], current["stage_id"]])
    wall_complete = stages["complete"] and interrupted["complete"] and not overlaps
    complete = collection["complete"] and wall_complete and attempt_accounting["all_attempt_costs_accounted"]
    experimental_wall = stages["known_stage_wall_seconds"] + (interrupted["known_archived_stage_wall_seconds"] if not overlaps else 0)
    physical = {k: known[k] for k in PHYSICAL if k in known}
    plus_technical = Counter(physical)
    plus_technical.update(technical["known_scientific_call_counts"])
    return {"schema": "full_study_end_to_end_costs_v1" if benchmark is None else "benchmark_phase_end_to_end_costs_v1",
            "benchmark_scope": benchmark or "full_study", "global_study_complete": False if benchmark is not None else None,
            "archived_interruption_scope": "all_observed_project_interruptions_conservatively_retained_not_allocated",
            "experimental_cost_accounting_complete": complete,
            "matched_final_budget": {"costs": dict(final), "methods": [{**dict(zip(("model_key", "benchmark", "method", "property"), key)), "costs": dict(value)} for key, value in sorted(matched.items())],
                                     "training_and_diagnostic_costs_excluded_from_this_comparison": True},
            "experimental_end_to_end": {"cost_categories": categories, "known_measurement_subtotals": dict(known), "known_physical_call_counts": physical,
                "stage_wall_seconds": experimental_wall if wall_complete else None, "known_stage_wall_seconds": experimental_wall,
                "overlapping_archived_stage_intervals": overlaps, "shared_costs_counted_once": True,
                "per_method_shared_cost_allocation": None, "episode_wall_seconds_not_added_to_stage_wall_seconds": True,
                "missing_hardware_costs": {"gpu_active_seconds": None, "energy_joules": None, "monetary_cost": None}},
            "shared_collection": collection, "offline_and_other_stage_times": stages, "archived_interrupted_work": interrupted,
            "technical_diagnostics_separate": technical, "known_physical_calls_including_technical": dict(plus_technical),
            "technical_unknowns_do_not_become_zero_or_main_scientific_results": True,
            "evidence_files": collection["evidence_files"] + interrupted["evidence_files"] + technical["evidence_files"]}
