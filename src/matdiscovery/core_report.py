"""Descriptive two-system paired core report, never full-study acceptance."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

from .accounting import file_sha256, fingerprint, write_json_atomic, validate_made_budget_scope
from .core_final import (
    METHODS, SCOPE, CoreFinalLedger, build_core_final_manifest,
    publish_once, read, require, verify_core_envelope,
)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def reliability(envelope, directory):
    """All proposal/failed/missing denominators retained; no independent-step CI."""
    from .metrics import calibration_metrics
    path = Path(directory) / "decisions.jsonl"
    matches = [item for item in envelope["artifacts"] if Path(item["path"]).resolve() == path.resolve()]
    require(len(matches) == 1 and file_sha256(path) == matches[0]["sha256"], "Core confidence data changed or is absent")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    require(len({row["decision_id"] for row in records}) == len(records), "Duplicate core confidence decisions")
    counts = Counter(total_proposed=len(records))
    pairs = []
    graph_required = envelope["execution_job"]["method"] == "esopt_graph_risk"
    for row in records:
        if graph_required:
            counts["graph_expected"] += 1
            state = row.get("graph_status")
            counts["graph_" + state if state in {"succeeded", "unavailable"} else "graph_missing_status"] += 1
        if row.get("disposition") != "executed":
            counts["not_executed_no_counterfactual_label"] += 1
            continue
        counts["executed"] += 1
        label, probability = row.get("label_future_failure"), row.get("predicted_failure_probability")
        if type(label) is not int or label not in (0, 1):
            counts["executed_without_observed_label"] += 1
            continue
        if not number(probability) or not 0 <= probability <= 1:
            counts["executed_without_valid_confidence"] += 1
            continue
        pairs.append((label, probability))
        counts["scored_executed"] += 1
    metrics = calibration_metrics(*zip(*pairs)) if pairs else None
    complete = bool(pairs) and len(pairs) == counts["executed"]
    graph_denominator = counts["graph_expected"]
    return {"counts": dict(counts), "executed_denominator": counts["executed"],
        "confidence_coverage": len(pairs) / counts["executed"] if counts["executed"] else None,
        "confidence_complete": complete, "metrics": metrics if complete else None,
        "observed_confidence_metrics": metrics, "missing_confidence_not_dropped_from_pair_metrics": True,
        "graph": {"applicable": graph_required, "expected_proposals": graph_denominator,
            "succeeded": counts["graph_succeeded"], "unavailable": counts["graph_unavailable"],
            "missing_status": counts["graph_missing_status"],
            "coverage": counts["graph_succeeded"] / graph_denominator if graph_denominator else None},
        "statistical_unit": "descriptive_decisions_within_one_episode; not_independent_inferential_samples"}


def pair_report(manifest, records):
    """Build only exact system pairs; missing outcomes remain explicit unknowns."""
    expected = {job["job_id"]: job for job in manifest["jobs"]}
    require(set(records) <= set(expected), "Report includes undeclared core jobs")
    systems, deltas = [], {key: [] for key in ("AUDC", "mSUN", "brier", "overconfident_error_rate_p_le_0_1")}
    for task in sorted({job["task_id"] for job in expected.values()}):
        by_method = {}
        for method in METHODS:
            job = next(j for j in expected.values() if j["task_id"] == task and j["method"] == method)
            record = records.get(job["job_id"])
            if record is None:
                by_method[method] = {"complete": False, "job_id": job["job_id"], "metrics": None, "reliability": None}
                continue
            episode = record["episode"]
            by_method[method] = {"complete": True, "job_id": job["job_id"], "status": episode["status"],
                "metrics": deepcopy(episode["metrics"]), "reliability": record["reliability"],
                "costs": deepcopy(episode["costs"]), "result": record["result"]}
        pair = {}
        for metric in deltas:
            values = []
            for method in METHODS:
                item = by_method[method]
                source = item["metrics"] if metric in {"AUDC", "mSUN"} else (item["reliability"] or {}).get("metrics")
                value = source.get(metric) if isinstance(source, dict) else None
                values.append(value)
            delta = values[1] - values[0] if all(number(value) for value in values) else None
            pair[metric] = {"full_minus_baseline": delta, "higher_is_better": metric in {"AUDC", "mSUN"},
                            "defined": delta is not None}
            if delta is not None:
                deltas[metric].append(delta)
        systems.append({"task_id": task, "seed": 1, "methods": by_method, "paired_deltas": pair})
    aggregate = {}
    for metric, values in deltas.items():
        complete = len(values) == 2
        oriented = values if metric in {"AUDC", "mSUN"} else [-value for value in values]
        interpretation = "no_conclusion_missing_paired_data"
        if complete:
            interpretation = ("positive_on_both_observed_systems" if all(value > 0 for value in oriented) else
                "negative_on_both_observed_systems" if all(value < 0 for value in oriented) else
                "no_observed_difference" if all(value == 0 for value in oriented) else "mixed_or_tied_system_outcomes")
        aggregate[metric] = {"mean_full_minus_baseline": sum(values) / 2 if complete else None,
            "paired_systems_observed": len(values), "paired_systems_expected": 2,
            "interpretation": interpretation, "inferential_p_value": None, "population_significance_claim": False}
    return {"systems": systems, "aggregate": aggregate, "independent_system_pairs": 2,
            "training_seeds": [1], "sample_scope": "two_predeclared_heldout_systems_one_seed",
            "uncertainty_effect_independently_identified": False,
            "method_contrast": "Baseline versus the combined ES and graph-risk controller; this pair does not isolate uncertainty or ES contributions.",
            "statistical_limit": "Descriptive paired outcomes only; N=2 does not establish general superiority or population significance."}


def _sum_costs(rows):
    total = Counter()
    for costs in rows:
        require(isinstance(costs, dict) and all(number(value) and value >= 0 for value in costs.values()), "Invalid actual core cost")
        total.update(costs)
    return dict(total)


def core_cost_report(corpus, es, records, *, expected_imported_jobs, expected_reused_development=False,
                     expected_made_budget=50, core_protocol=None):
    """Each physical episode contributes once, preserving historical reuse."""
    require(corpus.get("complete") is True and es.get("complete") is True, "Core corpus/ES cost evidence remains incomplete")
    verified = validate_made_budget_scope(expected_made_budget, core_protocol)
    if verified is not None:
        require(expected_imported_jobs == 3 and expected_reused_development,
                "FAST accounting must preserve all four historical B50 trajectories")
    require(type(expected_reused_development) is bool, "Core development origin must be explicit")
    reused_dev = corpus.get("reused_development")
    if expected_reused_development:
        require(isinstance(reused_dev, dict) and reused_dev.get("jobs") == reused_dev.get("episodes") == 1
                and reused_dev.get("execution_origin") == "historical_core_v1_completed_development"
                and len(reused_dev.get("receipts", [])) == 1,
                "Registered V2 development must retain its one historical physical receipt")
        new_dev = corpus.get("new_development")
        require(isinstance(new_dev, dict) and new_dev.get("jobs") == new_dev.get("episodes") == 0
                and new_dev.get("receipts") == [] and isinstance(new_dev.get("costs"), dict)
                and all(number(value) and value == 0 for value in new_dev["costs"].values()),
                "Reused development cannot also be charged as a new physical episode")
        require(isinstance(corpus.get("incremental_physical_costs"), dict)
                and all(number(value) and value == 0 for value in corpus["incremental_physical_costs"].values()),
                "A fully reused V2 corpus must have zero incremental physical costs")
    else:
        require(reused_dev is None, "Unregistered historical development cannot change V1 cost semantics")
    categories = {
        "imported_training_collection": corpus["imported_train"]["costs"],
        "new_development_collection": corpus["new_development"]["costs"],
        "es_training": es["training"]["costs"],
        "es_development": es["development"]["costs"],
        "matched_final_evaluation": _sum_costs([record["episode"]["costs"] for record in records.values()]),
    }
    historical_categories = ["imported_training_collection"]
    if expected_reused_development:
        categories["reused_development_collection"] = reused_dev["costs"]
        historical_categories.append("reused_development_collection")
        # The upstream corpus verifier validates each raw trajectory. Its
        # returned receipt must also be in the report's immutable evidence set.
        evidence = {(str(Path(item["path"]).resolve()), item["sha256"]) for item in corpus["evidence_files"]}
        require(all((str(Path(item["path"]).resolve()), item["sha256"]) in evidence for item in reused_dev["receipts"]),
                "Historical development receipt is absent from the corpus evidence")
        receipts = [item for group in (corpus["imported_train"], reused_dev) for item in group.get("receipts", [])]
        require(len({str(Path(item["path"]).resolve()) for item in receipts}) == len(receipts)
                and len({item["sha256"] for item in receipts}) == len(receipts),
                "A physical collection receipt was counted in multiple categories")
    require(len(records) == 4 and categories["matched_final_evaluation"].get("candidate_oracle_attempts") == 4 * expected_made_budget
            and all(record["episode"]["costs"].get("candidate_oracle_attempts") == expected_made_budget for record in records.values()),
            "Final costs need exactly four complete registered-budget jobs")
    require(expected_imported_jobs in {2, 3}, "Unexpected registered core import matrix")
    development_category = "reused_development_collection" if expected_reused_development else "new_development_collection"
    for category, expected in (("imported_training_collection", expected_imported_jobs * 50), (development_category, 50),
                               ("es_training", 4 * expected_made_budget), ("es_development", 2 * expected_made_budget)):
        require(categories[category].get("candidate_oracle_attempts") == expected
                and categories[category].get("dft_episode_attempts", 0) == 0,
                "Core source-cost category differs from the registered complete matrix: " + category)
    if "total" in corpus:
        actual = _sum_costs([corpus["imported_train"]["costs"], corpus["new_development"]["costs"]]
                            + ([reused_dev["costs"]] if expected_reused_development else []))
        require(Counter(actual) == Counter(_sum_costs([corpus["total"]["costs"]])),
                "Corpus total duplicates or omits a physical cost category")
    files = list(corpus["evidence_files"]) + list(es["evidence_files"])
    known = {}
    for item in files:
        path = str(Path(item["path"]).resolve())
        require(path not in known or known[path] == item["sha256"], "Contradictory source-cost artifact hashes")
        require(file_sha256(path) == item["sha256"], "Core cost source changed")
        known[path] = item["sha256"]
    return {"complete": True, "categories": deepcopy(categories),
        "new_episode_candidate_budget": expected_made_budget, "historical_collection_candidate_budget": 50,
        "incremental_core_costs": _sum_costs([value for key, value in categories.items() if key not in historical_categories]),
        "historical_reused_costs": _sum_costs([categories[key] for key in historical_categories]),
        "reused_plus_incremental_costs": _sum_costs(categories.values()),
        "historical_categories": historical_categories,
        "incremental_categories": [key for key in categories if key not in historical_categories],
        "development_collection_origin": "reused_completed_v1" if expected_reused_development else "new_core_execution",
        "imported_costs_are_historical_not_new_calls": True,
        "es_summary_total_not_added_again": True, "original_full_study_remaining_costs_included": False,
        "offline_representation_physical_oracle_calls": 0,
        "wall_time_note": "Episode wall times are not GPU-hours and exclude offline fit/loading overhead; do not add aggregate stage elapsed times again.",
        "evidence_files": [{"path": path, "sha256": digest} for path, digest in sorted(known.items())]}


def offline_transcoder_report(core, *, _historical_current_bank=None):
    """Retain both actual TC runs without adding wall time to oracle counts.

    V2 banks preserve a per-layer link to each real all-epoch precomputation.
    The common directory's result.json is accepted only when its exact receipt
    inventory agrees with all 32 bank links. No mtimes or GPU-hour estimates.
    """
    from .core_protocol import CAUSAL_GRAPH_REGISTRATIONS
    if core.get("registration") in CAUSAL_GRAPH_REGISTRATIONS:
        require(_historical_current_bank is None, "A reused bank is not a new raw fit")
        return offline_causal_graph_report(core)
    if core.get("normalization_amendment") is not None:
        require(_historical_current_bank is None, "Normalized main execution cannot be treated as raw history")
        return offline_normalized_transcoder_report(core)
    amendment = core.get("tc64_amendment")
    if amendment is None:
        return None  # Original V1 cost acceptance does not require a V2 record.
    require(isinstance(amendment, dict) and core["transcoder"]["epochs"] == 64,
            "TC cost amendment differs from the registered 64-epoch core")
    evidence = {}

    def source(item):
        path = str(Path(item["path"]).resolve())
        require(file_sha256(path) == item["sha256"], "Offline TC cost evidence changed")
        require(path not in evidence or evidence[path] == item["sha256"], "Conflicting offline TC cost evidence")
        evidence[path] = item["sha256"]
        return read(path)

    # Reconciliation and the predecessor registration are historical evidence,
    # not new training or scientific calls in the V2 workspace.
    source(amendment["predecessor_protocol"])
    source(amendment["failure_reconciliation"])
    prior = amendment.get("prior_tc16_execution")
    require(isinstance(prior, list) and prior, "V1 failed TC training execution evidence is missing")
    prior_records = [(item, source(item)) for item in prior]
    prior_results = [(item, record) for item, record in prior_records if Path(item["path"]).name == "result.json"]
    require(len(prior_results) == 1, "V1 TC execution must identify one real completed training result")
    prior_result_item, prior_result = prior_results[0]
    for item, record in prior_records:
        if Path(item["path"]).name == "run.json":
            require(record == prior_result, "V1 run/result records disagree; do not silently count a retry")

    def run(bank_item, expected_epochs, expected_core, *, passed, result_item=None):
        bank = source(bank_item)
        require(bank.get("bank_fingerprint") == fingerprint({key: value for key, value in bank.items() if key != "bank_fingerprint"})
                and bank.get("complete") is True and bank.get("ready_for_graphs") is passed
                and bank.get("configuration", {}).get("epochs") == expected_epochs
                and bank.get("configuration", {}).get("max_dev_fvu") == .5
                and len(bank.get("layers", [])) == 32,
                "Offline TC cost bank is incomplete or changed its epoch/fidelity contract")
        precomputations = []
        for index, layer in enumerate(bank["layers"]):
            metadata = layer.get("metadata", {})
            execution = metadata.get("execution_provenance", {})
            require(layer.get("layer_index") == index and metadata.get("epochs") == expected_epochs
                    and execution.get("epochs_scored") == expected_epochs
                    and execution.get("test_data_used") is False,
                    "Offline TC bank omitted an actual layer/epoch or read test data")
            item = execution.get("precomputation")
            require(isinstance(item, dict), "Offline TC layer lacks its real training receipt")
            precomputations.append(item)
        parents = {Path(item["path"]).resolve().parent.parent for item in precomputations}
        require(len(parents) == 1 and len({item["path"] for item in precomputations}) == 32,
                "Offline TC bank mixes or duplicates precomputation runs")
        inferred = next(iter(parents)).parent / "result.json"
        if result_item is None:
            require(inferred == Path(amendment["tc64_precompute_result_path"]).resolve(),
                    "Current TC cost result differs from its preregistered execution path")
            result_item = {"path": str(inferred), "sha256": file_sha256(inferred)}
        require(Path(result_item["path"]).resolve() == inferred, "TC execution result belongs to another training directory")
        record = source(result_item)
        require(record.get("complete") is True and record.get("core_fingerprint") == expected_core
                and record.get("layers") == 32 and record.get("epochs_per_layer") == expected_epochs
                and record.get("training_data_only") is True and record.get("new_policy_or_material_oracle_calls") == 0
                and record.get("layer_receipts") == precomputations,
                "TC runtime record and actual bank training inventory disagree")
        epoch_rows = 0
        for index, item in enumerate(precomputations):
            receipt = source(item)
            contract = receipt.get("contract", {})
            epochs = receipt.get("epochs", [])
            require(receipt.get("complete") is True and receipt.get("development_read") is False
                    and receipt.get("test_read") is False
                    and receipt.get("fingerprint") == fingerprint({key: value for key, value in receipt.items() if key != "fingerprint"})
                    and contract.get("epochs") == expected_epochs
                    and contract.get("registration", {}).get("core_fingerprint") == expected_core
                    and contract.get("registration", {}).get("layer_index") == index
                    and [row.get("epoch") for row in epochs] == list(range(expected_epochs))
                    and all(type(row.get("train_rows")) is int and row["train_rows"] > 0 for row in epochs),
                    "TC receipt does not preserve all real train-only epochs")
            epoch_rows += sum(row["train_rows"] for row in epochs)

        def elapsed(start_key):
            start, finish = record.get(start_key), record.get("finished_at")
            if start is None or finish is None:
                return None
            require(number(start) and number(finish) and finish >= start, "Invalid observed offline TC timing")
            return finish - start

        return {"core_fingerprint": expected_core, "epochs_per_layer": expected_epochs, "layers": 32,
            "observed_layer_epochs": 32 * expected_epochs, "observed_training_row_presentations": epoch_rows,
            "training_completed": True, "development_fidelity_outcome": "passed" if passed else "failed",
            "started_at": record.get("started_at"), "training_started_at": record.get("training_started_at"),
            "finished_at": record.get("finished_at"), "precompute_total_wall_seconds": elapsed("started_at"),
            "precompute_training_phase_wall_seconds": elapsed("training_started_at"),
            "development_selection_wall_seconds": None,
            "development_selection_timing_status": "not_independently_recorded; not_inferred_from_mtime",
            "gpu_hours": None, "gpu_hours_status": "not_measured; wall_time_is_not_gpu_time",
            "scientific_oracle_calls": 0, "bank": bank_item, "training_result": result_item}

    previous = run(amendment["failed_bank"], 16, amendment["predecessor_fingerprint"],
                   passed=False, result_item=prior_result_item)
    current_path = Path(core["workspace"]) / "experiments/transcoders/qwen35_4b/made/transcoder_manifest.json"
    current_item = _historical_current_bank or {"path": str(current_path), "sha256": file_sha256(current_path)}
    current = run(current_item, 64, core["fingerprint"], passed=_historical_current_bank is None)
    require(previous["training_result"]["sha256"] != current["training_result"]["sha256"],
            "Prior failed training cannot be counted as a newly executed 64-epoch run")
    return {"schema": "core_transcoder_attempt_costs_v1", "complete": True,
        "historical_failed_tc16": previous, "current_tc64": current,
        "observed_layer_epochs_across_attempts": 32 * (16 + 64),
        "scientific_oracle_calls": 0, "gpu_hours": None,
        "not_added_to_episode_wall_or_physical_cost_totals": True,
        "timing_note": "Training-phase wall is contained in precompute-total wall; do not add the two. Parallel worker wall is not GPU-hours. Missing phase timing remains unknown.",
        "evidence_files": [{"path": path, "sha256": digest} for path, digest in sorted(evidence.items())]}


def offline_normalized_transcoder_report(core):
    """Source-bound V3 history, diagnostics and full fit; never a GPU-hour sum."""
    amendment = core["normalization_amendment"]
    expected = {
        "raw_tc16": (list(range(32)), 16), "raw_tc64": (list(range(32)), 64),
        "raw128_four_layer_diagnostic": ([1, 12, 13, 16], 128),
        "normalized64_guard_failed": ([1, 12, 13, 16], 64),
        "fold_boundary_reproduction": ([1], 5),
        "normalized64_four_layer_success": ([1, 12, 13, 16], 64),
    }
    history = amendment.get("offline_history")
    require(isinstance(history, list) and len(history) == len(expected)
            and {item.get("run_id") for item in history} == set(expected),
            "Normalized cost history omitted or duplicated a registered attempt")
    evidence, documents, primary_owners = {}, {}, {}

    def source(item, *, parse=True):
        path = str(Path(item["path"]).resolve())
        require(file_sha256(path) == item["sha256"], "Normalized cost source changed")
        require(path not in evidence or evidence[path] == item["sha256"], "Conflicting normalized cost source hashes")
        evidence[path] = item["sha256"]
        if not parse or Path(path).suffix not in {".json", ".jsonl"}:
            return None
        if path not in documents:
            if Path(path).suffix == ".jsonl":
                data = Path(path).read_bytes()
                require(not data or data.endswith(b"\n"), "Closed diagnostic history has an incomplete row")
                documents[path] = [json.loads(line) for line in data.splitlines() if line]
            else:
                documents[path] = read(path)
        return documents[path]

    def own(run_id, item):
        # References to old attempts inside later provenance do not create
        # another execution. Only each attempt's own terminal records count.
        identity = item["sha256"]
        require(identity not in primary_owners or primary_owners[identity] == run_id,
                "One offline execution is charged to multiple attempt categories")
        primary_owners[identity] = run_id

    groups = {}
    for item in history:
        run_id = item["run_id"]
        require(item.get("role") == run_id and (item.get("expected_layers"), item.get("epochs_if_known")) == expected[run_id]
                and isinstance(item.get("artifacts"), list) and item["artifacts"],
                "Offline history changed the registered layer/epoch scope")
        require(len({str(Path(value["path"]).resolve()) for value in item["artifacts"]}) == len(item["artifacts"]),
                "Duplicated artifact inside an offline attempt")
        groups[run_id] = [(value, source(value)) for value in item["artifacts"]]

    def one(run_id, predicate):
        candidates = [(item, value) for item, value in groups[run_id] if isinstance(value, dict) and predicate(item, value)]
        require(len(candidates) == 1, "Missing or ambiguous primary record for " + run_id)
        own(run_id, candidates[0][0])
        return candidates[0]

    def wall(value):
        measured = value.get("elapsed_seconds")
        if measured is not None:
            require(number(measured) and measured >= 0, "Invalid measured offline wall time")
            return measured
        start, finish = value.get("started_at"), value.get("finished_at")
        if start is None or finish is None:
            return None
        require(number(start) and number(finish) and finish >= start, "Invalid offline timing endpoints")
        return finish - start

    def zero_calls(value, *, allow_counter_record=False):
        keys = ("new_policy_or_oracle_calls", "new_policy_or_material_oracle_calls", "policy_or_oracle_calls")
        observed = [value[key] for key in keys if key in value]
        if allow_counter_record and isinstance(value.get("counters"), dict):
            counters = value["counters"]
            required = ("candidate_oracle_attempts", "initialization_oracle_attempts", "surrogate_oracle_attempts", "dft_episode_attempts")
            require(all(type(counters.get(key)) is int and counters[key] == 0 for key in required)
                    and value.get("policy_loaded") is False, "Diagnostic counters are nonzero or unknown")
            observed.append(0)
        require(observed and all(type(value) is int and value == 0 for value in observed),
                "Offline attempt lacks an explicit zero-policy/oracle declaration")

    def epochs(rows, count):
        require(isinstance(rows, list) and [row.get("epoch") for row in rows] == list(range(count)),
                "Offline attempt omitted or duplicated completed epoch records")
        return count

    def base(run_id, outcome):
        return {"run_id": run_id, "category": "historical_failed_fit" if run_id.startswith("raw_tc") else "historical_diagnostic",
            "outcome": outcome, "planned_layers": expected[run_id][0], "planned_epochs_per_layer": expected[run_id][1],
            "wall_seconds": None, "gpu_hours": None, "scientific_oracle_calls": 0,
            "used_as_current_full_bank": False, "artifacts": [item for item, _ in groups[run_id]]}

    previous_core = source(amendment["predecessor_protocol"])
    require(previous_core["fingerprint"] == amendment["predecessor_fingerprint"]
            and previous_core["tc64_amendment"] == core["tc64_amendment"], "Normalized history has another raw-core predecessor")
    raw64_bank, _ = one("raw_tc64", lambda item, value: value.get("configuration", {}).get("epochs") == 64 and "bank_fingerprint" in value)
    raw = offline_transcoder_report(previous_core, _historical_current_bank=raw64_bank)
    for item in raw["evidence_files"]:
        source(item, parse=False)
    runs = []
    for run_id, raw_run in (("raw_tc16", raw["historical_failed_tc16"]), ("raw_tc64", raw["current_tc64"])):
        listed = {(str(Path(item["path"]).resolve()), item["sha256"]) for item, _ in groups[run_id]}
        for key in ("bank", "training_result"):
            item = raw_run[key]
            require((str(Path(item["path"]).resolve()), item["sha256"]) in listed, "History omitted the actual raw fit source")
            own(run_id, item)
        result = {**base(run_id, "failed_development_fidelity"), **raw_run}
        result["wall_seconds"] = raw_run["precompute_total_wall_seconds"]
        selections = [value for item, value in groups[run_id] if Path(item["path"]).name == "selection.json"]
        require(len(selections) <= 1, "Duplicate raw selection phase")
        if selections:
            result["development_selection_wall_seconds"] = wall(selections[0])
            result["development_selection_timing_status"] = "measured_independent_phase" if wall(selections[0]) is not None else "unknown"
        runs.append(result)

    run_id = "raw128_four_layer_diagnostic"
    summary_item, summary = one(run_id, lambda item, value: Path(item["path"]).name == "summary.json")
    zero_calls(summary, allow_counter_record=True)
    require(summary.get("complete") is True and summary.get("full_32_layer_bank") is False
            and summary.get("test_data_used") is False and [row.get("index") for row in summary.get("layers", [])] == expected[run_id][0],
            "Raw128 evidence is not the registered four-layer diagnostic")
    results = []
    listed = {str(Path(item["path"]).resolve()) for item, _ in groups[run_id]}
    for case in summary["layers"]:
        item = case["result"]; require(str(Path(item["path"]).resolve()) in listed, "Raw128 layer result omitted from history")
        value = source(item); own(run_id, item)
        require(value.get("complete") is True and value.get("index") == case["index"] and value["metadata"]["epochs"] == 128,
                "Raw128 layer execution is incomplete")
        epochs(value["metadata"]["history"], 128)
        for key in ("checkpoint", "sidecar"):
            source(value[key], parse=False)
        results.append({"layer": case["index"], "completed_epochs": 128, "worker_wall_seconds": wall(value),
                        "fidelity_gate_passed": value["fidelity_gate_passed"]})
    runs.append({**base(run_id, "diagnostic_only_not_adopted"), "wall_seconds": wall(summary),
                 "observed_layer_epochs": 512, "epoch_count_exact": True, "cases": results})

    run_id = "normalized64_guard_failed"
    failures = [(item, value) for item, value in groups[run_id] if Path(item["path"]).name == "failure.json"]
    logs = {Path(item["path"]).parent.name: value for item, value in groups[run_id] if Path(item["path"]).name == "history.jsonl"}
    require(len(failures) == 4 and len(logs) == 4, "Guard-failure history omitted a worker")
    cases = []
    for item, value in failures:
        own(run_id, item); zero_calls(value)
        index = value["task"]["layer_index"]
        require(value.get("complete") is False and index in expected[run_id][0] and value.get("error"), "Guard failure is not preserved")
        rows = logs[Path(item["path"]).parent.name]; epochs(rows, len(rows))
        require(len(rows) < 64, "Failed guard cannot masquerade as a full 64-epoch fit")
        cases.append({"layer": index, "completed_logged_epochs_lower_bound": len(rows),
            "unlogged_or_partial_training_work": "unknown", "worker_wall_seconds": wall(value), "error": value["error"]})
    require(sorted(row["layer"] for row in cases) == expected[run_id][0], "Guard failure duplicated a layer")
    runs.append({**base(run_id, "failed_auxiliary_numerical_guard"), "cases": sorted(cases, key=lambda row: row["layer"]),
        "observed_layer_epochs_lower_bound": sum(row["completed_logged_epochs_lower_bound"] for row in cases), "epoch_count_exact": False,
        "timing_status": "unknown_when_absent; no_mtime_or_planned_epoch_cost_imputation"})

    run_id = "fold_boundary_reproduction"
    _, value = one(run_id, lambda item, value: Path(item["path"]).name == "result.json")
    zero_calls(value)
    require(value.get("fixed_epochs_complete") == 5, "Boundary reproduction lost its actual five-epoch work")
    runs.append({**base(run_id, "numerical_boundary_reproduction_not_fidelity_acceptance"), "wall_seconds": wall(value),
        "observed_layer_epochs": 5, "epoch_count_exact": True})

    run_id = "normalized64_four_layer_success"
    _, summary = one(run_id, lambda item, value: Path(item["path"]).name == "summary.json")
    zero_calls(summary)
    require(summary.get("complete") is True and summary.get("all_four_passed") is True
            and [row.get("layer") for row in summary.get("cases", [])] == expected[run_id][0], "Normalized diagnostic is not complete for exactly four layers")
    terminal = [(item, value) for item, value in groups[run_id] if Path(item["path"]).name == "result.json"]
    require(len(terminal) == 4, "Normalized diagnostic omitted layer results")
    cases = []
    for item, value in terminal:
        own(run_id, item); zero_calls(value)
        require(value.get("complete") is True and value.get("epochs") == 64 and value.get("test_data_used") is False
                and value.get("fidelity_gate_passed") is True, "Normalized diagnostic layer is incomplete or failed")
        epochs(value["history"], 64)
        source(value["weights"], parse=False); source(value["statistics"]["tensor_file"], parse=False)
        cases.append({"layer": value["task"]["layer_index"], "completed_epochs": 64, "worker_wall_seconds": wall(value)})
    require(sorted(row["layer"] for row in cases) == expected[run_id][0], "Normalized diagnostic duplicated layers")
    runs.append({**base(run_id, "four_layer_diagnostic_pass_not_main_acceptance"), "wall_seconds": wall(summary),
        "observed_layer_epochs": 256, "epoch_count_exact": True, "cases": sorted(cases, key=lambda row: row["layer"])})

    current = _normalized_full_fit_cost(core, source)
    own("normalized64_full32_current_fit", current["result"])
    return {"schema": "core_normalized_transcoder_attempt_costs_v1", "complete": True,
        "historical_attempts": runs, "current_full_fit": current, "scientific_oracle_calls": 0, "gpu_hours": None,
        "physical_cost_categories_unchanged": True, "not_added_to_episode_wall_or_physical_cost_totals": True,
        "wall_time_total": None, "wall_time_total_status": "Not summed: some phases are unknown and worker intervals overlap aggregate run intervals.",
        "observed_layer_epochs_lower_bound": sum(row.get("observed_layer_epochs", row.get("observed_layer_epochs_lower_bound", 0)) for row in runs) + current["observed_layer_epochs"],
        "layer_epoch_total_exact": False,
        "evidence_files": [{"path": path, "sha256": digest} for path, digest in sorted(evidence.items())]}


def _normalized_full_fit_cost(core, source):
    """Require the actual full fit and installed bank, not four diagnostic layers."""
    amendment = core["normalization_amendment"]
    result_path = Path(amendment["normalized_precompute_result_path"]).resolve()
    require(result_path == Path(amendment["normalized_precompute_output_root"]).resolve() / "result.json",
            "Normalized execution path differs from registration")
    result_item = {"path": str(result_path), "sha256": file_sha256(result_path)}
    result = source(result_item)
    require(result.get("complete") is True and result.get("core_fingerprint") == core["fingerprint"]
            and result.get("layers") == 32 and result.get("epochs_per_layer") == 64
            and result.get("original_resume_verified") is True
            and result.get("all_bank_hashes_and_mtimes_unchanged_on_resume") is True
            and type(result.get("new_policy_or_material_oracle_calls")) is int
            and result["new_policy_or_material_oracle_calls"] == 0
            and isinstance(result.get("layer_receipts"), list) and len(result["layer_receipts"]) == 32,
            "Main normalized cost gate needs a complete real 32-layer/64-epoch run")
    external_path = Path(amendment["normalized_bank_path"]).resolve() / "transcoder_manifest.json"
    installed_path = Path(core["workspace"]).resolve() / "experiments/transcoders/qwen35_4b/made/transcoder_manifest.json"
    require(Path(result["bank"]["path"]).resolve() in {external_path, installed_path},
            "Normalized execution references an unregistered bank")
    source(result["bank"])
    external_item = {"path": str(external_path), "sha256": file_sha256(external_path)}
    installed_item = {"path": str(installed_path), "sha256": file_sha256(installed_path)}
    external, installed = source(external_item), source(installed_item)
    for bank in (external, installed):
        require(bank.get("bank_fingerprint") == fingerprint({key: value for key, value in bank.items() if key != "bank_fingerprint"})
                and bank.get("complete") is True and bank.get("ready_for_graphs") is True
                and bank.get("configuration", {}).get("epochs") == 64
                and bank.get("configuration", {}).get("max_dev_fvu") == .5
                and len(bank.get("layers", [])) == 32, "Full normalized bank is incomplete or failed fidelity")
    for key in ("configuration", "collection_fingerprint", "checkpoint_hash", "policy_runtime", "policy_configuration_fingerprint"):
        require(external.get(key) == installed.get(key), "Installed normalized bank changed a scientific input")
    receipts, cases, seen = result["layer_receipts"], [], set()
    for index, (original, layer, item) in enumerate(zip(external["layers"], installed["layers"], receipts, strict=True)):
        require(original.get("layer_index") == layer.get("layer_index") == index
                and original.get("metadata") == layer.get("metadata")
                and original.get("checkpoint_sha256") == layer.get("checkpoint_sha256")
                and original.get("transcoder_hash") == layer.get("transcoder_hash"),
                "Installed normalized layer is not the actual selected raw export")
        path = str(Path(item["path"]).resolve())
        require(path not in seen and Path(path) == result_path.parent / "layers" / f"layer_{index:02d}" / "result.json",
                "Duplicated or unregistered normalized worker receipt")
        seen.add(path)
        worker = source(item)
        require(worker.get("schema") == "parallel_normalized_layer_v1" and worker.get("complete") is True
                and worker.get("fingerprint") == fingerprint({key: value for key, value in worker.items() if key != "fingerprint"})
                and worker.get("layer_index") == index and worker.get("epochs") == worker.get("trained_epochs") == worker.get("history_length") == 64
                and worker.get("fit_recipe") == "train_centered_scalar_rms_fp32_export_v1"
                and worker.get("statistics_train_only") is True,
                "Normalized worker is partial, reused diagnostic, or another fitting recipe")
        checkpoint_path = Path(worker["checkpoint"]["path"]).resolve()
        require(checkpoint_path.parent == Path(path).parent
                and Path(worker["metadata"]["path"]).resolve() == Path(str(checkpoint_path) + ".json"),
                "Normalized worker checkpoint/metadata are outside its registered execution")
        metadata = source(worker["metadata"])
        request = worker.get("request", {})
        require(request.get("layer_index") == index and Path(request.get("output_dir", "")).resolve() == Path(path).parent
                and request.get("kwargs", {}).get("epochs") == 64
                and request["kwargs"].get("seed") == metadata.get("seed")
                and request["kwargs"].get("registration") == metadata.get("registration")
                and isinstance(request.get("input_files"), list) and request["input_files"]
                and isinstance(request.get("source_files"), list) and request["source_files"],
                "Normalized worker request differs from the actual fitted metadata")
        for input_file in request["input_files"] + request["source_files"]:
            source(input_file, parse=False)
        require(layer["metadata"].get("execution_provenance", {}).get("worker_receipt") == item
                and {key: value for key, value in layer["metadata"].items() if key != "execution_provenance"} == metadata
                and worker.get("transcoder_hash") == layer["transcoder_hash"],
                "Bank is not bound to the actual normalized worker metadata")
        source(worker["checkpoint"], parse=False)
        checkpoints = {value["checkpoint"]: value["checkpoint_sha256"] for value in (original, layer)}
        for path, digest in checkpoints.items():
            source({"path": path, "sha256": digest}, parse=False)
        normalization = metadata.get("normalization", {})
        history = metadata.get("history", [])
        require(metadata.get("epochs") == 64 and metadata.get("fit_recipe") == worker["fit_recipe"]
                and metadata.get("test_data_used") is False and metadata.get("max_dev_fvu") == .5
                and metadata.get("registration", {}).get("core_fingerprint") == core["fingerprint"]
                and metadata.get("runtime", {}).get("parameter_dtype") == "torch.float32"
                and normalization.get("train_only") is True
                and isinstance(normalization.get("sources"), list) and normalization["sources"]
                and normalization.get("sources") == metadata.get("source_files", {}).get("train")
                and normalization.get("export", {}).get("dtype") == "torch.float32"
                and [row.get("epoch") for row in history] == list(range(64))
                and all(type(row.get("train_rows")) is int and row["train_rows"] > 0
                        and row["train_rows"] == normalization.get("rows")
                        and number(row.get("dev", {}).get("output_mse")) and row["dev"]["output_mse"] >= 0 for row in history),
                "Normalized main metadata lacks complete train-only statistics/raw-space history")
        selected = min(range(64), key=lambda epoch: history[epoch]["dev"]["output_mse"])
        dev = history[selected]["dev"]
        require(metadata.get("selected_epoch") == selected and metadata.get("dev") == dev
                and metadata.get("fidelity_gate_passed") is True and dev.get("fvu_undefined") == 0
                and number(dev.get("output_fvu")) and 0 <= dev["output_fvu"] <= .5,
                "Main normalized layer failed unchanged raw development selection/fidelity")
        start, finish, elapsed = worker.get("started_at"), worker.get("finished_at"), worker.get("elapsed_seconds")
        require(number(start) and number(finish) and finish >= start and number(elapsed) and elapsed >= 0,
                "Normalized worker lacks actual completed timing")
        cases.append({"layer": index, "completed_epochs": 64, "worker_wall_seconds": elapsed,
            "started_at": start, "finished_at": finish, "receipt": item,
            "training_row_presentations": sum(row["train_rows"] for row in history)})
    require(number(result.get("started_at")) and number(result.get("finished_at"))
            and result["finished_at"] >= result["started_at"] and number(result.get("elapsed_seconds"))
            and result["elapsed_seconds"] >= 0, "Full normalized run lacks its measured wall time")
    return {"run_id": "normalized64_full32_current_fit", "category": "current_full_fit", "complete": True,
        "layers": 32, "epochs_per_layer": 64, "observed_layer_epochs": 2048, "epoch_count_exact": True,
        "wall_seconds": result["elapsed_seconds"], "started_at": result["started_at"], "finished_at": result["finished_at"],
        "worker_cases": cases, "development_evaluation_included_in_each_epoch_wall": True,
        "separate_development_selection_wall_seconds": None, "no_additional_dev_phase_claimed": True,
        "scientific_oracle_calls": 0, "gpu_hours": None, "result": result_item,
        "registered_bank": external_item, "core_bank": installed_item,
        "registered_bank_equals_core_bank": external_path == installed_path}


def _source_v3_cost_context(core):
    """Use the unchanged source core's cost implementation and bank verifier."""
    import importlib
    import sys
    from .passed_bank_reuse import verify_reused_bank, _frozen_modules, _Evidence
    reference = core["passed_bank_reuse_contract"]
    require(file_sha256(reference["path"]) == reference["sha256"], "Cost reuse contract changed")
    contract = read(reference["path"])
    require(contract["target_workspace"] == core["workspace"], "Cost reuse contract targets another core")
    bank, collection, proof = verify_reused_bank(contract)
    require(collection is None and proof["target_corpus_verified"] is False, "Source-only cost verification changed scope")
    source_core = read(contract["source_core_protocol"]["path"])
    modules = _frozen_modules(Path(source_core["workspace"]), source_core, _Evidence())
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        source_report = importlib.import_module(modules.namespace + ".core_report")
    finally:
        sys.dont_write_bytecode = old_bytecode
    history = source_report.offline_normalized_transcoder_report(source_core)
    require(history.get("complete") is True and history["current_full_fit"]["result"] == contract["source_run_result"]
            and history["current_full_fit"]["core_bank"] == contract["source_bank_manifest"],
            "Source TC cost record is not the explicitly reused all32 bank")
    return contract, source_core, bank, history


def offline_causal_graph_report(core):
    """V4 offline costs, independently usable before NN/ES/final completion.

    The original V3 32-layer fit occurs exactly once among historical attempts.
    Diagnostic phase times are contained in their attempt wall, never added to
    it. Missing CPU/worker timing and unpersisted graph work remain unknown.
    """
    from .core_protocol import CAUSAL_GRAPH_REGISTRATIONS
    require(core.get("registration") in CAUSAL_GRAPH_REGISTRATIONS, "Unregistered causal graph cost scope")
    contract, source_core, bank, prior = _source_v3_cost_context(core)
    amendment = core["causal_graph_amendment"]
    require(amendment.get("source_core_protocol") == contract["source_core_protocol"]
            and amendment.get("pause_closure") == contract["pause_closure"], "Causal cost history lost its exact source/closure")
    evidence, documents = {}, {}

    def source(item, *, parse=True):
        path = str(Path(item["path"]).resolve())
        require(file_sha256(path) == item["sha256"], "Causal offline cost evidence changed: " + path)
        require(path not in evidence or evidence[path] == item["sha256"], "Conflicting causal cost artifact hashes")
        evidence[path] = item["sha256"]
        if not parse or Path(path).suffix not in {".json", ".jsonl"}:
            return None
        if path not in documents:
            if Path(path).suffix == ".jsonl":
                data = Path(path).read_bytes()
                require(not data or data.endswith(b"\n"), "Closed graph/diagnostic file has an incomplete row")
                documents[path] = [json.loads(line) for line in data.splitlines() if line]
            else:
                documents[path] = read(path)
        return documents[path]

    def descendants(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                source(value, parse=False)
            for item in value.values(): descendants(item)
        elif isinstance(value, list):
            for item in value: descendants(item)

    def elapsed(value):
        wall = value.get("elapsed_seconds")
        start, finish = value.get("started_at"), value.get("finished_at")
        if wall is not None:
            require(number(wall) and wall >= 0, "Invalid measured causal diagnostic wall")
        if start is not None and finish is not None:
            require(number(start) and number(finish) and finish >= start, "Invalid causal diagnostic time endpoints")
            if wall is None: wall = finish - start
            else: require(abs(wall - (finish - start)) <= max(.01, abs(wall) * 1e-4), "Diagnostic wall/endpoints disagree")
        return wall

    for item in contract["source_artifacts"] + prior["evidence_files"] + [core["passed_bank_reuse_contract"]]:
        source(item, parse=False)
    # Move, rather than copy into a second cost category, the formerly current
    # V3 fit. Its provenance still names V3 and the original execution result.
    historical = deepcopy(prior["historical_attempts"])
    reused_fit = deepcopy(prior["current_full_fit"])
    reused_fit.update(run_id="normalized64_full32_source_fit", original_run_id=prior["current_full_fit"]["run_id"],
        category="historical_reused_passed_bank_fit", source_core_fingerprint=source_core["fingerprint"],
        new_training_in_this_core=False)
    historical.append(reused_fit)
    require(len(historical) == 7 and len({row["run_id"] for row in historical}) == 7,
            "V4 omitted or duplicated one of the six historical attempts or original full32 fit")

    closure = source(contract["pause_closure"])
    require(closure.get("closed") is True and closure.get("new_policy_or_material_oracle_calls") == 0
            and closure.get("processes_exited") and all(value is True for value in closure["processes_exited"].values()),
            "Old graph work is not closed")
    closure_files = {}
    for item in closure["artifacts"]:
        source(item, parse=False)
        source({"path": item["preserved"], "sha256": item["sha256"]}, parse=False)
        require(item["path"] not in closure_files, "Duplicated closure artifact")
        closure_files[item["path"]] = item
    report_path = Path(source_core["workspace"]) / "experiments/graph_reports/qwen35_4b/made/report.json"
    require(str(report_path) in closure_files, "Graph pause omitted its final report")
    graph_report = source(closure_files[str(report_path)])
    require(graph_report.get("transcoder_bank_fingerprint") == bank["bank_fingerprint"]
            and graph_report.get("expected_decisions") == contract["source_counts"]["decisions"]
            and graph_report.get("complete") is False and graph_report.get("status") == "failed"
            and graph_report.get("successful_graphs") == 0, "Original failed graph scope/outcome changed")
    rows, seen = [], set()
    for manifest in contract["source_manifest_paths"]:
        path = Path(manifest).parent / "graph_features.jsonl"
        if not path.exists(): continue
        require(str(path) in closure_files, "Pause omitted a persisted original graph journal")
        for row in source(closure_files[str(path)]):
            require(row.get("decision_id") not in seen and row.get("complete") is True
                    and row.get("graph_status") == "unavailable"
                    and row.get("transcoder_bank_fingerprint") == bank["bank_fingerprint"]
                    and row.get("graph_pipeline_fingerprint") == graph_report["graph_pipeline_fingerprint"]
                    and row.get("record_fingerprint") == fingerprint({k: v for k, v in row.items() if k != "record_fingerprint"}),
                    "Failed graph cost journal is duplicated, nonterminal or changed")
            seen.add(row["decision_id"]); rows.append(row)
    require(len(rows) == graph_report.get("completed_decisions") == graph_report.get("unavailable_graphs")
            and dict(Counter(row["failure_kind"] for row in rows)) == graph_report["failures_by_kind"],
            "Failed graph report and all persisted journals disagree")
    stage = closure["pipeline"]["stages"]["core-graphs"]
    require(stage.get("status") == "failed", "Original graph stage is not terminal failed")
    stage_wall = elapsed(stage)
    row_walls = [row.get("elapsed_seconds") for row in rows]
    require(all(value is None or (number(value) and value >= 0) for value in row_walls), "Invalid graph row timing")
    known_row_wall = sum(value for value in row_walls if value is not None)
    row_wall = known_row_wall if all(value is not None for value in row_walls) else None
    require(stage_wall is None or row_wall is None or row_wall <= stage_wall + .01,
            "Serial graph decision wall exceeds its containing stage")
    graph = {"run_id": "normalized_v3_failed_graph_stage", "category": "historical_failed_offline_graph",
        "execution_closed": True, "method_graph_acceptance": False, "phase_wall_seconds": stage_wall,
        "persisted_decision_wall_seconds": row_wall, "known_persisted_decision_wall_lower_bound": known_row_wall,
        "unattributed_stage_wall_seconds": stage_wall - row_wall if stage_wall is not None and row_wall is not None else None,
        "unpersisted_or_in_flight_computation": "unknown; not inferred from the number of unprocessed decisions",
        "expected_decisions": graph_report["expected_decisions"], "persisted_terminal_decisions": len(rows),
        "not_terminal_decisions": graph_report["expected_decisions"] - len(rows), "successful_graphs": 0,
        "failed_graphs": len(rows), "failures_by_kind": graph_report["failures_by_kind"],
        "report": {"path": str(report_path), "sha256": closure_files[str(report_path)]["sha256"]},
        "closure": contract["pause_closure"], "cpu_seconds": None, "gpu_hours": None, "scientific_oracle_calls": 0,
        "decision_wall_is_contained_in_phase_wall": True}

    history = amendment.get("diagnostic_history")
    require(isinstance(history, list) and history
            and len({str(Path(item["path"]).resolve()) for item in history}) == len(history),
            "Causal diagnostics are missing or duplicated")
    listed = {str(Path(item["path"]).resolve()): item for item in history}
    validated = amendment["validated_probe"]
    require(str(Path(validated["path"]).resolve()) in listed and listed[str(Path(validated["path"]).resolve())] == validated,
            "The actual validated probe must appear once in diagnostic history")
    schemas = {"single_prefix_native_feature_probe_v1", "causal_native_single_prefix_validation_v1"}
    attempts, cpu_audits = {}, []
    for item in history:
        value = source(item)
        if not isinstance(value, dict): continue
        descendants(value)
        if value.get("schema") == "passed_bank_source_audit_v1":
            require(value.get("complete") is True and value.get("source_core_fingerprint") == source_core["fingerprint"]
                    and value.get("target_corpus_verified") is False
                    and all(type(value.get(key)) is int and value[key] == 0 for key in
                            ("scientific_oracle_calls", "new_policy_or_material_oracle_calls", "gpu_model_calls")),
                    "Source-bank CPU audit is incomplete or changes scope/cost")
            measured = elapsed(value)
            require(measured is not None and value.get("started_at_utc") and value.get("finished_at_utc"),
                    "Source-bank CPU audit timing is missing")
            identity = fingerprint({"source": value["audit_source"], "started_at_utc": value["started_at_utc"],
                                    "finished_at_utc": value["finished_at_utc"]})
            require(not any(row["execution_identity"] == identity for row in cpu_audits), "Duplicate source-bank CPU audit")
            phases = []
            for key in ("build_elapsed_seconds", "verify_elapsed_seconds"):
                phase_wall = value.get(key)
                require(phase_wall is None or number(phase_wall) and 0 <= phase_wall <= measured,
                        "Invalid contained CPU audit timing")
                phases.append({"phase": key, "wall_seconds": phase_wall, "included_in_attempt_wall": True})
            cpu_audits.append({"run_id": "source_bank_cpu_audit:" + identity, "execution_identity": identity,
                "category": "read_only_cpu_admission", "cpu_task_wall_seconds": measured,
                "cpu_seconds": None, "gpu_hours": None, "scientific_oracle_calls": 0,
                "gpu_model_calls": 0, "contained_phases": phases, "terminal": item,
                "timing_note": "CPU task elapsed wall includes I/O; it is not measured process CPU time or TC training."})
            continue
        if value.get("schema") not in schemas: continue
        name = Path(item["path"]).name
        if name not in {"started.json", "failure.json", "result.json"}: continue
        require(isinstance(value.get("source"), dict) and number(value.get("started_at"))
                and type(value.get("pid")) is int, "Diagnostic attempt lacks its execution identity")
        identity = fingerprint({key: value[key] for key in ("schema", "source", "pid", "started_at")})
        entry = attempts.setdefault(identity, {"started": [], "terminal": []})
        entry["started" if name == "started.json" else "terminal"].append((item, value))
    diagnostics, outcomes = [], Counter()
    for identity, entry in attempts.items():
        require(len(entry["started"]) == 1 and len(entry["terminal"]) <= 1, "Missing/duplicate diagnostic start or terminal record")
        start_item, started = entry["started"][0]
        terminal_item, value = entry["terminal"][0] if entry["terminal"] else (None, started)
        is_gate = value["schema"] == "causal_native_single_prefix_validation_v1"
        required_zero = "new_policy_or_oracle_calls" if is_gate else "new_oracle_calls"
        explicit_zero = type(value.get(required_zero)) is int and value[required_zero] == 0
        zero_keys = ("new_oracle_calls", "new_policy_or_oracle_calls", "new_material_oracle_calls", "new_policy_generation_calls")
        require(explicit_zero and all(type(value[key]) is int and value[key] == 0 for key in zero_keys if key in value),
                "Diagnostic physical cost is nonzero, contradictory or undeclared")
        if "fingerprint" in value:
            require(value["fingerprint"] == fingerprint({k: v for k, v in value.items() if k != "fingerprint"}), "Changed diagnostic record")
        if is_gate:
            require(value.get("source_core_fingerprint") == source_core["fingerprint"]
                    and value.get("source_bank_manifest") == contract["source_bank_manifest"], "Gate diagnostic used another core/bank")
        elif value.get("complete") is True:
            require(value.get("core_fingerprint") == source_core["fingerprint"]
                    and value.get("bank_fingerprint") == bank["bank_fingerprint"]
                    and value.get("main_graph_files_written") is False
                    and value.get("native_FD_gate_run") is False, "Structural probe was confused with main/FD acceptance")
        state = "not_terminal_unknown_work" if terminal_item is None else "completed_diagnostic" if value.get("complete") is True else "failed_diagnostic"
        role = "causal_native_gate" if is_gate else "native_structure_probe"
        outcomes[(role, state)] += 1
        phases = []
        for row in value.get("assemblies", []):
            measured = row.get("elapsed_seconds")
            require(measured is None or number(measured) and measured >= 0, "Invalid contained assembly time")
            phases.append({"phase": "assembly", "feature_cap": row.get("feature_cap"), "wall_seconds": measured,
                           "included_in_attempt_wall": True})
        if terminal_item == validated:
            require(is_gate and value.get("complete") is True and value.get("passed") is True
                    and value.get("validation", {}).get("passed") is True, "Registered validation is not a passed real gate")
        diagnostics.append({"run_id": role + ":" + identity, "execution_identity": identity, "category": "historical_diagnostic",
            "role": role, "outcome": state, "failed_phase": value.get("failed_phase"), "error": value.get("error"),
            "phase_wall_seconds": elapsed(value) if terminal_item else None, "cpu_seconds": None, "worker_wall_seconds": None,
            "gpu_hours": None, "scientific_oracle_calls": 0,
            "policy_generation_calls": value.get("new_policy_generation_calls"),
            "capture_calls": value.get("capture_calls"), "additional_score_VJP_calls": value.get("additional_score_VJP_calls"),
            "validation_attempts": value.get("validation_attempts"), "attribute_attempts": value.get("attribute_attempts"),
            "contained_phases": phases, "method_graph_acceptance": False,
            "single_prefix_gate_passed": value.get("passed") if is_gate else None,
            "started_at": value["started_at"], "finished_at": value.get("finished_at"),
            "start": start_item, "terminal": terminal_item,
            "unrecorded_work": "unknown" if terminal_item is None else "no additional work inferred from terminal record"})
    require(outcomes[("native_structure_probe", "failed_diagnostic")] >= 1
            and outcomes[("native_structure_probe", "completed_diagnostic")] >= 1
            and outcomes[("causal_native_gate", "failed_diagnostic")] >= 1
            and any(item["terminal"] == validated for item in diagnostics),
            "Causal history omitted failed/completed structural probes or the failed/passed gate attempts")
    diagnostics.sort(key=lambda item: (item["started_at"], item["run_id"]))
    return {"schema": "core_causal_graph_offline_costs_v1", "complete": True,
        "source_core_fingerprint": source_core["fingerprint"], "historical_transcoder_attempts": historical,
        "current_transcoder_training": {"training_calls": 0, "layers_fitted": 0, "fit_wall_seconds": 0,
            "bank_reused": contract["source_bank_manifest"], "reuse_validation_cpu_seconds": None,
            "reuse_validation_timing_status": "unknown; source-admission audit duration is not new TC training"},
        "historical_graph_attempts": [graph], "diagnostic_attempts": diagnostics, "cpu_admission_audits": cpu_audits,
        "scientific_oracle_calls": 0, "gpu_hours": None, "wall_time_total": None,
        "timing_note": "Phase wall, contained decision/assembly wall, parallel worker wall, and unknown CPU time stay separate. No GPU-hour inference or repeated TC charge.",
        "not_added_to_episode_wall_or_physical_cost_totals": True, "full_graphs_NN_ES_or_final_acceptance_claimed": False,
        "evidence_files": [{"path": path, "sha256": digest} for path, digest in sorted(evidence.items())]}


def prior_interruption_costs(core):
    """The separately archived partial is historical overhead, not reused data."""
    source = core["reconciliation"]
    require(file_sha256(source["path"]) == source["sha256"], "Core transition reconciliation changed")
    record = read(source["path"])
    require(record.get("schema") == "interruption_reconciliation_v1" and record.get("all_requests_resolved") is True
            and record.get("unknown_physical_outcomes", 0) == 0 and record.get("used_for_training") is False
            and record.get("used_for_final_evaluation") is False
            and record.get("additional_incurred_costs_not_subtracted_from_main_budgets") is True,
            "Prior core-transition costs are unknown or mixed into main data")
    seen = set()
    for item in record["artifacts"]:
        path = Path(item["path_after"]).resolve()
        require(path not in seen and Path(source["path"]).resolve().parent in path.parents
                and file_sha256(path) == item["sha256"], "Prior interrupted raw evidence is missing, duplicated or changed")
        seen.add(path)
    require(bool(seen), "Prior interruption has no raw evidence")
    return {"costs": _sum_costs([record["observed_physical_costs"]]), "receipt": source,
            "included_in_matched_core_budgets": False, "new_core_scientific_calls": False,
            "scope": "only_the_registered_transition_partial; other_older_full_study_diagnostics_not_included"}


def report_core(project, *, output=None):
    from .core_protocol import read_core, final_jobs, core_execution_budget
    from .core_collection import audit_core_corpus_costs
    from .core_es import audit_core_es_costs
    core = read_core(project)
    budget = core_execution_budget(core)
    workspace = Path(core["workspace"]).resolve()
    destination = Path(output).resolve() if output else workspace / "experiments/core_reports"
    require(destination.is_relative_to(workspace / "experiments"), "Core report must remain in the independent workspace")
    final = workspace / "experiments/core_final"
    manifest = build_core_final_manifest(core["fingerprint"], final_jobs(core), core_protocol=core)
    require(read(final / "manifest.json") == manifest, "Stored core final matrix differs")
    execution = read(final / "execution_inputs.json")
    require(all(file_sha256(path) == digest for path, digest in execution["inputs"].items()), "Frozen core final source/evaluator inputs changed")
    ledger = CoreFinalLedger(final / "ledger.json", manifest, readonly=True)
    tasks = read(workspace / "configs/benchmark_tasks.json")
    records, profiles, missing = {}, {}, []
    for job in manifest["jobs"]:
        state = ledger.data["jobs"][job["job_id"]]
        if state["state"] != "succeeded":
            missing.append({"job_id": job["job_id"], "state": state["state"], "halt": state.get("halt")})
            continue
        expected_path = final / "jobs" / job["job_id"] / state["attempt_id"] / "result.json"
        require(Path(state["result_path"]).resolve() == expected_path.resolve(), "Core result is not a fresh final trajectory under its own attempt")
        proof = verify_core_envelope(expected_path, job, manifest, tasks=tasks)
        require(proof["result_sha256"] == state["result_sha256"], "Core final ledger result hash mismatch")
        envelope = read(expected_path)
        require(len(envelope["episodes"]) == 1, "MADE core job must have one complete episode")
        records[job["job_id"]] = {"episode": envelope["episodes"][0], "reliability": reliability(envelope, expected_path.parent),
            "result": {"path": str(expected_path), "sha256": proof["result_sha256"]}}
        profile = envelope["execution_profile"]
        require(profile["execution_inputs_fingerprint"] == fingerprint(execution), "Final profile refers to another execution input record")
        if job["method"] in profiles:
            require(profiles[job["method"]] == profile, "Same core method changed policy/controller between paired systems")
        profiles[job["method"]] = profile
    report = {"schema": "core_paired_report_v1", "scope": SCOPE, "core_protocol_fingerprint": core["fingerprint"],
        "core_complete": not missing, "global_study_complete": False, "expected_counts": deepcopy(manifest["expected_counts"]),
        "completed_jobs": len(records), "missing_jobs": missing, "ledger": ledger.completion(),
        "paired_results": pair_report(manifest, records), "execution_profiles": profiles,
        "costs": None, "scientific_improvement_assumed": False}
    if missing:
        report["status"] = "incomplete_no_core_acceptance"
        write_json_atomic(destination / "report.json", report)
        return report
    final_completion = read(final / "completion.json")
    require(final_completion.get("complete") is True and final_completion.get("core_final_complete") is True
            and final_completion.get("core_protocol_fingerprint") == core["fingerprint"]
            and final_completion.get("manifest_fingerprint") == manifest["fingerprint"]
            and final_completion.get("expected_counts") == manifest["expected_counts"] and final_completion.get("completed_jobs") == 4,
            "Final stage has not published its complete four-job acceptance")
    require(profiles["baseline"]["initial_actual_model_state_hash"] == profiles["esopt_graph_risk"]["initial_actual_model_state_hash"]
            and profiles["baseline"]["policy_configuration_fingerprint"] == profiles["esopt_graph_risk"]["policy_configuration_fingerprint"],
            "Paired methods do not share the same initial model and decoding configuration")
    corpus = audit_core_corpus_costs(core)
    es = audit_core_es_costs(workspace / "experiments/core_es", core)
    if budget == 10:
        selected = profiles["esopt_graph_risk"]["selected_es"]
        fields = ("evolution_signal_observed", "evolution_status", "nonzero_text_parameter_update_verified",
                  "zero_update_proof", "generation_curve", "initial_dev_evaluated")
        require(all(key in es and selected.get(key) == es[key] for key in fields),
                "FAST final selected policy differs from independently audited ES evolution evidence")
        curve = es["generation_curve"]
        require([row["generation"] for row in curve] == [1, 2]
                and len(curve[0]["dev_environment_seeds"]) == 1
                and curve[0]["dev_environment_seeds"] == curve[1]["dev_environment_seeds"]
                and es["initial_dev_evaluated"] is False,
                "FAST development curve must use one fixed dev seed for G1/G2, without an invented G0 evaluation")
        report.update({key: deepcopy(es[key]) for key in fields})
        report.update(registered_execution_scope=core["scope"], execution_budget=budget,
            original_b50_core_complete=False, initial_dev_metric=None,
            evolution_interpretation=("Complete registered execution with no selected-policy evolution; not evidence of method effectiveness."
                if not es["evolution_signal_observed"] else
                "Measured selected text-parameter change; effectiveness is assessed only by the independent paired outcomes."))
    report["costs"] = core_cost_report(corpus, es, records, expected_imported_jobs=len(core["imported_train"]),
                                     expected_reused_development=bool(core.get("imported_development")),
                                     expected_made_budget=budget, core_protocol=core if budget == 10 else None)
    offline = offline_transcoder_report(core)
    if offline is not None:
        key = "offline_representation_attempts" if offline.get("schema") == "core_causal_graph_offline_costs_v1" else "offline_transcoder_attempts"
        report["costs"][key] = offline
        evidence = {item["path"]: item["sha256"] for item in report["costs"]["evidence_files"]}
        for item in offline["evidence_files"]:
            require(item["path"] not in evidence or evidence[item["path"]] == item["sha256"],
                    "Offline and physical cost evidence disagree")
            evidence[item["path"]] = item["sha256"]
        report["costs"]["evidence_files"] = [{"path": path, "sha256": digest} for path, digest in sorted(evidence.items())]
    report["costs"]["prior_interrupted"] = prior_interruption_costs(core)
    report["costs"]["all_recorded_costs_with_transition_overhead"] = _sum_costs([
        report["costs"]["reused_plus_incremental_costs"], report["costs"]["prior_interrupted"]["costs"]])
    report["status"] = "complete_registered_core_only"
    report["result_files"] = [record["result"] for record in records.values()]
    write_json_atomic(destination / "report.json", report)
    completion = {"schema": "core_completion_v1", "complete": True, "core_complete": True, "global_study_complete": False,
        "scope": SCOPE, "core_protocol_fingerprint": core["fingerprint"], "expected_counts": deepcopy(manifest["expected_counts"]), "completed_jobs": 4,
        "population_superiority_or_significance_claim": False, "original_full_study_complete": False,
        "report": {"path": str(destination / "report.json"), "sha256": file_sha256(destination / "report.json")}}
    if budget == 10:
        completion.update(execution_budget=10, registered_execution_scope=core["scope"],
            original_b50_core_complete=False, evolution_status=es["evolution_status"],
            evolution_signal_observed=es["evolution_signal_observed"], scientific_improvement_assumed=False)
    publish_once(destination / "core_completion.json", completion)
    stage = {"schema": "deadline_core_stage_receipt_v1", "stage": "report", "core_fingerprint": core["fingerprint"],
        "complete": True, "scope": SCOPE, "original_full_study_complete": False,
        "core_complete": True, "global_study_complete": False, "core_protocol_fingerprint": core["fingerprint"],
        "completion": {"path": str(destination / "core_completion.json"), "sha256": file_sha256(destination / "core_completion.json")},
        "artifacts": [{"path": str(destination / name), "sha256": file_sha256(destination / name)}
                      for name in ("report.json", "core_completion.json")]}
    stage["fingerprint"] = fingerprint(stage)
    publish_once(workspace / "experiments/core_stage_receipts/report.json", stage)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = report_core(args.project, output=args.output)
    print(json.dumps({"status": result["status"], "core_complete": result["core_complete"], "global_study_complete": False,
                      "completed_jobs": result["completed_jobs"]}, indent=2))
    return 0 if result["core_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
