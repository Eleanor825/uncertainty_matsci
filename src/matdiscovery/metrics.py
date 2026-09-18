"""Episode-level reporting with explicit missingness and paired cluster inference.

One row is one complete episode, never one atomic action or oracle-query step.
The estimand averages episodes within seed, seeds within task, then tasks equally.
No failed observation is removed from denominators. Undefined properties after a
failed evaluator remain missing, so their unconditional means/CIs are undefined.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .accounting import (
    AccountingError, COUNT_KEYS, episode_key, expected_episodes, validate_episode,
    validate_manifest,
)


class IncompleteResultsError(AccountingError):
    def __init__(self, report: Mapping):
        self.report = dict(report)
        super().__init__(
            f"Final report requires the immutable full matrix: "
            f"{report['missing_episodes']} missing, {report['incomplete_episodes']} incomplete, "
            f"{report['budget_mismatch_episodes']} budget-mismatched episodes"
        )


def read_episode_summaries(paths: Iterable[str | Path]) -> list[dict]:
    """Read JSON envelopes/lists or JSONL episode summaries, retaining every row."""
    result = []
    for path in paths:
        path = Path(path)
        if path.suffix == ".jsonl":
            result.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        else:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                result.extend(data)
            elif isinstance(data, dict) and "episodes" in data:
                result.extend(data["episodes"])
            else:
                result.append(data)
    return result


def _index(rows: Iterable[Mapping]) -> dict[tuple, Mapping]:
    index = {}
    for row in rows:
        validate_episode(row)
        key = episode_key(row)
        if key in index:
            raise AccountingError(f"Duplicate episode (steps must not be independent rows): {key}")
        index[key] = row
    return index


def audit_summaries(summaries: Iterable[Mapping], manifest: Mapping, *, final: bool = False) -> dict:
    """Check the complete declared denominator, including failures and unobserved rows."""
    validate_manifest(manifest)
    index, expected = _index(summaries), expected_episodes(manifest)
    unexpected = set(index) - set(expected)
    if unexpected:
        raise AccountingError(f"Results outside the immutable manifest: {sorted(unexpected)[:3]}")
    missing = sorted(set(expected) - set(index))
    incomplete, mismatches = [], []
    counts, costs, all_costs, jobs = Counter(), Counter(), Counter(), defaultdict(list)
    for key, row in index.items():
        want = expected[key]
        jobs[want["job_id"]].append(key)
        counts[row["status"]] += 1
        all_costs.update(row["costs"])
        if not row["complete"]:
            incomplete.append(key)
        wrong_budget = False
        for name in COUNT_KEYS[1:]:
            actual = row["costs"].get(name, 0)
            costs[name] += actual
            wrong_budget |= type(actual) is not int or actual != want[name]
        if "environment_seed" in row and row["environment_seed"] != want["environment_seed"]:
            raise AccountingError(f"Unexpected packed environment seed: {key}")
        if wrong_budget:
            mismatches.append(key)
    complete_keys = set(index) - set(incomplete) - set(mismatches)
    complete_jobs = sum(all(episode_key({**j, "episode_id": e}) in complete_keys for e in j["episode_ids"]) for j in manifest["jobs"])
    report = {
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "expected_counts": dict(manifest["expected_counts"]),
        "observed_counts": {"jobs": len(jobs), "episodes": len(index), **{name: costs[name] for name in COUNT_KEYS[1:]}},
        "observed_costs": dict(all_costs),
        "completed_jobs": complete_jobs, "completed_episodes": len(complete_keys),
        "missing_episodes": len(missing), "incomplete_episodes": len(incomplete),
        "budget_mismatch_episodes": len(mismatches), "status_counts": dict(counts),
        "failed_episode_count": counts["failed"],
        "failure_denominator_expected": len(expected),
        "failure_denominator_observed": len(index),
        "observed_failed_fraction": counts["failed"] / len(index) if index else None,
        "missing_episode_keys": [list(k) for k in missing],
        "incomplete_episode_keys": [list(k) for k in incomplete],
        "budget_mismatch_episode_keys": [list(k) for k in mismatches],
        "complete": not missing and not incomplete and not mismatches,
    }
    if final and not report["complete"]:
        raise IncompleteResultsError(report)
    return report


def _number(value) -> float | None:
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return float(value)


def summarize_results(summaries: Iterable[Mapping], manifest: Mapping, *, final: bool = True) -> dict:
    """Report per-task metrics; do not merge unlike physical property units.

    `observed_value_mean` is explicitly conditional on a metric being defined.
    `mean` is null whenever any expected episode is missing/incomplete or lacks
    the metric, including failed episodes. Completion is independent of success.
    """
    rows = list(summaries)
    audit = audit_summaries(rows, manifest, final=final)
    index = _index(rows)
    groups = defaultdict(list)
    for key in expected_episodes(manifest):
        groups[key[:4]].append(key)
    reports = []
    for identity, keys in sorted(groups.items()):
        observed = [index[k] for k in keys if k in index]
        names = sorted({name for row in observed for name in row["metrics"]})
        report = dict(zip(("benchmark", "model_key", "method", "task_id"), identity))
        report.update({
            "expected_episodes": len(keys), "observed_episodes": len(observed),
            "complete_episodes": sum(row["complete"] for row in observed),
            "failed_episodes": sum(row["status"] == "failed" for row in observed),
            "failure_denominator": len(keys), "metrics": {},
        })
        for name in names:
            by_seed = defaultdict(list)
            missing = 0
            for key in keys:
                row = index.get(key)
                value = _number(row["metrics"].get(name)) if row and row["complete"] else None
                if value is None:
                    missing += 1
                else:
                    by_seed[key[4]].append(value)
            values = [v for vv in by_seed.values() for v in vv]
            seed_means = {str(seed): float(np.mean(v)) for seed, v in sorted(by_seed.items())}
            report["metrics"][name] = {
                "mean": float(np.mean(list(seed_means.values()))) if not missing else None,
                "observed_value_mean": float(np.mean(values)) if values else None,
                "observed_values": len(values), "missing_values": missing,
                "expected_denominator": len(keys), "observed_seed_means": seed_means,
            }
        reports.append(report)
    return {"report_kind": "final" if final else "partial", "accounting": audit, "tasks": reports}


def paired_cluster_bootstrap(
    summaries: Iterable[Mapping], *, benchmark: str, model_key: str,
    method: str, reference: str, metric: str, manifest: Mapping | None = None,
    task_ids: Iterable[str] | None = None, n_bootstrap: int = 10000,
    confidence: float = 0.95, seed: int = 0,
) -> dict:
    """Paired percentile CI, resampling tasks then seed clusters within each task.

    All episodes within a sampled task/seed remain together. The same resampled
    indices are used for both methods. Without a manifest this is an explicitly
    exploratory comparison: jointly absent episodes cannot be detected. Final
    reports must also pass audit_summaries(..., final=True).
    """
    if method == reference:
        raise AccountingError("Paired comparison needs two different methods")
    if type(n_bootstrap) is not int or n_bootstrap < 1 or not 0 < confidence < 1:
        raise AccountingError("Require positive bootstrap count and confidence in (0,1)")
    selected_tasks = set(task_ids) if task_ids is not None else None
    def selected(key):
        return key[0] == benchmark and key[1] == model_key and key[2] in {method, reference} and (selected_tasks is None or key[3] in selected_tasks)
    index = {key: row for key, row in _index(summaries).items() if selected(key)}
    if manifest is not None:
        validate_manifest(manifest)
        wanted = {key for key in expected_episodes(manifest) if selected(key)}
        if set(index) != wanted:
            raise AccountingError("Paired inference is missing expected episodes or has unexpected rows")
    arms = {arm: {(*key[:2], *key[3:]): row for key, row in index.items() if key[2] == arm} for arm in (method, reference)}
    if not arms[method] or set(arms[method]) != set(arms[reference]):
        raise AccountingError("Methods must have exactly the same complete paired episodes")
    if selected_tasks is not None and {key[2] for key in arms[method]} != selected_tasks:
        raise AccountingError("Requested tasks are missing from the paired comparison")
    physical_metrics = {"property", "property_value", "property_abs_error", "absolute_error", "abs_error"}
    if benchmark == "crystalgym" and metric in physical_metrics:
        properties = {key[2].split(":", 1)[0] for key in arms[method]}
        if len(properties) != 1:
            raise AccountingError("Physical metrics with different units require a separate comparison per property")
    differences, failed = defaultdict(lambda: defaultdict(list)), Counter()
    for key in sorted(arms[method]):
        first, second = arms[method][key], arms[reference][key]
        values = []
        for arm, row in ((method, first), (reference, second)):
            if not row["complete"]:
                raise AccountingError("Incomplete episodes cannot enter paired inference")
            value = _number(row["metrics"].get(metric))
            if value is None:
                raise AccountingError(f"Metric {metric} is missing/undefined; dropping failed episodes is prohibited: {key}")
            failed[arm] += row["status"] == "failed"
            values.append(value)
        differences[key[2]][key[3]].append(values[0] - values[1])
    clusters = [np.asarray([np.mean(v) for _, v in sorted(seeds.items())], dtype=float) for _, seeds in sorted(differences.items())]
    estimate = float(np.mean([values.mean() for values in clusters]))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for draw in range(n_bootstrap):
        sampled_tasks = rng.integers(0, len(clusters), size=len(clusters))
        task_means = []
        for task in sampled_tasks:
            values = clusters[task]
            task_means.append(values[rng.integers(0, len(values), size=len(values))].mean())
        bootstrap[draw] = np.mean(task_means)
    tail = (1 - confidence) / 2
    low, high = np.quantile(bootstrap, [tail, 1 - tail])
    return {
        "benchmark": benchmark, "model_key": model_key, "method": method,
        "reference": reference, "metric": metric, "delta": estimate,
        "ci": [float(low), float(high)], "confidence": confidence,
        "direction": "method_minus_reference", "n_tasks": len(clusters),
        "n_seed_pairs": sum(len(values) for values in clusters),
        "n_episodes_per_method": len(arms[method]),
        "failed_episodes": {arm: failed[arm] for arm in (method, reference)},
        "bootstrap_seed": seed, "n_bootstrap": n_bootstrap,
        "resampling": "paired_task_then_seed_clusters_all_episodes_retained",
        "coverage_verified_against_manifest": manifest is not None,
    }


def calibration_metrics(labels, probabilities, *, bins: int = 10) -> dict:
    """Reuse the project's risk implementation, including overconfident-error rate."""
    from .uncertainty import risk_metrics
    return risk_metrics(labels, probabilities, bins=bins)
