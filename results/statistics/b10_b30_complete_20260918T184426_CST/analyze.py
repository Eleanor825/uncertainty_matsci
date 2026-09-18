"""Descriptive statistics across paired chemical systems; no experiment calls."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
FIELDS = ["budget", "metric", "group", "n", "mean", "sample_variance_ddof1",
          "sample_sd_ddof1", "population_variance_ddof0", "seed", "variance_axis"]


def main():
    output, sources, systems = [], [], []
    for budget in (10, 30):
        path = ROOT / f"pairs_B{budget}.csv"
        rows = list(csv.DictReader(path.open(newline="")))
        task_ids = {row["task_id"] for row in rows}
        assert len(rows) == len(task_ids) == 30
        assert all(int(row.get("seed", 1)) == 1 for row in rows)
        assert all(int(row.get("budget", budget)) == budget for row in rows)
        systems.append(task_ids)
        sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for metric in ("SUN", "mSUN", "AUDC"):
            key, divisor = ("SUN", budget) if metric == "mSUN" else (metric, 1)
            values = {arm: [float(row[f"{arm}_{key}"]) / divisor for row in rows]
                      for arm in ("baseline", "full")}
            values["full_minus_baseline"] = [o - b for o, b in zip(values["full"], values["baseline"])]
            for group, data in values.items():
                assert all(math.isfinite(value) for value in data)
                mean = statistics.mean(data)
                variance = statistics.variance(data)
                direct = sum((value - mean) ** 2 for value in data) / (len(data) - 1)
                assert math.isclose(variance, direct, rel_tol=1e-12, abs_tol=1e-14)
                output.append(dict(zip(FIELDS, [budget, metric, group, len(data), mean, variance,
                    statistics.stdev(data), statistics.pvariance(data), 1, "chemical_system"])))
    assert systems[0] == systems[1], "Budget comparisons must use the same official system set"
    for metric in ("SUN", "mSUN", "AUDC"):
        for group in ("baseline", "full", "full_minus_baseline"):
            output.append(dict(zip(FIELDS, [50, metric, group, 0, None, None, None, None, 1, "chemical_system"])))
    with (ROOT / "statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    summary = {
        "schema": "MADE_single_seed_cross_system_mean_variance_v1",
        "as_of_CST": "2026-09-18T18:44:26+08:00",
        "benchmark": "MADE", "model": "Qwen3.5-4B", "seed": 1,
        "methods": {"baseline": "original baseline", "full": "fixed G2 ESOpt plus graph-risk policy"},
        "B10_and_B30_complete_pairs_each": 30,
        "B50": {"completed": 0, "claimed_uncompleted": 10, "unclaimed": 50, "complete_pairs": 0},
        "sources": sources, "statistics": output,
        "limitations": [
            "Variance is across chemical systems, not repeated random seeds.",
            "Sample variance uses n-1; population variance uses n. SD is the square root of sample variance.",
            "Paired difference is full minus baseline on each matched system.",
            "The uncertainty of the mean, calibrated uncertainty, and repeatability are not measured by this variance.",
            "Missing B50 statistics are null, not zero performance.",
        ],
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"paired_systems_each": 30, "budgets_complete": [10, 30], "rows": len(output)}))


if __name__ == "__main__":
    main()
