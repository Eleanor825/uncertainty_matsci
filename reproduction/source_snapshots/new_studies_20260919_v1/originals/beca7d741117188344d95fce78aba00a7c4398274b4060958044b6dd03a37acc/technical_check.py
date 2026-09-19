"""Exactly four authorized technical ODE attempts: two direct plus two RPC.

No policy, DFT or materials oracle. Never substitutes for a formal benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.client import OracleClient
from oracle.official import file_sha, load_official
from oracle.snar_rpc import unique_json

POINTS = [{"tau": 1.25, "equiv_pldn": 3.0, "conc_dfnb": 0.3, "temperature": 75.0},
          {"tau": 0.5, "equiv_pldn": 1.0, "conc_dfnb": 0.1, "temperature": 30.0}]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    unique_json(output / "plan.json", {"classification": "technical_not_main", "scientific_main_result": False,
                                      "seed": 1729, "points": POINTS, "max_total_ode_attempts": 4,
                                      "script_sha256": file_sha(__file__)})
    started = time.time()
    direct_count = 0
    direct_returned = 0
    client = None
    results = []
    try:
        cls, DataSet, _, _, proof = load_official()
        import numpy as np
        benchmark = cls(noise_level=0.0)
        benchmark.rng = np.random.default_rng(1729)
        for i, point in enumerate(POINTS):
            unique_json(output / (f"reference_{i}.intent.json"), {"query_id": str(i), "parameters": point, "source": proof})
            direct_count += 1
            then = time.perf_counter()
            raw = benchmark.run_experiments(DataSet([list(point.values())], columns=list(point)))
            values = {k: float(raw[k].iloc[-1]) for k in ("sty", "e_factor")}
            direct_returned += 1
            unique_json(output / (f"reference_{i}.result.json"), {"objectives": values, "elapsed_seconds": time.perf_counter() - then})
            results.append(values)
        client = OracleClient(sys.executable, output / "rpc", seed=1729, max_queries=2, classification="technical_not_main")
        for i, point in enumerate(POINTS):
            actual = client.evaluate("parity-" + str(i), point)
            assert actual["objectives"] == results[i], (actual, results[i])
            cached = client.evaluate("parity-" + str(i), point)
            assert cached["cache_hit"] and cached["physical_oracle_calls_this_request"] == 0
            assert cached["objectives"] == results[i]
        child = client.close()
        assert child["oracle_attempts"] == 2 and child["unknown_outcomes"] == 0
        unique_json(output / "summary.json", {"classification": "technical_not_main", "scientific_main_result": False,
                    "passed": True, "status": "complete", "exact_rpc_direct_parity": True,
                    "counters": {"snar_ode_attempts": direct_count + child["oracle_attempts"], "direct_reference_attempts": direct_count,
                                 "rpc_oracle_attempts": child["oracle_attempts"], "dft_episode_attempts": 0},
                    "unknown_outcomes": 0, "results": results, "source": proof,
                    "elapsed_seconds": time.time() - started, "rpc_summary_sha256": file_sha(output / "rpc/summary.json")})
        print(json.dumps({"passed": True, "actual_total_ode_attempts": 4, "results": results, "elapsed_seconds": time.time() - started}))
    except BaseException:
        import traceback
        # No retry or speculative completion after partial failure.
        unique_json(output / "failure.json", {"classification": "technical_not_main", "scientific_main_result": False,
                    "direct_attempts": direct_count, "direct_returned": direct_returned,
                    "rpc_costs_source": "rpc/oracle_attempts.jsonl", "traceback": traceback.format_exc(),
                    "elapsed_seconds": time.time() - started})
        raise


if __name__ == "__main__":
    main()
