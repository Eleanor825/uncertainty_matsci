"""CPU administrative tests. All query evaluations here are explicit unit stubs."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from oracle.official import BOUNDS, parameters
from oracle.snar_rpc import NeedsReconciliation, SnarOracle, encode, unique_json

POINT = {k: (a + b) / 2 for k, (a, b) in BOUNDS.items()}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="unit_only_no_oracle_")
        self.path = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def oracle(self, **kwargs):
        return SnarOracle(self.path, seed=7, max_queries=1,
                          classification="technical_not_main", **kwargs)

    @staticmethod
    def stub(conditions):
        import pandas as pd
        return pd.DataFrame({"sty": [123.0], "e_factor": [456.0]})

    def test_parameter_contract(self):
        self.assertEqual(set(parameters(POINT)), set(BOUNDS))
        for point in [dict(POINT, tau=True), dict(POINT, tau=float("nan")),
                      dict(POINT, tau=0.0), dict(POINT, extra=1.0)]:
            with self.assertRaises(ValueError):
                parameters(point)

    def test_cache_budget_and_changed_request(self):
        oracle = self.oracle()
        with patch.object(oracle.benchmark, "run_experiments", side_effect=self.stub) as call:
            first = oracle.evaluate("q1", POINT)
            second = oracle.evaluate("q1", POINT)
            self.assertEqual(call.call_count, 1)
            self.assertEqual(first["objectives"], second["objectives"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(second["physical_oracle_calls_this_request"], 0)
            with self.assertRaises(ValueError):
                oracle.evaluate("q1", dict(POINT, tau=0.5))
            with self.assertRaises(ValueError):
                oracle.evaluate("q2", POINT)
        summary = oracle.close()
        self.assertEqual((summary["oracle_attempts"], summary["completed_queries"], summary["unknown_outcomes"]), (1, 1, 0))

    def test_closed_cache_reopen_does_not_change_original_journal(self):
        oracle = self.oracle()
        with patch.object(oracle.benchmark, "run_experiments", side_effect=self.stub):
            original = oracle.evaluate("q1", POINT)
        oracle.close()
        before = (self.path / "oracle_attempts.jsonl").read_bytes()
        other = self.oracle()
        with patch.object(other.benchmark, "run_experiments", side_effect=AssertionError("No replay")):
            result = other.evaluate("q1", POINT)
            self.assertEqual(result["objectives"], original["objectives"])
        other.close()
        self.assertEqual(before, (self.path / "oracle_attempts.jsonl").read_bytes())

    def test_pending_intent_stops_without_call(self):
        oracle = self.oracle()
        request = {"query_id": "q1", "parameters": parameters(POINT), "seed": 7,
                   "noise_level": 0.0, "source_lock_sha256": oracle.identity["source"]["source_lock_sha256"]}
        unique_json(oracle.queries / (hashlib.sha256(b"q1").hexdigest() + ".intent.json"),
                    dict(request, request_sha256=hashlib.sha256(encode(request).encode()).hexdigest(), attempt_id=oracle.session_id + ":1"))
        oracle._refresh()  # Simulate reconstruction at a fresh-owner boundary.
        with patch.object(oracle.benchmark, "run_experiments", side_effect=AssertionError("No replay")):
            with self.assertRaises(NeedsReconciliation):
                oracle.evaluate("q1", POINT)
            with self.assertRaises(NeedsReconciliation):
                oracle.evaluate("new", POINT)
        summary = oracle.close()
        self.assertEqual((summary["oracle_attempts"], summary["reserved_queries"], summary["unknown_outcomes"]), (0, 1, 1))

    def test_official_call_exception_is_charged_once(self):
        oracle = self.oracle()
        with patch.object(oracle.benchmark, "run_experiments", side_effect=RuntimeError("unit-only failure")) as call:
            with self.assertRaises(RuntimeError):
                oracle.evaluate("q1", POINT)
            with self.assertRaises(NeedsReconciliation):
                oracle.evaluate("q1", POINT)
            self.assertEqual(call.call_count, 1)
        summary = oracle.close()
        self.assertEqual((summary["oracle_attempts"], summary["errors"], summary["unknown_outcomes"]), (1, 1, 0))

    def test_different_seed_cannot_reopen(self):
        oracle = self.oracle()
        oracle.close()
        with self.assertRaises(ValueError):
            SnarOracle(self.path, seed=8, max_queries=1, classification="technical_not_main")

    def test_pause_releases_owner_and_permits_later_new_query(self):
        oracle = self.oracle()
        receipt = oracle.pause()
        self.assertTrue(receipt["paused"])
        self.assertFalse((self.path / "summary.json").exists())
        other = self.oracle()
        with patch.object(other.benchmark, "run_experiments", side_effect=self.stub) as call:
            other.evaluate("q1", POINT)
            self.assertEqual(call.call_count, 1)
        other.close()


if __name__ == "__main__":
    unittest.main()
