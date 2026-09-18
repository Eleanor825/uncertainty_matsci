"""Transport/configuration contracts only: no materials experiment or proxy rewards.

Small objects below are test doubles for error/serialization boundaries. Scientific
validity, model loading and real QE/ORB execution require the fresh benchmark envs.
"""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from matdiscovery.benchmark_adapters import (
    AdapterError, BaseAdapter, CrystalGymAdapter, MADEAdapter, file_sha256,
    jsonable, load_snapshot, score_order, score_rankable,
)
from matdiscovery.env_server import RPCSession, serve


class EchoContractDouble:
    def initial_result(self):
        print("library progress must never enter the JSON protocol")
        return {"initialized": True}

    def dispatch(self, op, args):
        if op == "step":
            raise AdapterError("contract_error", "deliberate error", {"count": 1})
        return {"op": op}

    def status(self):
        return {"contract_double": True}


class BenchmarkRPCContracts(unittest.TestCase):
    def test_jsonl_stdout_isolated_and_errors_keep_correlation(self):
        session = RPCSession("made", factory=lambda name, args: EchoContractDouble())
        incoming = io.StringIO('\n'.join([
            '{"id":1,"op":"init","args":{}}',
            '{"id":"failure","op":"step","args":{}}',
            'not json',
            '{"id":3,"op":"close"}',
            '{"id":4,"op":"observe"}',
        ]) + '\n')
        output = io.StringIO()
        serve(incoming, output, session)
        responses = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(responses), 4)
        self.assertEqual(responses[0], {"id": 1, "ok": True, "result": {"initialized": True}})
        self.assertEqual(responses[1]["id"], "failure")
        self.assertEqual(responses[1]["error"]["code"], "contract_error")
        self.assertEqual(responses[2]["error"]["code"], "invalid_json")
        self.assertTrue(session.closed)

    def test_uninitialized_and_mixed_benchmark_requests_fail(self):
        session = RPCSession("made", factory=lambda *unused: self.fail("factory should not run"))
        self.assertEqual(session.handle({"id": 1, "op": "observe"})["error"]["code"], "not_initialized")
        response = session.handle({"id": 2, "op": "init", "args": {"benchmark": "crystalgym"}})
        self.assertEqual(response["error"]["code"], "benchmark_mismatch")
        self.assertEqual(session.handle({"id": {}, "op": "observe"})["error"]["code"], "invalid_request")

    def test_real_server_module_starts_without_scientific_packages(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        process = subprocess.run([sys.executable, "-m", "matdiscovery.env_server", "--benchmark", "made"],
            input='{"id":1,"op":"observe"}\n{"id":2,"op":"close"}\n', capture_output=True, text=True, env=env, check=True)
        responses = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual([r["id"] for r in responses], [1, 2])
        self.assertEqual(responses[0]["error"]["code"], "not_initialized")

    def test_missing_or_scalar_only_snapshot_never_becomes_a_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.json"
            path.write_text(json.dumps({"source": "materials_project", "complete": True,
                "elements": ["Li", "O"], "thermo_types": ["GGA_GGA+U"],
                "entries": [{"material_id": "label-only", "formula": "Li2O", "e_above_hull": 0}]}))
            with self.assertRaises(AdapterError) as failure:
                load_snapshot(str(path), file_sha256(path), ["Li", "O"])
            self.assertEqual(failure.exception.code, "invalid_snapshot")
            with self.assertRaises(AdapterError) as failure:
                load_snapshot(str(path), "0"*64, ["Li", "O"])
            self.assertEqual(failure.exception.code, "asset_hash_mismatch")
            with self.assertRaises(AdapterError) as failure:
                load_snapshot(str(path), None, ["Li", "O"])
            self.assertEqual(failure.exception.code, "snapshot_not_frozen")

    def test_nonfinite_values_are_preserved_without_json_nan(self):
        result = jsonable({"score": float("inf"), "bad": float("nan")})
        encoded = json.dumps(result, allow_nan=False)
        self.assertEqual(json.loads(encoded)["score"], {"nonfinite": "inf"})

    def test_official_negative_infinity_does_not_discard_finite_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.make_adapter_shell(MADEAdapter, directory)
            values = [float("nan"), -float("inf"), -2.0, 1.0, float("inf")]
            entries = [{"hash": str(i), "scores": {}} for i in range(len(values))]
            outcomes = adapter._record_scores("oracle", entries, values, [{"score": value} for value in values])
            self.assertEqual([entry["hash"] for entry in entries], ["3", "2", "1", "0", "4"])
            self.assertEqual(outcomes[1]["status"], "official_negative_infinity_sentinel")
            self.assertTrue(outcomes[1]["rankable"])
            self.assertEqual(adapter.counts, {"score_failures": 2, "score_sentinels": 1})
            self.assertFalse(score_rankable(float("nan")))
            self.assertFalse(score_rankable(float("inf")))
            bottom = sorted(entries, key=lambda entry: score_order(entry["scores"]["oracle"], descending=False))
            self.assertEqual([entry["hash"] for entry in bottom], ["1", "2", "3", "0", "4"])
            json.dumps(jsonable(outcomes), allow_nan=False)

    def test_all_negative_infinity_scores_preserve_official_stable_tie_order(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.make_adapter_shell(MADEAdapter, directory)
            entries = [{"hash": name, "scores": {}} for name in ("first", "second")]
            adapter._record_scores("oracle", entries, [-float("inf")]*2, [{}, {}])
            self.assertEqual([entry["hash"] for entry in entries], ["first", "second"])
            self.assertTrue(all(entry["score_status"]["oracle"]["rankable"] for entry in entries))

    def make_adapter_shell(self, cls, directory, budget=2):
        adapter = object.__new__(cls)
        BaseAdapter.__init__(adapter, {"vendor_root": directory, "work_dir": str(Path(directory).parent / "new_output"), "seed": 1, "budget": budget})
        return adapter

    def test_oracle_exception_consumes_attempt_and_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.make_adapter_shell(MADEAdapter, directory)
            adapter.env = SimpleNamespace(query_count=0, is_done=lambda: False)

            def fail_oracle(structure):
                adapter.count("candidate_oracle_attempts")
                raise RuntimeError("deliberate oracle exception boundary")

            adapter.env.step = fail_oracle
            for attempt in (1, 2):
                adapter.selected = {"structure": object(), "hash": str(attempt)}
                with self.assertRaises(AdapterError) as failure:
                    adapter.step({})
                self.assertEqual(failure.exception.code, "oracle_or_environment_exception")
                self.assertEqual(adapter.episode_counts["candidate_oracle_attempts"], attempt)
                self.assertIsNone(adapter.selected)
                self.assertEqual(adapter.env.query_count, 0)
            self.assertTrue(adapter.done())
            with self.assertRaises(AdapterError) as failure:
                adapter.step({})
            self.assertEqual(failure.exception.code, "budget_exhausted")
            self.assertEqual(adapter.counts["candidate_oracle_attempts"], 2)

    def test_crystal_observation_does_not_expose_original_species_or_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = self.make_adapter_shell(CrystalGymAdapter, directory)
            adapter.budget_unit, adapter.property, adapter.target = "dft_episode_attempts", "band_gap", 2.0
            adapter.episode_done = False
            adapter.filled = ["Li", None]
            adapter.env = SimpleNamespace(
                t=1, n_sites=2, sample_ind=630,
                traversal=[SimpleNamespace(item=lambda: 0), SimpleNamespace(item=lambda: 1)],
                data={"cif": "hidden_original_Au_crystal", "band_gap": "secret_label"},
                state=object(),
                graph_to_dict_complete=lambda state: {"lengths": [4, 4, 4], "angles": [90, 90, 90],
                    "frac_coords": [[0, 0, 0], [.5, .5, .5]], "atom_types": [79, 79]},
            )
            observation = adapter.observe()
            encoded = json.dumps(observation)
            self.assertEqual(observation["filled_elements"], ["Li", None])
            self.assertEqual(observation["focus_site"], 1)
            self.assertNotIn("atom_types", encoded)
            self.assertNotIn("hidden_original", encoded)
            self.assertNotIn("secret_label", encoded)
            self.assertNotIn('"Au"', encoded)

    def test_complete_task_matrix_and_unlocked_training_budget(self):
        path = Path(__file__).resolve().parents[1]/"configs/benchmark_tasks.json"
        config = json.loads(path.read_text())
        made, crystal = config["made"], config["crystalgym"]
        self.assertEqual(len(made["systems"]), 30)
        self.assertEqual(len({s["id"] for s in made["systems"]}), 30)
        self.assertEqual(made["oracle_query_budget_per_episode"], 50)
        self.assertEqual(made["seeds"], [1, 2, 3, 4, 5])
        self.assertEqual({p["id"] for p in crystal["properties"]}, {"bm", "density", "band_gap"})
        self.assertEqual(set(crystal["train_indices"]) & set(crystal["heldout_indices"]), set())
        self.assertEqual(len(crystal["train_indices"]), 5)
        self.assertEqual(len(crystal["heldout_indices"]), 2)
        self.assertEqual(crystal["evaluation"]["rollouts_per_seed_per_heldout_prototype"], 5)
        self.assertEqual(crystal["training"]["official_rl_yaml_total_timesteps"], 500000)
        self.assertIsNone(crystal["training"]["main_protocol_budget"])
        self.assertFalse(crystal["training"]["full_training_may_be_claimed"])


if __name__ == "__main__":
    unittest.main()
