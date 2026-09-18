"""CPU scheduling/accounting contracts; no MACE, ORB, torch or scientific calls.

Run the pinned official Oracle scheduler with simple non-scientific test doubles.
These tests do not establish numerical parity of the production MACE helper.
"""

import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from matdiscovery.benchmark_adapters import (
    AdapterError, BaseAdapter, MADEAdapter, RELAX, score_order,
)
from matdiscovery.mace_parallel import (
    AuditedOracleExecutor, TorchFXGuard,
    audit_oracle_attempt_journal, mace_execution_metadata, official_chunk_count,
)


def official_oracle_class():
    source = Path(__file__).resolve().parents[1] / "vendor/MADE/src/made/oracles/base.py"
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != "32305857b9ef5e5f782942d1868f89c653c9c656bff6ded5989f02027e87956b":
        raise RuntimeError("Tests require the exact pinned official MADE scheduler")
    tree = ast.parse(raw)
    tree.body = [node for node in tree.body if not
                 (isinstance(node, ast.ImportFrom) and node.module == "pymatgen.core.structure")]
    namespace = {"Structure": object}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["Oracle"]


OfficialOracle = official_oracle_class()


class FakeGuard:
    def __init__(self):
        self.checked = []

    def check(self, calculator=None):
        if calculator is not None:
            self.checked.append(calculator)


class ContractOracle(OfficialOracle):
    def __init__(self, num_workers=1, function=None, factory=None, **kwargs):
        super().__init__(num_workers=num_workers)
        self.options = kwargs
        self._calculator_factory = factory or object
        self.calculator = self._calculator_factory()
        self._thread_local = threading.local()
        self.function = function or (lambda value, calculator: value)

    def evaluate(self, value):
        if self.num_workers > 1:
            if not hasattr(self._thread_local, "calculator"):
                self._thread_local.calculator = self._calculator_factory()
            calculator = self._thread_local.calculator
        else:
            calculator = self.calculator
        return self.function(value, calculator)


class MACEParallelContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.vendor = self.directory / "vendor"
        self.vendor.mkdir()
        self.adapter = object.__new__(MADEAdapter)
        BaseAdapter.__init__(self.adapter, {"vendor_root": str(self.vendor),
            "work_dir": str(self.directory / "output"), "seed": 1, "budget": 50})
        # Do not change process cwd as a running server would.
        self.adapter.work_dir.mkdir()
        self.adapter.event_file = self.adapter.work_dir / "env_events.jsonl"
        self.adapter.oracle_attempt_file = self.adapter.work_dir / "oracle_attempts.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def install(self, oracle, role="mace", guard=None):
        return AuditedOracleExecutor(oracle, role=role, candidate_hash=lambda s: f"test-{s}",
            invoke=self.adapter.invoke_oracle, record=self.adapter.oracle_audit,
            guard=guard or (FakeGuard() if oracle.num_workers == 4 else None))

    def rows(self):
        return [json.loads(line) for line in self.adapter.oracle_attempt_file.read_text().splitlines()]

    def test_dynamic_native_batches_preserve_exact_order_and_private_objects(self):
        for size in (1, 4, 5, 57):
            with self.subTest(size=size):
                constructed, used, order = [], {}, []
                lock = threading.Lock()
                chunks = official_chunk_count(size, 4)
                barrier = threading.Barrier(chunks)
                chunk_size = math.ceil(size / min(4, size))

                def factory():
                    calculator = object()
                    constructed.append(calculator)
                    order.append("construct")
                    return calculator

                def compute(value, calculator):
                    self.assertFalse(self.adapter._state_lock._is_owned())
                    with lock:
                        order.append("evaluate")
                        used.setdefault(threading.get_ident(), set()).add(id(calculator))
                    if value % chunk_size == 0:
                        barrier.wait(timeout=5)
                    return {"energy": float(value), "natoms": 1}

                oracle = ContractOracle(4, function=compute, factory=factory)
                initial = oracle.calculator
                self.install(oracle)
                result = oracle.batch_evaluate(list(range(size)))
                self.assertEqual([r["energy"] for r in result], list(range(size)))
                self.assertEqual(len(constructed), 1 + chunks)
                self.assertEqual(order[:1 + chunks], ["construct"] * (1 + chunks))
                self.assertEqual(order[1 + chunks:], ["evaluate"] * size)
                self.assertEqual(len(used), chunks)
                self.assertTrue(all(len(objects) == 1 for objects in used.values()))
                self.assertEqual(len(set.union(*used.values())), chunks)
                self.assertNotIn(id(initial), set.union(*used.values()))
                self.assertIs(oracle._calculator_factory, factory)
        snapshot = self.adapter.oracle_audit_snapshot()
        self.assertEqual(snapshot["attempts_started"], 67)
        self.assertEqual(snapshot["terminals"], {"returned": 67, "exception": 0})
        self.assertFalse(snapshot["pending"])
        rows = self.rows()
        self.assertEqual([r["sequence"] for r in rows], list(range(len(rows))))
        self.assertEqual(len({r["attempt_id"] for r in rows if r["kind"] == "oracle_attempt_started"}), 67)

    def test_duplicate_inputs_get_distinct_indices_and_original_identity(self):
        candidate = object()
        seen = []
        oracle = ContractOracle(1, function=lambda value, calc: seen.append(value) or {"energy": -1.0})
        self.install(oracle)
        result = oracle.batch_evaluate([candidate, candidate])
        self.assertEqual(len(result), 2)
        self.assertTrue(all(value is candidate for value in seen))
        starts = [r for r in self.rows() if r["kind"] == "oracle_attempt_started"]
        self.assertEqual([r["batch_input_index"] for r in starts], [0, 1])
        self.assertEqual(starts[0]["candidate_hash"], starts[1]["candidate_hash"])
        self.assertNotEqual(starts[0]["attempt_id"], starts[1]["attempt_id"])

    def test_serial_still_uses_one_calculator_and_stops_at_first_exception(self):
        used = []
        def compute(value, calculator):
            used.append(calculator)
            if value == 1:
                raise ValueError("serial contract failure")
            return {"energy": value}
        oracle = ContractOracle(1, function=compute)
        self.install(oracle)
        with self.assertRaisesRegex(ValueError, "serial contract failure"):
            oracle.batch_evaluate(list(range(8)))
        self.assertEqual(used, [oracle.calculator, oracle.calculator])
        snapshot = self.adapter.oracle_audit_snapshot()
        self.assertEqual(snapshot["counts"], {"surrogate_oracle_attempts": 2})
        self.assertEqual(snapshot["terminals"], {"returned": 1, "exception": 1})
        self.assertFalse(snapshot["pending"])
        self.assertFalse(any(r["kind"] == "calculator_constructor_started" for r in self.rows()))
        # Ordinary tool events retain their original schema and sequencing.
        self.assertEqual([e["kind"] for e in self.adapter.events], ["oracle_evaluation", "oracle_exception"])
        self.assertNotIn("attempt_id", self.adapter.events[0])

    def test_parallel_exception_retains_other_chunks_actual_extra_calls(self):
        barrier = threading.Barrier(4)
        def compute(value, calculator):
            if value % 2 == 0:
                barrier.wait(timeout=5)
            if value == 0:
                raise ValueError("parallel contract failure")
            return {"energy": value}
        oracle = ContractOracle(4, function=compute)
        self.install(oracle)
        with self.assertRaisesRegex(ValueError, "parallel contract failure"):
            oracle.batch_evaluate(list(range(8)))
        snapshot = self.adapter.oracle_audit_snapshot()
        self.assertEqual(snapshot["counts"], {"surrogate_oracle_attempts": 7})
        self.assertEqual(snapshot["terminals"], {"returned": 6, "exception": 1})
        self.assertFalse(snapshot["pending"])
        rows = self.rows()
        failure = [r for r in rows if r["kind"] == "oracle_attempt_exception"][0]
        self.assertEqual(failure["batch_input_index"], 0)
        self.assertIn("raise ValueError", failure["traceback"])
        batch = rows[-1]
        self.assertEqual(batch["kind"], "oracle_batch_exception")
        self.assertTrue(batch["parallel_workers_joined"])
        self.assertFalse(batch["exception_execution_set_equivalent_to_serial"])
        self.assertEqual(len({r["attempt_id"] for r in rows if r["kind"] == "oracle_attempt_started"}), 7)

    def test_factory_failure_is_not_an_oracle_attempt(self):
        constructed = []
        def factory():
            if len(constructed) == 2:
                raise RuntimeError("constructor failed")
            obj = object()
            constructed.append(obj)
            return obj
        oracle = ContractOracle(4, factory=factory, function=lambda *a: self.fail("No evaluation allowed"))
        original_factory = oracle._calculator_factory
        self.install(oracle)
        with self.assertRaisesRegex(RuntimeError, "constructor failed"):
            oracle.batch_evaluate(list(range(4)))
        self.assertEqual(self.adapter.counts, {})
        self.assertEqual(self.adapter.oracle_audit_snapshot()["attempts_started"], 0)
        self.assertIs(oracle._calculator_factory, original_factory)
        self.assertEqual(self.rows()[-1]["stage"], "calculator_construction")
        self.assertEqual(sum(r["kind"] == "calculator_constructor_exception" for r in self.rows()), 1)

    def test_shared_calculator_or_dirty_fx_blocks_all_evaluation(self):
        for shared in (True, False):
            with self.subTest(shared=shared):
                obj = object()
                guard = FakeGuard()
                if not shared:
                    def check(calculator=None):
                        if calculator is not None:
                            raise RuntimeError("FX remains patched")
                    guard.check = check
                oracle = ContractOracle(4, factory=(lambda: obj) if shared else object,
                    function=lambda *a: self.fail("No evaluation allowed"))
                self.install(oracle, guard=guard)
                with self.assertRaises(RuntimeError):
                    oracle.batch_evaluate(list(range(4)))
                self.assertEqual(self.adapter.counts, {})

    def test_new_batch_reconstructs_after_last_forward_and_empty_is_free(self):
        sequence = []
        def factory():
            sequence.append("constructor")
            return object()
        oracle = ContractOracle(4, factory=factory,
            function=lambda v, c: sequence.append("evaluate") or v)
        self.install(oracle)
        oracle.batch_evaluate([1])
        oracle.batch_evaluate([])
        oracle.batch_evaluate([2])
        self.assertEqual(sequence, ["constructor", "constructor", "evaluate", "constructor", "evaluate"])
        self.assertEqual(self.adapter.counts["surrogate_oracle_attempts"], 2)

    def test_direct_orb_initialization_and_candidate_calls_have_exact_phase(self):
        oracle = ContractOracle(1)
        self.install(oracle, role="orb")
        self.assertEqual(oracle.evaluate(10), 10)
        self.adapter._phase = "candidate"
        self.assertEqual(oracle.evaluate(11), 11)
        self.assertEqual(self.adapter.counts, {"initialization_oracle_attempts": 1, "candidate_oracle_attempts": 1})
        self.assertFalse(self.adapter.oracle_audit_snapshot()["pending"])
        with self.assertRaisesRegex(ValueError, "ORB requires one"):
            self.install(ContractOracle(4), role="orb")

    def test_count_event_and_snapshot_are_thread_safe_without_numerical_lock(self):
        def worker(value):
            for i in range(100):
                self.adapter.count("test_only")
                self.adapter.event("test", value=value, i=i)
                snapshot = self.adapter.status()
                self.assertEqual(snapshot["counts"], snapshot["episode_counts"])
                snapshot["counts"]["test_only"] = -1
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(8)))
        self.assertEqual(self.adapter.count_value("test_only"), 800)
        rows = [json.loads(line) for line in self.adapter.event_file.read_text().splitlines()]
        self.assertEqual([r["sequence"] for r in rows], list(range(800)))
        oracle = ContractOracle(1)
        def compute(value, calculator):
            # Snapshot from another thread can finish while native evaluate runs.
            with ThreadPoolExecutor(max_workers=1) as pool:
                snap = pool.submit(self.adapter.oracle_audit_snapshot).result(timeout=2)
            self.assertEqual(len(snap["pending"]), 1)
            self.assertFalse(self.adapter._state_lock._is_owned())
            return value
        oracle.function = compute
        self.install(oracle)
        oracle.evaluate(1)

    def test_nonfinite_raw_values_and_original_stable_score_order_are_preserved(self):
        oracle = ContractOracle(1, function=lambda value, c: {"energy": value})
        self.install(oracle)
        results = oracle.batch_evaluate([1.0, -math.inf, math.nan])
        self.assertEqual(results[1]["energy"], -math.inf)
        self.assertTrue(math.isnan(results[2]["energy"]))
        terminal = [r for r in self.rows() if r["kind"] == "oracle_attempt_returned"]
        self.assertEqual(terminal[1]["result"]["energy"], {"nonfinite": "-inf"})
        self.assertEqual(sorted(range(3), key=lambda i: score_order(results[i]["energy"])), [0, 1, 2])

    def test_worker_configuration_rejects_bool_and_unsupported_values(self):
        for value in (True, False, 0, 2, 8, 4.0, "4", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mace_execution_metadata(value)
        self.assertEqual(mace_execution_metadata(1)["lifecycle"], "original_single_calculator")
        self.assertEqual(mace_execution_metadata(4)["num_workers"], 4)
        for value in (True, 2, "4"):
            args = {**self.adapter.args, "mace_num_workers": value}
            with self.assertRaises(AdapterError) as failure:
                MADEAdapter(args)
            self.assertEqual(failure.exception.code, "invalid_config")

    def test_real_guard_logic_rejects_each_fx_global_and_wrong_parameter_runtime(self):
        # Run the production guard logic against tiny metadata-only modules,
        # importing no torch package and constructing no numerical model.
        torch = ModuleType("torch")
        torch.__path__ = []
        torch.float32 = "float32"
        torch.get_default_dtype = lambda: "float32"
        module = SimpleNamespace(__call__=object(), __getattr__=object())
        torch.nn = SimpleNamespace(Module=module)
        fx = ModuleType("torch.fx")
        fx.__path__ = []
        symbolic = ModuleType("torch.fx._symbolic_trace")
        symbolic.CURRENT_PATCHER, symbolic._is_fx_tracing_flag = None, False
        fx._symbolic_trace, torch.fx = symbolic, fx
        parameter = SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype="float32", is_floating_point=lambda: True)
        calc = SimpleNamespace(models=[SimpleNamespace(parameters=lambda: [parameter])])
        with patch.dict(sys.modules, {"torch": torch, "torch.fx": fx, "torch.fx._symbolic_trace": symbolic}):
            guard = TorchFXGuard()
            guard.check(calc)
            for obj, attr, bad in ((module, "__call__", object()), (module, "__getattr__", object()),
                                   (symbolic, "CURRENT_PATCHER", object()), (symbolic, "_is_fx_tracing_flag", True),
                                   (parameter, "dtype", "float64"), (parameter.device, "type", "cuda"),
                                   (torch, "get_default_dtype", lambda: "float64")):
                with self.subTest(attr=attr), patch.object(obj, attr, bad), self.assertRaises(RuntimeError):
                    guard.check(calc)

    def test_constructor_failure_restores_orb_loader_without_scientific_count(self):
        modules = {}
        for name in ("made", "made.oracles", "made.oracles.orb", "made.oracles.orb.orb_oracle",
                     "made.oracles.mace", "made.oracles.mace.mace_oracle", "made.utils",
                     "made.utils.structure_hash", "orb_models", "orb_models.forcefield"):
            modules[name] = ModuleType(name)
            modules[name].__path__ = []
        def failing_constructor(**kwargs):
            raise ValueError("initial constructor failed")
        modules["made.oracles.orb.orb_oracle"].ORBOracle = failing_constructor
        modules["made.oracles.mace.mace_oracle"].MACEOracle = failing_constructor
        modules["made.utils.structure_hash"].structure_hash = str
        loader = lambda **kwargs: object()
        pretrained = SimpleNamespace(orb_v3_conservative_inf_omat=loader)
        modules["orb_models.forcefield"].pretrained = pretrained
        self.adapter.device, self.adapter.mace_num_workers = "cpu", 1
        self.adapter.assets = {key: {"path": "/test/" + key} for key in
            ("orb_checkpoint", "mace_checkpoint", "element_reference_energies")}
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(ValueError, "initial constructor failed"):
                self.adapter._make_oracle("orb")
        self.assertIs(pretrained.orb_v3_conservative_inf_omat, loader)
        self.assertFalse(self.adapter.counts)
        self.assertEqual([r["kind"] for r in self.rows()], ["oracle_constructor_started", "oracle_constructor_exception"])

    def test_make_oracle_keeps_orb_one_and_exact_scientific_options(self):
        modules = {}
        for name in ("made", "made.oracles", "made.oracles.orb", "made.oracles.orb.orb_oracle",
                     "made.oracles.mace", "made.oracles.mace.mace_oracle", "made.utils",
                     "made.utils.structure_hash", "orb_models", "orb_models.forcefield"):
            modules[name] = ModuleType(name)
            modules[name].__path__ = []
        modules["made.oracles.orb.orb_oracle"].ORBOracle = ContractOracle
        modules["made.oracles.mace.mace_oracle"].MACEOracle = ContractOracle
        modules["made.utils.structure_hash"].structure_hash = lambda s: f"test-{s}"
        loader = lambda **kwargs: object()
        pretrained = SimpleNamespace(orb_v3_conservative_inf_omat=loader)
        modules["orb_models.forcefield"].pretrained = pretrained
        self.adapter.device = "cpu"
        self.adapter.mace_num_workers = 4
        self.adapter.assets = {key: {"path": "/test/" + key} for key in
            ("orb_checkpoint", "mace_checkpoint", "element_reference_energies")}
        with patch.dict(sys.modules, modules), patch("matdiscovery.benchmark_adapters.TorchFXGuard", FakeGuard):
            orb, mace = self.adapter._make_oracle("orb"), self.adapter._make_oracle("mace")
            self.assertEqual((orb.num_workers, mace.num_workers), (1, 4))
            self.assertEqual(orb.options["model_name"], "orb-v3-conservative-inf-omat")
            self.assertEqual(orb.options["relax_kwargs"], RELAX)
            self.assertEqual(mace.options, {"model_name": "custom", "model_path": "/test/mace_checkpoint",
                "default_dtype": "float32", "dispersion": False, "device": "cpu", "relax": True,
                "relax_kwargs": RELAX, "element_reference_energies_path": "/test/element_reference_energies"})
            self.assertIs(pretrained.orb_v3_conservative_inf_omat, loader)
            self.assertEqual(orb.evaluate(7), 7)
            self.assertEqual(mace.batch_evaluate([2, 3]), [2, 3])

    def test_readonly_journal_audit_role_phase_episode_and_invoked_failures(self):
        oracle = ContractOracle(4, function=lambda value, c: {"energy": value})
        self.install(oracle)
        oracle.batch_evaluate([1, 2, 3, 4])
        self.adapter.episode_index = 1
        self.adapter._phase = "candidate"
        orb = ContractOracle(1, function=lambda value, c: (_ for _ in ()).throw(ValueError("invoked failure")))
        self.install(orb, role="orb")
        with self.assertRaises(ValueError):
            orb.evaluate(5)
        path = self.adapter.oracle_attempt_file
        before = path.read_bytes()
        report = audit_oracle_attempt_journal(path)
        self.assertEqual(path.read_bytes(), before)
        self.assertTrue(report["valid"])
        self.assertTrue(report["closed"])
        self.assertFalse(report["successful"])
        self.assertEqual(report["counts"], {"surrogate_oracle_attempts": 4, "candidate_oracle_attempts": 1})
        self.assertEqual(len(report["oracle_errors"]), 1)
        self.assertEqual(report["constructor_errors"], [])
        self.assertEqual(report["by_role_phase_episode"][1], {"role": "orb", "phase": "candidate",
                         "episode_index": 1, "attempts": 1, "returned": 0, "exceptions": 1, "unknown": 0})

    def test_auditor_rejects_duplicate_missing_terminal_and_counter_tampering(self):
        oracle = ContractOracle(1)
        self.install(oracle)
        oracle.evaluate(1)
        original = self.rows()
        start = next(r for r in original if r["kind"] == "oracle_attempt_started")
        terminal = next(r for r in original if r["kind"] == "oracle_attempt_returned")
        for mutation in ("duplicate_start", "duplicate_terminal", "missing_terminal", "counter", "sequence", "unhashable"):
            with self.subTest(mutation=mutation):
                rows = json.loads(json.dumps(original))
                if mutation == "duplicate_start": rows.append(dict(start))
                elif mutation == "duplicate_terminal": rows.append(dict(terminal))
                elif mutation == "missing_terminal": rows = [r for r in rows if r["kind"] != "oracle_attempt_returned"]
                elif mutation == "counter":
                    for row in rows:
                        if row["kind"].startswith("oracle_attempt_"): row["counter"] = "candidate_oracle_attempts"
                elif mutation == "sequence": rows[1]["sequence"] = "bad"
                else: rows[-1]["batch_id"] = []
                if mutation != "sequence":
                    for i, row in enumerate(rows): row["sequence"] = i
                path = self.directory / (mutation + ".jsonl")
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                report = audit_oracle_attempt_journal(path)
                self.assertFalse(report["closed"])
                if mutation == "missing_terminal": self.assertEqual(report["unknown_attempt_ids"], [start["attempt_id"]])
                else: self.assertTrue(report["errors"])
                self.assertFalse(report["counts_reliable"])

    def test_auditor_constructor_failure_is_closed_and_separate_from_call_cost(self):
        def factory():
            raise RuntimeError("construction only")
        oracle = ContractOracle(4)
        oracle._calculator_factory = factory
        self.install(oracle)
        with self.assertRaises(RuntimeError): oracle.batch_evaluate([1])
        report = audit_oracle_attempt_journal(self.adapter.oracle_attempt_file)
        self.assertTrue(report["valid"])
        self.assertTrue(report["closed"])
        self.assertFalse(report["successful"])
        self.assertEqual(report["counts"], {})
        self.assertEqual(report["attempts_started"], 0)
        self.assertEqual(len(report["constructor_errors"]), 1)
        self.assertEqual(report["oracle_errors"], [])

    def test_auditor_missing_or_truncated_file_never_means_verified_zero(self):
        path = self.directory / "missing.jsonl"
        self.assertFalse(audit_oracle_attempt_journal(path)["counts_reliable"])
        path.write_text('{"sequence":0')
        self.assertFalse(audit_oracle_attempt_journal(path)["valid"])


if __name__ == "__main__":
    unittest.main()
