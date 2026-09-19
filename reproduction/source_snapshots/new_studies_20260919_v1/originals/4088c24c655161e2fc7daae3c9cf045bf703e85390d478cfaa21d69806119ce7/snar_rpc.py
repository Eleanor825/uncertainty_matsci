"""One-process JSON-RPC oracle: official CPU SnAr, no automatic retries."""
import argparse
import contextlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from oracle.official import BOUNDS, UNITS, file_sha, load_official, parameters


def encode(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def unique_json(path, value):
    path = Path(path)
    temp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x") as stream:
        stream.write(encode(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)  # Atomic no-replace publication; never replace old receipts.
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temp.unlink()


class NeedsReconciliation(RuntimeError):
    pass


class SnarOracle:
    def __init__(self, session_dir, *, seed, max_queries, classification,
                 protocol_path=None, protocol_sha256=None, source_root=None):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seed must be an unsigned32 integer")
        if type(max_queries) is not int or max_queries <= 0:
            raise ValueError("Positive max_queries required")
        if classification not in {"technical_not_main", "benchmark"}:
            raise ValueError("Explicit classification required")
        if classification == "technical_not_main" and max_queries > 4:
            raise ValueError("Technical session cap is four official oracle attempts")
        protocol = None
        if protocol_path is not None:
            if not protocol_sha256 or file_sha(protocol_path) != protocol_sha256:
                raise ValueError("Protocol source hash differs")
            json.loads(Path(protocol_path).read_text())
            protocol = {"path": str(Path(protocol_path).resolve()), "sha256": protocol_sha256}
        if classification == "benchmark" and protocol is None:
            raise ValueError("Formal caller must supply its frozen protocol reference")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("Dedicated CPU worker requires CUDA_VISIBLE_DEVICES='' before import")
        self.directory = Path(session_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.seed, self.limit, self.classification = seed, max_queries, classification
        self.attempts = self.returned = self.errors = 0
        self.query_ids, self.closed, self.paused = set(), False, False
        self.session_id = uuid.uuid4().hex
        self.journal = self.directory / "oracle_attempts.jsonl"
        self.queries = self.directory / "queries"
        self.queries.mkdir(exist_ok=True)
        # A nonblocking process lock excludes concurrent owners. Kernel releases
        # it after a crash; durable per-query intents still prevent replay.
        import fcntl
        self._lock = (self.directory / ".oracle.lock").open("a+")
        fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with contextlib.redirect_stdout(sys.stderr):
            cls, self.DataSet, _, _, source = load_official(source_root)
            import numpy as np
            self.benchmark = cls(noise_level=0.0)
            self.benchmark.rng = np.random.default_rng(seed)
        actual = {v.name: [float(x) for x in v.bounds] for v in self.benchmark.domain.input_variables}
        if actual != {k: list(v) for k, v in BOUNDS.items()}:
            raise ValueError("Official input domain differs")
        self.identity = {"schema": "summit_snar_oracle_session_v1", "session_id": self.session_id,
                         "classification": classification, "scientific_main_result": classification == "benchmark",
                         "seed": seed, "noise_level": 0.0, "max_queries": max_queries, "protocol": protocol,
                         "source": source, "parameter_bounds": actual, "parameter_units": UNITS,
                         "raw_objectives": {"sty": {"maximize": True, "unit": "kg/m^3/h", "declared_bounds": [0, 13000], "floor": 1e-6, "upper_clip": None},
                                            "e_factor": {"maximize": False, "declared_bounds": [0, 500], "no_product_value": 1000, "upper_clip": 1000}},
                         "runtime": {"python": sys.version.split()[0], "executable": sys.executable,
                                     "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                     "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                                     "packages": {k: importlib.metadata.version(k) for k in ("numpy", "pandas", "scipy")}},
                         "query_cost_definition": "one attempted official run_experiments call containing exactly one parameter row",
                         "dft_calls": 0, "new_model_calls": 0}
        identity_path = self.directory / "identity.json"
        if identity_path.exists():
            previous = json.loads(identity_path.read_text())
            if {k: v for k, v in previous.items() if k != "session_id"} != {k: v for k, v in self.identity.items() if k != "session_id"}:
                self._lock.close()
                raise ValueError("Existing journal source/runtime/seed/protocol differs")
            if not self.journal.is_file():
                self._lock.close()
                raise NeedsReconciliation("Existing identity has no durable journal")
            self.identity, self.session_id = previous, previous["session_id"]
            self.closed = (self.directory / "summary.json").exists()
            self._refresh()
        else:
            if self.journal.exists() or any(self.queries.iterdir()):
                self._lock.close()
                raise NeedsReconciliation("Unidentified partial session cannot be resumed")
            unique_json(identity_path, self.identity)
            self.journal.touch(exist_ok=False)
            self.reserved_queries, self.pending = 0, []

    def _refresh(self):
        intents = sorted(self.queries.glob("*.intent.json"))
        events = [json.loads(line) for line in self.journal.read_text().splitlines()]
        starts = [e["attempt_id"] for e in events if e["event"] == "attempt_started"]
        if len(starts) != len(set(starts)):
            raise NeedsReconciliation("Duplicate started attempt evidence")
        intent_ids = {json.loads(p.read_text())["attempt_id"] for p in intents}
        if not set(starts) <= intent_ids:
            raise NeedsReconciliation("Started attempt has no immutable intent")
        if len(list(self.queries.glob("*.terminal.json"))) > len(intents):
            raise NeedsReconciliation("Orphan terminal evidence")
        terminals = []
        self.pending = []
        for path in intents:
            request = json.loads(path.read_text())
            terminal_path = path.with_name(path.name.replace(".intent.json", ".terminal.json"))
            if not terminal_path.exists():
                self.pending.append(request["query_id"])
                continue
            terminal = json.loads(terminal_path.read_text())
            if terminal["request_sha256"] != request["request_sha256"] or terminal["query_id"] != request["query_id"]:
                raise NeedsReconciliation("Query intent/terminal mismatch")
            if request["attempt_id"] not in starts or terminal["status"] not in {"succeeded", "failed"}:
                raise NeedsReconciliation("Terminal has no start or invalid status")
            terminals.append(terminal)
        self.reserved_queries = len(intents)
        self.attempts = len(starts)
        self.returned = sum(t["status"] == "succeeded" for t in terminals)
        self.errors = sum(t["status"] == "failed" for t in terminals)
        self.query_ids = {json.loads(p.read_text())["query_id"] for p in intents}
        return intents

    def event(self, kind, **fields):
        record = {"schema": "summit_snar_oracle_event_v1", "event": kind,
                  "session_id": self.session_id, "time_epoch": time.time(), **fields}
        with self.journal.open("a") as stream:
            stream.write(encode(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def evaluate(self, query_id, raw_parameters):
        if self.paused:
            raise RuntimeError("Paused owner cannot execute; reopen the durable journal")
        if not isinstance(query_id, str) or not query_id or len(query_id) > 256:
            raise ValueError("Nonempty query_id string required")
        point = parameters(raw_parameters)
        import hashlib
        request = {"query_id": query_id, "parameters": point, "seed": self.seed,
                   "noise_level": 0.0, "source_lock_sha256": self.identity["source"]["source_lock_sha256"]}
        request_sha = hashlib.sha256(encode(request).encode()).hexdigest()
        key = hashlib.sha256(query_id.encode()).hexdigest()
        intent = self.queries / (key + ".intent.json")
        terminal = self.queries / (key + ".terminal.json")
        if intent.exists():
            old = json.loads(intent.read_text())
            if old["request_sha256"] != request_sha:
                raise ValueError("query_id already binds different request bytes")
            if not terminal.exists():
                raise NeedsReconciliation("Pending query has no terminal; never rerun its ODE")
            saved = json.loads(terminal.read_text())
            if saved["status"] != "succeeded":
                raise NeedsReconciliation("Original query failed; no automatic retry")
            if not self.closed:
                self.event("cache_hit", query_id=query_id, request_sha256=request_sha,
                           physical_oracle_calls_this_request=0)
            return dict(saved["result"], cache_hit=True, physical_oracle_calls_this_request=0)
        if self.pending:
            raise NeedsReconciliation("A pending query must be reconciled before further work")
        if self.closed:
            raise RuntimeError("Session already closed")
        if self.reserved_queries >= self.limit:
            raise ValueError("Oracle query budget exhausted")
        conditions = self.DataSet([[point[name] for name in BOUNDS]], columns=list(BOUNDS))
        attempt_id = self.session_id + ":" + str(self.attempts + 1)
        unique_json(intent, dict(request, request_sha256=request_sha, attempt_id=attempt_id))
        self.reserved_queries += 1
        self.pending.append(query_id)
        self.event("attempt_started", attempt_id=attempt_id, query_id=query_id, parameters=point, request_sha256=request_sha)
        self.query_ids.add(query_id)
        self.attempts += 1
        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                out = self.benchmark.run_experiments(conditions)
            values = {name: float(out[name].iloc[-1]) for name in ("sty", "e_factor")}
            if not all(math.isfinite(x) for x in values.values()):
                raise ValueError("Official oracle returned a nonfinite objective")
        except BaseException as exc:
            self.errors += 1
            if not terminal.exists():
                unique_json(terminal, {"query_id": query_id, "request_sha256": request_sha,
                                       "status": "failed", "error_type": type(exc).__name__,
                                       "error": str(exc), "traceback": traceback.format_exc()})
            self.event("attempt_error", attempt_id=attempt_id, query_id=query_id,
                       error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                       elapsed_seconds=time.perf_counter() - started)
            self.pending.remove(query_id)
            raise
        result = {"query_id": query_id, "parameters": point, "objectives": values,
                  "oracle_attempt_id": attempt_id, "oracle_attempts": self.attempts,
                  "elapsed_seconds": time.perf_counter() - started, "noise_level": 0.0, "seed": self.seed,
                  "cache_hit": False, "physical_oracle_calls_this_request": 1,
                  "request_sha256": request_sha}
        # Persistence failures after a successful ODE must not be relabeled as
        # physical failures or cause hidden reruns. The intent remains durable.
        unique_json(terminal, {"query_id": query_id, "request_sha256": request_sha,
                               "status": "succeeded", "result": result})
        self.event("attempt_returned", attempt_id=attempt_id, **result)
        self.returned += 1
        self.pending.remove(query_id)
        return result

    def pause(self, reason="requested"):
        if not self.paused:
            self._refresh()
            if not self.closed:
                self.event("session_paused", reason=reason, oracle_attempts=self.attempts,
                           pending_query_ids=self.pending)
            self.paused = True
            self._lock.close()
        return {"schema": "summit_snar_oracle_pause_v1", "paused": True,
                "already_closed": self.closed, "session_id": self.session_id,
                "oracle_attempts": self.attempts, "reserved_queries": self.reserved_queries,
                "pending_query_ids": self.pending, "new_oracle_calls": 0,
                "journal_sha256": file_sha(self.journal)}

    def close(self, reason="requested"):
        if self.paused:
            raise RuntimeError("Paused owner cannot finalize a journal owned by a later process")
        if not self.closed:
            self._refresh()
            self.closed = True
            self.event("session_closed", reason=reason, oracle_attempts=self.attempts,
                       returned=self.returned, errors=self.errors)
            value = {"schema": "summit_snar_oracle_summary_v1", "classification": self.classification,
                     "scientific_main_result": self.classification == "benchmark", "status": "closed",
                     "session_id": self.session_id, "oracle_attempts": self.attempts,
                     "completed_queries": self.returned, "errors": self.errors,
                     "unknown_outcomes": len(self.pending),
                     "reserved_queries": self.reserved_queries,
                     "reserved_without_started_event": self.reserved_queries - self.attempts,
                     "pending_query_ids": self.pending,
                     "budget_exhausted": self.attempts == self.limit,
                     "elapsed_seconds": time.time() - self.started, "dft_calls": 0,
                     "identity_sha256": file_sha(self.directory / "identity.json"),
                     "journal_sha256": file_sha(self.journal)}
            unique_json(self.directory / "summary.json", value)
        self._lock.close()
        return json.loads((self.directory / "summary.json").read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", "--session-dir", dest="session_dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--max-queries", type=int, required=True)
    parser.add_argument("--classification", choices=["technical_not_main", "benchmark"], required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    oracle = None
    try:
        for line in sys.stdin:
            request = None
            try:
                def reject_constant(value):
                    raise ValueError("Non-JSON numeric constant: " + value)
                def no_duplicate_keys(pairs):
                    obj = {}
                    for key, value in pairs:
                        if key in obj:
                            raise ValueError("Duplicate JSON key: " + key)
                        obj[key] = value
                    return obj
                request = json.loads(line, parse_constant=reject_constant, object_pairs_hook=no_duplicate_keys)
                if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or set(request) != {"jsonrpc", "id", "method", "params"}:
                    raise ValueError("Strict JSON-RPC2.0 envelope required")
                if request["method"] == "initialize" and isinstance(request["params"], dict) and set(request["params"]) == {"seed", "noise_level"}:
                    if type(request["params"]["noise_level"]) not in (int, float) or request["params"]["noise_level"] != 0:
                        raise ValueError("Frozen noise_level must be zero")
                    if oracle is None:
                        oracle = SnarOracle(args.session_dir, seed=request["params"]["seed"], max_queries=args.max_queries,
                                            classification=args.classification, protocol_path=args.protocol,
                                            protocol_sha256=args.protocol_sha256, source_root=args.source_root)
                    elif request["params"]["seed"] != oracle.seed:
                        raise ValueError("Initialized seed differs")
                    result = oracle.identity
                elif oracle is not None and request["method"] == "evaluate" and isinstance(request["params"], dict) and set(request["params"]) == {"query_id", "parameters"}:
                    result = oracle.evaluate(request["params"]["query_id"], request["params"]["parameters"])
                elif oracle is not None and request["method"] == "close" and request["params"] == {}:
                    result = oracle.close()
                elif oracle is not None and request["method"] == "pause" and request["params"] == {}:
                    result = oracle.pause()
                else:
                    raise ValueError("Unknown method or argument fields")
                response = {"jsonrpc": "2.0", "id": request["id"], "result": result}
            except Exception as exc:
                response = {"jsonrpc": "2.0", "id": request.get("id") if isinstance(request, dict) else None,
                            "error": {"type": type(exc).__name__, "message": str(exc)}}
                if oracle is not None and not oracle.closed:
                    oracle.event("request_rejected_or_failed", rpc_id=response["id"], error=response["error"], oracle_attempts=oracle.attempts)
                else:
                    print(traceback.format_exc(), file=sys.stderr, flush=True)
            print(encode(response), flush=True)
            if oracle is not None and isinstance(request, dict) and request.get("method") in {"close", "pause"} and "result" in response:
                break
    finally:
        if oracle is not None:
            if not oracle.closed and not oracle.paused:
                oracle.pause(reason="stdin_eof")
            else:
                oracle._lock.close()


if __name__ == "__main__":
    main()
