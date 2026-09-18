"""Audited scheduling around the pinned MADE Oracle, without changing numerics.

The original ``Oracle.batch_evaluate`` still owns chunking, order and exceptions.
For MACE/4, all private calculators are constructed on the calling thread before
any worker evaluates a structure. This prevents our constructors' process-global
Torch FX patches from overlapping our numerical forwards. No scientific packages
are imported until a real parallel MACE adapter requests ``TorchFXGuard``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time
import traceback
from typing import Any, Callable
import uuid


def audit_oracle_attempt_journal(path) -> dict[str, Any]:
    """Read-only accounting and closure check; malformed evidence never means zero.

    ``valid`` concerns journal integrity. ``closed`` also requires every started
    attempt, constructor and batch to have one terminal. ``successful`` additionally
    excludes recorded failures. Counts include *invoked* exceptions, but exclude
    constructor failures. Callers must compare ``counts`` with their RPC/episode
    receipt and planned budget; this function does not infer those external totals.
    ``counts_reliable=False`` means the reported unique-start totals are diagnostic,
    not an accepted physical-cost total. The function never writes or retries.
    """
    path = Path(path)
    errors, rows = [], []
    digest = hashlib.sha256()
    def invalid(message):
        errors.append(message)
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key " + key)
            value[key] = item
        return value
    try:
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                digest.update(raw)
                try:
                    if not raw.endswith(b"\n"):
                        raise ValueError("incomplete JSONL line")
                    row = json.loads(raw, object_pairs_hook=pairs,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON " + value)))
                    if not isinstance(row, dict):
                        raise ValueError("record is not an object")
                    rows.append(row)
                except (ValueError, UnicodeError) as exc:
                    invalid(f"line {line_number}: {exc}")
    except OSError as exc:
        invalid(f"journal unreadable: {type(exc).__name__}: {exc}")

    starts, terminals, batches, batch_terminals = {}, {}, {}, {}
    constructors, constructor_terminals, ready = {}, {}, {}
    run_ids, start_record_count = set(), 0
    positions = {id(row): position for position, row in enumerate(rows)}
    context_keys = ("role", "phase", "episode_index", "batch_id", "batch_input_index",
                    "batch_input_count", "candidate_hash", "counter", "num_workers", "index_scope")
    known = {"oracle_batch_started", "oracle_batch_finished", "oracle_batch_exception",
             "oracle_constructor_started", "oracle_constructor_finished", "oracle_constructor_exception",
             "calculator_constructor_started", "calculator_constructor_finished", "calculator_constructor_exception",
             "all_calculators_ready", "calculator_assigned", "oracle_attempt_started",
             "oracle_attempt_returned", "oracle_attempt_exception"}
    for sequence, row in enumerate(rows):
        kind = row.get("kind")
        if not isinstance(kind, str) or kind not in known:
            invalid(f"record {sequence}: unknown schema/kind")
            continue
        if row.get("schema") != "matdiscovery.oracle_attempts.v1":
            invalid(f"record {sequence}: unknown schema/kind")
        if type(row.get("sequence")) is not int or row["sequence"] != sequence:
            invalid(f"record {sequence}: noncontiguous sequence")
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            invalid(f"record {sequence}: invalid run_id")
        else:
            run_ids.add(run_id)
        if type(row.get("episode_index")) is not int or row["episode_index"] < 0:
            invalid(f"record {sequence}: invalid episode")
        if kind in {"oracle_batch_started", "oracle_batch_finished", "oracle_batch_exception"}:
            key = row.get("batch_id")
            target = batches if kind == "oracle_batch_started" else batch_terminals
            if not isinstance(key, str) or not key or key in target:
                invalid(f"record {sequence}: invalid/duplicate batch identity")
            else:
                target[key] = row
            if kind != "oracle_batch_started" and (not isinstance(key, str) or key not in batches):
                invalid(f"record {sequence}: batch terminal precedes start")
        elif kind in {"oracle_constructor_started", "oracle_constructor_finished", "oracle_constructor_exception",
                      "calculator_constructor_started", "calculator_constructor_finished", "calculator_constructor_exception"}:
            if kind.startswith("oracle_"):
                key = ("initial", row.get("constructor_id"))
                if not isinstance(key[1], str) or not key[1]:
                    invalid(f"record {sequence}: invalid constructor identity")
            else:
                key = ("batch", row.get("batch_id"), row.get("slot"))
                if not isinstance(key[1], str) or key[1] not in batches or type(key[2]) is not int or key[2] < 0:
                    invalid(f"record {sequence}: invalid batch constructor identity")
            # JSON values may be malformed/unhashable. Reject without losing the
            # remainder of the accounting report to a parser exception.
            key = repr(key)
            target = constructors if kind.endswith("_started") else constructor_terminals
            if key in target:
                invalid(f"record {sequence}: duplicate constructor start/terminal")
            else:
                target[key] = row
            if not kind.endswith("_started") and key not in constructors:
                invalid(f"record {sequence}: constructor terminal precedes start")
        elif kind == "all_calculators_ready":
            key = row.get("batch_id")
            if not isinstance(key, str) or key not in batches or key in ready:
                invalid(f"record {sequence}: invalid/duplicate ready batch")
            else:
                ready[key] = row
        elif kind in {"oracle_attempt_started", "oracle_attempt_returned", "oracle_attempt_exception"}:
            key = row.get("attempt_id")
            target = starts if kind == "oracle_attempt_started" else terminals
            if kind == "oracle_attempt_started":
                if key != f"{run_id}:{start_record_count}":
                    invalid(f"record {sequence}: unexpected attempt ID sequence")
                start_record_count += 1
            if not isinstance(key, str) or not key or key in target:
                invalid(f"record {sequence}: invalid/duplicate attempt start/terminal")
                continue
            target[key] = row
            if kind != "oracle_attempt_started":
                if kind == "oracle_attempt_returned" and not isinstance(row.get("result"), dict):
                    invalid(f"record {sequence}: returned attempt lacks raw result dictionary")
                if kind == "oracle_attempt_exception" and not all(isinstance(row.get(field), str) and row[field]
                                                                   for field in ("error_type", "traceback")):
                    invalid(f"record {sequence}: failed attempt lacks exception evidence")
                if key not in starts:
                    invalid(f"record {sequence}: attempt terminal precedes start")
                elif any(row.get(field) != starts[key].get(field) for field in context_keys):
                    invalid(f"record {sequence}: attempt terminal identity/counter mismatch")
    if len(run_ids) > 1:
        invalid("Journal mixes multiple adapter run IDs")

    counts, groups = {}, {}
    for attempt_id, row in starts.items():
        role, phase, episode = row.get("role"), row.get("phase"), row.get("episode_index")
        expected_counter = "surrogate_oracle_attempts" if role == "mace" else f"{phase}_oracle_attempts"
        if (role not in ("mace", "orb") or phase not in ("initialization", "candidate")
                or row.get("counter") != expected_counter):
            invalid(f"attempt {attempt_id}: invalid role/phase/counter mapping")
        if role in ("mace", "orb") and phase in ("initialization", "candidate"):
            counts[expected_counter] = counts.get(expected_counter, 0) + 1
        batch_id = row.get("batch_id")
        batch = batches.get(batch_id) if isinstance(batch_id, str) else None
        index, size = row.get("batch_input_index"), row.get("batch_input_count")
        if (batch is None or row.get("role") != batch.get("role")
                or row.get("num_workers") != batch.get("num_workers")
                or size != batch.get("input_count") or type(size) is not int
                or type(index) is not int or not 0 <= index < size
                or not isinstance(row.get("candidate_hash"), str) or not row["candidate_hash"]):
            invalid(f"attempt {attempt_id}: invalid batch/input identity")
        group_key = (str(role), str(phase), str(episode))
        group = groups.setdefault(group_key, {"role": role, "phase": phase, "episode_index": episode,
                                             "attempts": 0, "returned": 0, "exceptions": 0, "unknown": 0})
        group["attempts"] += 1
        terminal = terminals.get(attempt_id)
        group["unknown" if terminal is None else "returned" if terminal["kind"] == "oracle_attempt_returned" else "exceptions"] += 1

    for batch_id, batch in batches.items():
        if (batch.get("role") not in ("mace", "orb") or type(batch.get("num_workers")) is not int
                or batch["num_workers"] not in (1, 4)
                or (batch.get("role") == "orb" and batch["num_workers"] != 1)):
            invalid(f"batch {batch_id}: invalid role/worker configuration")
        batch_attempts = [r for r in starts.values() if r.get("batch_id") == batch_id]
        indices = [r.get("batch_input_index") for r in batch_attempts]
        if len({repr(i) for i in indices}) != len(indices):
            invalid(f"batch {batch_id}: duplicate input index invocations")
        terminal = batch_terminals.get(batch_id)
        if terminal:
            if any(r["attempt_id"] not in terminals or
                   positions[id(terminals[r["attempt_id"]])] >= positions[id(terminal)] for r in batch_attempts):
                invalid(f"batch {batch_id}: terminal precedes joined oracle outcomes")
            if (terminal.get("role") != batch.get("role")
                    or terminal.get("num_workers") != batch.get("num_workers")):
                invalid(f"batch {batch_id}: terminal role/worker mismatch")
            if terminal["kind"] == "oracle_batch_finished":
                if (type(batch.get("input_count")) is not int or terminal.get("output_count") != batch["input_count"]
                        or len(batch_attempts) != batch["input_count"]
                        or any(terminals.get(r["attempt_id"], {}).get("kind") != "oracle_attempt_returned" for r in batch_attempts)):
                    invalid(f"batch {batch_id}: successful batch count/terminal mismatch")
            elif terminal.get("stage") in ("input_identity", "calculator_construction") and batch_attempts:
                invalid(f"batch {batch_id}: constructor/preflight failure incorrectly counted as invocation")
        if batch.get("num_workers") == 4 and batch_attempts:
            marker = ready.get(batch_id)
            completed = [r for r in constructor_terminals.values()
                         if r.get("batch_id") == batch_id and r["kind"] == "calculator_constructor_finished"]
            expected = official_chunk_count(batch.get("input_count", 0), 4) if type(batch.get("input_count")) is int and batch["input_count"] > 0 else -1
            if (marker is None or marker.get("count") != expected or len(completed) != expected
                    or marker.get("fx_globals_restored") is not True
                    or any(r.get("fx_globals_restored") is not True for r in completed)
                    or (marker and (any(positions[id(r)] >= positions[id(marker)] for r in completed)
                                    or any(positions[id(r)] <= positions[id(marker)] for r in batch_attempts)))):
                invalid(f"batch {batch_id}: invalid construction/forward phase boundary")

    unknown = sorted(set(starts) - set(terminals))
    constructor_unknown = sorted(set(constructors) - set(constructor_terminals))
    batch_unknown = sorted(set(batches) - set(batch_terminals))
    oracle_errors = [r for r in terminals.values() if r["kind"] == "oracle_attempt_exception"]
    constructor_errors = [r for r in constructor_terminals.values() if r["kind"].endswith("_exception")]
    batch_errors = [r for r in batch_terminals.values() if r["kind"] == "oracle_batch_exception"]
    valid = not errors
    closed = valid and not (unknown or constructor_unknown or batch_unknown)
    return {"schema": "matdiscovery.oracle_attempt_audit.v1", "path": str(path),
            "sha256": digest.hexdigest(), "valid": valid, "closed": closed,
            "successful": closed and not (oracle_errors or constructor_errors or batch_errors),
            "counts_reliable": valid, "counts": counts, "by_role_phase_episode": list(groups.values()),
            "attempts_started": len(starts), "start_record_count": start_record_count,
            "unique_ids": len(starts), "attempt_ids": list(starts), "starts": list(starts.values()),
            "terminals": list(terminals.values()), "unknown_attempt_ids": unknown,
            "unknown_constructor_ids": constructor_unknown, "unknown_batch_ids": batch_unknown,
            "oracle_errors": oracle_errors, "constructor_errors": constructor_errors,
            "constructor_counts": {"started": len(constructors),
                                   "finished": len(constructor_terminals) - len(constructor_errors),
                                   "exceptions": len(constructor_errors), "unknown": len(constructor_unknown)},
            "batch_errors": batch_errors, "errors": errors, "record_count": len(rows)}


# Not an audit/logging lock: workers do not acquire it. It serializes complete
# constructor/forward lifecycles across adapter instances within this process.
ORACLE_LIFECYCLE_LOCK = threading.RLock()


def mace_execution_metadata(num_workers: int) -> dict[str, Any]:
    if type(num_workers) is not int or num_workers not in (1, 4):
        raise ValueError("mace_num_workers must be the integer 1 or 4")
    return {
        "executor": "made.Oracle.batch_evaluate",
        "num_workers": num_workers,
        "lifecycle": ("original_single_calculator" if num_workers == 1 else
                      "serial_private_construction_then_parallel_evaluation_v2"),
        "composition_grouping": "unchanged_one_composition_per_scorer_call",
        "constructor_forward_overlap": False,
        "exception_execution_set_equivalent_to_serial": num_workers == 1,
        "attempt_journal": "oracle_attempts.jsonl",
        "attempt_journal_schema": "matdiscovery.oracle_attempts.v1",
    }


def official_chunk_count(size: int, workers: int) -> int:
    """Number of tasks actually submitted by MADE bffda0c's ceiling chunk rule."""
    if size == 0:
        return 0
    groups = min(workers, size)
    chunk_size = (size + groups - 1) // groups
    return (size + chunk_size - 1) // chunk_size


class TorchFXGuard:
    """Check restored FX globals and the already fixed CPU/FP32 MACE parameters."""

    def __init__(self):
        import torch
        import torch.fx._symbolic_trace as symbolic_trace

        self.torch, self.symbolic_trace = torch, symbolic_trace
        self.original_call = torch.nn.Module.__call__
        self.original_getattr = torch.nn.Module.__getattr__
        self.check()

    def check(self, calculator=None) -> None:
        torch, fx = self.torch, self.symbolic_trace
        if (torch.nn.Module.__call__ is not self.original_call
                or torch.nn.Module.__getattr__ is not self.original_getattr
                or fx.CURRENT_PATCHER is not None or fx._is_fx_tracing_flag is not False):
            raise RuntimeError("Torch FX globals are not restored; oracle evaluation forbidden")
        if calculator is not None:
            if torch.get_default_dtype() != torch.float32:
                raise RuntimeError("Parallel MACE did not retain the fixed float32 default dtype")
            models = calculator.models
            if not models:
                raise RuntimeError("MACE calculator has no models to verify")
            for model in models:
                for parameter in model.parameters():
                    if (parameter.device.type != "cpu" or
                            (parameter.is_floating_point() and parameter.dtype != torch.float32)):
                        raise RuntimeError("Parallel MACE requires the verified CPU/float32 placement")


class PreparedCalculatorBatch:
    """A fresh private calculator per participating thread; no worker constructs."""

    def __init__(self, factory: Callable, count: int, *, batch_id: str,
                 record: Callable, guard: Any, forbidden_identities=()):
        if type(count) is not int or count < 1:
            raise ValueError("Positive calculator count required")
        self.batch_id, self.count, self.record = batch_id, count, record
        self._lock = threading.Lock()
        self._assignments: dict[int, tuple[int, Any]] = {}
        calculators, identities = [], set()
        forbidden_identities = set(forbidden_identities)
        started = time.monotonic()
        guard.check()
        for slot in range(count):
            record("calculator_constructor_started", batch_id=batch_id, slot=slot)
            before = time.monotonic()
            try:
                calculator = factory()
                if id(calculator) in identities or id(calculator) in forbidden_identities:
                    raise RuntimeError("Calculator factory returned a shared object")
                guard.check(calculator)
            except BaseException as exc:
                record("calculator_constructor_exception", batch_id=batch_id, slot=slot,
                       elapsed_seconds=time.monotonic() - before,
                       error_type=type(exc).__name__, message=str(exc),
                       traceback=traceback.format_exc())
                raise
            identities.add(id(calculator))
            calculators.append((slot, calculator))
            record("calculator_constructor_finished", batch_id=batch_id, slot=slot,
                   object_identity=id(calculator), fx_globals_restored=True,
                   elapsed_seconds=time.monotonic() - before)
        guard.check()
        self.constructor_seconds = time.monotonic() - started
        self._available = deque(calculators)
        record("all_calculators_ready", batch_id=batch_id, count=count,
               unique_objects=len(identities), constructor_seconds=self.constructor_seconds,
               fx_globals_restored=True)

    def acquire(self):
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id not in self._assignments:
                if not self._available:
                    raise RuntimeError("More worker threads than preconstructed calculators")
                slot, calculator = self._available.popleft()
                self._assignments[thread_id] = (slot, calculator)
                self.record("calculator_assigned", batch_id=self.batch_id, slot=slot,
                            object_identity=id(calculator))
            return self._assignments[thread_id][1]

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            return {"prepared": self.count, "assigned": len(self._assignments),
                    "unused_prepared": len(self._available),
                    "constructor_seconds": self.constructor_seconds,
                    "constructor_forward_overlap": False}


@dataclass(frozen=True)
class _BatchItem:
    # Only the pinned scheduler sees this envelope (it slices and iterates).
    # Native evaluate always receives the original Structure object. Explicit
    # indices also distinguish duplicate objects/hashes without mutating inputs.
    structure: Any
    context: dict[str, Any]
    owner: Any


class AuditedOracleExecutor:
    """Install on a newly constructed oracle; audit callback owns exact counts.

    ``invoke(original_evaluate, original_structure, **context)`` must durably log
    a start and terminal record and never put numerical work under its log lock.
    An official parallel exception may leave extra attempts in other chunks;
    the native executor joins all workers before the failed batch is recorded.
    """

    def __init__(self, oracle, *, role: str, candidate_hash: Callable,
                 invoke: Callable, record: Callable, guard=None):
        if role not in {"mace", "orb"}:
            raise ValueError("Unknown oracle role")
        workers = oracle.num_workers
        if type(workers) is not int or workers not in (1, 4) or (role == "orb" and workers != 1):
            raise ValueError("ORB requires one worker; MACE supports one or four")
        if workers == 4 and guard is None:
            raise ValueError("Parallel MACE requires an explicit FX/parameter guard")
        self.oracle, self.role, self.workers = oracle, role, workers
        self.hash, self.invoke, self.record, self.guard = candidate_hash, invoke, record, guard
        self._native_evaluate = oracle.evaluate
        self._native_batch = oracle.batch_evaluate
        self._native_factory = getattr(oracle, "_calculator_factory", None)
        if workers == 4 and not callable(self._native_factory):
            raise ValueError("Parallel MACE requires the official calculator factory")
        self._owner = object()
        oracle.evaluate = self.evaluate
        oracle.batch_evaluate = self.batch_evaluate

    def evaluate(self, structure):
        if isinstance(structure, _BatchItem):
            if structure.owner is not self._owner:
                raise RuntimeError("Unexpected oracle batch context")
            return self.invoke(self._native_evaluate, structure.structure, **structure.context)
        # Official elemental initialization uses direct evaluate. Give it the
        # same durable evidence and, for MACE/4, the same two-phase lifecycle.
        return self.batch_evaluate([structure])[0]

    def batch_evaluate(self, structures):
        with ORACLE_LIFECYCLE_LOCK:
            if self.oracle.num_workers != self.workers:
                raise RuntimeError("Oracle worker count changed after installation")
            if not structures:
                return self._native_batch(structures)
            batch_id = uuid.uuid4().hex
            started = time.monotonic()
            pool = None
            stage = "input_identity"
            self.record("oracle_batch_started", batch_id=batch_id, role=self.role,
                        num_workers=self.workers, input_count=len(structures))
            try:
                tagged = [_BatchItem(s, {"batch_id": batch_id, "role": self.role,
                          "num_workers": self.workers, "batch_input_index": i,
                          "batch_input_count": len(structures),
                          "index_scope": "official_oracle_batch_after_scorer_cache_filter",
                          "candidate_hash": self.hash(s)},
                          self._owner) for i, s in enumerate(structures)]
                if self.workers == 4:
                    stage = "calculator_construction"
                    pool = PreparedCalculatorBatch(self._native_factory,
                        official_chunk_count(len(structures), self.workers),
                        batch_id=batch_id, record=self.record, guard=self.guard,
                        forbidden_identities=(id(self.oracle.calculator),))
                    self.oracle._thread_local = threading.local()
                    self.oracle._calculator_factory = pool.acquire
                stage = "evaluation"
                results = self._native_batch(tagged)
            except BaseException as exc:
                self.record("oracle_batch_exception", batch_id=batch_id, role=self.role,
                            num_workers=self.workers, stage=stage,
                            elapsed_seconds=time.monotonic() - started,
                            error_type=type(exc).__name__, message=str(exc),
                            traceback=traceback.format_exc(),
                            construction=pool.metadata() if pool else None,
                            parallel_workers_joined=stage == "evaluation",
                            exception_execution_set_equivalent_to_serial=self.workers == 1)
                raise
            else:
                self.record("oracle_batch_finished", batch_id=batch_id, role=self.role,
                            num_workers=self.workers, output_count=len(results),
                            elapsed_seconds=time.monotonic() - started,
                            construction=pool.metadata() if pool else None)
                return results
            finally:
                if self.workers == 4:
                    self.oracle._calculator_factory = self._native_factory
                    # Native ThreadPoolExecutor.__exit__ has joined every worker,
                    # including on exception. Do not retain completed batch models.
                    self.oracle._thread_local = threading.local()
