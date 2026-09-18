"""Dependency-lazy adapters around fresh official MADE and CrystalGym code.

Init requires absolute vendor_root/work_dir, seed, budget and pinned assets.
MADE additionally requires mp_cache_path and mp_cache_sha256. Snapshot schema:
source='materials_project', elements, thermo_types=['GGA_GGA+U'], complete=true,
entries=[ComputedStructureEntry MSON or PDEntry with attribute.structure].

MADE budget counts candidate oracle *attempts*, including exceptions, per episode.
CrystalGym requires budget_unit='atomic_action_attempts' or 'dft_episode_attempts';
that session budget persists across reset. All raw official scientific failures
are distinct from RPC/step exceptions. Models are never replaced with mock values.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any
import uuid

from matdiscovery.mace_parallel import (
    AuditedOracleExecutor, ORACLE_LIFECYCLE_LOCK, TorchFXGuard,
    mace_execution_metadata,
)


MADE_COMMIT = "bffda0c9a1fa1904b35c0ef67b455881c975df86"
CRYSTALGYM_COMMIT = "d4191966ea6662f60d6f58f84e3c2e1457750075"
CRYSTAL_VAL_SHA256 = "7b5f4085464b3eac9fd2123acee97ec372214004ce7e983b535b33395abc3e29"
PSEUDODICT_SHA256 = "82d4ed34ed6e3d39f25bf90a917ad1b18dd582882e57771c5bd54d0132fa79bd"
TRAIN_INDICES = [630, 2271, 8354, 8666, 8906]
HELDOUT_INDICES = [3403, 2190]
SMALL_ELEMENTS = ["Li", "Na", "K", "Rb", "Be", "Ca", "Mg", "Sr", "H", "C", "N", "O", "P", "S", "Se", "F", "Cl", "Br"]
RELAX = {"optimizer": "fire", "fmax": 0.02, "steps": 500, "relax_unit_cell": True}


def jsonable(value: Any) -> Any:
    """Preserve nonfinite evidence explicitly instead of emitting invalid JSON NaN."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "detach"):
        return jsonable(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "as_dict"):
        return jsonable(value.as_dict())
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


class AdapterError(Exception):
    def __init__(self, code: str, message: str, details: Any = None):
        super().__init__(message)
        self.code, self.details = code, details or {}

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "type": type(self).__name__, "message": str(self), "details": jsonable(self.details)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def absolute_path(value: Any, label: str, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise AdapterError("invalid_path", f"{label} must be an absolute path.")
    path = Path(value).resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise AdapterError("missing_asset", f"{label} does not exist: {path}")
    return path


def asset(value: Any, label: str, expected: str | None = None) -> dict[str, Any]:
    spec = value if isinstance(value, dict) else {"path": value}
    path = absolute_path(spec.get("path"), label)
    digest = file_sha256(path)
    expected = expected or spec.get("sha256")
    if expected and digest != expected:
        raise AdapterError("asset_hash_mismatch", f"SHA256 mismatch for {label}", {"path": str(path), "expected": expected, "actual": digest})
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AdapterError("invalid_config", f"{label} must be a positive integer.")
    return value


def score_rankable(value: float) -> bool:
    """Official negative-infinity failure sentinels remain rankable last."""
    return math.isfinite(value) or value == -math.inf


def score_order(value: float, descending: bool = True) -> tuple[int, float]:
    # Invalid NaN/+inf never win or precede valid entries, including bottom queries.
    return (0, -value if descending else value) if score_rankable(value) else (1, 0.0)


def seed_all(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def verify_vendor(root: Path, package: str, commit: str) -> dict[str, str]:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode or result.stdout.strip() != commit:
        raise AdapterError("vendor_revision_mismatch", f"Expected fresh {package} commit {commit}.")
    dirty = subprocess.run(["git", "-C", str(root), "diff", "--name-only", "HEAD"], capture_output=True, text=True, check=True)
    if dirty.stdout.strip():
        raise AdapterError("vendor_modified", "Tracked official source has modifications.", {"files": dirty.stdout.splitlines()})
    sys.path.insert(0, str(root / "src" if package == "made" else root))
    return {"root": str(root), "commit": commit}


def verify_import(module_name: str, root: Path) -> Any:
    module = importlib.import_module(module_name)
    location = Path(module.__file__).resolve()
    if root not in location.parents:
        raise AdapterError("wrong_vendor_import", f"{module_name} loaded outside fresh vendor root: {location}")
    return module


def load_snapshot(path_value: Any, digest: Any, elements: list[str]) -> tuple[dict, dict]:
    if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
        raise AdapterError("snapshot_not_frozen", "mp_cache_sha256 must explicitly freeze the raw MP snapshot.")
    info = asset(path_value, "mp_cache_path", digest.lower())
    with open(info["path"], encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("source") != "materials_project" or payload.get("complete") is not True:
        raise AdapterError("invalid_snapshot", "Snapshot must identify complete materials_project raw data.")
    if sorted(payload.get("elements", [])) != sorted(elements) or payload.get("thermo_types") != ["GGA_GGA+U"]:
        raise AdapterError("snapshot_task_mismatch", "Snapshot chemistry/thermo_types does not match the fixed task.")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise AdapterError("invalid_snapshot", "Snapshot has no official MSON entries.")
    for entry in entries:
        structure = entry.get("structure") if isinstance(entry, dict) else None
        if not structure and isinstance(entry, dict) and isinstance(entry.get("attribute"), dict):
            structure = entry["attribute"].get("structure")
        if not isinstance(structure, dict) or not structure.get("sites"):
            raise AdapterError("invalid_snapshot", "Every raw entry must contain a crystal structure; scalar/candidate tables are not supported.")
    info.update(entry_count=len(entries), retrieved_at=payload.get("retrieved_at"), database_version=payload.get("database_version"))
    return payload, info


def serialized_operation(method):
    """Serialize public environment mutations, separately from the audit lock."""
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._operation_lock:
            return method(self, *args, **kwargs)
    return wrapped


class BaseAdapter:
    name = "base"

    def __init__(self, args: dict[str, Any]):
        self.args = copy.deepcopy(args)
        self.vendor_root = absolute_path(args.get("vendor_root"), "vendor_root", directory=True)
        work = args.get("work_dir")
        if not isinstance(work, str) or not Path(work).is_absolute():
            raise AdapterError("invalid_path", "work_dir must be an absolute new-project path.")
        self.work_dir = Path(work).resolve()
        if self.work_dir == self.vendor_root or self.vendor_root in self.work_dir.parents:
            raise AdapterError("invalid_path", "Runtime outputs must be outside the fresh vendor checkout.")
        self.seed = args.get("seed")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise AdapterError("invalid_config", "seed must be an integer in [0, 2**32).")
        self.budget = positive_int(args.get("budget"), "budget")
        self._state_lock = threading.RLock()
        self._operation_lock = threading.RLock()
        self._audit_run_id = uuid.uuid4().hex
        self._audit_sequence = self._attempt_sequence = 0
        self._pending_oracle_attempts: dict[str, dict] = {}
        self._oracle_terminals = {"returned": 0, "exception": 0}
        self.oracle_audit_records: list[dict] = []
        self.episode_index = 0
        self.counts: dict[str, int] = {}
        self.episode_counts: dict[str, int] = {}
        self.events: list[dict] = []
        self.event_sequence = 0
        self.metadata: dict[str, Any] = {}
        self.closed = self.faulted = False
        self._phase = "initialization"

    def start_output(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.event_file = self.work_dir / "env_events.jsonl"
        self.oracle_attempt_file = self.work_dir / "oracle_attempts.jsonl"
        if any(path.exists() and path.stat().st_size for path in (self.event_file, self.oracle_attempt_file)):
            raise AdapterError("output_collision", "Use a fresh work_dir; existing environment events cannot be overwritten.")
        os.chdir(self.work_dir)

    def count(self, key: str) -> int:
        with self._state_lock:
            self.counts[key] = self.counts.get(key, 0) + 1
            self.episode_counts[key] = self.episode_counts.get(key, 0) + 1
            return self.counts[key]

    def count_value(self, key: str, *, episode: bool = False) -> int:
        with self._state_lock:
            return (self.episode_counts if episode else self.counts).get(key, 0)

    def event(self, kind: str, **fields: Any) -> dict:
        converted = jsonable(fields)
        with self._state_lock:
            record = {"sequence": self.event_sequence, "episode_index": self.episode_index, "kind": kind, **converted}
            self.event_sequence += 1
            self.events.append(record)
            del self.events[:-128]  # Full evidence is durable JSONL, not an unbounded training-time list.
            if hasattr(self, "event_file"):
                with self.event_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            return copy.deepcopy(record)

    def oracle_audit(self, kind: str, **fields: Any) -> dict:
        """Private durable lifecycle evidence, absent from policy-visible events."""
        converted = jsonable(fields)
        with self._state_lock:
            record = {"schema": "matdiscovery.oracle_attempts.v1",
                      "run_id": self._audit_run_id, "sequence": self._audit_sequence,
                      "episode_index": self.episode_index, "kind": kind,
                      "unix_time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                      "thread_id": threading.get_ident(), "pid": os.getpid(), **converted}
            if hasattr(self, "oracle_attempt_file"):
                with self.oracle_attempt_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            self._audit_sequence += 1
            self.oracle_audit_records.append(record)
            del self.oracle_audit_records[:-128]
            return copy.deepcopy(record)

    def oracle_audit_snapshot(self) -> dict:
        with self._state_lock:
            return {"run_id": self._audit_run_id, "attempts_started": self._attempt_sequence,
                    "terminals": dict(self._oracle_terminals),
                    "pending": copy.deepcopy(self._pending_oracle_attempts),
                    "counts": dict(self.counts), "episode_counts": dict(self.episode_counts),
                    "recent_records": copy.deepcopy(self.oracle_audit_records)}

    def invoke_oracle(self, evaluate, structure, **context):
        """One attempted native call, including exceptions; no numerical lock."""
        with self._state_lock:
            role, phase = context["role"], self._phase
            key = "surrogate_oracle_attempts" if role == "mace" else f"{phase}_oracle_attempts"
            attempt_id = f"{self._audit_run_id}:{self._attempt_sequence}"
            details = {**context, "attempt_id": attempt_id, "phase": phase, "counter": key}
            self.oracle_audit("oracle_attempt_started", **details)
            self._attempt_sequence += 1
            self.count(key)
            self._pending_oracle_attempts[attempt_id] = details
        started = time.monotonic()
        try:
            result = evaluate(structure)
        except BaseException as exc:
            elapsed = time.monotonic() - started
            with self._state_lock:
                self.oracle_audit("oracle_attempt_exception", **details, elapsed_seconds=elapsed,
                                  error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
                self._oracle_terminals["exception"] += 1
                del self._pending_oracle_attempts[attempt_id]
            self.event("oracle_exception", role=role, phase=phase, elapsed_seconds=elapsed,
                       error_type=type(exc).__name__, message=str(exc))
            raise
        elapsed = time.monotonic() - started
        with self._state_lock:
            self.oracle_audit("oracle_attempt_returned", **details, elapsed_seconds=elapsed, result=result)
            self._oracle_terminals["returned"] += 1
            del self._pending_oracle_attempts[attempt_id]
        self.event("oracle_evaluation", role=role, phase=phase, elapsed_seconds=elapsed, result=result)
        return result

    def status(self) -> dict:
        with self._state_lock:
            return {"benchmark": self.name, "seed": self.seed, "episode_index": self.episode_index,
                    "episode_seed": (self.seed + self.episode_index) % (2**32), "budget": self.budget,
                    "counts": dict(self.counts), "episode_counts": dict(self.episode_counts),
                    "faulted": self.faulted, "closed": self.closed}

    @serialized_operation
    def initial_result(self) -> dict:
        return {"initialized": True, "metadata": copy.deepcopy(self.metadata), "observation": self.observe()}

    @serialized_operation
    def dispatch(self, op: str, args: dict) -> Any:
        self.count("rpc_attempts")
        if self.closed:
            raise AdapterError("session_closed", "Adapter is closed.", self.status())
        if op == "close":
            self.closed = True
            if hasattr(self, "env") and hasattr(self.env, "close"):
                self.env.close()
            return self.status()
        if self.faulted and op not in {"observe", "reset"}:
            raise AdapterError("environment_faulted", "Reset is required after a partial environment failure.", self.status())
        try:
            return getattr(self, op)(args) if op in {"reset", "tool", "step"} else self.observe()
        except AdapterError:
            raise
        except Exception as exc:
            record = self.event("operation_exception", operation=op, error_type=type(exc).__name__, message=str(exc))
            raise AdapterError("operation_exception", str(exc), {"event": record, **self.status()}) from exc


class MADEAdapter(BaseAdapter):
    name = "made"

    def __init__(self, args: dict):
        super().__init__(args)
        self.mace_num_workers = args.get("mace_num_workers", 1)
        try:
            mace_execution = mace_execution_metadata(self.mace_num_workers)
        except ValueError as exc:
            raise AdapterError("invalid_config", str(exc)) from exc
        self.elements = args.get("elements")
        if not isinstance(self.elements, list) or not self.elements or not all(isinstance(x, str) for x in self.elements) or len(set(self.elements)) != len(self.elements):
            raise AdapterError("invalid_config", "elements must be distinct chemical symbols.")
        # Fail on absent/partial raw data before importing any heavyweight dependencies.
        payload, snapshot = load_snapshot(args.get("mp_cache_path"), args.get("mp_cache_sha256"), self.elements)
        assets = args.get("assets", {})
        self.assets = {key: asset(assets.get(key), key) for key in ("orb_checkpoint", "mace_checkpoint", "chemeleon_checkpoint", "element_reference_energies")}
        self.tolerance = args.get("stability_tolerance", 0.1)
        if self.tolerance not in (0.1, 0.01):
            raise AdapterError("invalid_config", "MADE supported paper thresholds are 0.1 and 0.01 eV/atom.")
        self.device = args.get("device", "cuda")
        if self.device not in {"cuda", "cpu"}:
            raise AdapterError("invalid_config", "Specify a fixed cpu or cuda device.")
        if self.mace_num_workers == 4 and self.device != "cpu":
            raise AdapterError("invalid_config", "MACE workers=4 is verified only for the fixed CPU/float32 oracle.")
        self.generator_device = args.get("generator_device", self.device)
        if self.generator_device not in {"cuda", "cpu"}:
            raise AdapterError("invalid_config", "generator_device must be cpu or cuda.")
        self.metadata = {"vendor": verify_vendor(self.vendor_root, "made", MADE_COMMIT), "snapshot": snapshot,
                         "assets": self.assets, "oracle": {"model": "orb-v3-conservative-inf-omat", "relax": True, **RELAX},
                         "oracle_device": self.device, "generator_device": self.generator_device,
                         "mace_num_workers": self.mace_num_workers, "orb_num_workers": 1,
                         "mace_execution": mace_execution,
                         "stability_tolerance": self.tolerance, "max_stoichiometry": 20,
                         "budget_unit": "candidate_oracle_attempts_per_episode", "seed_schedule": "(seed + episode_index) mod 2**32",
                         "deviation": "Exceptions consume attempted-query budget; raw official accepted-query metrics are also preserved."}
        self.start_output()
        seed_all(self.seed)
        verify_import("made.envs.convex_hull", self.vendor_root)
        from made.data.chemical_system import PhaseDiagramDataset
        from made.envs.convex_hull import ConvexHullEnvironment
        from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
        from pymatgen.entries.computed_entries import ComputedStructureEntry
        from pymatgen.core import Structure

        decoded = []
        for record in payload["entries"]:
            if record.get("structure"):
                original = ComputedStructureEntry.from_dict(record)
                structure, energy = original.structure, original.energy
            else:
                original = PDEntry.from_dict(record)
                structure, energy = Structure.from_dict(record["attribute"]["structure"]), original.energy
            if not math.isfinite(float(energy)) or not {str(e) for e in structure.composition.elements}.issubset(self.elements):
                raise AdapterError("invalid_snapshot", "Nonfinite energy or out-of-system entry in snapshot.")
            decoded.append(PDEntry(structure.composition, energy, attribute={"structure": structure.as_dict()}))

        class FrozenPhaseDiagramDataset(PhaseDiagramDataset):
            def _load_dataset(inner_self):
                return PhaseDiagram(decoded)

        self.dataset = FrozenPhaseDiagramDataset(elements=self.elements, thermo_types=["GGA_GGA+U"])
        self.oracle = self._make_oracle("orb")
        self.env = ConvexHullEnvironment(dataset=self.dataset, oracle=self.oracle, budget=self.budget,
            start_with_all_stable=True, compute_elemental_from_oracle=True, stability_tolerance=self.tolerance,
            include_near_stable_from_ground_truth=True, filter_by_smact=True, max_stoichiometry=20,
            structure_matcher_ltol=0.2, structure_matcher_stol=0.3, structure_matcher_angle_tol=5,
            structure_matcher_primitive_cell=True)
        self._build_tools()
        self._phase = "candidate"
        self.event("initialized", metadata=self.metadata)

    def _make_oracle(self, kind: str):
        if kind not in {"orb", "mace"}:
            raise AdapterError("invalid_oracle", "Only the fixed ORB and MACE oracles are supported.")
        from made.oracles.orb.orb_oracle import ORBOracle
        from made.oracles.mace.mace_oracle import MACEOracle
        from made.utils.structure_hash import structure_hash

        workers = 1 if kind == "orb" else self.mace_num_workers
        options = dict(device=self.device, element_reference_energies_path=self.assets["element_reference_energies"]["path"],
                       relax=True, relax_kwargs=dict(RELAX), num_workers=workers)
        constructor_id = uuid.uuid4().hex
        with ORACLE_LIFECYCLE_LOCK:
            guard = TorchFXGuard() if workers == 4 else None
            self.oracle_audit("oracle_constructor_started", constructor_id=constructor_id,
                              role=kind, num_workers=workers)
            started = time.monotonic()
            try:
                if kind == "orb":
                    from orb_models.forcefield import pretrained
                    original = pretrained.orb_v3_conservative_inf_omat
                    # Pin the official loader only while constructing. ORB always
                    # evaluates using this same initial calculator, with one worker.
                    pretrained.orb_v3_conservative_inf_omat = functools.partial(original, weights_path=self.assets["orb_checkpoint"]["path"])
                    try:
                        oracle = ORBOracle(model_name="orb-v3-conservative-inf-omat", **options)
                    finally:
                        pretrained.orb_v3_conservative_inf_omat = original
                else:
                    oracle = MACEOracle(model_name="custom", model_path=self.assets["mace_checkpoint"]["path"],
                                        default_dtype="float32", dispersion=False, **options)
                if guard is not None:
                    guard.check(oracle.calculator)
            except BaseException as exc:
                self.oracle_audit("oracle_constructor_exception", constructor_id=constructor_id,
                                  role=kind, num_workers=workers, elapsed_seconds=time.monotonic()-started,
                                  error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
                raise
            self.oracle_audit("oracle_constructor_finished", constructor_id=constructor_id,
                              role=kind, num_workers=workers, elapsed_seconds=time.monotonic()-started,
                              fx_globals_restored=True if guard is not None else None)
            # Retain the official evaluate/batch implementations and result schema.
            # The private executor only supplies calculator lifetimes and evidence.
            oracle._matdiscovery_executor = AuditedOracleExecutor(oracle, role=kind,
                candidate_hash=structure_hash, invoke=self.invoke_oracle,
                record=self.oracle_audit, guard=guard)
        return oracle

    def _build_tools(self) -> None:
        from made.agents.generators.random import RandomGenerator
        from made.agents.generators.chemeleon import ChemeleonGenerator
        from made.agents.filters.chain import FilterChain
        from made.agents.filters.min_distance import MinDistanceFilter
        from made.agents.filters.smact import SMACTValidityFilter
        from made.agents.filters.uniqueness import UniquenessFilter
        from made.agents.scorers.random import RandomSelector
        from made.agents.scorers.diversity import CompositionDiversity
        from made.agents.llm_react_orchestrator import OrchestratorTools
        checkpoint, device = self.assets["chemeleon_checkpoint"]["path"], self.generator_device

        class PinnedChemeleon(ChemeleonGenerator):
            def setup(inner_self):
                if inner_self.dm is None:
                    from chemeleon_dng.diffusion.diffusion_module import DiffusionModule
                    inner_self.dm = DiffusionModule.load_from_checkpoint(checkpoint, map_location=device)

        seed = (self.seed + self.episode_index) % (2**32)
        self.generators = {"random": RandomGenerator(seed=seed), "chemeleon": PinnedChemeleon(task="csp", batch_size=32, device=device)}
        self.static_filter = FilterChain([MinDistanceFilter(min_distance_threshold=0.5), SMACTValidityFilter()])
        self.uniqueness_filter = UniquenessFilter(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True)
        self.scorers = {"random": RandomSelector(seed=seed), "diversity": CompositionDiversity(seed=seed)}
        self.buffer, self.structure_cache, self.selected = {}, {}, None
        # Reuse official read/query tools. Mutation tools below use the same official
        # components but propagate exceptions and exact filter records without fallback.
        self.read_tools = OrchestratorTools(self.generators, self.static_filter, self.uniqueness_filter,
            self.scorers, self.elements, 20, self.buffer, self.structure_cache, self.env.get_state())

    def done(self) -> bool:
        return self.count_value("candidate_oracle_attempts", episode=True) >= self.budget or self.env.is_done()

    @serialized_operation
    def observe(self) -> dict:
        return {**self.status(), "done": self.done(), "state": jsonable(self.env.get_state()),
                "buffer": {comp: [{k: jsonable(v) for k, v in entry.items() if k != "structure"} for entry in entries] for comp, entries in self.buffer.items()},
                "selected": self.selected is not None}

    @serialized_operation
    def reset(self, args: dict) -> dict:
        if args:
            raise AdapterError("immutable_config", "reset takes no overrides; task, seed schedule and evaluator are frozen.")
        if not self.done() and not self.faulted:
            raise AdapterError("episode_incomplete", "MADE reset cannot discard an unfinished query budget.", self.status())
        with self._state_lock:
            self.episode_index += 1
            self.episode_counts = {}
        seed_all((self.seed + self.episode_index) % (2**32))
        self._phase = "initialization"
        try:
            self.env.reset()
            self._build_tools()
        except Exception:
            self.faulted = True
            raise
        finally:
            self._phase = "candidate"
        self.faulted = False
        self.event("reset", status=self.status())
        return self.observe()

    def _validate_structure(self, structure) -> None:
        if not 1 <= len(structure) <= 20 or not {str(e) for e in structure.composition.elements}.issubset(self.elements):
            raise AdapterError("invalid_structure", "Structure must use allowed elements and at most 20 atoms.", self.status())
        if not math.isfinite(structure.volume) or structure.volume <= 0:
            raise AdapterError("invalid_structure", "Structure must have finite positive volume.")

    def _record_scores(self, scorer_name, entries, scores, records):
        """Retain official scores individually; do not reject a batch for -inf.

        OracleScorer negates safe_e_above_hull's documented +inf sentinel when
        hull decomposition is undefined. This is distinct from an MLIP exception.
        NaN/+inf are retained as explicit, unrankable candidate score failures.
        """
        if len(scores) != len(entries) or len(records) != len(entries):
            raise AdapterError("score_length_mismatch", "Official scorer results do not match the candidate batch.")
        values = [float(score) for score in scores]
        outcomes = []
        for entry, value, record in zip(entries, values, records, strict=True):
            status = ("finite" if math.isfinite(value) else
                      "official_negative_infinity_sentinel" if value == -math.inf else
                      "invalid_nan" if math.isnan(value) else "invalid_positive_infinity")
            outcome = {"structure_hash": entry["hash"], "score": value,
                       "status": status, "rankable": score_rankable(value), "official_result": jsonable(record)}
            entry["scores"][scorer_name] = value
            entry.setdefault("score_status", {})[scorer_name] = {"status": status, "rankable": score_rankable(value)}
            outcomes.append(outcome)
            if not math.isfinite(value):
                self.count("score_sentinels" if value == -math.inf else "score_failures")
                self.event("score_sentinel" if value == -math.inf else "score_failure",
                           scorer=scorer_name, **outcome)
        entries.sort(key=lambda entry: score_order(entry["scores"][scorer_name]))
        return outcomes

    def _buffer_add(self, structure, source: str) -> dict:
        from made.utils.structure_hash import structure_hash
        self._validate_structure(structure)
        digest = structure_hash(structure)
        if digest in self.structure_cache:
            return {"accepted": False, "reason": "already_in_buffer", "hash": digest}
        records = []
        for filter_object in (self.static_filter, self.uniqueness_filter):
            passed, results = filter_object.filter([structure], self.env.get_state(), return_results=True)
            records.extend(jsonable(results))
            if not passed:
                return {"accepted": False, "reason": "official_filter_rejection", "hash": digest, "filters": records}
        comp = structure.composition.reduced_formula
        entry = {"structure": structure, "hash": digest, "composition": comp,
                 "full_formula": structure.composition.formula.replace(" ", ""), "source": source,
                 "scores": {}, "num_sites": len(structure)}
        self.buffer.setdefault(comp, []).append(entry)
        self.structure_cache[digest] = entry
        return {"accepted": True, "composition": comp, "structure_index": len(self.buffer[comp])-1, "hash": digest, "filters": records}

    @serialized_operation
    def tool(self, args: dict) -> dict:
        self.count("tool_attempts")
        if self.done():
            raise AdapterError("budget_exhausted", "MADE attempted-query budget exhausted.", self.status())
        aliases = {"generate": "generate_structures", "create": "create_structure", "score": "score_buffer", "query": "query_structures", "select": "select_for_evaluation"}
        name = args.get("name", args.get("tool"))
        name = aliases.get(name, name)
        params = args.get("arguments", {})
        if not isinstance(params, dict):
            raise AdapterError("invalid_tool", "tool.arguments must be an object.")
        started = time.monotonic()
        try:
            result = self._tool(name, params)
        except Exception as exc:
            event = self.event("tool_exception", tool=name, elapsed_seconds=time.monotonic()-started,
                               error_type=type(exc).__name__, message=str(exc))
            raise AdapterError("tool_exception", str(exc), {"event": event, **self.status()}) from exc
        event = self.event("tool_result", tool=name, elapsed_seconds=time.monotonic()-started, result=result)
        return {"tool": name, "output": result, "event": event, "status": self.status()}

    def _tool(self, name: str, params: dict) -> Any:
        from made.agents.base import Plan
        from pymatgen.core import Composition, Lattice, Structure
        if name == "generate_structures":
            generator_name = params.get("generator_name", "chemeleon")
            if generator_name not in self.generators:
                raise AdapterError("invalid_generator", "Allowed generators: chemeleon, random.")
            formulas = params.get("compositions", [])
            if isinstance(formulas, str):
                formulas = [x.strip() for x in formulas.split(",") if x.strip()]
            if not isinstance(formulas, list) or not formulas:
                raise AdapterError("invalid_composition", "Explicit compositions are required; no fallback composition is generated.")
            compositions = [Composition(formula) for formula in formulas]
            for comp in compositions:
                if not {str(e) for e in comp.elements}.issubset(self.elements) or not 1 <= comp.num_atoms <= 20 or any(float(v) != int(v) for v in comp.values()):
                    raise AdapterError("invalid_composition", "Compositions need integral allowed-element counts and 1–20 atoms.")
            count = positive_int(params.get("num_candidates", 32), "num_candidates")
            if count > 1024 or len(compositions) > 1024:
                raise AdapterError("invalid_config", "At most 1024 requested candidates/compositions per tool call.")
            structures = self.generators[generator_name].generate(Plan(compositions=compositions, num_candidates=count,
                constraints={"elements": self.elements, "max_stoichiometry": 20}), self.env.get_state())
            records = [self._buffer_add(s, generator_name) for s in structures]
            return {"generated": len(structures), "accepted": sum(r["accepted"] for r in records), "records": records}
        if name == "create_structure":
            if "structure" in params:
                structure = Structure.from_dict(params["structure"])
            else:
                species = params.get("species", [])
                coords = params.get("frac_coords", [])
                if isinstance(species, str):
                    species = [x.strip() for x in species.split(",") if x.strip()]
                if isinstance(coords, str):
                    coords = [[float(v) for v in row.split(",")] for row in coords.split(";") if row.strip()]
                lattice = Lattice.from_parameters(**{k: params.get(k, 90.0 if k in {"alpha", "beta", "gamma"} else None) for k in ("a", "b", "c", "alpha", "beta", "gamma")})
                structure = Structure(lattice, species, coords)
            return self._buffer_add(structure, "controller_created")
        if name == "score_buffer":
            scorer_name = params.get("scorer_name", "oracle")
            if scorer_name == "oracle" and "oracle" not in self.scorers:
                from made.agents.scorers.oracle import OracleScorer
                self.scorers["oracle"] = OracleScorer(self._make_oracle("mace"), score_function="e_above_hull", enable_cache=True, rerank_on_state_change=True)
            if scorer_name not in self.scorers:
                raise AdapterError("invalid_scorer", "Allowed scorers: oracle, random, diversity. No LLM scorer is called.")
            groups = [params["composition"]] if params.get("composition") else list(self.buffer)
            result = []
            for comp in groups:
                if comp not in self.buffer:
                    raise AdapterError("missing_composition", f"No buffered composition {comp}.")
                entries = self.buffer[comp]
                before = self.count_value("surrogate_oracle_attempts")
                scores, records = self.scorers[scorer_name].score_candidates([e["structure"] for e in entries], self.env.get_state(), return_results=True)
                outcomes = self._record_scores(scorer_name, entries, scores, records)
                result.append({"composition": comp, "scores": scores, "records": jsonable(records),
                               "candidates": outcomes,
                               "surrogate_oracle_attempts": self.count_value("surrogate_oracle_attempts")-before,
                               "score_policy": "finite descending, official -inf last; NaN/+inf unrankable; stable order for ties"})
            return result
        if name == "select_for_evaluation":
            if self.selected is not None:
                raise AdapterError("selection_pending", "Evaluate the selected candidate before selecting another.")
            comp = params.get("composition")
            if comp not in self.buffer:
                raise AdapterError("missing_composition", "Select an existing buffer composition.")
            entries = self.buffer[comp]
            scorer = params.get("scorer_name")
            selected_hash = params.get("structure_hash")
            if selected_hash:
                matches = [i for i, entry in enumerate(entries) if entry["hash"] == selected_hash]
                if not matches:
                    raise AdapterError("invalid_selection_hash", "The selected hash is not in this composition buffer.")
                index = matches[0]
            elif scorer:
                if any(scorer not in entry["scores"] for entry in entries):
                    raise AdapterError("unscored_candidates", "Score every candidate before selecting by that scorer.")
                entries.sort(key=lambda entry: score_order(entry["scores"][scorer]))
                index = params.get("structure_index", 0)
            else:
                index = params.get("structure_index", 0)
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(entries):
                raise AdapterError("invalid_selection_index", "Buffer selection index is out of range.")
            if scorer and (scorer not in entries[index]["scores"] or not score_rankable(entries[index]["scores"][scorer])):
                raise AdapterError("unrankable_score", "This candidate has no usable score from the requested scorer; NaN/+inf cannot select a winner.")
            self.selected = entries.pop(index)
            self.structure_cache.pop(self.selected["hash"], None)
            if not entries:
                del self.buffer[comp]
            return {k: jsonable(v) for k, v in self.selected.items() if k != "structure"}
        if name == "query_structures":
            comp = params.get("composition")
            if comp not in self.buffer:
                raise AdapterError("missing_composition", "Query an existing buffer composition.")
            k = positive_int(params.get("k", 5), "k")
            mode, scorer = params.get("mode", "all"), params.get("scorer_name")
            entries = list(enumerate(self.buffer[comp]))
            if mode in {"top", "bottom"}:
                if not scorer or any(scorer not in entry["scores"] for _, entry in entries):
                    raise AdapterError("unscored_candidates", "This query requires a score on every candidate.")
                entries.sort(key=lambda pair: score_order(pair[1]["scores"][scorer], descending=mode == "top"))
            elif mode == "random":
                random.shuffle(entries)
            elif mode != "all":
                raise AdapterError("invalid_query_mode", "Use all, random, top or bottom.")
            # Unlike the upstream text display's rank indices, these canonical indices
            # and stable hashes refer to the actual buffer and cannot select a different item.
            result = []
            for index, entry in entries[:k]:
                item = {key: jsonable(value) for key, value in entry.items() if key != "structure"}
                item["structure_index"] = index
                item["structure_hash"] = entry["hash"]
                if params.get("include_structure_details", False):
                    item["structure"] = jsonable(entry["structure"])
                result.append(item)
            return {"composition": comp, "candidates": result, "selection_key": "structure_hash"}
        if name in {"list_compositions", "get_buffer_stats"}:
            self.read_tools.state = self.env.get_state()
            return getattr(self.read_tools, name)(**params)
        raise AdapterError("unknown_tool", f"Unsupported MADE tool: {name!r}")

    @serialized_operation
    def step(self, args: dict) -> dict:
        self.count("step_attempts")
        if self.done():
            raise AdapterError("budget_exhausted", "No more candidate oracle attempts are allowed.", self.status())
        if args:
            raise AdapterError("invalid_step", "MADE step takes no arguments; explicitly create/generate and select a candidate first.", self.status())
        if self.selected is None:
            raise AdapterError("no_selection", "No candidate selected; no fallback will be made.", self.status())
        selected, self.selected = self.selected, None
        before = self.env.query_count
        started = time.monotonic()
        try:
            observation, _ = self.env.step(selected["structure"])
        except Exception as exc:
            # A failure after env.query_count changed may leave a partially mutated hull.
            self.faulted = self.env.query_count != before
            event = self.event("step_exception", candidate_hash=selected["hash"], error_type=type(exc).__name__, message=str(exc), elapsed_seconds=time.monotonic()-started)
            raise AdapterError("oracle_or_environment_exception", str(exc), {"event": event, "done": self.done(), **self.status()}) from exc
        metrics = self.env.metrics_history[-1] if self.env.metrics_history else {}
        public_keys = {"queries_used", "num_newly_discovered_structures", "num_newly_discovered_stable",
                       "discovery_efficiency", "stable_discovery_efficiency"}
        public_metrics = {key: value for key, value in metrics.items() if key in public_keys or
                          key.startswith(("validity_", "diversity_", "novelty_", "discovery_curve_"))}
        event = self.event("scientific_result", candidate_hash=selected["hash"], observation=observation,
                           metrics=public_metrics, elapsed_seconds=time.monotonic()-started)
        self.event("evaluation_only_metrics", metrics=metrics, policy_visible=False)
        return {"observation": self.observe(), "official_observation": jsonable(observation),
                "official_metrics": jsonable(public_metrics), "event": event, "done": self.done()}


class CrystalGymAdapter(BaseAdapter):
    name = "crystalgym"

    def __init__(self, args: dict):
        super().__init__(args)
        self.budget_unit = args.get("budget_unit")
        if self.budget_unit not in {"atomic_action_attempts", "dft_episode_attempts"}:
            raise AdapterError("invalid_config", "CrystalGym requires an explicit budget_unit: atomic_action_attempts or dft_episode_attempts.")
        self.property = args.get("property")
        if self.property not in {"bm", "density", "band_gap"}:
            raise AdapterError("invalid_config", "property must be bm, density or band_gap.")
        self.target = args.get("target")
        if isinstance(self.target, bool) or not isinstance(self.target, (int, float)) or not math.isfinite(self.target) or self.target <= 0:
            raise AdapterError("invalid_config", "target must be a finite positive property value.")
        self.split = args.get("split", "train")
        self.index = args.get("prototype_index")
        if self.split == "heldout":
            if self.index not in HELDOUT_INDICES:
                raise AdapterError("invalid_split", "Held-out evaluation requires prototype_index 3403 or 2190.")
        elif self.split == "train":
            if self.index is not None and self.index not in TRAIN_INDICES:
                raise AdapterError("invalid_split", "Training cannot use held-out prototypes.")
        else:
            raise AdapterError("invalid_split", "Supported final-benchmark split values are train and heldout.")
        assets = args.get("assets", {})
        self.assets = {
            "mp20_val": asset(assets.get("mp20_val", str(self.vendor_root/"crystal_gym/data/mp_20/val.csv")), "mp20_val", CRYSTAL_VAL_SHA256),
            "pseudodict": asset(assets.get("pseudodict", str(self.vendor_root/"crystal_gym/files/pseudodict.pkl")), "pseudodict", PSEUDODICT_SHA256),
        }
        pseudo_dir = absolute_path(assets.get("pseudo_dir"), "pseudo_dir", directory=True)
        qe_dir = absolute_path(assets.get("qe_dir"), "qe_dir", directory=True)
        self.assets["pw_x"] = asset(str(qe_dir/"bin/pw.x"), "pw.x")
        self.assets["ev_x"] = asset(str(qe_dir/"bin/ev.x"), "ev.x")
        for name in ("pw_x", "ev_x"):
            if not os.access(self.assets[name]["path"], os.X_OK):
                raise AdapterError("missing_executable", f"{name} is not executable.")
        mpi = shutil.which("mpirun")
        if not mpi:
            raise AdapterError("missing_executable", "Official CrystalGym requires mpirun in PATH.")
        import pickle
        # Only deserialize the SHA256-verified file from the fresh official source.
        with open(self.assets["pseudodict"]["path"], "rb") as handle:
            pseudo_map = pickle.load(handle)
        self.assets["pseudopotentials"] = {element: asset(str(pseudo_dir/pseudo_map[element]), f"{element} UPF") for element in SMALL_ELEMENTS}
        fixed_qe = dict(prefix="myprefix", electron_maxstep=300, tstress=False, tprnfor=False,
                        ecutwfc=50, ecutrho=400, verbosity="high", diagonalization="david", smearing="gaussian",
                        mixing_mode="plain", mixing_beta=0.7, degauss=0.001, nspin=1, kppa=1000,
                        calculation="vc-relax" if self.property == "density" else "scf",
                        occupations="fixed" if self.property == "band_gap" else "smearing")
        for key, value in args.get("qe", {}).items():
            if key not in fixed_qe or fixed_qe[key] != value:
                raise AdapterError("evaluator_override", f"QE field {key} would change the fixed official evaluator.")
        self.qe = {**fixed_qe, "qe_dir": str(qe_dir), "pseudo_dir": str(pseudo_dir),
                   "pseudodict": self.assets["pseudodict"]["path"], "outdir": str(self.work_dir/"qe_scratch")}
        self.metadata = {"vendor": verify_vendor(self.vendor_root, "crystal_gym", CRYSTALGYM_COMMIT), "assets": self.assets,
                         "qe": self.qe, "mpirun": mpi, "property": self.property, "target": self.target,
                         "split": self.split, "prototype_index": self.index, "budget_unit": self.budget_unit,
                         "seed_schedule": "(seed + episode_index) mod 2**32", "vocab": SMALL_ELEMENTS,
                         "original_rl_yaml_atomic_timesteps": 500000,
                         "budget_interpretation": "This configured budget is explicit; no claim of full training unless the main protocol locks it."}
        self.start_output()
        seed_all(self.seed)
        module = verify_import("crystal_gym.env.crystal_env", self.vendor_root)
        env_options = {"dataset": "mp_20", "data_path": self.assets["mp20_val"]["path"],
                       "mode": "cubic-mini" if self.index is None else "single", "index": self.index,
                       "p_hat": self.target, "seed": self.seed, "property": self.property, "reward_min": -5.0,
                       "substitution": False, "vocab": "small", "agent": "MEGNetRL", "run_name": "episode_000000"}
        self.env = module.CrystalGymEnv(kwargs={"env": env_options, "qe": self.qe})
        self.episode_done = False
        self.filled = [None] * int(self.env.n_sites)
        self._phase = "candidate"
        self.event("initialized", metadata=self.metadata)

    def done(self) -> bool:
        return self.count_value(self.budget_unit) >= self.budget

    @serialized_operation
    def observe(self) -> dict:
        geometry = self.env.graph_to_dict_complete(self.env.state)
        focus = None if self.episode_done or self.env.t >= self.env.n_sites else int(self.env.traversal[self.env.t].item())
        # Never serialize env.data, original CIF/species, graph atom_types, or unfilled
        # placeholder identities. The only chemical identities here are controller actions.
        return {**self.status(), "done": self.done(), "episode_done": self.episode_done,
                "prototype_index": int(self.env.sample_ind), "property": self.property, "target": self.target,
                "lattice_lengths": jsonable(geometry["lengths"]), "lattice_angles": jsonable(geometry["angles"]),
                "fractional_coordinates": jsonable(geometry["frac_coords"]), "filled_elements": list(self.filled),
                "focus_site": focus, "legal_actions": [{"index": i, "element": symbol} for i, symbol in enumerate(SMALL_ELEMENTS)],
                "num_sites": int(self.env.n_sites), "filled_count": int(self.env.t)}

    @serialized_operation
    def reset(self, args: dict) -> dict:
        if args:
            raise AdapterError("immutable_config", "reset does not accept seed, prototype or evaluator overrides.")
        if self.done():
            raise AdapterError("budget_exhausted", "Session budget persists across CrystalGym resets.", self.status())
        if not self.episode_done and not self.faulted:
            raise AdapterError("episode_incomplete", "Complete the current crystal before reset; dropping prefixes is not allowed.", self.status())
        with self._state_lock:
            self.episode_index += 1
            self.episode_counts = {}
        seed = (self.seed + self.episode_index) % (2**32)
        seed_all(seed)
        self.env.run_name = f"episode_{self.episode_index:06d}"
        self.env.reset(seed=seed)
        self.filled = [None] * int(self.env.n_sites)
        self.episode_done = self.faulted = False
        self.event("reset", status=self.status())
        return self.observe()

    @serialized_operation
    def tool(self, args: dict) -> dict:
        self.count("tool_attempts")
        raise AdapterError("unsupported_tool", "CrystalGym actions are element indices supplied via step; no alternate scientific tools are exposed.", self.status())

    @serialized_operation
    def step(self, args: dict) -> dict:
        self.count("step_attempts")
        if self.done():
            raise AdapterError("budget_exhausted", "CrystalGym session budget exhausted.", self.status())
        if self.episode_done:
            raise AdapterError("episode_done", "Call reset before another action.", self.status())
        self.count("atomic_action_attempts")
        action = args.get("action")
        if isinstance(action, str) and action in SMALL_ELEMENTS:
            action = SMALL_ELEMENTS.index(action)
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < len(SMALL_ELEMENTS):
            event = self.event("invalid_action", action=action)
            raise AdapterError("invalid_action", "Action must be a legal element symbol or zero-based integer index.", {"event": event, **self.status()})
        if set(args) != {"action"}:
            raise AdapterError("invalid_step", "CrystalGym step accepts only action.", self.status())
        site = int(self.env.traversal[self.env.t].item())
        terminal_attempt = self.env.t + 1 == self.env.n_sites
        if terminal_attempt:
            number = self.count("dft_episode_attempts")
            self.env.run_name = f"episode_{self.episode_index:06d}/dft_{number:08d}"
            self.env.qe_inputs["outdir"] = str(self.work_dir/"qe_scratch"/f"dft_{number:08d}")
            Path(self.env.qe_inputs["outdir"]).mkdir(parents=True, exist_ok=False)
        self.filled[site] = SMALL_ELEMENTS[action]
        started = time.monotonic()
        try:
            _, reward, terminated, truncated, info = self.env.step(action)
        except Exception as exc:
            self.faulted = True
            event = self.event("step_exception", site=site, action=action, dft_attempt=terminal_attempt,
                               elapsed_seconds=time.monotonic()-started, error_type=type(exc).__name__, message=str(exc))
            raise AdapterError("dft_or_environment_exception", str(exc), {"event": event, **self.status()}) from exc
        elapsed = time.monotonic()-started
        self.count("successful_action_steps")
        self.episode_done = bool(terminated or truncated)
        terminal_info = info.get("final_info", [{}])[0].get("episode", {}) if terminal_attempt else {}
        flag = terminal_info.get("error_flag")
        labels = {"band_gap": {1: "dft_nonconvergence", 2: "dft_charge", 3: "dft_output_or_calculation_error"},
                  "density": {3: "dft_output_or_calculation_error"},
                  "bm": {2: "eos_parse_error", 3: "qe_or_eos_calculation_error"}}
        label = ("dft_success" if flag == 0 else labels[self.property].get(flag, "unclassified_official_dft_failure")) if terminal_attempt else None
        property_value = terminal_info.get(self.property)
        scientific = {"terminal": self.episode_done, "reward": float(reward), "property": self.property,
                      "property_value": property_value, "official_error_flag": flag, "scientific_label": label,
                      "absolute_target_error": abs(float(property_value)-self.target) if property_value is not None else None,
                      "dft_success": flag == 0 if terminal_attempt else None,
                      "elapsed_seconds": elapsed, "official_info": info,
                      "terminated": bool(terminated), "truncated": bool(truncated)}
        event = self.event("scientific_result" if terminal_attempt else "action_result", site=site, action=action, **scientific)
        return {"observation": self.observe(), "scientific_result": jsonable(scientific), "event": event,
                "done": self.done(), "episode_done": self.episode_done}


def create_adapter(benchmark: str, args: dict) -> BaseAdapter:
    cls = {"made": MADEAdapter, "crystalgym": CrystalGymAdapter}.get(benchmark)
    if cls is None:
        raise AdapterError("unknown_benchmark", "Specify made or crystalgym at process launch or init.")
    adapter = object.__new__(cls)
    try:
        cls.__init__(adapter, args)
    except AdapterError:
        raise
    except Exception as exc:
        details = adapter.status() if hasattr(adapter, "counts") else {}
        raise AdapterError("initialization_failed", str(exc), {"cause_type": type(exc).__name__, **details}) from exc
    return adapter
