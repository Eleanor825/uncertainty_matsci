"""Immutable full final-evaluation matrix and conservative, restart-safe accounting.

The manifest describes evaluation only. It does not assert that training, fitted
uncertainty models, model checkpoints, or physical evaluators are ready. A worker
must atomically claim a job *before* invoking an evaluator. Lost workers become
orphaned, never runnable: a lease expiring does not prove a remote QE call ended.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import time
from typing import Any, Mapping
import uuid


METHODS = (
    "baseline", "entropy_risk", "hidden_risk", "graph_risk", "esopt", "esopt_graph_risk"
)
MODEL_KEYS = ("qwen35_4b", "qwen35_9b")
SEEDS = (1, 2, 3, 4, 5)
STATES = ("pending", "running", "succeeded", "failed", "orphaned")
TERMINAL_EPISODE_STATES = {"succeeded", "failed"}
COUNT_KEYS = ("episodes", "candidate_oracle_attempts", "dft_episode_attempts")
FULL_COUNTS = {
    "jobs": 2160, "episodes": 3600,
    "candidate_oracle_attempts": 90000, "dft_episode_attempts": 1800,
}


class AccountingError(ValueError):
    """An identity, completeness, or restart-safety invariant was violated."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    from .immutable_hash_cache import maybe_cached_file_sha256
    cached = maybe_cached_file_sha256(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Publish a complete JSON file using fsync and same-directory replacement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def pack_environment_seed(
    training_seed: int, property_id: str, prototype_id: str, rollout_id: int,
    *, partition: str = "final_test",
) -> int:
    """The bit-field schedule already locked in configs/benchmark_tasks.json."""
    partitions = {
        "uncertainty_fit_and_es_train": 0,
        "uncertainty_calibration_dev": 1,
        "final_test": 2,
    }
    properties = {"bm": 0, "density": 1, "band_gap": 2}
    prototypes = {"C1": 0, "C7": 6, "training_mixture": 7}
    if type(training_seed) is not int or training_seed not in SEEDS:
        raise AccountingError("training_seed must be one of the five prespecified integers")
    if type(rollout_id) is not int or not 0 <= rollout_id < 2**22:
        raise AccountingError("rollout_id exceeds its 22-bit field")
    if partition not in partitions or property_id not in properties or prototype_id not in prototypes:
        raise AccountingError("Unknown seed partition/property/prototype")
    if (partition == "final_test") != (prototype_id in {"C1", "C7"}):
        raise AccountingError("Held-out prototypes belong only to the final-test partition")
    return (
        (partitions[partition] << 30) | ((training_seed - 1) << 27)
        | (properties[property_id] << 25) | (prototypes[prototype_id] << 22) | rollout_id
    )


def _protocol(tasks: Mapping, splits: Mapping) -> dict:
    made, crystal = tasks["made"], tasks["crystalgym"]
    systems = sorted(made["systems"], key=lambda x: x["id"])
    system_sets = {tuple(sorted(s["elements"])) for s in systems}
    split_sets = {
        name: sorted([sorted(system) for system in splits["splits"][name]])
        for name in ("train", "dev", "test")
    }
    if len(systems) != 30 or len(system_sets) != 30:
        raise AccountingError("Full MADE evaluation requires exactly 30 distinct systems")
    if Counter(s["system_size"] for s in systems) != {3: 10, 4: 10, 5: 10}:
        raise AccountingError("MADE requires ten systems in each size stratum")
    for system in systems:
        if system["id"] != "-".join(sorted(system["elements"])):
            raise AccountingError("Chemical-system id must match its elements")
        if len(set(system["elements"])) != system["system_size"]:
            raise AccountingError("Chemical-system size is inconsistent")
    if {tuple(s) for s in split_sets["test"]} != system_sets or len(split_sets["test"]) != 30:
        raise AccountingError("Every official main system must remain in the test split")
    train, dev = ({tuple(s) for s in split_sets[name]} for name in ("train", "dev"))
    if train & dev or (train | dev) & system_sets:
        raise AccountingError("MADE train/dev/test chemical-system groups overlap")
    if sorted(made["seeds"]) != list(SEEDS) or sorted(crystal["seeds"]) != list(SEEDS):
        raise AccountingError("Full evaluation requires seeds 1..5 in both benchmarks")
    if made["oracle_query_budget_per_episode"] != 50 or made["episodes_per_system_per_method"] != 5:
        raise AccountingError("MADE main evaluation requires five B=50 episodes per system")
    if made["primary_stability_tolerance_ev_per_atom"] != 0.1:
        raise AccountingError("The primary matrix uses the locked 0.1 eV/atom threshold")
    properties = sorted(crystal["properties"], key=lambda x: x["id"])
    if {p["id"]: p["target"] for p in properties} != {"bm": 500.0, "density": 5.0, "band_gap": 2.0}:
        raise AccountingError("CrystalGym final evaluation requires all three OOD property targets")
    heldout = sorted((p for p in crystal["prototypes"] if p["split"] == "heldout"), key=lambda x: x["id"])
    if {p["id"]: p["index"] for p in heldout} != {"C1": 3403, "C7": 2190}:
        raise AccountingError("CrystalGym requires exactly C1 and C7 as held-out prototypes")
    evaluation = crystal["evaluation"]
    if (evaluation["rollouts_per_seed_per_heldout_prototype"] != 5
            or sorted(evaluation["rollout_ids"]) != list(range(5))
            or evaluation["budget_per_condition"] != 5
            or evaluation["budget_unit"] != "dft_episode_attempts"):
        raise AccountingError("Each CrystalGym condition requires all five DFT episodes")
    schedule = crystal["environment_seed_schedule"]
    if (schedule["name"] != "disjoint_partition_bitfields_v1"
            or schedule["property_indices"] != {"bm": 0, "density": 1, "band_gap": 2}
            or schedule["prototype_codes"] != {
                "C1_heldout_index3403": 0, "C7_heldout_index2190": 6,
                "official_five_prototype_training_mixture": 7,
            }
            or sorted(schedule["training_seeds"]) != list(SEEDS)
            or schedule["rollout_id_min"] != 0
            or schedule["rollout_id_max_exclusive"] != 2**22
            or schedule["partitions"] != {
                "uncertainty_fit_and_es_train": 0,
                "uncertainty_calibration_dev": 1, "final_test": 2,
            }):
        raise AccountingError("The configured environment seed schedule changed")
    return {
        "made": {
            "systems": [{**s, "elements": sorted(s["elements"])} for s in systems],
            "seeds": list(SEEDS), "budget": 50, "splits": split_sets,
            "official_commit": made["official_commit"],
            "stability_tolerance_ev_per_atom": made["primary_stability_tolerance_ev_per_atom"],
            "oracle": made["oracle"], "phase_diagram": made["phase_diagram"],
            "structure_matcher": made["structure_matcher"], "max_atoms": made["max_atoms"],
            "failure_budget_rule": made["failure_budget_rule"],
        },
        "crystalgym": {
            "properties": properties, "heldout_prototypes": heldout,
            "train_indices": sorted(crystal["train_indices"]), "seeds": list(SEEDS),
            "rollout_ids": list(range(5)), "budget": 5,
            "official_commit": crystal["official_commit"], "task": crystal["task"],
            "evaluator": crystal["evaluator"], "action_space": crystal["action_space"],
            "seed_schedule": "disjoint_partition_bitfields_v1",
            "failure_rule": evaluation["failure_rule"],
        },
    }


def _models(source: Mapping) -> list[dict]:
    selected = source["selection"]["ordered_model_keys"]
    if sorted(selected) != sorted(MODEL_KEYS):
        raise AccountingError("This full matrix is locked to the two selected smaller Qwen models")
    available = {m["key"]: m for m in source["models"]}
    result = []
    for key in MODEL_KEYS:
        model = available[key]
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise AccountingError("A model requires its immutable repository revision")
        result.append({
            "key": key, "model_id": model["model_id"], "revision": model["revision"],
            "tokenizer_revision": model["revision"], "dtype": model["dtype"],
            "files": sorted([
                {"path": f["path"], "sha256": f["sha256"], "size_bytes": f["size_bytes"]}
                for f in model["weight_files"] + model["metadata_files"]
            ], key=lambda f: f["path"]),
        })
    return result


def _jobs(protocol: Mapping, models: list[dict]) -> list[dict]:
    jobs = []
    for model in models:
        for method in METHODS:
            common = {
                "stage": "final_eval", "model_key": model["key"],
                "model_id": model["model_id"], "model_revision": model["revision"],
                "tokenizer_revision": model["tokenizer_revision"],
                "model_artifact_fingerprint": fingerprint(model), "method": method,
            }
            for task in protocol["made"]["systems"]:
                for seed in SEEDS:
                    jobs.append({
                        **common, "benchmark": "made", "task_id": task["id"],
                        "task": task, "seed": seed, "environment_seeds": [seed],
                        "episode_ids": ["0"], "budget": 50,
                        "budget_unit": "candidate_oracle_attempts_per_episode",
                        "expected_counts": {"episodes": 1, "candidate_oracle_attempts": 50, "dft_episode_attempts": 0},
                        "evaluator_fingerprint": fingerprint(protocol["made"]),
                    })
            for prop in protocol["crystalgym"]["properties"]:
                for prototype in protocol["crystalgym"]["heldout_prototypes"]:
                    for seed in SEEDS:
                        jobs.append({
                            **common, "benchmark": "crystalgym",
                            "task_id": f"{prop['id']}:{prototype['id']}",
                            "task": {"property": prop, "prototype": prototype}, "seed": seed,
                            "environment_seeds": [pack_environment_seed(seed, prop["id"], prototype["id"], i) for i in range(5)],
                            "episode_ids": [str(i) for i in range(5)], "budget": 5,
                            "budget_unit": "dft_episode_attempts",
                            "expected_counts": {"episodes": 5, "candidate_oracle_attempts": 0, "dft_episode_attempts": 5},
                            "evaluator_fingerprint": fingerprint(protocol["crystalgym"]),
                        })
    for job in jobs:
        job["job_id"] = f"final-{job['benchmark']}-{fingerprint(job)[:24]}"
    return sorted(jobs, key=lambda j: (j["model_key"], j["method"], j["benchmark"], j["task_id"], j["seed"]))


def build_final_manifest(project_root: str | Path) -> dict:
    """Build the full 2,160-job final-evaluation matrix from locked local inputs."""
    root = Path(project_root)
    def read(name):
        return json.loads((root / "configs" / f"{name}.json").read_text())
    return build_final_manifest_from_configs(read("benchmark_tasks"), read("made_splits"), read("model_manifest"))


def build_final_manifest_from_configs(tasks: Mapping, splits: Mapping, models: Mapping) -> dict:
    """Pure counterpart of build_final_manifest; rejects reduced experimental scope."""
    protocol, selected = _protocol(tasks, splits), _models(models)
    manifest = {
        "schema_version": 1, "scope": "full_main_final_evaluation_two_qwen_six_methods",
        "training_jobs_included": False,
        "training_prerequisite": "The separate main training protocol must be locked and completed before final evaluation.",
        "methods": list(METHODS), "models": selected, "protocol": protocol,
        "jobs": _jobs(protocol, selected), "expected_counts": dict(FULL_COUNTS),
        "expected_by_benchmark": {
            "made": {"jobs": 1800, "episodes": 1800, "candidate_oracle_attempts": 90000, "dft_episode_attempts": 0},
            "crystalgym": {"jobs": 360, "episodes": 1800, "candidate_oracle_attempts": 0, "dft_episode_attempts": 1800},
        },
    }
    manifest["manifest_fingerprint"] = fingerprint(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping) -> None:
    """Check both content hash and complete Cartesian product, not reported totals."""
    payload = {k: v for k, v in manifest.items() if k != "manifest_fingerprint"}
    if manifest.get("manifest_fingerprint") != fingerprint(payload):
        raise AccountingError("Manifest fingerprint mismatch")
    if manifest.get("methods") != list(METHODS) or [m["key"] for m in manifest["models"]] != list(MODEL_KEYS):
        raise AccountingError("Manifest omits a selected model or method")
    if manifest.get("training_jobs_included") is not False:
        raise AccountingError("This manifest cannot claim unspecified training jobs")
    if manifest.get("expected_counts") != FULL_COUNTS:
        raise AccountingError("Expected counts cannot be reduced")
    expected = _jobs(manifest["protocol"], manifest["models"])
    if manifest["jobs"] != expected or len({j["job_id"] for j in expected}) != FULL_COUNTS["jobs"]:
        raise AccountingError("Manifest does not contain the exact full job matrix")
    actual = {"jobs": len(expected), **{k: sum(j["expected_counts"][k] for j in expected) for k in COUNT_KEYS}}
    if actual != FULL_COUNTS:
        raise AccountingError("Actual matrix size does not match full expected counts")
    for benchmark in ("made", "crystalgym"):
        subset = [j for j in expected if j["benchmark"] == benchmark]
        counts = {"jobs": len(subset), **{k: sum(j["expected_counts"][k] for j in subset) for k in COUNT_KEYS}}
        if manifest["expected_by_benchmark"].get(benchmark) != counts:
            raise AccountingError("Benchmark-specific expected counts cannot be reduced")


def episode_key(summary: Mapping) -> tuple:
    """Policy seed is the independent replicate; packed environment seed is separate."""
    try:
        seed, episode = summary["seed"], summary["episode_id"]
        if type(seed) is not int or not isinstance(episode, (int, str)) or isinstance(episode, bool):
            raise AccountingError("seed must be an integer and episode_id a string/integer")
        keys = tuple(summary[k] for k in ("benchmark", "model_key", "method", "task_id"))
        if not all(isinstance(k, str) and k for k in keys):
            raise AccountingError("Episode identity fields must be nonempty strings")
        return (*keys, seed, str(episode))
    except KeyError as exc:
        raise AccountingError(f"Missing episode identity field: {exc}") from exc


def validate_made_budget_scope(expected_made_budget=50, core_protocol=None):
    """B10 is available only through the independently sealed FAST core.

    The caller's job or a numerical override alone cannot reduce a budget.  The
    original full-study ledger still validates its unchanged B50 manifest.
    """
    if type(expected_made_budget) is not int or expected_made_budget not in (10, 50):
        raise AccountingError("Unregistered MADE budget")
    if expected_made_budget == 50:
        return None
    if not isinstance(core_protocol, Mapping) or not core_protocol.get("workspace"):
        raise AccountingError("B10 requires a read_core-verified FAST protocol")
    from .core_protocol import FAST_CAUSAL_GRAPH_REGISTRATION, core_execution_budget, read_core
    try:
        verified = read_core(core_protocol["workspace"])
        if (verified != core_protocol or verified.get("registration") != FAST_CAUSAL_GRAPH_REGISTRATION
                or core_execution_budget(verified) != 10 or verified.get("budget") != 50):
            raise AccountingError("B10 scope does not match its sealed FAST protocol")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AccountingError("Unverifiable FAST MADE budget scope: " + str(exc)) from exc
    return verified


def expected_episodes(manifest: Mapping, *, expected_made_budget=50, core_protocol=None,
                      evaluation_extension=None, made_all30_extension=None, made_budget_sweep=None) -> dict[tuple, dict]:
    if made_budget_sweep is not None:
        if evaluation_extension is not None or made_all30_extension is not None:
            raise AccountingError("Cannot mix budget sweep and other evaluation admission scopes")
        from .made_budget_sweep import validate_job_scope
        try:
            jobs = manifest["jobs"]
            if not jobs or len({job["job_id"] for job in jobs}) != len(jobs):
                raise AccountingError("Missing or duplicate sweep jobs")
            for job in jobs:
                validate_job_scope(made_budget_sweep, job, expected_made_budget=expected_made_budget,
                                   core_protocol=core_protocol)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AccountingError("Unverifiable budget sweep admission: " + str(exc)) from exc
        verified = None
    elif made_all30_extension is not None:
        if evaluation_extension is not None:
            raise AccountingError("Cannot mix cumulative-five and all30 admission scopes")
        from .made_all30_extension import validate_job_scope
        try:
            jobs = manifest["jobs"]
            if not jobs or len({job["job_id"] for job in jobs}) != len(jobs):
                raise AccountingError("Missing or duplicate all30 jobs")
            for job in jobs:
                validate_job_scope(made_all30_extension, job, expected_made_budget=expected_made_budget,
                                   core_protocol=core_protocol)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AccountingError("Unverifiable all30 admission: " + str(exc)) from exc
        verified = None
    elif evaluation_extension is not None:
        from .evaluation_extension import validate_extension
        from .core_protocol import read_core
        try:
            extension = validate_extension(evaluation_extension)
            parent = read_core(extension["parent_core"]["workspace"])
            if core_protocol is not None and core_protocol != parent:
                raise AccountingError("Evaluation extension differs from supplied parent core")
            if type(expected_made_budget) is not int or expected_made_budget != 10:
                raise AccountingError("Evaluation extension requires its exact B10 budget")
            validate_made_budget_scope(expected_made_budget, parent)
            jobs = manifest["jobs"]
            if (len({job["job_id"] for job in jobs}) != len(jobs)
                    or any(job not in extension["new_jobs"] for job in jobs)):
                raise AccountingError("Job is not in the exact six new evaluation extension jobs")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AccountingError("Unverifiable evaluation extension: " + str(exc)) from exc
        verified = None
    else:
        verified = validate_made_budget_scope(expected_made_budget, core_protocol)
    if verified is not None:
        from .core_protocol import final_jobs
        registered_jobs = final_jobs(verified)
        if any(job not in registered_jobs for job in manifest["jobs"]):
            raise AccountingError("B10 result job is not in the sealed FAST final matrix")
    result = {}
    for job in manifest["jobs"]:
        if job["benchmark"] == "made" and (job.get("budget") != expected_made_budget
                or job.get("expected_counts", {}).get("candidate_oracle_attempts")
                != expected_made_budget * len(job["episode_ids"])):
            raise AccountingError("MADE job differs from the verified episode budget")
        for episode, env_seed in zip(job["episode_ids"], job["environment_seeds"], strict=True):
            result[episode_key({**job, "episode_id": episode})] = {
                "job_id": job["job_id"], "environment_seed": env_seed,
                "candidate_oracle_attempts": expected_made_budget if job["benchmark"] == "made" else 0,
                "dft_episode_attempts": 1 if job["benchmark"] == "crystalgym" else 0,
            }
    return result


def validate_episode(summary: Mapping, *, require_complete: bool = False) -> None:
    episode_key(summary)
    if type(summary.get("complete")) is not bool:
        raise AccountingError("Episode complete must be an explicit boolean")
    if not isinstance(summary.get("status"), str):
        raise AccountingError("Episode status is required")
    if summary["complete"] and summary["status"] not in TERMINAL_EPISODE_STATES:
        raise AccountingError("Complete episodes must have succeeded/failed terminal status")
    if require_complete and not summary["complete"]:
        raise AccountingError("Incomplete episode cannot count as a verified result")
    if not isinstance(summary.get("metrics"), dict) or not isinstance(summary.get("costs"), dict):
        raise AccountingError("Every episode needs metrics and costs dictionaries")
    if summary["complete"] and not summary["metrics"]:
        raise AccountingError("Complete episodes need metrics, including explicit nulls for undefined quantities")
    for name, value in summary["costs"].items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise AccountingError(f"Invalid nonnegative cost {name}")


def verify_result(path: str | Path, job: Mapping, manifest_fingerprint: str, *,
                  expected_made_budget=50, core_protocol=None, evaluation_extension=None,
                  made_all30_extension=None, made_budget_sweep=None) -> dict:
    """Verify result identity, every episode/budget, and every referenced raw artifact.

    Official scientific failures are terminal observations and may be verified.
    A job that stopped before its query budget was spent is incomplete. Artifacts
    must include raw attempts/logs; this is integrity verification, not a claim
    that hash checking itself proves a scientific measurement is valid.
    """
    try:
        path = Path(path).resolve()
        raw_result = path.read_bytes()
        result_hash = hashlib.sha256(raw_result).hexdigest()
        data = json.loads(raw_result)
        if not isinstance(data, dict):
            raise AccountingError("Result must be a JSON object envelope")
        if data.get("job_id") != job["job_id"] or data.get("manifest_fingerprint") != manifest_fingerprint:
            raise AccountingError("Result belongs to a different job or manifest")
        if data.get("model_revision") != job["model_revision"] or data.get("complete") is not True:
            raise AccountingError("Result model revision/complete marker is invalid")
        rows = data.get("episodes")
        if not isinstance(rows, list) or len(rows) != job["expected_counts"]["episodes"]:
            raise AccountingError("Result is missing expected episodes")
        expected = expected_episodes({"jobs": [job]}, expected_made_budget=expected_made_budget,
                                     core_protocol=core_protocol, evaluation_extension=evaluation_extension,
                                     made_all30_extension=made_all30_extension, made_budget_sweep=made_budget_sweep)
        seen, costs, all_costs, failed = set(), Counter(), Counter(), 0
        for row in rows:
            validate_episode(row, require_complete=True)
            key = episode_key(row)
            if key not in expected or key in seen:
                raise AccountingError("Unexpected or duplicated episode identity")
            seen.add(key)
            if row.get("environment_seed") != expected[key]["environment_seed"]:
                raise AccountingError("Episode environment seed does not match the frozen schedule")
            for name in COUNT_KEYS[1:]:
                actual = row["costs"].get(name, 0)
                if type(actual) is not int or actual != expected[key][name]:
                    raise AccountingError(f"Episode did not account for the exact {name} budget")
                costs[name] += actual
            all_costs.update(row["costs"])
            failed += row["status"] == "failed"
        artifacts = data.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise AccountingError("Result needs hashed raw evidence artifacts")
        artifact_paths = set()
        for artifact in artifacts:
            target = Path(artifact["path"])
            target = target if target.is_absolute() else path.parent / target
            target = target.resolve()
            if target == path or target in artifact_paths:
                raise AccountingError("Raw evidence cannot be the result itself or a duplicate")
            artifact_paths.add(target)
            if file_sha256(target) != artifact["sha256"]:
                raise AccountingError(f"Raw artifact hash mismatch: {target}")
        if file_sha256(path) != result_hash:
            raise AccountingError("Result changed during verification")
        return {"result_path": str(path), "result_sha256": result_hash,
                "episodes": len(rows), "failed_episodes": failed,
                "costs": dict(all_costs), **dict(costs)}
    except (OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise AccountingError(f"Unverifiable result {path}: {exc}") from exc


class RunLedger:
    """SQLite claims and append-only events; failed/orphaned work is never auto-retried.

    Keep this database on a filesystem with working SQLite locking (not copied
    independently to workers). Workers report the returned attempt_id with every
    update. Retain failed attempt artifacts when explicitly authorizing a retry.
    """

    def __init__(self, path: str | Path, manifest: Mapping):
        validate_manifest(manifest)
        self.manifest = json.loads(canonical_json(manifest))
        self.jobs = {j["job_id"]: j for j in self.manifest["jobs"]}
        self.fingerprint = self.manifest["manifest_fingerprint"]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, state TEXT NOT NULL, attempt_number INTEGER NOT NULL DEFAULT 0,
                attempt_id TEXT, owner TEXT, heartbeat REAL, result_path TEXT, result_sha256 TEXT,
                error TEXT, updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                at REAL NOT NULL, event TEXT NOT NULL, details TEXT NOT NULL
            );
        """)
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()
            if existing is not None and existing[0] != canonical_json(self.manifest):
                raise AccountingError("Ledger manifest is immutable; scope/budget changes require a distinct study ledger")
            if existing is None:
                self.connection.execute("INSERT INTO metadata VALUES ('manifest', ?)", (canonical_json(self.manifest),))
                self.connection.executemany("INSERT INTO jobs(job_id,state,updated_at) VALUES (?,'pending',?)", [(job_id, time.time()) for job_id in self.jobs])
            stored = {r[0] for r in self.connection.execute("SELECT job_id FROM jobs")}
            if stored != set(self.jobs):
                raise AccountingError("Ledger job membership differs from the immutable manifest")
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def get(self, job_id: str) -> dict:
        row = self.connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise AccountingError("Unknown job id")
        return dict(row)

    def _event(self, job_id: str, event: str, details: Mapping) -> None:
        self.connection.execute("INSERT INTO events(job_id,at,event,details) VALUES (?,?,?,?)", (job_id, time.time(), event, canonical_json(details)))

    def claim(self, job_id: str, owner: str) -> dict | None:
        """Return the exclusive attempt token, or None if the job is not pending."""
        if not owner:
            raise AccountingError("A worker owner identifier is required")
        self.get(job_id)
        token, now = uuid.uuid4().hex, time.time()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.connection.execute("UPDATE jobs SET state='running',attempt_number=attempt_number+1,attempt_id=?,owner=?,heartbeat=?,updated_at=?,error=NULL WHERE job_id=? AND state='pending'", (token, owner, now, now, job_id))
            if cursor.rowcount:
                self._event(job_id, "claimed", {"owner": owner, "attempt_id": token})
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return {"job": json.loads(canonical_json(self.jobs[job_id])), "attempt_id": token} if cursor.rowcount else None

    def heartbeat(self, job_id: str, attempt_id: str) -> None:
        now = time.time()
        result = self.connection.execute("UPDATE jobs SET heartbeat=?,updated_at=? WHERE job_id=? AND state='running' AND attempt_id=?", (now, now, job_id, attempt_id))
        if result.rowcount != 1:
            raise AccountingError("Heartbeat does not own a running attempt")

    def _finish(self, job_id: str, attempt_id: str, state: str, details: Mapping) -> None:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.connection.execute("UPDATE jobs SET state=?,result_path=?,result_sha256=?,error=?,updated_at=? WHERE job_id=? AND state='running' AND attempt_id=?", (state, details.get("result_path"), details.get("result_sha256"), details.get("error"), time.time(), job_id, attempt_id))
            if cursor.rowcount != 1:
                raise AccountingError("Only the owner of a running attempt can finish it")
            self._event(job_id, state, {"attempt_id": attempt_id, **details})
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def succeed(self, job_id: str, attempt_id: str, result_path: str | Path) -> dict:
        verification = verify_result(result_path, self.jobs[job_id], self.fingerprint)
        self._finish(job_id, attempt_id, "succeeded", verification)
        return verification

    def fail(
        self, job_id: str, attempt_id: str, error: str, *, execution_stopped: bool,
        costs: Mapping | None = None, evidence_paths=(),
    ) -> None:
        """Known stopped attempts only; an uncertain remote timeout must be orphaned."""
        if execution_stopped is not True or not error:
            raise AccountingError("Failure needs evidence that physical execution stopped")
        if costs is not None and any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in costs.values()):
            raise AccountingError("Failed-attempt costs must be finite and nonnegative")
        artifacts = [{"path": str(Path(p).resolve()), "sha256": file_sha256(p)} for p in evidence_paths]
        self._finish(job_id, attempt_id, "failed", {
            "error": error, "execution_stopped": True,
            "costs": dict(costs) if costs is not None else None, "artifacts": artifacts,
        })

    def events(self, job_id: str) -> list[dict]:
        """Include old attempt evidence/costs after a deliberate retry."""
        self.get(job_id)
        return [{**dict(row), "details": json.loads(row["details"])} for row in self.connection.execute("SELECT * FROM events WHERE job_id=? ORDER BY event_id", (job_id,))]

    def mark_orphaned(self, *, stale_before: float) -> list[str]:
        """Quarantine stale running jobs; never assume lease expiry cancels computation."""
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            rows = list(self.connection.execute("SELECT job_id,attempt_id FROM jobs WHERE state='running' AND heartbeat < ?", (stale_before,)))
            for row in rows:
                self.connection.execute("UPDATE jobs SET state='orphaned',updated_at=? WHERE job_id=?", (time.time(), row["job_id"]))
                self._event(row["job_id"], "orphaned", {"attempt_id": row["attempt_id"], "reason": "heartbeat_expired_execution_state_unknown"})
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return [row["job_id"] for row in rows]

    def reconcile(self, job_id: str, *, evidence: str, execution_stopped: bool, result_path: str | Path | None = None) -> None:
        """Resolve an orphan to verified success or known failure; does not retry it."""
        if execution_stopped is not True or not evidence:
            raise AccountingError("Explicit reconciliation evidence of stopped execution is required")
        details = verify_result(result_path, self.jobs[job_id], self.fingerprint) if result_path else {}
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            if self.get(job_id)["state"] != "orphaned":
                raise AccountingError("Only orphaned attempts require reconciliation")
            state = "succeeded" if result_path else "failed"
            self.connection.execute("UPDATE jobs SET state=?,result_path=?,result_sha256=?,error=?,updated_at=? WHERE job_id=?", (state, details.get("result_path"), details.get("result_sha256"), None if result_path else evidence, time.time(), job_id))
            self._event(job_id, "reconciled", {"evidence": evidence, "execution_stopped": True, "state": state, **details})
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def retry_failed(self, job_id: str, *, evidence: str, previous_attempts_accounted: bool) -> None:
        """Explicit retry authorization; past calls/costs remain in attempt artifacts/events."""
        if not evidence or previous_attempts_accounted is not True:
            raise AccountingError("Retry requires explicit accounting for all previous physical calls")
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            if self.get(job_id)["state"] != "failed":
                raise AccountingError("Only known-stopped failed attempts can be retried")
            self.connection.execute("UPDATE jobs SET state='pending',updated_at=? WHERE job_id=?", (time.time(), job_id))
            self._event(job_id, "explicit_retry_authorized", {"evidence": evidence, "previous_attempts_accounted": True})
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def resume_plan(self) -> dict:
        """Only verified complete results are skippable; corrupt results are quarantined."""
        plan = {"runnable": [], "skip_verified": [], "running": [], "failed": [], "orphaned": []}
        for job_id in self.jobs:
            row = self.get(job_id)
            if row["state"] == "succeeded":
                try:
                    result = verify_result(row["result_path"], self.jobs[job_id], self.fingerprint)
                    if result["result_sha256"] != row["result_sha256"]:
                        raise AccountingError("Previously recorded result file changed")
                except AccountingError as exc:
                    self.connection.execute("BEGIN IMMEDIATE")
                    try:
                        changed = self.connection.execute("UPDATE jobs SET state='orphaned',error=?,updated_at=? WHERE job_id=? AND state='succeeded'", (str(exc), time.time(), job_id))
                        if changed.rowcount:
                            self._event(job_id, "verification_failed", {"error": str(exc)})
                        self.connection.execute("COMMIT")
                    except Exception:
                        self.connection.execute("ROLLBACK")
                        raise
                    row = self.get(job_id)
                else:
                    plan["skip_verified"].append(job_id)
                    continue
            plan["runnable" if row["state"] == "pending" else row["state"]].append(job_id)
        return plan

    def completion(self) -> dict:
        plan = self.resume_plan()
        verified = Counter({key: 0 for key in COUNT_KEYS})
        failed_episodes = 0
        costs = Counter()
        for job_id in plan["skip_verified"]:
            row = self.get(job_id)
            result = verify_result(row["result_path"], self.jobs[job_id], self.fingerprint)
            verified.update({key: result.get(key, 0) for key in COUNT_KEYS})
            failed_episodes += result["failed_episodes"]
            costs.update(result["costs"])
        return {
            "manifest_fingerprint": self.fingerprint, "expected_counts": dict(FULL_COUNTS),
            "verified_counts": {"jobs": len(plan["skip_verified"]), **dict(verified)},
            "failed_scientific_episodes": failed_episodes,
            "verified_result_costs": dict(costs),
            "previous_attempt_costs_policy": "Inspect immutable attempt events and raw evidence; verified-result costs exclude failed/retried attempts.",
            "states": {key: len(value) for key, value in plan.items()},
            "complete": len(plan["skip_verified"]) == FULL_COUNTS["jobs"],
        }
