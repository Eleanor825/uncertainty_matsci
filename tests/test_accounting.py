"""Synthetic bookkeeping checks only; no material evaluation is performed."""

import copy
import json
from pathlib import Path
import time

import pytest

from matdiscovery.accounting import (
    AccountingError, FULL_COUNTS, RunLedger, build_final_manifest,
    build_final_manifest_from_configs, file_sha256, fingerprint,
    pack_environment_seed, validate_manifest, verify_result, write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def manifest():
    return build_final_manifest(ROOT)


def result_file(directory, job, manifest, *, failed=False):
    """Clearly synthetic test evidence, never a benchmark result."""
    directory.mkdir(exist_ok=True)
    raw = directory / "synthetic_attempts.jsonl"
    raw.write_text('{"test_fixture_only":true}\n')
    rows = []
    for episode, env_seed in zip(job["episode_ids"], job["environment_seeds"]):
        rows.append({
            **{key: job[key] for key in ("benchmark", "model_key", "method", "task_id", "seed")},
            "episode_id": episode, "environment_seed": env_seed, "complete": True,
            "status": "failed" if failed else "succeeded",
            "metrics": {"reward": -1.0 if failed else 1.0, "failure_rate": int(failed)},
            "costs": {"candidate_oracle_attempts": 50 if job["benchmark"] == "made" else 0,
                      "dft_episode_attempts": int(job["benchmark"] == "crystalgym")},
        })
    data = {"job_id": job["job_id"], "manifest_fingerprint": manifest["manifest_fingerprint"],
            "model_revision": job["model_revision"], "complete": True, "episodes": rows,
            "artifacts": [{"path": raw.name, "sha256": file_sha256(raw)}]}
    path = directory / "result.json"
    write_json_atomic(path, data)
    return path


def test_complete_main_matrix_and_disjoint_seed_schedule(manifest):
    assert manifest["expected_counts"] == FULL_COUNTS
    assert len(manifest["jobs"]) == 2160
    assert len({job["job_id"] for job in manifest["jobs"]}) == 2160
    assert manifest["training_jobs_included"] is False
    for model in ("qwen35_4b", "qwen35_9b"):
        for method in manifest["methods"]:
            made = [j for j in manifest["jobs"] if (j["model_key"], j["method"], j["benchmark"]) == (model, method, "made")]
            crystal = [j for j in manifest["jobs"] if (j["model_key"], j["method"], j["benchmark"]) == (model, method, "crystalgym")]
            assert len(made) == 150
            assert len(crystal) == 30
            assert sum(j["expected_counts"]["candidate_oracle_attempts"] for j in made) == 7500
            assert sum(j["expected_counts"]["episodes"] for j in crystal) == 150
            assert len({s for j in crystal for s in j["environment_seeds"]}) == 150
    assert pack_environment_seed(1, "band_gap", "C1", 0) == 2214592512
    assert pack_environment_seed(1, "band_gap", "C7", 0) == 2239758336
    assert pack_environment_seed(1, "band_gap", "training_mixture", 0, partition="uncertainty_calibration_dev") == 1170210816
    with pytest.raises(AccountingError):
        pack_environment_seed(1, "bm", "C1", 0, partition="uncertainty_fit_and_es_train")


def test_manifest_and_ids_are_independent_of_input_enumeration_order(manifest):
    tasks, splits, models = [json.loads((ROOT / "configs" / f"{name}.json").read_text()) for name in ("benchmark_tasks", "made_splits", "model_manifest")]
    for field in ("systems", "seeds"):
        tasks["made"][field].reverse()
    for field in ("properties", "prototypes", "seeds", "train_indices"):
        tasks["crystalgym"][field].reverse()
    tasks["crystalgym"]["evaluation"]["rollout_ids"].reverse()
    for systems in splits["splits"].values():
        systems.reverse()
        for elements in systems:
            elements.reverse()
    models["models"].reverse()
    models["selection"]["ordered_model_keys"].reverse()
    for model in models["models"]:
        model["weight_files"].reverse()
        model["metadata_files"].reverse()
    rebuilt = build_final_manifest_from_configs(tasks, splits, models)
    assert rebuilt == manifest


def test_scope_reduction_cannot_redefine_completion(manifest, tmp_path):
    reduced = copy.deepcopy(manifest)
    reduced["jobs"].pop()
    reduced["expected_counts"]["jobs"] -= 1
    reduced["manifest_fingerprint"] = fingerprint({k: v for k, v in reduced.items() if k != "manifest_fingerprint"})
    with pytest.raises(AccountingError, match="reduced"):
        validate_manifest(reduced)
    with RunLedger(tmp_path / "ledger.db", manifest) as ledger:
        assert ledger.completion()["complete"] is False
    changed = copy.deepcopy(manifest)
    changed["protocol"]["made"]["max_atoms"] = 19
    # Rebuilding an otherwise valid manifest still must not replace an existing ledger.
    from matdiscovery.accounting import _jobs
    changed["jobs"] = _jobs(changed["protocol"], changed["models"])
    changed["manifest_fingerprint"] = fingerprint({k: v for k, v in changed.items() if k != "manifest_fingerprint"})
    with pytest.raises(AccountingError, match="immutable"):
        RunLedger(tmp_path / "ledger.db", changed)


def test_verified_resume_and_modified_raw_evidence_quarantine(manifest, tmp_path):
    job = manifest["jobs"][0]
    path = result_file(tmp_path, job, manifest, failed=True)
    with RunLedger(tmp_path / "ledger.db", manifest) as ledger:
        claimed = ledger.claim(job["job_id"], "worker-1")
        assert ledger.claim(job["job_id"], "worker-2") is None
        verification = ledger.succeed(job["job_id"], claimed["attempt_id"], path)
        assert verification["failed_episodes"] == len(job["episode_ids"])
        plan = ledger.resume_plan()
        assert plan["skip_verified"] == [job["job_id"]]
        assert ledger.completion()["failed_scientific_episodes"] == len(job["episode_ids"])
    with RunLedger(tmp_path / "ledger.db", manifest) as resumed:
        assert resumed.resume_plan()["skip_verified"] == [job["job_id"]]
        (tmp_path / "synthetic_attempts.jsonl").write_text("corrupted\n")
        plan = resumed.resume_plan()
        assert plan["skip_verified"] == []
        assert plan["orphaned"] == [job["job_id"]]
        assert resumed.claim(job["job_id"], "worker-3") is None


def test_exclusive_claims_across_connections_and_orphan_reconciliation(manifest, tmp_path):
    job = manifest["jobs"][0]
    with RunLedger(tmp_path / "ledger.db", manifest) as first, RunLedger(tmp_path / "ledger.db", manifest) as second:
        claim = first.claim(job["job_id"], "worker-1")
        assert second.claim(job["job_id"], "worker-2") is None
        with pytest.raises(AccountingError):
            second.heartbeat(job["job_id"], "incorrect-token")
        first.heartbeat(job["job_id"], claim["attempt_id"])
        assert second.mark_orphaned(stale_before=time.time() + 1) == [job["job_id"]]
        assert first.claim(job["job_id"], "worker-3") is None
        with pytest.raises(AccountingError):
            first.retry_failed(job["job_id"], evidence="only a timeout", previous_attempts_accounted=True)
        with pytest.raises(AccountingError):
            first.reconcile(job["job_id"], evidence="only a timeout", execution_stopped=False)
        first.reconcile(job["job_id"], evidence="synthetic worker and child process joined; no evaluator call started", execution_stopped=True)
        assert first.get(job["job_id"])["state"] == "failed"
        assert first.claim(job["job_id"], "worker-3") is None
        first.retry_failed(job["job_id"], evidence="synthetic attempt evidence audited; zero calls", previous_attempts_accounted=True)
        assert second.claim(job["job_id"], "worker-3")
        assert first.get(job["job_id"])["attempt_number"] == 2


@pytest.mark.parametrize("mutation", ["missing_episode", "wrong_seed", "wrong_revision", "budget_shortfall", "missing_raw"])
def test_result_completeness_is_verified_before_success(manifest, tmp_path, mutation):
    job = next(j for j in manifest["jobs"] if j["benchmark"] == "crystalgym")
    path = result_file(tmp_path, job, manifest)
    data = json.loads(path.read_text())
    if mutation == "missing_episode":
        data["episodes"].pop()
    elif mutation == "wrong_seed":
        data["episodes"][0]["environment_seed"] += 1
    elif mutation == "wrong_revision":
        data["model_revision"] = "unlocked"
    elif mutation == "budget_shortfall":
        data["episodes"][0]["costs"]["dft_episode_attempts"] = 0
    else:
        data["artifacts"] = []
    write_json_atomic(path, data)
    with pytest.raises(AccountingError):
        verify_result(path, job, manifest["manifest_fingerprint"])


def test_changed_summary_is_not_skipped_even_when_new_summary_is_valid(manifest, tmp_path):
    job = manifest["jobs"][0]
    path = result_file(tmp_path, job, manifest)
    with RunLedger(tmp_path / "ledger.db", manifest) as ledger:
        claim = ledger.claim(job["job_id"], "worker")
        ledger.succeed(job["job_id"], claim["attempt_id"], path)
        data = json.loads(path.read_text())
        data["episodes"][0]["metrics"]["reward"] = 999
        write_json_atomic(path, data)
        assert ledger.resume_plan()["orphaned"] == [job["job_id"]]


def test_claim_cannot_mutate_manifest_and_failed_attempt_costs_are_preserved(manifest, tmp_path):
    job = manifest["jobs"][0]
    with RunLedger(tmp_path / "ledger.db", manifest) as ledger:
        claim = ledger.claim(job["job_id"], "worker")
        claim["job"]["budget"] = 1
        assert ledger.jobs[job["job_id"]]["budget"] == 5
        log = tmp_path / "synthetic_failed_attempt.json"
        log.write_text('{"test_fixture_only":true}')
        ledger.fail(job["job_id"], claim["attempt_id"], "synthetic stopped process", execution_stopped=True, costs={"dft_episode_attempts": 1, "elapsed_seconds": 3.5}, evidence_paths=[log])
        ledger.retry_failed(job["job_id"], evidence="costs retained and reviewed", previous_attempts_accounted=True)
        event = next(e for e in ledger.events(job["job_id"]) if e["event"] == "failed")
        assert event["details"]["costs"]["elapsed_seconds"] == 3.5
        assert event["details"]["artifacts"][0]["sha256"] == file_sha256(log)


@pytest.mark.parametrize("contents", [b"[]", b"not json", b"\xff\xfe"])
def test_malformed_result_envelopes_are_rejected(manifest, tmp_path, contents):
    path = tmp_path / "malformed.json"
    path.write_bytes(contents)
    with pytest.raises(AccountingError):
        verify_result(path, manifest["jobs"][0], manifest["manifest_fingerprint"])
