"""Synthetic CPU extension contracts; no real materials, models, or oracle calls."""
from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from matdiscovery import evaluation_extension as extmod
from matdiscovery.accounting import AccountingError, expected_episodes, file_sha256, fingerprint, verify_result, write_json_atomic
from matdiscovery.evaluation_extension import build_extension, extension_jobs, read_extension, validate_extension
from test_fast_core_accounting import fast_core, fast_manifest, zero_selected
from test_training_jobs import project


@pytest.fixture
def complete_parent(fast_core, tmp_path, monkeypatch):
    """Actual shared envelope/RPC verifier, tiny policy and scalar fake oracle."""
    from matdiscovery.esopt import tensor_state_hash
    from test_core_final_integration import runner_fixture
    from test_fast_es_budget import b10_trajectory
    parent = Path(fast_core["workspace"])
    monkeypatch.setattr(extmod, "PARENT_CORE_FINGERPRINT", fast_core["fingerprint"])
    write_json_atomic(parent / "source_manifest.json", {"classification": "synthetic unit source only"})
    calls = []
    runner = runner_fixture(parent, tmp_path, calls)
    runner.output = parent / "experiments/core_final"
    runner.core, runner.manifest = fast_core, fast_manifest(fast_core)
    def selected(policy, directory, **kwargs):
        base = tensor_state_hash(dict(policy.model.state_dict()))
        value = zero_selected(base); value["artifacts"] = {}
        policy.mark_state("reload", generation=1)
        return value
    runner.selected_loader = selected
    class UnitRollout:
        def __init__(self, root, policy, **kwargs): self.project = root
        def run(self, job, output, *, collection):
            assert not collection
            calls.append(job["job_id"])
            b10_trajectory(self.project, job, output, discovered=job["task_id"] == "Al-V-Zn")
    runner.rollout_factory = UnitRollout
    runner.run()
    assert len(calls) == 4
    return {"core": fast_core, "runner": runner, "calls": calls, "workspace": tmp_path / "extension"}


@pytest.fixture
def extension(complete_parent):
    return build_extension(complete_parent["core"], complete_parent["workspace"])


def reseal(value):
    value["fingerprint"] = value["extension_fingerprint"] = fingerprint(
        {k: v for k, v in value.items() if k not in {"fingerprint", "extension_fingerprint"}})
    write_json_atomic(Path(value["workspace"]) / "configs/evaluation_extension.json", value)
    return value


def new_envelope(fixture, extension, tmp_path, *, method="baseline", failed_last=False):
    """Produce a registered B10 trajectory with the existing real audit recorder."""
    from matdiscovery.final_evaluation import execution_job
    from matdiscovery.training_jobs import TrainingJobCallbacks, reconstruct_scientific_evidence
    from test_fast_es_budget import b10_trajectory
    job = next(j for j in extension["new_jobs"] if j["method"] == method)
    profile = deepcopy(extension["execution_profiles"][method])
    from types import SimpleNamespace
    policy = SimpleNamespace(configuration_fingerprint=profile["policy_configuration_fingerprint"],
        checkpoint_hash=profile["initial_checkpoint_manifest_hash"],
        get_state_id=lambda: "synthetic extension session",
        model_stamp=SimpleNamespace(generation=profile["selected_es"]["selected_generation"] if profile["selected_es"] else 0))
    actual = execution_job(job, policy, profile)
    actual["group_id"] = job["group_id"]
    actual["source"] = "registered_evaluation_extension"
    output = tmp_path / "new-trajectory"
    b10_trajectory(Path(fixture["core"]["workspace"]), actual, output, discovered=True, failed_last=failed_last)
    proof = reconstruct_scientific_evidence(output, actual, tasks=fixture["runner"].tasks,
        partition="final_test", expected_mace_num_workers=4, expected_made_budget=10, core_protocol=fixture["core"])
    value = {"job_id": job["job_id"], "manifest_fingerprint": extension["fingerprint"],
        "model_revision": job["model_revision"], "complete": True,
        "episodes": json.loads((output / "episodes.json").read_text()),
        "artifacts": TrainingJobCallbacks._inventory(output), "evidence_schema": "final_official_rpc_v1",
        "scientific_evidence": proof, "execution_job": actual, "execution_profile": profile,
        "execution_profile_fingerprint": fingerprint(profile)}
    path = output / "result.json"; write_json_atomic(path, value)
    return job, path


def test_exact_cumulative_matrix_immutable_parent_and_idempotent_build(complete_parent):
    parent = Path(complete_parent["core"]["workspace"])
    before = {p: (file_sha256(p), p.stat().st_mtime_ns) for p in parent.rglob("*") if p.is_file()}
    extension = build_extension(complete_parent["core"], complete_parent["workspace"])
    assert tuple(t["id"] for t in extension["new_tasks"]) == extmod.NEW_TASKS
    assert len(extension_jobs(extension)) == 6
    assert len(extension_jobs(extension, include_imported=True)) == 10
    assert extension["expected_new_counts"]["candidate_oracle_attempts"] == 60
    assert extension["expected_cumulative_counts"]["candidate_oracle_attempts"] == 100
    assert extension["training_jobs"] == extension["es_jobs"] == []
    assert extension["registration"]["expansion_registered_after_initial_two_system_results_known"] is True
    assert not extension["registration"]["checkpoint_reselection"]
    path = complete_parent["workspace"] / "configs/evaluation_extension.json"
    initial = (path.read_bytes(), path.stat().st_mtime_ns)
    assert read_extension(path) == build_extension(complete_parent["core"], complete_parent["workspace"]) == extension
    assert (path.read_bytes(), path.stat().st_mtime_ns) == initial
    assert all((file_sha256(p), p.stat().st_mtime_ns) == value for p, value in before.items())
    assert len(complete_parent["calls"]) == 4  # importing never invokes another rollout


@pytest.mark.parametrize("field", ["budget", "task", "seed", "method", "count", "omit", "duplicate", "profile", "disclosure", "train"])
def test_rehashed_scope_or_job_mutation_is_rejected(extension, field):
    bad = deepcopy(extension)
    if field == "budget": bad["new_jobs"][0]["budget"] = 50
    elif field == "task": bad["new_jobs"][0]["task"]["elements"] = ["Al", "Li", "V"]
    elif field == "seed": bad["new_jobs"][0]["environment_seeds"] = [2]
    elif field == "method": bad["new_jobs"][0]["method"] = "esopt"
    elif field == "count": bad["expected_cumulative_counts"]["jobs"] = 6
    elif field == "omit": bad["new_jobs"].pop()
    elif field == "duplicate": bad["new_jobs"][1] = deepcopy(bad["new_jobs"][0])
    elif field == "profile": bad["execution_profiles"]["esopt_graph_risk"]["actual_model_state_hash"] = "another policy"
    elif field == "disclosure": bad["registration"]["expansion_registered_after_initial_two_system_results_known"] = False
    else: bad["training_jobs"] = [{"unregistered": True}]
    reseal(bad)
    with pytest.raises(ValueError, match="changed"):
        validate_extension(bad)


@pytest.mark.parametrize("target", ["result", "raw", "completion", "stage_receipt", "parent_protocol", "parent_source"])
def test_parent_byte_changes_or_missing_proofs_fail_closed(extension, target):
    if target == "result": path = Path(extension["imported_results"][0]["result"]["path"])
    elif target == "raw":
        result = json.loads(Path(extension["imported_results"][0]["result"]["path"]).read_text())
        path = Path(result["artifacts"][0]["path"])
    elif target == "parent_protocol": path = Path(extension["parent_core"]["path"])
    elif target == "parent_source": path = Path(extension["scientific_parent_inputs"][-1]["path"])
    else: path = Path(extension["parent_final"][target]["path"])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError): validate_extension(extension)


def test_wrong_parent_and_parent_nested_workspace_are_rejected(complete_parent, monkeypatch):
    monkeypatch.setattr(extmod, "PARENT_CORE_FINGERPRINT", "f" * 64)
    with pytest.raises(ValueError, match="exact completed FAST"):
        build_extension(complete_parent["core"], complete_parent["workspace"])
    monkeypatch.setattr(extmod, "PARENT_CORE_FINGERPRINT", complete_parent["core"]["fingerprint"])
    with pytest.raises(ValueError, match="separate"):
        build_extension(complete_parent["core"], Path(complete_parent["core"]["workspace"]) / "new-extension")


def test_budget_verifier_opt_in_membership_and_legacy_defaults(complete_parent, extension, tmp_path):
    job, path = new_envelope(complete_parent, extension, tmp_path)
    kwargs = {"expected_made_budget": 10, "core_protocol": complete_parent["core"]}
    with pytest.raises(AccountingError): verify_result(path, job, extension["fingerprint"])
    with pytest.raises(AccountingError, match="sealed FAST final matrix"):
        verify_result(path, job, extension["fingerprint"], **kwargs)
    verified = verify_result(path, job, extension["fingerprint"], **kwargs, evaluation_extension=extension)
    assert verified["candidate_oracle_attempts"] == 10
    for budget in (True, 9, 50):
        with pytest.raises(AccountingError):
            expected_episodes({"jobs": [job]}, expected_made_budget=budget, evaluation_extension=extension)
    for invalid in ([{**job, "seed": 2}], [job, job], [extension["imported_results"][0]["job"]]):
        with pytest.raises(AccountingError, match="six new"):
            expected_episodes({"jobs": invalid}, **kwargs, evaluation_extension=extension)
    badparent = deepcopy(complete_parent["core"]); badparent["execution_budget"] = 50
    with pytest.raises(AccountingError, match="parent core"):
        expected_episodes({"jobs": [job]}, expected_made_budget=10, core_protocol=badparent, evaluation_extension=extension)
    assert len(expected_episodes({"jobs": [job]}, expected_made_budget=10, evaluation_extension=extension)) == 1


@pytest.mark.parametrize("method", extmod.METHODS)
def test_shared_final_verifier_uses_exact_parent_profile_and_real_b10_rpc(complete_parent, extension, tmp_path, method):
    from matdiscovery.final_evaluation import verify_final_envelope
    job, path = new_envelope(complete_parent, extension, tmp_path, method=method, failed_last=True)
    kwargs = {"tasks": complete_parent["runner"].tasks, "expected_mace_num_workers": 4,
        "expected_made_budget": 10, "core_protocol": complete_parent["core"], "evaluation_extension": extension}
    verified = verify_final_envelope(path, job, extension["fingerprint"], **kwargs)
    assert verified["candidate_oracle_attempts"] == 10  # real-shaped failed attempt stays in B10 denominator
    wrong = deepcopy(extension["execution_profiles"][method]); wrong["actual_model_state_hash"] = "reselected"
    with pytest.raises(ValueError, match="reselect"):
        verify_final_envelope(path, job, extension["fingerprint"], expected_profile=wrong, **kwargs)


def test_validation_and_additional_source_are_hashed_without_claiming_science(complete_parent, tmp_path):
    validation = tmp_path / "cpu.json"; write_json_atomic(validation, {"passed": True, "classification": "unit test only"})
    source = tmp_path / "runner.py"; source.write_text("# synthetic unit runner; never scientific execution\n")
    extension = build_extension(complete_parent["core"], complete_parent["workspace"], source_files=[source], validation=validation)
    assert extension["implementation_validation_supplied"] is True
    source.write_text("# changed\n")
    with pytest.raises(ValueError, match="artifact changed"): validate_extension(extension)


def shards_for(fixture):
    jobs = extmod.planned_extension_jobs(fixture["core"])
    primary = [j["job_id"] for j in jobs if j["method"] == "baseline" or j["task_id"] == "Au-K-Tb"]
    secondary = [j["job_id"] for j in jobs if j["job_id"] not in primary]
    return [{"shard_id": name, "execution_site": name,
             "output_workspace": str(fixture["workspace"] / "shards" / name), "job_ids": ids}
            for name, ids in (("primary", primary), ("secondary", secondary))]


def test_static_two_sites_have_exact_disjoint_union_and_no_runtime_reassignment(complete_parent):
    shards = shards_for(complete_parent)
    extension = build_extension(complete_parent["core"], complete_parent["workspace"], execution_shards=shards)
    assert [len(extension_jobs(extension, shard_id=name)) for name in ("primary", "secondary")] == [4, 2]
    assert len(extension_jobs(extension)) == 6
    assert extension["expected_cumulative_counts"]["jobs"] == 10
    assert build_extension(complete_parent["core"], complete_parent["workspace"], execution_shards=shards) == extension
    with pytest.raises(ValueError, match="cannot be replaced"):
        build_extension(complete_parent["core"], complete_parent["workspace"])
    with pytest.raises(ValueError, match="imported"):
        extension_jobs(extension, shard_id="primary", include_imported=True)
    with pytest.raises(ValueError, match="Unregistered"):
        extension_jobs(extension, shard_id="third")


@pytest.mark.parametrize("mutation", ["overlap", "missing", "imported", "shared_output", "nested_output", "outside"])
def test_invalid_static_shards_cannot_be_sealed(complete_parent, mutation):
    shards = shards_for(complete_parent)
    if mutation == "overlap": shards[1]["job_ids"][0] = shards[0]["job_ids"][0]
    elif mutation == "missing": shards[1]["job_ids"].pop()
    elif mutation == "imported": shards[1]["job_ids"][0] = complete_parent["runner"].manifest["jobs"][0]["job_id"]
    elif mutation == "shared_output": shards[1]["output_workspace"] = shards[0]["output_workspace"]
    elif mutation == "nested_output": shards[1]["output_workspace"] = str(Path(shards[0]["output_workspace"]) / "child")
    else: shards[1]["output_workspace"] = str(Path(complete_parent["core"]["workspace"]))
    with pytest.raises(ValueError):
        build_extension(complete_parent["core"], complete_parent["workspace"], execution_shards=shards)
    assert not (complete_parent["workspace"] / "configs/evaluation_extension.json").exists()


@pytest.mark.parametrize("mutation", ["same_size_restored_mtime", "atomic_replace", "symlink_retarget"])
def test_profile_cache_rejects_artifact_changes_even_when_size_and_mtime_match(tmp_path, mutation):
    from matdiscovery.final_evaluation import _verify_profile_artifacts
    from matdiscovery.immutable_hash_cache import immutable_hash_cache
    original = tmp_path / "weights.bin"; original.write_bytes(b"AAAA")
    path = tmp_path / "link.bin" if mutation == "symlink_retarget" else original
    if path != original: path.symlink_to(original)
    before = original.stat()
    profile = {"artifacts": {str(path): file_sha256(path)}}
    with immutable_hash_cache():
        _verify_profile_artifacts(profile)
        if mutation == "same_size_restored_mtime":
            original.write_bytes(b"BBBB")
            os.utime(original, ns=(before.st_atime_ns, before.st_mtime_ns))
        else:
            replacement = tmp_path / "replacement.bin"; replacement.write_bytes(b"BBBB")
            os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
            if mutation == "atomic_replace": replacement.replace(original)
            else:
                path.unlink(); path.symlink_to(replacement)
        assert path.stat().st_size == before.st_size and path.stat().st_mtime_ns == before.st_mtime_ns
        with pytest.raises(ValueError, match="checksum"):
            _verify_profile_artifacts(profile)


def test_profile_cache_rejects_mutation_after_hash_before_publication(tmp_path, monkeypatch):
    from matdiscovery import final_evaluation
    path = tmp_path / "weights.bin"; path.write_bytes(b"AAAA")
    expected = file_sha256(path)
    def hash_then_change(target):
        digest = file_sha256(target)
        target.write_bytes(b"BBBB")
        return digest
    monkeypatch.setattr(final_evaluation, "file_sha256", hash_then_change)
    with pytest.raises(ValueError, match="changed during hashing"):
        final_evaluation._verify_profile_artifacts({"artifacts": {str(path): expected}})
