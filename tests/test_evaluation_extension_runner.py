"""Static evaluation shard/recovery contracts using tiny CPU, synthetic RPC only.

The shared final verifier and real oracle-attempt journal recorder remain in the
path. Fixtures never run a scientific oracle, GPU, pretrained model, or experiment.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from matdiscovery import evaluation_extension_runner as mod
from matdiscovery.accounting import AccountingError, file_sha256, fingerprint, write_json_atomic
from matdiscovery.evaluation_extension import build_extension
from matdiscovery.evaluation_extension_runner import (
    EvaluationExtensionLedger, EvaluationExtensionRunner, aggregate_extension,
    build_shard_manifest, validate_shard_manifest, verify_extension_envelope,
)
from test_evaluation_extension import complete_parent, shards_for
from test_fast_core_accounting import fast_core
from test_training_jobs import project


def snapshots(root):
    return {str(p): (file_sha256(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def setup(complete_parent, monkeypatch):
    f = complete_parent
    parent = Path(f["core"]["workspace"])
    source = Path(mod.__file__).resolve().parent
    frozen = parent / "src/matdiscovery"
    frozen.mkdir(parents=True, exist_ok=True)
    # Replace the generic callback fixture marker with the actual frozen Python inventory.
    marker = frozen / "mock.py"
    assert marker.read_text() == "# synthetic callback fixture\n"
    marker.unlink()
    for name in mod.FROZEN_RUNTIME_MODULES:
        (frozen / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, frozen / name)
    # Admission/provenance is independently covered by the real extension fixture;
    # isolate construction of a tiny policy provider, not the envelope verifier.
    monkeypatch.setattr(mod, "CoreFinalRunner", lambda *args, **kwargs: f["runner"])
    extension = build_extension(f["core"], f["workspace"], execution_shards=shards_for(f))
    f.update(extension=extension, new_calls=[], parent_before=snapshots(parent))
    return f


def runner(f, shard="primary", *, interrupt=False):
    from test_fast_es_budget import b10_trajectory
    class UnitRollout:
        def __init__(self, project, policy, *, settings, risk_model, attributor):
            assert project == Path(f["core"]["workspace"])
            assert settings.failure_aware_control is True
            assert settings.max_generation_retries == 2
            self.project = project
        def run(self, job, output, *, collection):
            assert collection is False
            assert job in [dict(candidate, **{k: v for k, v in job.items() if k not in candidate})
                           for candidate in f["extension"]["new_jobs"]]
            assert job["budget"] == 10 and job["group_id"] == job["task_id"]
            f["new_calls"].append(job["job_id"])
            b10_trajectory(self.project, job, output,
                discovered=job["method"] == "esopt_graph_risk")
            if interrupt:
                raise RuntimeError("synthetic closed physical transcript interrupted before envelope")
    return EvaluationExtensionRunner(f["workspace"], shard=shard, rollout_factory=UnitRollout)


def test_static_shards_are_disjoint_and_only_aggregate_accepts_six(setup):
    f = setup
    first, second = runner(f), runner(f, "secondary")
    assert len(first.manifest["jobs"]) == 4 and len(second.manifest["jobs"]) == 2
    old = {item["job"]["job_id"] for item in f["extension"]["imported_results"]}
    assert not old & {j["job_id"] for j in first.manifest["jobs"] + second.manifest["jobs"]}
    one = first.run()
    assert one["complete"] and one["shard_complete"] and not one["extension_complete"]
    pending = aggregate_extension(f["workspace"])
    assert not pending["complete"] and pending["missing_shards"] == ["secondary"]
    assert not (f["workspace"] / "experiments/evaluation_extension_stage_receipts/final.json").exists()
    two = second.run()
    assert two["complete"] and not two["extension_complete"]
    report = aggregate_extension(f["workspace"])
    assert report["complete"] and report["extension_complete"] and not report["global_study_complete"]
    assert report["expected_new_counts"]["jobs"] == 6
    assert report["expected_cumulative_counts"]["candidate_oracle_attempts"] == 100
    assert report["initial_two_known_before_expansion"]["paired_systems"] == 2
    assert report["new_three_out_of_sample_after_expansion"]["paired_systems"] == 3
    assert report["cumulative_five_descriptive"]["paired_systems"] == 5
    assert report["cumulative_five_descriptive"]["inferential_p_value"] is None
    for method in ("baseline", "esopt_graph_risk"):
        assert report["costs"]["new_six_by_method"][method]["candidate_oracle_attempts"] == 30
        assert report["costs"]["original_four_by_method"][method]["candidate_oracle_attempts"] == 20
        assert report["costs"]["cumulative_ten_by_method"][method]["candidate_oracle_attempts"] == 50
    assert report["costs"]["additional_training_or_ES_cost"] == 0
    assert len(f["new_calls"]) == len(set(f["new_calls"])) == 6 and len(f["calls"]) == 4
    assert snapshots(Path(f["core"]["workspace"])) == f["parent_before"]


def test_completed_resume_and_aggregation_preserve_receipts_without_policy_reload(setup):
    f = setup
    a, b = runner(f), runner(f, "secondary")
    a.run(); b.run(); aggregate_extension(f["workspace"])
    paths = [p for p in f["workspace"].rglob("*.json") if p.name in
             {"completion.json", "stage_receipt.json", "ledger.json", "report.json", "final.json"}]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    f["runner"].policy_factory = lambda _: pytest.fail("complete resume must not load a policy")
    a.run(); b.run(); aggregate_extension(f["workspace"])
    assert len(f["new_calls"]) == 6
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items())
    assert snapshots(Path(f["core"]["workspace"])) == f["parent_before"]


def test_closed_failed_job_is_preserved_and_never_replayed(setup):
    f = setup
    a = runner(f, interrupt=True)
    with pytest.raises(RuntimeError, match="synthetic closed"):
        a.run()
    assert len(f["new_calls"]) == 1
    f["runner"].policy_factory = lambda _: pytest.fail("reconciliation must precede model load")
    with pytest.raises(ValueError, match="reconciliation"):
        runner(f).run()
    assert len(f["new_calls"]) == 1 and not (a.output / "completion.json").exists()


def test_committed_raw_envelope_is_reconciled_after_commit_interruption(setup, monkeypatch):
    f = setup
    a = runner(f)
    original = EvaluationExtensionLedger.finish
    monkeypatch.setattr(EvaluationExtensionLedger, "finish", lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt): a.run()
    assert len(f["new_calls"]) == 1
    monkeypatch.setattr(EvaluationExtensionLedger, "finish", original)
    done = runner(f).run()
    assert done["shard_complete"] and len(f["new_calls"]) == len(set(f["new_calls"])) == 4
    ledger = json.loads((a.output / "ledger.json").read_text())
    assert any(e["event"] == "reconciled_complete" for e in ledger["events"])
    assert all(s["attempt_number"] == 1 for s in ledger["jobs"].values())


def test_partial_running_attempt_halts_before_loading_or_replaying(setup):
    f = setup
    a = runner(f)
    a.output.mkdir(parents=True)
    ledger = EvaluationExtensionLedger(a.output / "ledger.json", a.manifest, extension=a.extension)
    job = a.manifest["jobs"][0]
    attempt = ledger.claim(job["job_id"])
    folder = a.output / "jobs" / job["job_id"] / attempt
    folder.mkdir(parents=True)
    (folder / "partial.txt").write_text("unit unresolved call")
    f["runner"].policy_factory = lambda _: pytest.fail("must reject before model load")
    with pytest.raises(ValueError, match="partial.*reconciliation"):
        a.run()
    assert not f["new_calls"] and (folder / "partial.txt").exists()
    assert json.loads((a.output / "ledger.json").read_text())["jobs"][job["job_id"]]["state"] == "orphaned"


def test_manifest_and_ledger_cannot_adopt_other_shard_or_parent_jobs(setup):
    f = setup
    a = runner(f)
    bad = deepcopy(a.manifest)
    bad["jobs"][0] = deepcopy(f["extension"]["imported_results"][0]["job"])
    bad["fingerprint"] = fingerprint({k: v for k, v in bad.items() if k != "fingerprint"})
    with pytest.raises(ValueError, match="ownership"):
        validate_shard_manifest(bad, a.extension)
    ledger = EvaluationExtensionLedger(a.output / "ledger.json", a.manifest, extension=a.extension)
    other = build_shard_manifest(a.extension, "secondary")
    with pytest.raises(ValueError, match="another"):
        EvaluationExtensionLedger(ledger.path, other, extension=a.extension)
    with pytest.raises(ValueError, match="sealed ownership"):
        runner(f, "not-a-sealed-shard")
    with pytest.raises(ValueError, match="Output override"):
        EvaluationExtensionRunner(f["workspace"], output=a.output / "extra")


def test_frozen_source_or_profile_drift_stops_before_new_physics(setup):
    f = setup
    a = runner(f)
    source = Path(f["core"]["workspace"]) / "src/matdiscovery/native_attribution.py"
    source.write_text(source.read_text() + "\n# unit tamper\n")
    with pytest.raises(ValueError, match="frozen parent runtime"):
        a.run()
    assert not f["new_calls"]


def test_loaded_profile_cannot_reselect_weights_or_controllers(setup):
    f = setup
    a = runner(f)
    f["runner"].execution["unexpected_unit_profile_change"] = True
    with pytest.raises(ValueError, match="profile differs"):
        a.run()
    assert not f["new_calls"]


def test_raw_result_tamper_is_not_reconciled_or_accepted(setup):
    f = setup
    a, b = runner(f), runner(f, "secondary")
    a.run(); b.run()
    raw = next((b.output / "jobs").glob("*/*/episodes.json"))
    raw.write_text("[]")
    f["runner"].policy_factory = lambda _: pytest.fail("no reload for corrupted complete evidence")
    with pytest.raises(ValueError): b.run()
    with pytest.raises(ValueError): aggregate_extension(f["workspace"])
    assert len(f["new_calls"]) == 6
    assert not (f["workspace"] / "experiments/evaluation_extension_stage_receipts/final.json").exists()


def test_new_result_needs_explicit_extension_and_correct_shard(setup):
    from matdiscovery.final_evaluation import verify_final_envelope
    f = setup
    a = runner(f)
    a.run()
    job = a.manifest["jobs"][0]
    path = next((a.output / "jobs" / job["job_id"]).glob("*/result.json"))
    with pytest.raises(AccountingError, match="sealed FAST final matrix"):
        verify_final_envelope(path, job, a.manifest["fingerprint"], tasks=a.tasks,
            expected_profile=a.extension["execution_profiles"][job["method"]],
            expected_made_budget=10, core_protocol=a.core)
    with pytest.raises(ValueError, match="another shard"):
        verify_extension_envelope(path, job, build_shard_manifest(a.extension, "secondary"),
            a.extension, tasks=a.tasks, core=a.core)
    assert len(f["new_calls"]) == 4 and len(f["calls"]) == 4


def test_runner_lock_excludes_concurrent_workers_without_affecting_other_shard(setup):
    f = setup
    a, b = runner(f), runner(f, "secondary")
    with mod.exclusive_lock(a.output / ".runner.lock"):
        with pytest.raises(ValueError, match="Another worker"):
            a.run()
        with mod.exclusive_lock(b.output / ".runner.lock"):
            pass
    assert not f["new_calls"]


@pytest.mark.parametrize("field", ["source", "selected_generation"])
def test_resealed_envelope_cannot_mislabel_source_or_generation(setup, field):
    f = setup
    a = runner(f, "secondary")
    a.run()
    job = a.manifest["jobs"][0]
    path = next((a.output / "jobs" / job["job_id"]).glob("*/result.json"))
    value = json.loads(path.read_text())
    value["execution_job"][field] = "unregistered source" if field == "source" else 999
    write_json_atomic(path, value)
    with pytest.raises(ValueError, match="source|generation"):
        verify_extension_envelope(path, job, a.manifest, a.extension, tasks=a.tasks, core=a.core)


def test_extra_runtime_module_not_in_parent_snapshot_is_rejected(setup):
    f = setup
    a = runner(f)
    path = Path(f["core"]["workspace"]) / "src/matdiscovery/rpc.py"
    assert path.exists()  # recursively bound, even though the first design omitted it
    path.write_text(path.read_text() + "\n# simulated RPC drift\n")
    with pytest.raises(ValueError, match="frozen parent runtime.*rpc"):
        a.run()
    assert not f["new_calls"]


def test_rehashed_wrong_shard_receipt_schema_is_not_global_acceptance(setup):
    f = setup
    a, b = runner(f), runner(f, "secondary")
    a.run(); b.run()
    path = b.output / "stage_receipt.json"
    value = json.loads(path.read_text()); value["schema"] = "unrelated_scope"
    value["fingerprint"] = fingerprint({k: v for k, v in value.items() if k != "fingerprint"})
    write_json_atomic(path, value)
    with pytest.raises(ValueError, match="shard stage"):
        aggregate_extension(f["workspace"])
    assert not (f["workspace"] / "experiments/evaluation_extension_stage_receipts/final.json").exists()
