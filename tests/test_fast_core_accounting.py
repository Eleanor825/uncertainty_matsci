"""FAST B10 bookkeeping fixtures only; no model/oracle or study measurements."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from matdiscovery.accounting import (AccountingError, expected_episodes, fingerprint,
    verify_result, write_json_atomic)
from matdiscovery.core_final import (CoreFinalError, CoreFinalLedger, build_core_final_manifest,
    validate_core_final_manifest, verify_selected_evolution)
from matdiscovery.core_report import core_cost_report, pair_report
from test_accounting import result_file
from test_core_final_report import unit_jobs, unit_manifest, records
from test_core_report_reused_development import reused_inputs
from test_training_jobs import project


@pytest.fixture
def fast_core(project, monkeypatch):
    from matdiscovery import core_protocol
    root = project
    core = {"workspace": str(root), "registration": deepcopy(core_protocol.FAST_CAUSAL_GRAPH_REGISTRATION),
        "budget": 50, "execution_budget": 10, "model_key": "qwen35_4b", "methods": ["baseline", "esopt_graph_risk"],
        "study_id": core_protocol.FAST_CAUSAL_GRAPH_REGISTRATION["study_id"],
        "scope": core_protocol.FAST_CAUSAL_GRAPH_REGISTRATION["scope"],
        "imported_train": [{}, {}, {}], "imported_development": {"synthetic_reused_development": True},
        "model": {"model_id": "unit/tiny-model", "revision": "unit-revision"},
        "test_tasks": [{"id": "Al-Li-V", "elements": ["Al", "Li", "V"]},
                       {"id": "Al-V-Zn", "elements": ["Al", "V", "Zn"]}]}
    core["fingerprint"] = fingerprint(core)
    write_json_atomic(root / "configs/deadline_core_protocol.json", core)
    # This isolates the already separately tested physical/source admission.
    # Budget code must still call this boundary and match its whole result.
    def admitted(path):
        assert Path(path).resolve() == root.resolve()
        return deepcopy(core)
    monkeypatch.setattr(core_protocol, "read_core", admitted)
    return core


def fast_manifest(core):
    from matdiscovery.core_protocol import final_jobs
    return build_core_final_manifest(core["fingerprint"], final_jobs(core), core_protocol=core)


def fast_result(tmp_path, core):
    manifest = fast_manifest(core)
    path = result_file(tmp_path, manifest["jobs"][0], {"manifest_fingerprint": manifest["fingerprint"]})
    data = json.loads(path.read_text())
    data["episodes"][0]["costs"]["candidate_oracle_attempts"] = 10
    write_json_atomic(path, data)
    return manifest, path


def test_budget10_is_not_authorized_by_self_reported_job_or_numeric_override(tmp_path, fast_core):
    manifest, path = fast_result(tmp_path, fast_core)
    job = manifest["jobs"][0]
    with pytest.raises(AccountingError, match="budget"):
        verify_result(path, job, manifest["fingerprint"])
    with pytest.raises(AccountingError, match="read_core"):
        verify_result(path, job, manifest["fingerprint"], expected_made_budget=10)
    altered = deepcopy(fast_core); altered["budget"] = 10
    with pytest.raises(AccountingError, match="sealed FAST"):
        verify_result(path, job, manifest["fingerprint"], expected_made_budget=10, core_protocol=altered)
    with pytest.raises(AccountingError):
        expected_episodes({"jobs": [job]}, expected_made_budget=True, core_protocol=fast_core)


def test_bound_fast_result_and_four_job_ledger_use_exact40_without_reducing_full_scope(tmp_path, fast_core):
    manifest, path = fast_result(tmp_path, fast_core)
    job = manifest["jobs"][0]
    proof = verify_result(path, job, manifest["fingerprint"], expected_made_budget=10, core_protocol=fast_core)
    assert proof["candidate_oracle_attempts"] == 10
    ledger = CoreFinalLedger(tmp_path / "ledger.json", manifest)
    assert ledger.completion()["expected_counts"]["candidate_oracle_attempts"] == 40
    assert ledger.completion()["global_study_complete"] is False
    changed = deepcopy(job); changed["task_id"] = "unregistered-system"
    with pytest.raises(AccountingError, match="sealed FAST final matrix"):
        expected_episodes({"jobs": [changed]}, expected_made_budget=10, core_protocol=fast_core)
    assert unit_manifest()["expected_counts"]["candidate_oracle_attempts"] == 200
    with pytest.raises(CoreFinalError):
        build_core_final_manifest(fast_core["fingerprint"], manifest["jobs"])


@pytest.mark.parametrize("mutation", ["source", "budget", "count", "job", "remove_binding"])
def test_fast_manifest_cannot_be_rehashed_to_another_scope(fast_core, mutation):
    manifest = fast_manifest(fast_core)
    if mutation == "source":
        Path(manifest["core_protocol"]["path"]).write_text("changed")
    elif mutation == "budget": manifest["execution_budget"] = 50
    elif mutation == "count": manifest["expected_counts"]["candidate_oracle_attempts"] = 4
    elif mutation == "job": manifest["jobs"][0]["task"]["elements"] = ["Co", "Ni"]
    else: manifest.pop("core_protocol")
    manifest["fingerprint"] = fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
    with pytest.raises((CoreFinalError, AccountingError)):
        validate_core_final_manifest(manifest)


def zero_selected(base="base"):
    curve = [{"generation": gen, "fitness": [0., 0.], "normalized_rewards": [0., 0.],
        "population_fitness_tied": True, "parameter_update_observed": False,
        "text_parameter_update_observed": False, "all_parameter_deltas_zero": True,
        "actual_model_state_hash_before": base, "actual_model_state_hash_after": base,
        "dev_mean": 0., "dev_environment_seeds": [1729]} for gen in (1, 2)]
    proof = {"schema": "verified_fast_es_zero_selected_update_v1", "complete": True,
        "selected_generation": 1, "selected_matches_initial": True,
        "initial_actual_model_state_hash": base, "generations": curve,
        "original_formula_verified": True, "initial_dev_evaluated": False}
    return {"selected_generation": 1, "initial_actual_model_state_hash": base,
        "actual_model_state_hash": base, "evolution_signal_observed": False,
        "nonzero_text_parameter_update_verified": False, "evolution_status": "no_evolution_signal",
        "initial_dev_evaluated": False, "zero_update_proof": proof, "generation_curve": curve}


def test_zero_updated_checkpoint_is_explicit_fast_null_and_legacy_still_rejects():
    selected = zero_selected()
    verify_selected_evolution(selected, "base", "base", fast=True)
    with pytest.raises(CoreFinalError, match="genuinely updated"):
        verify_selected_evolution(selected, "base", "base")
    for key, value in (("zero_update_proof", None), ("evolution_signal_observed", True),
                       ("initial_dev_evaluated", True), ("evolution_status", "effective")):
        bad = deepcopy(selected); bad[key] = value
        with pytest.raises(CoreFinalError):
            verify_selected_evolution(bad, "base", "base", fast=True)


def test_fast_costs_keep_historical200_new100_and_no_extra_es_summary_total(tmp_path, fast_core):
    corpus, es = reused_inputs(tmp_path)
    es["training"]["costs"]["candidate_oracle_attempts"] = 40
    es["development"]["costs"]["candidate_oracle_attempts"] = 20
    actual = records(fast_manifest(fast_core))
    for record in actual.values(): record["episode"]["costs"]["candidate_oracle_attempts"] = 10
    result = core_cost_report(corpus, es, actual, expected_imported_jobs=3,
        expected_reused_development=True, expected_made_budget=10, core_protocol=fast_core)
    assert result["historical_reused_costs"]["candidate_oracle_attempts"] == 200
    assert result["incremental_core_costs"]["candidate_oracle_attempts"] == 100
    assert result["reused_plus_incremental_costs"]["candidate_oracle_attempts"] == 300
    assert result["categories"]["matched_final_evaluation"]["candidate_oracle_attempts"] == 40
    assert result["historical_collection_candidate_budget"] == 50
    assert result["es_summary_total_not_added_again"] is True
    with pytest.raises(AccountingError):
        core_cost_report(corpus, es, actual, expected_imported_jobs=3,
            expected_reused_development=True, expected_made_budget=10)
    with pytest.raises(CoreFinalError):
        core_cost_report(corpus, es, actual, expected_imported_jobs=3, expected_reused_development=True)
    paired = pair_report(fast_manifest(fast_core), actual)
    assert paired["uncertainty_effect_independently_identified"] is False


def test_four_real_shape_b10_envelopes_resume_and_report_honest_null(tmp_path, fast_core, monkeypatch):
    """Use the physical-evidence verifier and real recorder, with scalar mocks."""
    from matdiscovery import core_collection, core_es, core_report
    from matdiscovery.esopt import tensor_state_hash
    from test_core_final_integration import runner_fixture
    from test_fast_es_budget import b10_trajectory
    project = Path(fast_core["workspace"])
    calls = []; runner = runner_fixture(project, tmp_path, calls)
    runner.output = project / "experiments/core_final"
    runner.core, runner.manifest = fast_core, fast_manifest(fast_core)
    selected_profiles = []
    def selected(policy, directory, **kwargs):
        base = tensor_state_hash(dict(policy.model.state_dict()))
        value = zero_selected(base); value["artifacts"] = {}
        policy.mark_state("reload", generation=1)
        selected_profiles.append(value)
        return value
    runner.selected_loader = selected
    class UnitRollout:
        def __init__(self, root, policy, **kwargs): self.project = root
        def run(self, job, output, *, collection):
            assert not collection
            calls.append(job["job_id"])
            b10_trajectory(self.project, job, output, discovered=False)
            rows = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
            for row in rows:
                row.update(disposition="executed", label_future_failure=1, predicted_failure_probability=.2)
            (output / "decisions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    runner.rollout_factory = UnitRollout
    completion = runner.run()
    assert completion["expected_counts"]["candidate_oracle_attempts"] == 40
    assert completion["evolution_status"] == "no_evolution_signal"
    assert completion["original_b50_core_complete"] is False
    assert completion["scientific_improvement_assumed"] is False
    original = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in runner.output.glob("jobs/*/*/result.json")}
    runner.run()
    assert len(calls) == len(set(calls)) == 4
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in original.items())
    corpus, es = reused_inputs(tmp_path)
    es["training"]["costs"]["candidate_oracle_attempts"] = 40
    es["development"]["costs"]["candidate_oracle_attempts"] = 20
    selected = selected_profiles[-1]
    fields = ("evolution_signal_observed", "evolution_status", "nonzero_text_parameter_update_verified",
              "zero_update_proof", "generation_curve", "initial_dev_evaluated")
    es.update({key: selected[key] for key in fields})
    # Real source-ancestry/ES math are covered separately. Here the complete
    # final physical verifier remains active; no outcome/parameter is measured.
    monkeypatch.setattr(core_collection, "audit_core_corpus_costs", lambda core: corpus)
    monkeypatch.setattr(core_es, "audit_core_es_costs", lambda directory, core: es)
    monkeypatch.setattr(core_report, "offline_transcoder_report", lambda core: None)
    monkeypatch.setattr(core_report, "prior_interruption_costs", lambda core: {"costs": {}})
    report = core_report.report_core(project)
    assert report["core_complete"] and report["evolution_status"] == "no_evolution_signal"
    assert report["costs"]["reused_plus_incremental_costs"]["candidate_oracle_attempts"] == 300
    assert report["initial_dev_evaluated"] is False and report["initial_dev_metric"] is None
    assert [row["generation"] for row in report["generation_curve"]] == [1, 2]
    assert report["paired_results"]["aggregate"]["AUDC"]["interpretation"] == "no_observed_difference"
    assert not report["paired_results"]["uncertainty_effect_independently_identified"]
    es["evolution_status"] = "fabricated_improvement"
    with pytest.raises(CoreFinalError, match="independently audited ES"):
        core_report.report_core(project)
