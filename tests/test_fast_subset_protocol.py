"""Explicit budget separation; these are CPU schema tests, not experiment data."""
import copy
import json
from pathlib import Path

import pytest

from matdiscovery.core_protocol import (REGISTRATION, TC64_REGISTRATION, NORMALIZED_REGISTRATION,
    CAUSAL_GRAPH_REGISTRATION, FAST_CAUSAL_GRAPH_REGISTRATION, NORMALIZED_RECIPE,
    CoreProtocolError, _derived_main, _registered_epochs, core_execution_budget,
    collection_jobs, final_jobs)
from matdiscovery.graph_recovery import AMENDMENT_SPEC, FAST_AMENDMENT_SPEC, _amendment_spec


def core_for(registration):
    core = {"registration": copy.deepcopy(registration), "study_id": registration["study_id"],
        "scope": registration["scope"], "model_key": "qwen35_4b", "methods": registration["methods"],
        "training_seed": 1, "budget": 50, "model": {"model_id": "synthetic-model", "revision": "synthetic-revision"},
        "train_tasks": [{"id": "Al-Au-Hf", "elements": ["Al", "Au", "Hf"]}],
        "dev_tasks": [{"id": "Al-Pd-Sm", "elements": ["Al", "Pd", "Sm"]}],
        "test_tasks": [{"id": "Al-Li-V", "elements": ["Al", "Li", "V"]},
                       {"id": "Al-V-Zn", "elements": ["Al", "V", "Zn"]}],
        "imported_train": [{"job": {"job_id": f"synthetic-train-{seed}", "seed": seed, "budget": 50}}
                           for seed in (1, 2, 3)],
        "imported_development": {"job": {"job_id": "synthetic-dev", "seed": 1, "budget": 50}},
        "fit_recipe": NORMALIZED_RECIPE, "passed_bank_reuse_contract": {}, "causal_graph_amendment": {}}
    if registration == FAST_CAUSAL_GRAPH_REGISTRATION:
        core["execution_budget"] = 10
    return core


def test_fast_new_rollouts_are_B10_and_historical_jobs_keep_exact_B50_identity():
    core = core_for(FAST_CAUSAL_GRAPH_REGISTRATION)
    assert _registered_epochs(core) == 64 and core_execution_budget(core) == 10
    historical = collection_jobs(core)
    assert historical == [item["job"] for item in core["imported_train"]] + [core["imported_development"]["job"]]
    assert len(historical) == 4 and all(job["budget"] == 50 for job in historical)
    final = final_jobs(core)
    assert len(final) == 4 and len({job["job_id"] for job in final}) == 4
    assert all(job["budget"] == job["expected_counts"]["candidate_oracle_attempts"] == 10 for job in final)
    assert sum(job["expected_counts"]["candidate_oracle_attempts"] for job in final) == 40
    assert {(job["method"], job["task_id"], job["seed"]) for job in final} == {
        (method, system, 1) for method in ("baseline", "esopt_graph_risk") for system in ("Al-Li-V", "Al-V-Zn")}


def test_subset_main_changes_execution_accounting_only():
    parent = json.loads((Path(__file__).parents[1] / "configs/main_protocol.json").read_text())
    old, fast = core_for(CAUSAL_GRAPH_REGISTRATION), core_for(FAST_CAUSAL_GRAPH_REGISTRATION)
    before, after = _derived_main(parent, old), _derived_main(parent, fast)
    assert before["graph"] == after["graph"] and before["collection"] == after["collection"]
    assert before["transcoder"] == after["transcoder"] and after["transcoder"]["epochs"] == 64
    assert {k: v for k, v in before["esopt"].items() if k != "full_training_and_development_candidate_oracle_attempts"} == {
        k: v for k, v in after["esopt"].items() if k != "full_training_and_development_candidate_oracle_attempts"}
    assert after["esopt"]["full_training_and_development_candidate_oracle_attempts"] == 60
    assert after["final_evaluation"]["MADE_candidate_oracle_attempts"] == 40
    assert after["execution_budget"] == 10
    assert all(job["budget"] == 50 for job in final_jobs(old))


@pytest.mark.parametrize("registration", [REGISTRATION, TC64_REGISTRATION, NORMALIZED_REGISTRATION, CAUSAL_GRAPH_REGISTRATION])
def test_old_registrations_cannot_silently_become_B10(registration):
    core = {"registration": registration, "execution_budget": 10}
    with pytest.raises(CoreProtocolError, match="prior registration"):
        core_execution_budget(core)


@pytest.mark.parametrize("budget", [None, 0, True, 10.0, 20, 50])
def test_FAST_budget_is_exact_not_an_arbitrary_override(budget):
    core = core_for(FAST_CAUSAL_GRAPH_REGISTRATION)
    if budget is None: core.pop("execution_budget")
    else: core["execution_budget"] = budget
    with pytest.raises(CoreProtocolError, match="execution budget"):
        _registered_epochs(core)


def test_fast_amendment_discloses_reduced_new_budget_and_null_evolution_outcomes():
    assert _amendment_spec(CAUSAL_GRAPH_REGISTRATION) == AMENDMENT_SPEC
    assert _amendment_spec(FAST_CAUSAL_GRAPH_REGISTRATION) == FAST_AMENDMENT_SPEC
    assert FAST_AMENDMENT_SPEC["historical_collection_budget"] == 50
    assert FAST_AMENDMENT_SPEC["new_ES_and_test_episode_budget"] == 10
    assert not FAST_AMENDMENT_SPEC["monotonic_improvement_required_or_claimed"]
    assert not FAST_AMENDMENT_SPEC["generation_zero_full_method_evaluation_available"]
    assert "model_runtime_evaluator_chemistries_seeds_B50_ES_and_tests_unchanged" not in FAST_AMENDMENT_SPEC
