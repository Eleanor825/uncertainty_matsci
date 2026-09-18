"""Synthetic executor declarations; these tests perform no scientific calls."""
import copy
import json

import pytest

from matdiscovery.execution_contract import declared_mace_workers, verify_mace_workers
from matdiscovery.mace_parallel import mace_execution_metadata
from matdiscovery.training_jobs import reconstruct_scientific_evidence
from test_training_jobs import project, setup_request, write_mock_scientific_trajectory


def test_legacy_serial_and_explicit_parallel_are_distinct():
    assert declared_mace_workers({}) == 1
    verify_mace_workers({}, {}, 1)
    parallel = {"mace_num_workers": 4, "orb_num_workers": 1, "mace_execution": mace_execution_metadata(4)}
    assert declared_mace_workers({"made_execution": parallel}) == 4
    verify_mace_workers({"mace_num_workers": 4}, parallel, 4)
    with pytest.raises(ValueError, match="fixed paired"):
        verify_mace_workers({}, {}, 4)


@pytest.mark.parametrize("value", [True, False, 0, 2, 8, 4.0, "4", None])
def test_executor_does_not_coerce_unsupported_values(value):
    with pytest.raises(ValueError):
        declared_mace_workers({"made_execution": {"mace_num_workers": value}})


@pytest.mark.parametrize("mutation", ["request", "metadata", "orb", "lifecycle", "missing_lifecycle"])
def test_parallel_evidence_requires_the_same_declared_executor(mutation):
    args = {"mace_num_workers": 4}
    meta = {"mace_num_workers": 4, "orb_num_workers": 1, "mace_execution": mace_execution_metadata(4)}
    if mutation == "request":
        args["mace_num_workers"] = 1
    elif mutation == "metadata":
        meta["mace_num_workers"] = 1
    elif mutation == "orb":
        meta["orb_num_workers"] = 4
    elif mutation == "lifecycle":
        meta["mace_execution"]["lifecycle"] = "concurrent_constructor_not_validated"
    else:
        del meta["mace_execution"]
    with pytest.raises(ValueError):
        verify_mace_workers(args, meta, 4)


def test_full_fifty_step_evidence_does_not_hide_an_executor_mismatch(project, tmp_path):
    _, _, _, job = setup_request(project)
    output = tmp_path / "synthetic_trajectory"
    write_mock_scientific_trajectory(project, job, output)
    tasks = json.loads((project / "configs/benchmark_tasks.json").read_text())
    expected = declared_mace_workers(json.loads((project / "configs/main_protocol.json").read_text()))
    proof = reconstruct_scientific_evidence(output, job, tasks=tasks, expected_mace_num_workers=expected)
    assert proof["costs"]["candidate_oracle_attempts"] == 50
    wrong = 1 if expected == 4 else 4
    with pytest.raises(ValueError, match="fixed paired"):
        reconstruct_scientific_evidence(output, job, tasks=tasks, expected_mace_num_workers=wrong)


def test_environment_arguments_pass_only_registered_workers(project, tmp_path):
    from matdiscovery.rollouts import make_environment_arguments, RolloutSettings
    _, _, _, job = setup_request(project)
    index = project / "data/raw/materials_project/index.json"
    key = "-".join(sorted(job["task"]["elements"]))
    index.write_text(json.dumps({"snapshots": {key: {"path": "/synthetic/snapshot", "sha256": "synthetic-sha"}}}))
    before = copy.deepcopy(job)
    result = make_environment_arguments(project, job, tmp_path / "output", RolloutSettings())
    expected = declared_mace_workers(json.loads((project / "configs/main_protocol.json").read_text()))
    assert result["mace_num_workers"] == expected
    assert result["budget"] == 50 and result["seed"] == job["environment_seeds"][0]
    assert job == before


def test_complete_rpc_cannot_hide_a_partial_oracle_journal(project, tmp_path):
    _, _, _, job = setup_request(project)
    output = tmp_path / "synthetic_incomplete_journal"
    write_mock_scientific_trajectory(project, job, output)
    journal = output / "environment/oracle_attempts.jsonl"
    lines = journal.read_text().splitlines()
    journal.write_text("\n".join(lines[:-1]) + "\n")
    tasks = json.loads((project / "configs/benchmark_tasks.json").read_text())
    with pytest.raises(ValueError, match="valid closed journal"):
        reconstruct_scientific_evidence(output, job, tasks=tasks, expected_mace_num_workers=4)
