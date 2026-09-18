"""CPU admission and source-snapshot tests; no real scientific results."""
import copy
import json
from pathlib import Path

import pytest

from matdiscovery.accounting import file_sha256, write_json_atomic
from matdiscovery.core_protocol import (CoreProtocolError, REGISTRATION, _core_hash, collection_jobs,
    collection_manifest_paths, corpus_contract, final_jobs, manifest_input_files,
    prepare_core_workspace, read_core, validate_corpus_contract)
from matdiscovery.core_collection import import_training
from core_corpus_fixture import make_core_fixture, complete_core_dev


@pytest.mark.parametrize("completed", [2, 3])
def test_registered_scope_and_exact_corpus_snapshot(tmp_path, completed):
    f = make_core_fixture(tmp_path, completed=completed)
    core = f["core"]; workspace = Path(core["workspace"])
    assert core["dev_tasks"] == [{"id": "Al-Pd-Sm", "elements": ["Al", "Pd", "Sm"]}]
    assert [x["id"] for x in core["test_tasks"]] == ["Al-Li-V", "Al-V-Zn"]
    assert len(collection_jobs(core)) == completed + 1 and len(final_jobs(core)) == 4
    assert [x["job"]["seed"] for x in core["imported_train"]] == list(range(1, completed + 1))
    assert core["graph"]["max_feature_nodes"] == 32 and core["graph"]["max_backward_targets"] == 42
    assert core["transcoder"]["epochs"] == 16 and core["transcoder"]["max_development_output_fvu"] == .5
    assert core["deadline_utc"] == "2026-09-17T17:25:00Z" and core["original_full_study_complete"] is False
    assert not (workspace / "configs/stage_queue.json").exists()
    assert all(workspace in path.parents or f["parent"] in path.parents for path in manifest_input_files(core))
    assert prepare_core_workspace(f["parent"], workspace, reconciliation_path=f["reconciliation"]) == core
    import_training(core); complete_core_dev(f)
    manifests = collection_manifest_paths(core)
    contract = corpus_contract(core, manifests)
    assert contract["expected_collection_jobs"] == completed + 1 and contract["test_used_for_fit"] is False
    assert validate_corpus_contract(contract, manifest_paths=manifests, model_key="qwen35_4b") == contract
    for wrong in (manifests[:-1], manifests + manifests[:1], tuple(reversed(manifests[:-1]))):
        with pytest.raises(CoreProtocolError, match="omitted|duplicated|added"):
            corpus_contract(core, wrong)


@pytest.mark.parametrize("mutation", ["source", "extra_source", "graph", "model", "deadline", "original_receipt"])
def test_core_source_or_registration_mutation_is_rejected(tmp_path, mutation):
    f = make_core_fixture(tmp_path); core = copy.deepcopy(f["core"]); workspace = Path(core["workspace"])
    if mutation == "source":
        (workspace / "src/matdiscovery/fixture.py").write_text("changed source")
    elif mutation == "extra_source":
        (workspace / "src/unregistered.py").write_text("# unregistered source")
    elif mutation == "original_receipt":
        (f["directories"][0] / "completion.json").write_text("{}")
    else:
        if mutation == "graph": core["graph"]["max_feature_nodes"] = 1
        elif mutation == "model": core["model"]["revision"] = "changed"
        else: core["deadline_utc"] = "2099-01-01T00:00:00Z"
        core["fingerprint"] = core["core_fingerprint"] = _core_hash(core)
        write_json_atomic(workspace / "configs/deadline_core_protocol.json", core)
    with pytest.raises(CoreProtocolError):
        read_core(workspace)


def test_default_failure_fit_stays_full_matrix_core_requires_exact_contract(tmp_path, monkeypatch):
    import matdiscovery.failure_fit as fit
    f = make_core_fixture(tmp_path, completed=2); core = f["core"]
    import_training(core); complete_core_dev(f)
    manifests = collection_manifest_paths(core); contract = corpus_contract(core, manifests)
    with pytest.raises(fit.RiskDataError, match="default 210"):
        fit.fit_failure_models(manifests, tmp_path / "risk", tmp_path / "labels", model_key="qwen35_4b")
    def admitted(*args, **kwargs):
        raise RuntimeError("unit admission reached, no fit run")
    monkeypatch.setattr(fit, "load_risk_dataset", admitted)
    with pytest.raises(RuntimeError, match="unit admission reached"):
        fit.fit_failure_models(manifests, tmp_path / "risk", tmp_path / "labels", model_key="qwen35_4b", corpus_contract=contract)
    changed = copy.deepcopy(contract); changed["manifests"] = changed["manifests"][:-1]
    with pytest.raises(CoreProtocolError):
        fit.fit_failure_models(manifests, tmp_path / "risk", tmp_path / "labels", model_key="qwen35_4b", corpus_contract=changed)


def test_core_stage_queue_is_the_only_mutable_configuration(tmp_path):
    f = make_core_fixture(tmp_path); core = f["core"]; workspace = Path(core["workspace"])
    write_json_atomic(workspace / "configs/stage_queue.json", {"classification": "unit supervisor queue"})
    assert read_core(workspace) == core
    write_json_atomic(workspace / "configs/extra_science.json", {"changed": True})
    with pytest.raises(CoreProtocolError, match="unregistered"):
        read_core(workspace)
