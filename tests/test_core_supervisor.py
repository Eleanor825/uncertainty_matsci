"""Orchestration contracts only; no model or material evaluator execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location("core_supervisor_script",
    Path(__file__).resolve().parents[1] / "scripts/run_core_study.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def complete(root, name, **extra):
    path = core.receipt_path(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"complete": True, **extra}))


def test_declares_only_the_eight_core_stages_without_global_acceptance(tmp_path):
    queue = core.build_queue(tmp_path)
    assert [stage["id"] for stage in queue["stages"]] == ["core-" + n for n in core.NAMES]
    assert len(queue["stages"]) == 8
    assert not queue["final_acceptance_verified"] and not queue["original_full_study_complete"]
    assert all(stage["command"][-1] == name for stage, name in zip(queue["stages"], core.NAMES))
    with pytest.raises(ValueError):
        core.stage_command(tmp_path, "extra_test_after_seeing_outcomes")


def test_cannot_overwrite_a_prior_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "validate_workspace", lambda root: {"synthetic": "source"})
    path = core.declare_queue(tmp_path)
    before = path.read_bytes()
    assert core.declare_queue(tmp_path) == path and path.read_bytes() == before
    value = json.loads(before)
    value["stages"].pop()
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="silent replacement"):
        core.declare_queue(tmp_path)


def test_success_requires_a_domain_receipt_not_only_zero_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "validate_workspace", lambda root: {"synthetic": "source"})
    monkeypatch.setattr(core.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    with pytest.raises(FileNotFoundError):
        core.run_stage(tmp_path, "import")
    complete(tmp_path, "import")
    assert core.run_stage(tmp_path, "import") == 0
    proof = json.loads((tmp_path / "experiments/core_stage_verification/import.json").read_text())
    assert proof["complete"] and not proof["original_full_study_complete"]


def test_failure_does_not_publish_verification_or_repeat(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(core, "validate_workspace", lambda root: {"synthetic": "source"})
    def fail(*a, **kw):
        calls.append(a)
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(core.subprocess, "run", fail)
    assert core.run_stage(tmp_path, "import") == 7
    assert len(calls) == 1
    assert not (tmp_path / "experiments/core_stage_verification/import.json").exists()


def test_later_stage_requires_complete_prior_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "validate_workspace", lambda root: {"synthetic": "source"})
    monkeypatch.setattr(core.subprocess, "run", lambda *a, **kw: pytest.fail("must not dispatch"))
    with pytest.raises(FileNotFoundError):
        core.run_stage(tmp_path, "esopt")
    complete(tmp_path, "import", complete_value=False)
    path = core.receipt_path(tmp_path, "import")
    path.write_text('{"complete": false}')
    with pytest.raises(RuntimeError, match="verified evidence"):
        core.run_stage(tmp_path, "collect")


def test_source_mutation_prevents_success_receipt(tmp_path, monkeypatch):
    versions = iter(({"source": "before"}, {"source": "after"}))
    monkeypatch.setattr(core, "validate_workspace", lambda root: next(versions))
    monkeypatch.setattr(core.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    complete(tmp_path, "import")
    with pytest.raises(RuntimeError, match="changed during"):
        core.run_stage(tmp_path, "import")
    assert not (tmp_path / "experiments/core_stage_verification/import.json").exists()
