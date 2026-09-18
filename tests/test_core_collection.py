"""Raw recorder/closure/import tests, not material experiment outcomes."""
import json
from pathlib import Path

import pytest

from matdiscovery.accounting import file_sha256, write_json_atomic
from matdiscovery.collection_provenance import CollectionProvenanceError
from matdiscovery.core_protocol import CoreProtocolError, collection_manifest_paths
from matdiscovery.core_collection import (audit_core_corpus_costs, collect_development, import_training,
    stage_receipt_path, verify_import, verify_stage_receipt)
from core_corpus_fixture import make_core_fixture, complete_core_dev


def test_import_rebases_only_paths_and_never_writes_original(tmp_path):
    f = make_core_fixture(tmp_path, legacy=True); core = f["core"]
    originals = {p: (file_sha256(p), p.stat().st_mtime_ns) for d in f["directories"] for p in d.rglob("*") if p.is_file()}
    result = import_training(core)
    assert result["complete"] and result["new_physical_calls"] == 0
    imported = collection_manifest_paths(core)[0]
    data = json.loads(imported.read_text()); original = f["directories"][0]
    assert imported != original / "collection_manifest.json"
    assert data["core_import"]["original_manifest"]["sha256"] == file_sha256(original / "collection_manifest.json")
    assert all(Path(x["path"]).parent.parent == original for x in data["activation_shards"])
    row = json.loads((imported.parent / "decisions.jsonl").read_text())
    event = json.loads((imported.parent / "decision_events.jsonl").read_text())
    assert row["input_ids_file"] == event["input_ids_file"] == str(imported.parent / "prefix_tokens.pt")
    assert (imported.parent / "prefix_tokens.pt").read_bytes() == (original / "prefix_tokens.pt").read_bytes()
    assert row["label_future_failure"] is None
    (imported.parent / "graph_features.jsonl").write_text('{"classification":"unit-only derived sidecar"}\n')
    import_training(core)
    assert originals == {p: (file_sha256(p), p.stat().st_mtime_ns) for d in f["directories"] for p in d.rglob("*") if p.is_file()}


def test_import_partial_or_modified_copy_refused_without_overwrite(tmp_path):
    f = make_core_fixture(tmp_path); core = f["core"]
    destination = collection_manifest_paths(core)[0].parent
    destination.mkdir(parents=True)
    (destination / "partial").write_text("interrupted")
    with pytest.raises(CoreProtocolError, match="Partial core import"):
        import_training(core)
    assert (destination / "partial").read_text() == "interrupted"


def test_import_artifact_tamper_and_receipt_rehash_cannot_change_outcome(tmp_path):
    f = make_core_fixture(tmp_path); core = f["core"]; import_training(core)
    destination = collection_manifest_paths(core)[0].parent
    (destination / "decisions.jsonl").write_text('{"label_future_failure":0}\n')
    with pytest.raises(CoreProtocolError, match="changed"):
        verify_import(core, core["imported_train"][0])


def test_core_costs_accept_closed_real_recorder_and_count_imports_separately(tmp_path):
    f = make_core_fixture(tmp_path); core = f["core"]
    import_training(core); complete_core_dev(f)
    no_model = lambda *args: pytest.fail("complete evidence must not load a model")
    result = collect_development(core, policy_factory=no_model)
    assert result["candidate_oracle_attempts"] == 50
    report = audit_core_corpus_costs(core)
    assert report["imported_train"]["costs"]["candidate_oracle_attempts"] == 150
    assert report["new_development"]["costs"]["candidate_oracle_attempts"] == 50
    assert report["total"]["costs"]["initialization_oracle_attempts"] == 8
    assert report["complete"] and not report["original_full_study_complete"]
    assert any(x["path"].endswith("oracle_attempts.jsonl") for x in report["evidence_files"])
    receipt = stage_receipt_path(core, "collect"); before = file_sha256(receipt), receipt.stat().st_mtime_ns
    collect_development(core, policy_factory=no_model)
    assert before == (file_sha256(receipt), receipt.stat().st_mtime_ns)


@pytest.mark.parametrize("partial", ["directory", "started"])
def test_unknown_development_is_never_replayed(tmp_path, partial):
    f = make_core_fixture(tmp_path); core = f["core"]; import_training(core)
    if partial == "directory": collection_manifest_paths(core)[-1].parent.mkdir()
    else: write_json_atomic(stage_receipt_path(core, "collect").with_suffix(".started.json"), {"complete": False})
    with pytest.raises(CoreProtocolError, match="blind physical replay"):
        collect_development(core, policy_factory=lambda *args: pytest.fail("model must not load"))


def test_worker4_missing_journal_cannot_enter_core_corpus(tmp_path):
    f = make_core_fixture(tmp_path); core = f["core"]; import_training(core)
    directory = complete_core_dev(f)
    (directory / "environment/oracle_attempts.jsonl").write_text("")
    with pytest.raises(CollectionProvenanceError, match="changed"):
        audit_core_corpus_costs(core)


@pytest.mark.parametrize("alias_manifest", [False, True])
def test_real_frozen_source_symlink_receipt_and_prefix_aliases(tmp_path, alias_manifest):
    # Legacy worker1 proof and current recorder fixtures both preserve physical
    # path aliases in source bytes; the import must compare resolved identity.
    f = make_core_fixture(tmp_path, completed=2, legacy=True,
                          frozen_alias=True, alias_manifest=alias_manifest)
    core = f["core"]
    original = f["directories"][0]
    originals = {p: (file_sha256(p), p.stat().st_mtime_ns)
                 for directory in f["directories"] for p in directory.rglob("*") if p.is_file()}
    receipt = json.loads((original / "completion.json").read_text())
    assert all("/frozen_sources/" in x["path"] for x in receipt["artifacts"])
    assert "/frozen_sources/" in json.loads((original / "decisions.jsonl").read_text())["input_ids_file"]
    assert import_training(core)["complete"] is True
    destination = collection_manifest_paths(core)[0].parent
    derived = json.loads((destination / "collection_manifest.json").read_text())
    assert len(derived["activation_shards"]) == 32
    assert not (destination / "activation_shards").exists()
    assert all(Path(x["path"]).parent.parent == original for x in derived["activation_shards"])
    row = json.loads((destination / "decisions.jsonl").read_text())
    event = json.loads((destination / "decision_events.jsonl").read_text())
    assert row["input_ids_file"] == event["input_ids_file"] == str(destination / "prefix_tokens.pt")
    assert (destination / "prefix_tokens.pt").read_bytes() == (original / "prefix_tokens.pt").read_bytes()
    verify_import(core, core["imported_train"][0])
    assert originals == {p: (file_sha256(p), p.stat().st_mtime_ns)
                         for directory in f["directories"] for p in directory.rglob("*") if p.is_file()}
