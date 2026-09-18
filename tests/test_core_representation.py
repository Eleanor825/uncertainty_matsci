"""Core stage routing unit checks; no TC training, graph model or oracle run."""
from pathlib import Path

import pytest

from matdiscovery.core_protocol import CoreProtocolError
from matdiscovery.core_collection import import_training, collect_development
from matdiscovery.core_representation import graph_config, paths_for, run_stage, transcoder_config
from core_corpus_fixture import make_core_fixture, complete_core_dev


def test_core_keeps_full_transcoder_fidelity_and_configures_all_prefix_graph_cap(tmp_path):
    core = make_core_fixture(tmp_path)["core"]
    tc, graph = transcoder_config(core), graph_config(core)
    assert tc.epochs == 16 and tc.max_dev_fvu == .5 and tc.device == "cuda:0"
    assert graph.max_feature_nodes == 32 and graph.max_backward_targets == 42
    assert (graph.validation_epsilon, graph.validation_rtol, graph.validation_atol) == (.001, .05, .0001)
    assert graph.transcoder_device == "cpu" and graph.dtype == "float32"
    assert all(Path(core["workspace"]) in path.parents for path in paths_for(core).values())


def test_core_representation_cannot_fit_without_complete_dev(tmp_path, monkeypatch):
    core = make_core_fixture(tmp_path)["core"]; import_training(core)
    monkeypatch.setattr("torch.cuda.set_per_process_memory_fraction", lambda *a: pytest.fail("must reject before GPU configuration"))
    with pytest.raises(FileNotFoundError):
        run_stage(core, "transcoders")


def test_core_tc_runner_forwards_entire_corpus_and_strict_resume(tmp_path, monkeypatch):
    import matdiscovery.core_representation as module
    f = make_core_fixture(tmp_path); core = f["core"]
    import_training(core); complete_core_dev(f); collect_development(core)
    calls = []
    def full_stage(manifests, destination, *, model_key, config, resume):
        calls.append((manifests, model_key, config, resume))
        raise RuntimeError("unit routing reached; never train TC")
    monkeypatch.setattr(module, "train_transcoders_from_collections", full_stage)
    monkeypatch.setattr("torch.cuda.set_per_process_memory_fraction", lambda *a: None)
    with pytest.raises(RuntimeError, match="never train TC"):
        run_stage(core, "transcoders")
    manifests, model, config, resume = calls[0]
    assert len(manifests) == 4 and model == "qwen35_4b" and config.epochs == 16 and resume is True
    assert not (Path(core["workspace"]) / "experiments/core_stage_receipts/transcoders.json").exists()
