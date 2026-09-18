"""Tiny completed-run retention/replay tests; no real training or physics."""
import copy
import json

import pytest

from matdiscovery.checkpoint_retention import retire_completed_es_checkpoints, verify_retired_checkpoint
from matdiscovery.es_training import ESTrainingConfig
from matdiscovery.esopt import AgenticESOpt
from test_es_training import build


def test_retire_completed_only_keep_best_final_resume_and_exact_replay(tmp_path):
    config = ESTrainingConfig(generations=5, population=2, dev_every=1, noise_chunk_size=5)
    driver, callback, base = build(tmp_path, config=config)
    summary = driver.run()
    count = len(callback.executed)
    root = tmp_path / "run"
    report = retire_completed_es_checkpoints(root)
    assert report["retained_generations"] == [4, 5]
    assert report["retired_generations"] == [0, 1, 2, 3] and report["retired_bytes"] > 0
    for generation in range(6):
        folder = root / "checkpoints" / f"generation_{generation:04d}"
        assert (folder / "policy.pt").exists() == (generation in {4, 5})
        assert (folder / "complete.json").exists() and (folder / "es_history.json").exists()
    # Reconstruct an old checkpoint from the exact original model and retained
    # update history, without replaying any scientific trajectory.
    replay = AgenticESOpt(copy.deepcopy(base), policy_model_id="fixture/tiny-hf@fixture-revision",
                         parameter_scope="full", noise_chunk_size=5)
    replay.replay_history(root / "checkpoints/generation_0002/es_history.json")
    marker = json.loads((root / "checkpoints/generation_0002/complete.json").read_text())
    assert replay.model_state_hash() == marker["actual_model_state_hash"]
    resumed, _, _ = build(tmp_path, config=config, base=base, callbacks=callback)
    again = resumed.run(resume=True)
    assert again["best_actual_model_state_hash"] == summary["best_actual_model_state_hash"]
    assert len(callback.executed) == count
    assert retire_completed_es_checkpoints(root)["removed_bytes_this_call"] == 0


def test_unexplained_missing_tensor_never_becomes_retired(tmp_path):
    driver, callback, _ = build(tmp_path)
    driver.run()
    root = tmp_path / "run"
    (root / "checkpoints/generation_0000/policy.pt").unlink()
    with pytest.raises(ValueError, match="Unexplained"):
        retire_completed_es_checkpoints(root)
    assert not (root / "checkpoint_retirement.json").exists()
    assert (root / "checkpoints/generation_0001/policy.pt").exists()


def test_retirement_receipt_tamper_and_missing_retained_weights_rejected(tmp_path):
    driver, _, _ = build(tmp_path)
    driver.run()
    root = tmp_path / "run"
    retire_completed_es_checkpoints(root)
    marker = json.loads((root / "checkpoints/generation_0000/complete.json").read_text())
    path = root / "checkpoint_retirement.json"
    original = path.read_text()
    receipt = json.loads(original)
    receipt["retired"]["0"]["policy_pt_sha256"] = "0" * 64
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="modified"):
        verify_retired_checkpoint(root, 0, marker)
    path.write_text(original)
    (root / "checkpoints/generation_0002/policy.pt").unlink()
    with pytest.raises(ValueError, match="best/final"):
        verify_retired_checkpoint(root, 0, marker)


def test_no_retirement_before_complete_reload_acceptance(tmp_path):
    driver, _, _ = build(tmp_path)
    driver.run()
    root = tmp_path / "run"
    path = root / "training_summary.json"
    summary = json.loads(path.read_text())
    summary["clean_reload"]["allclose"] = False
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="reload"):
        retire_completed_es_checkpoints(root)
    assert all((root / "checkpoints" / f"generation_{i:04d}/policy.pt").exists() for i in range(3))
