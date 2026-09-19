"""Isolated tiny CPU contracts; no CUDA, policy model or chemical oracle."""
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest
import torch

from matdiscovery.normalized_transcoders import train_normalized_layer
from matdiscovery.transcoders import TranscoderConfig, write_activation_shard
from method.tc_worker import (GIB, make_request, validate_request, fingerprint,
                              require_capacity, run_layer_subprocess)


@pytest.fixture(autouse=True)
def restore_runtime():
    before = (torch.get_num_threads(), torch.get_float32_matmul_precision(),
              torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
              torch.get_default_dtype())
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_num_threads(before[0]); torch.set_float32_matmul_precision(before[1])
    torch.backends.cuda.matmul.allow_tf32 = before[2]; torch.backends.cudnn.allow_tf32 = before[3]
    torch.set_default_dtype(before[4])


def fixture_inputs(tmp_path, dev_split="dev"):
    config = TranscoderConfig(2, 2, 4, 2)
    generator = torch.Generator().manual_seed(17)
    paths = {}
    for split, rows in (("train", 11), (dev_split, 7)):
        x = torch.randn(rows, 2, generator=generator)*.2 + torch.tensor([.9, -.3])
        y = x @ torch.tensor([[.02, .01], [-.01, .03]]) + torch.tensor([.1, -.2])
        p = tmp_path/(split+".pt")
        write_activation_shard(p, x, y, group_ids=[split]*rows,
            prefix_hashes=[split+str(i) for i in range(rows)], split=split,
            policy_fingerprint="unit-policy", layer_path="unit.mlp")
        paths[split] = [p]
    return config, paths["train"], paths[dev_split]


def arguments(tmp_path):
    return dict(policy_fingerprint="unit-policy", layer_path="unit.mlp",
                output_path=tmp_path/"bank"/"layer.pt", seed=1729,
                registration={"unit_only": True}, device="cpu", unit_test=True,
                epochs=64, batch_size=8, learning_rate=4e-4)


def test_real_cpu64_subprocess_matches_original_and_same_process_group(tmp_path):
    config, train, dev = fixture_inputs(tmp_path)
    kwargs = arguments(tmp_path)
    model, metadata = run_layer_subprocess(config, train, dev,
        worker_directory=tmp_path/"worker", **kwargs)
    original_kwargs = {k:v for k,v in kwargs.items() if k != "unit_test"}
    original_kwargs["output_path"] = tmp_path/"reference.pt"
    original, expected = train_normalized_layer(config, train, dev, **original_kwargs)
    assert model.checkpoint_hash() == original.checkpoint_hash()
    assert metadata["history"] == expected["history"] and len(metadata["history"]) == 64
    assert metadata["selected_epoch"] == expected["selected_epoch"]
    assert metadata["source_files"] == expected["source_files"]
    receipt = json.loads((tmp_path/"worker/receipt.json").read_text())
    assert receipt["complete"] and receipt["epochs"] == 64
    assert receipt["pid"] != os.getpid() and receipt["parent_pid"] == os.getpid()
    assert receipt["pgid"] == os.getpgrp() and receipt["policy_models_loaded"] == 0
    assert receipt["runtime"]["device"] == "cpu" and receipt["unit_test"]
    assert receipt["fidelity_gate_passed"] == metadata["fidelity_gate_passed"]
    assert receipt["status"] == ("succeeded" if metadata["fidelity_gate_passed"] else "failed_fidelity")
    original_bytes = {p: p.read_bytes() for p in (tmp_path/"bank").iterdir()}
    with pytest.raises(FileExistsError):
        run_layer_subprocess(config, train, dev, worker_directory=tmp_path/"worker", **kwargs)
    assert all(p.read_bytes() == data for p,data in original_bytes.items())


def test_changed_source_math_and_unsafe_bank_inventory_are_rejected(tmp_path):
    config, train, dev = fixture_inputs(tmp_path)
    kwargs = arguments(tmp_path)
    request = make_request(config, train, dev, **kwargs)
    for name, value in (("epochs", 63), ("gpu_memory_bytes", 3*GIB), ("max_dev_fvu", .6), ("worker_source_sha256", "0"*64)):
        changed = copy.deepcopy(request); changed[name] = value
        changed["fingerprint"] = fingerprint({k:v for k,v in changed.items() if k!="fingerprint"})
        with pytest.raises(ValueError): validate_request(changed)
    with pytest.raises(ValueError, match="outside bank inventory"):
        run_layer_subprocess(config, train, dev, worker_directory=tmp_path/"bank"/"worker", **kwargs)
    train[0].write_bytes(train[0].read_bytes()+b"tamper")
    with pytest.raises(ValueError, match="source bytes changed"):
        validate_request(request)


def test_test_shards_rejected_by_original_validator_without_fit(tmp_path):
    config, train, test = fixture_inputs(tmp_path)
    # The public writer already rejects test. Construct an explicitly invalid
    # unit payload to check the child's independent read-time rejection too.
    payload = torch.load(test[0], weights_only=True)
    payload["metadata"]["split"] = "test"
    torch.save(payload, test[0])
    with pytest.raises(RuntimeError, match="fit failed"):
        run_layer_subprocess(config, train, test, worker_directory=tmp_path/"worker", **arguments(tmp_path))
    assert not (tmp_path/"bank/layer.pt").exists()
    failure = json.loads((tmp_path/"worker/receipt.json.failure.json").read_text())
    assert not failure["complete"] and failure["scientific_oracle_calls"] == 0


def test_capacity_gate_keeps_original_allocator_and_extra_context_headroom():
    request = {"expected_gpu_uuid":"GPU-unit", "minimum_free_bytes":12*GIB}
    require_capacity(request, {"uuid":"GPU-unit", "free_bytes":12*GIB})
    for snapshot in ({"uuid":"GPU-unit", "free_bytes":12*GIB-1},
                     {"uuid":"GPU-other", "free_bytes":100*GIB}):
        with pytest.raises(RuntimeError): require_capacity(request, snapshot)
