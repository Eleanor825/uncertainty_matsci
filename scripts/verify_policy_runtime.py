#!/usr/bin/env python3
"""A labelled technical gate on real weights, never a main-experiment result."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from matdiscovery.policy import QwenPolicyAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    output = root / "technical_not_main" / args.model_key
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.30)
    started = time.monotonic()
    policy = QwenPolicyAdapter.from_verified_checkpoint(root / "configs/model_manifest.json", args.model_key, root / "data/models" / args.model_key, max_input_tokens=3072)
    print("VERIFIED_MODEL_LOADED", args.model_key, flush=True)
    generated = policy.generate_action([
        {"role": "system", "content": 'Select one allowed element for a materials-agent interface check. Output only a JSON object with one key "action".'},
        {"role": "user", "content": 'Allowed elements: Li, Na, K. Output one selection, for example {"action":"Li"}.'},
    ], seed=91017, legal_schema={"type": "object", "required": ["action"], "properties": {"action": {"type": "string", "enum": ["Li", "Na", "K"]}}, "additionalProperties": False})
    if not generated.success:
        raise RuntimeError(f"Real model action failed: {generated.failure_code}: {generated.error}")
    capture = policy.capture_prefix(generated.input_ids_with_completion, stamp=generated.model_stamp)
    leaves = []
    def inject(module, arguments, result):
        leaf = result.detach().requires_grad_(True)
        leaves.append(leaf)
        return leaf
    flags = [(p, p.requires_grad) for p in policy.model.parameters()]
    handle = None
    try:
        for p, _ in flags:
            p.requires_grad_(False)
        handle = policy.model.get_submodule(policy.mlp_paths[0]).register_forward_hook(inject)
        ids = generated.input_ids_with_completion.to("cuda:0")
        with torch.enable_grad():
            logits = policy.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1]
            target = logits.argmax()
            gradient, = torch.autograd.grad(logits[target], leaves[0])
        if not torch.isfinite(gradient).all() or not bool(gradient.abs().sum() > 0):
            raise RuntimeError("Real Qwen GDN/native Jacobian is absent or nonfinite")
        norm = float(gradient.float().norm())
    finally:
        if handle is not None:
            handle.remove()
        for p, flag in flags:
            p.requires_grad_(flag)
    result = {"classification": "technical_not_main", "model_key": args.model_key, "generated": generated.to_record(), "capture_layers": len(capture.mlp_inputs), "capture_parity_max_abs": capture.parity_max_abs, "native_jacobian_finite_nonzero": True, "gradient_norm": norm, "elapsed_seconds": time.monotonic()-started, "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(), "parameter_report": policy.parameter_report(), "transcoder_graph_validated": False, "scientific_oracle_calls": 0}
    (output / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["classification", "model_key", "capture_layers", "capture_parity_max_abs", "native_jacobian_finite_nonzero", "elapsed_seconds", "max_gpu_allocated_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
