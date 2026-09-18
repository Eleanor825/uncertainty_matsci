#!/usr/bin/env python3
"""CPU-only affine TopK reparameterization proof, never a training/main gate."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from matdiscovery.accounting import file_sha256, write_json_atomic
from matdiscovery.transcoders import TopKTranscoder, TranscoderConfig, _load_shard


def train_statistics(paths):
    """Global scalar centered RMS, calculated from EVERY train row in float64."""
    count = 0; sx = sy = xx = yy = None; policy = layer = None
    for path in paths:
        shard = _load_shard(path); meta = shard["metadata"]
        if meta["split"] != "train": raise ValueError("Statistics may read training shards only")
        if policy is not None and (policy, layer) != (meta["policy_fingerprint"], meta["layer_path"]):
            raise ValueError("Mixed policy/layer training statistics")
        policy, layer = meta["policy_fingerprint"], meta["layer_path"]
        x, y = shard["inputs"].double(), shard["outputs"].double()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise ValueError("Nonfinite train data")
        if sx is None:
            sx = torch.zeros(x.shape[1], dtype=torch.float64); sy = torch.zeros(y.shape[1], dtype=torch.float64)
            xx = yy = 0.
        sx += x.sum(0); sy += y.sum(0); xx += x.square().sum().item(); yy += y.square().sum().item(); count += len(x)
    if count < 2: raise ValueError("At least two complete training rows are required")
    mx, my = sx / count, sy / count
    vx = max(0., xx / (count * len(mx)) - mx.square().mean().item())
    vy = max(0., yy / (count * len(my)) - my.square().mean().item())
    if vx <= 1e-24 or vy <= 1e-24: raise ValueError("Undefined centered RMS must fail closed")
    return {"mean_x": mx, "mean_y": my, "scale_x": math.sqrt(vx), "scale_y": math.sqrt(vy),
        "rows": count, "layer_path": layer, "policy_fingerprint": policy,
        "sources": [{"path": str(Path(path).resolve()), "sha256": file_sha256(path)} for path in paths]}


def fold_to_original_topk(normalized, stats):
    """Positive common latent scale preserves ReLU/TopK and unit decoder columns."""
    sx, sy = stats["scale_x"], stats["scale_y"]
    if not all(math.isfinite(v) and v > 0 for v in (sx, sy)): raise ValueError("Both scalar scales must be finite and positive")
    source = normalized.encoder.weight
    mx = stats["mean_x"].to(source); my = stats["mean_y"].to(source)
    if len(mx) != normalized.config.input_dim or len(my) != normalized.config.output_dim: raise ValueError("Mean shape mismatch")
    if not torch.isfinite(mx).all() or not torch.isfinite(my).all(): raise ValueError("Mean is nonfinite")
    raw = TopKTranscoder(normalized.config, seed=0).to(source)
    with torch.no_grad():
        raw.encoder.weight.copy_((sy / sx) * normalized.encoder.weight)
        raw.encoder.bias.copy_(sy * (normalized.encoder.bias - normalized.encoder.weight @ mx / sx))
        raw.decoder.weight.copy_(normalized.decoder.weight)
        raw.decoder.bias.copy_(sy * normalized.decoder.bias + my)
    return raw


def verify(normalized, stats, inputs):
    normalized.requires_grad_(False)
    raw = fold_to_original_topk(normalized, stats).requires_grad_(False)
    x = inputs.to(normalized.encoder.weight).detach().clone().requires_grad_(True)
    mx, my = stats["mean_x"].to(x), stats["mean_y"].to(x); sx, sy = stats["scale_x"], stats["scale_y"]
    latent_n = normalized.encode((x - mx) / sx); latent_r = raw.encode(x)
    expected = sy * normalized((x - mx) / sx) + my; actual = raw(x)
    generator = torch.Generator().manual_seed(7123)
    cotangent = torch.randn(expected.shape, generator=generator, dtype=torch.float64).to(x)
    cotangent /= cotangent.norm()
    grad_expected = torch.autograd.grad((expected * cotangent).sum(), x, retain_graph=True)[0]
    grad_actual = torch.autograd.grad((actual * cotangent).sum(), x)[0]
    a, ai = latent_n.topk(normalized.config.top_k, dim=-1); b, bi = latent_r.topk(raw.config.top_k, dim=-1)
    # Tied zero entries carry no activation/contribution; compare positive ordering.
    positive_order_equal = torch.equal(ai[a > 0], bi[b > 0]) and torch.equal(a > 0, b > 0)
    tolerance = 1e-10 if x.dtype == torch.float64 else 2e-6
    output_error = (expected - actual).abs().max().item()
    gradient_error = (grad_expected - grad_actual).abs().max().item()
    latent_error = (sy * latent_n - latent_r).abs().max().item()
    decoder_equal = torch.equal(raw.decoder.weight, normalized.decoder.weight)
    passed = output_error <= tolerance and gradient_error <= tolerance and latent_error <= tolerance and positive_order_equal and decoder_equal
    return {"dtype": str(x.dtype), "rows_checked": len(x), "shape": asdict(normalized.config),
        "output_max_absolute_error": output_error, "input_gradient_max_absolute_error": gradient_error,
        "scaled_latent_max_absolute_error": latent_error, "positive_topk_order_equal": positive_order_equal,
        "positive_support_equal": torch.equal(latent_n > 0, latent_r > 0), "decoder_weights_bit_equal": decoder_equal,
        "decoder_max_unit_norm_error": (raw.decoder.weight.norm(dim=0) - 1).abs().max().item(),
        "absolute_tolerance": tolerance, "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precomputation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.set_float32_matmul_precision("highest")
    contract = json.loads(args.precomputation.read_text())["contract"]
    stats = train_statistics(contract["train_paths"])
    # All data establish statistics; sixteen fixed train rows exercise numerical
    # equivalence only. This is not a subsampled reconstruction/learning result.
    first = _load_shard(contract["train_paths"][0])["inputs"][:16]
    rows = []
    for dtype in (torch.float64, torch.float32):
        model = TopKTranscoder(TranscoderConfig(**contract["config"]), seed=contract["seed"]).to(dtype)
        # Nonzero biases exercise the complete affine formula, not only its easy zero-bias special case.
        with torch.no_grad():
            model.encoder.bias.fill_(.037); model.decoder.bias.fill_(-.021)
        rows.append(verify(model, stats, first))
    report = {"classification": "CPU_affine_equivalence_only_not_training_or_material_results",
        "passed": all(row["passed"] for row in rows), "optimizer_steps": 0, "GPU_calls": 0,
        "policy_or_oracle_calls": 0, "development_or_test_data_read": False,
        "statistics": {k: stats[k] for k in ("rows", "scale_x", "scale_y", "layer_path", "policy_fingerprint", "sources")},
        "checks": rows, "source_sha256": file_sha256(__file__)}
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]: raise RuntimeError("Affine equivalence check failed")


if __name__ == "__main__": main()
