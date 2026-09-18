"""Train-only scalar-normalized TopK fitting and audited FP32 raw export.

The statistical arithmetic, parameter-fold order and boundary audit are copied
from the frozen normalized64 diagnostic. Training optimizes normalized MSE;
selection always evaluates the actual exported raw-space FP32 model. Hard TopK
near ties preclude an arbitrary-input claim of pointwise FP32 equivalence.
"""
from __future__ import annotations
from dataclasses import asdict
import copy
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from .accounting import file_sha256, write_json_atomic
from .esopt import tensor_state_hash
from .transcoders import (TopKTranscoder, TranscoderConfig, _load_shard,
    iter_activation_batches, reconstruction_metrics, validate_shard_splits)

FIT_RECIPE = "train_centered_scalar_rms_fp32_export_v1"
DIAGNOSTIC_REFERENCE_SHA256 = "f65f495424498f567aa16c319f40264009d680c5f7e9a93bb733a1e58b1f6110"
STATISTICS_REFERENCE_SHA256 = "dbb48d6a17b8a78b5d64b8b4580ef29085eb8c90a79ab9bc39ed90f9653531fc"


def require(value, message):
    if not value:
        raise ValueError(message)


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


def validate_raw_mapping(model, raw, mx, my, sx, sy):
    expected = {"encoder.weight": (sy / sx) * model.encoder.weight,
        "encoder.bias": sy * (model.encoder.bias - model.encoder.weight @ mx / sx),
        "decoder.weight": model.decoder.weight, "decoder.bias": sy * model.decoder.bias + my}
    for key, value in raw.state_dict().items():
        require(bool(torch.isfinite(value).all()) and torch.equal(value, expected[key]),
            "Export parameter mapping is nonfinite or differs from the registered affine formula")


def audit_fold(model, raw, x, mx, my, sx, sy):
    """Audit affine arithmetic separately from discontinuous FP32 TopK ties.

    Raw-space evaluation and future native FD gates remain authoritative. This
    does not claim arbitrary-input FP32 equality between two arithmetic orders.
    """
    with torch.no_grad():
        pre_n = sy * model.encoder((x - mx) / sx)
        pre_r = raw.encoder(x)
        per_row_error = (pre_n - pre_r).abs().amax(1)
        require(bool(torch.isfinite(per_row_error).all()), "Nonfinite affine arithmetic")
        unit = torch.finfo(x.dtype).eps / 2
        gamma = lambda n: (n + 4) * unit / (1 - (n + 4) * unit)
        # Conservative dot-product forward-error scales, not an arbitrary
        # output-unit tolerance. These audits do not replace the native gate.
        pre_scale = sy * F.linear(((x - mx) / sx).abs(), model.encoder.weight.abs(), model.encoder.bias.abs())
        pre_scale += F.linear(x.abs(), raw.encoder.weight.abs(), raw.encoder.bias.abs())
        pre_scale += sy * F.linear((x.abs() + mx.abs()) / sx, model.encoder.weight.abs())
        pre_bound = gamma(x.shape[1]) * pre_scale.amax(1)
        z_n = sy * model.encode((x - mx) / sx); z_r = raw.encode(x)
        changed = (z_n > 0) != (z_r > 0); affected = changed.any(1)
        output_error = (raw(x) - (sy * model((x - mx) / sx) + my)).abs().amax(1)
        same_error = output_error[~affected].max().item() if bool((~affected).any()) else 0.
        require(bool(torch.isfinite(output_error).all()), "Nonfinite folded output")
        out_scale = F.linear(z_n.abs() + z_r.abs(), raw.decoder.weight.abs(), raw.decoder.bias.abs())
        out_bound = gamma(model.config.feature_dim) * out_scale
        out_bound += F.linear((z_n - z_r).abs(), raw.decoder.weight.abs())
        out_bound = out_bound.amax(1)
        cutoff = pre_n.topk(model.config.top_k, dim=-1).values[:, -1].clamp_min(0)
        margin = (pre_n - cutoff[:, None]).abs()
        band = 2 * per_row_error[:, None] + 2 * unit * pre_n.abs()
        outside_band = changed & (margin > band)
        sign_changed = (pre_n > 0) != (pre_r > 0)
        sign_outside = sign_changed & (pre_n.abs() > per_row_error[:, None] + 2 * unit * pre_n.abs())
        return {"batch_rows": len(x), "preactivation_max_absolute_error": per_row_error.max().item(),
            "output_max_absolute_error": output_error.max().item(), "same_support_output_max_error": same_error,
            "cutoff_rounding_affected_rows": int(affected.sum()), "support_difference_entries": int(changed.sum()),
            "preactivation_max_forward_error_bound": pre_bound.max().item(),
            "preactivation_max_error_bound_ratio": (per_row_error / pre_bound.clamp_min(torch.finfo(x.dtype).tiny)).max().item(),
            "same_support_max_error_bound_ratio": (output_error[~affected] / out_bound[~affected].clamp_min(torch.finfo(x.dtype).tiny)).max().item() if bool((~affected).any()) else 0.,
            "support_changes_outside_rounding_band": int(outside_band.sum()), "sign_changes_outside_rounding_band": int(sign_outside.sum()),
            "claim": "real_arithmetic_equivalence_FP32_cutoff_discontinuities_recorded"}


def _files(paths):
    return [{"path": str(Path(path).resolve()), "sha256": file_sha256(path)} for path in paths]


def _validate_raw_shards(train_paths, dev_paths, config, policy_fingerprint, layer_path):
    require(len(set(train_paths + dev_paths)) == len(train_paths) + len(dev_paths), "Duplicate raw shard path")
    for split, paths in (("train", train_paths), ("dev", dev_paths)):
        require(paths, "Complete train and development sources are required")
        for path in paths:
            shard = _load_shard(path); meta = shard["metadata"]
            x, y = shard["inputs"], shard["outputs"]
            require(meta["split"] == split and meta["policy_fingerprint"] == policy_fingerprint
                and meta["layer_path"] == layer_path, "Raw source split/policy/layer mismatch; test is forbidden")
            require(x.ndim == y.ndim == 2 and x.shape == (meta["rows"], config.input_dim)
                and y.shape == (meta["rows"], config.output_dim) and len(x) > 0
                and meta["input_dim"] == config.input_dim and meta["output_dim"] == config.output_dim
                and len(meta["group_ids"]) == len(meta["prefix_hashes"]) == len(x), "Raw source dimensions/rows differ")
            require(x.dtype == y.dtype == torch.float32 and bool(torch.isfinite(x).all())
                and bool(torch.isfinite(y).all()), "Raw activation sources must be finite FP32")


def _statistics_record(stats):
    tensors = {"mean_x": stats["mean_x"], "mean_y": stats["mean_y"],
        "scale_x": torch.tensor(stats["scale_x"], dtype=torch.float64),
        "scale_y": torch.tensor(stats["scale_y"], dtype=torch.float64)}
    return {"fit_recipe": FIT_RECIPE, "train_only": True, "input_space": "raw_captured_mlp_input_output",
        "statistics_dtype": "torch.float64", "runtime_mean_dtype": "torch.float32",
        "statistic": "per_coordinate_mean_and_global_centered_RMS_over_all_rows_and_coordinates",
        "rows": stats["rows"], "policy_fingerprint": stats["policy_fingerprint"], "layer_path": stats["layer_path"],
        "sources": copy.deepcopy(stats["sources"]), "mean_x": stats["mean_x"].tolist(), "mean_y": stats["mean_y"].tolist(),
        "scale_x": stats["scale_x"], "scale_y": stats["scale_y"],
        "train_statistics_tensor_hash": tensor_state_hash(tensors),
        "export": {"dtype": "torch.float32", "encoder_weight": "(scale_y / scale_x) * W",
            "encoder_bias": "scale_y * (b - W @ mean_x / scale_x)", "decoder_weight": "D unchanged",
            "decoder_bias": "scale_y * b_decoder + mean_y",
            "arbitrary_input_FP32_pointwise_equivalence_claimed": False,
            "cutoff_discontinuities": "recorded; actual raw FP32 model determines reconstruction and all future native gates"}}


def train_normalized_layer(config, train_paths, dev_paths, *, policy_fingerprint, layer_path,
                           output_path, seed, epochs=64, batch_size=256, learning_rate=4e-4,
                           device="cuda:0", max_dev_fvu=.5, cpu_threads=2,
                           gpu_memory_bytes=2 * 1024**3, registration=None):
    """Return the actual selected raw TopK model and original-schema metadata.

    Only ``output_path`` and its ``.json`` sidecar are written. Full-precision
    train-only means/scales are embedded in metadata, preserving the original
    bank file inventory. Failed fidelity remains an explicit saved result.
    ``train_output_mse`` means the raw end-of-epoch training MSE, never the
    normalized optimizer loss. Partial/existing output is never overwritten.
    """
    config.validate()
    require(type(epochs) is int and epochs == 64, "This registered recipe requires all64 epochs")
    require(type(batch_size) is int and batch_size > 0 and math.isfinite(learning_rate) and learning_rate > 0,
            "Invalid batch size/learning rate")
    require(type(seed) is int and seed >= 0 and type(cpu_threads) is int and cpu_threads == 2,
            "A nonnegative seed and fixed two-thread runtime are required")
    require(max_dev_fvu == .5 and gpu_memory_bytes == 2 * 1024**3, "Registered fidelity threshold/cap changed")
    require(torch.get_default_dtype() == torch.float32, "Recipe initialization must be native FP32")
    output = Path(output_path).resolve(); sidecar = Path(str(output) + ".json")
    require(not output.exists() and not sidecar.exists(), "Existing/partial normalized output requires reconciliation")
    paths = [str(Path(path).resolve()) for path in train_paths]
    development = [str(Path(path).resolve()) for path in dev_paths]
    _validate_raw_shards(paths, development, config, policy_fingerprint, layer_path)
    files = {"train": _files(paths), "dev": _files(development)}
    sources = {name: file_sha256(Path(__file__).with_name(name)) for name in
               ("normalized_transcoders.py", "transcoders.py", "esopt.py")}
    torch.set_num_threads(cpu_threads); torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    target = torch.device(device)
    require(target.type in {"cpu", "cuda"}, "Only the validated CPU/CUDA FP32 implementations are supported")
    if target.type == "cuda":
        torch.cuda.set_device(target)
        total = torch.cuda.get_device_properties(target).total_memory
        torch.cuda.set_per_process_memory_fraction(gpu_memory_bytes / total, target)
        torch.cuda.reset_peak_memory_stats(target)
    provenance = validate_shard_splits(paths, development, policy_fingerprint=policy_fingerprint,
                                       layer_path=layer_path, config=config)
    stats = train_statistics(paths)
    require(stats["rows"] == provenance["rows"]["train"] and stats["sources"] == files["train"], "Statistics omitted or changed a training source")
    normalization = _statistics_record(stats)
    model = TopKTranscoder(config, seed=seed).to(target)
    raw = fold_to_original_topk(model, stats).to(target)
    mx, my = stats["mean_x"].to(target).float(), stats["mean_y"].to(target).float()
    sx, sy = stats["scale_x"], stats["scale_y"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    history, best, best_state, best_normalized, best_epoch = [], float("inf"), None, None, None
    max_fold_error = 0.
    for epoch in range(epochs):
        model.train(); loss_sum, rows = 0., 0
        for x, y in iter_activation_batches(paths, batch_size, seed=seed + epoch, shuffle=True):
            x, y = x.to(target), y.to(target)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model((x - mx) / sx), (y - my) / sy)
            require(bool(torch.isfinite(loss)), "Nonfinite normalized training loss")
            loss.backward(); optimizer.step(); model.normalize_decoder_()
            loss_sum += loss.item() * len(x); rows += len(x)
        model.eval()
        with torch.no_grad():
            raw.encoder.weight.copy_((sy / sx) * model.encoder.weight)
            raw.encoder.bias.copy_(sy * (model.encoder.bias - model.encoder.weight @ mx / sx))
            raw.decoder.weight.copy_(model.decoder.weight)
            raw.decoder.bias.copy_(sy * model.decoder.bias + my)
            validate_raw_mapping(model, raw, mx, my, sx, sy)
            fold_audit = audit_fold(model, raw, x, mx, my, sx, sy)
            fold_error = fold_audit["output_max_absolute_error"]
            max_fold_error = max(max_fold_error, fold_error)
            train_raw = reconstruction_metrics(raw, iter_activation_batches(paths, batch_size, seed=seed, shuffle=False))
            dev_raw = reconstruction_metrics(raw, iter_activation_batches(development, batch_size, seed=seed, shuffle=False))
        require(rows == provenance["rows"]["train"] and train_raw["rows"] == rows
            and dev_raw["rows"] == provenance["rows"]["dev"], "An epoch omitted raw source rows")
        for metric in (train_raw, dev_raw):
            require(all(math.isfinite(value) and value >= 0 for value in metric.values()), "Nonfinite/invalid raw reconstruction metric")
        row = {"epoch": epoch, "train_rows": rows, "train_output_mse": train_raw["output_mse"],
            "train_normalized_batch_mse": loss_sum / rows, "train_raw_end_epoch": train_raw,
            "dev": dev_raw, "fold_max_absolute_error": fold_error, "fold_audit": fold_audit}
        history.append(row)
        if dev_raw["output_mse"] < best:
            best, best_epoch = dev_raw["output_mse"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in raw.state_dict().items()}
            best_normalized = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    require(best_state is not None and len(history) == 64, "Incomplete normalized selection")
    raw.load_state_dict(best_state); raw.eval()
    frequency = torch.zeros(config.feature_dim, dtype=torch.int64)
    zero_rows = 0
    with torch.no_grad():
        for x, _ in iter_activation_batches(paths, batch_size, seed=seed, shuffle=False):
            active = raw.encode(x.to(target)) > 0
            frequency += active.sum(0).cpu(); zero_rows += int((~active.any(1)).sum())
    if target.type == "cuda": torch.cuda.synchronize(target)
    dev = history[best_epoch]["dev"]
    runtime = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device": str(target),
        "parameter_dtype": "torch.float32", "torch_cpu_threads": torch.get_num_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "gpu_memory_bytes": gpu_memory_bytes}
    if target.type == "cuda":
        runtime.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(target),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(target))
        require(runtime["peak_reserved_bytes"] <= gpu_memory_bytes, "Normalized fitting exceeded its fixed allocator cap")
    require(files == {"train": _files(paths), "dev": _files(development)}
        and sources == {name: file_sha256(Path(__file__).with_name(name)) for name in sources}, "Source changed during normalized fitting")
    metadata = {"schema_version": 1, "method": "per_mlp_topk_input_to_output_v1", "fit_recipe": FIT_RECIPE,
        "config": asdict(config), "seed": seed, "epochs": epochs, "learning_rate": learning_rate, "batch_size": batch_size,
        "selected_epoch": best_epoch, "history": history, "dev": dev, "max_dev_fvu": max_dev_fvu,
        "fidelity_gate_passed": not dev["fvu_undefined"] and dev["output_fvu"] <= max_dev_fvu,
        "provenance": provenance, "transcoder_hash": raw.checkpoint_hash(), "normalization": normalization,
        "train_output_mse_semantics": "full_raw_train_end_epoch_MSE_not_normalized_optimizer_loss",
        "selected_normalized_state_hash": tensor_state_hash(best_normalized),
        "selected_train_firing": {"rows": provenance["rows"]["train"], "ever_positive": int((frequency > 0).sum()),
            "never_positive": int((frequency == 0).sum()), "zero_feature_rows": zero_rows,
            "mean_active_features": float(frequency.sum()) / provenance["rows"]["train"],
            "feature_frequencies": frequency.tolist()},
        "boundary_audit_scope": "last_consumed_training_batch_each_epoch; full_raw_train_and_dev_metrics_remain_authoritative",
        "max_fold_error": max_fold_error, "runtime": runtime, "source_sha256": sources,
        "frozen_diagnostic_reference": {"run_normalized_diagnostic_v2": DIAGNOSTIC_REFERENCE_SHA256,
            "statistics_and_fold_helper": STATISTICS_REFERENCE_SHA256},
        "source_files": files, "registration": copy.deepcopy(registration), "test_data_used": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "metadata": metadata}, output)
    write_json_atomic(sidecar, metadata)
    return raw, metadata


def verify_normalized_metadata(metadata, *, config=None, policy_fingerprint=None, layer_path=None,
                               train_paths=None, dev_paths=None, expected_registration=None):
    """Lightweight recipe/history/statistics/source verification; no fitting/GPU.

    Callers separately verify checkpoint bytes and the actual raw tensor hash
    with the original loader. When paths are supplied, their SHA256s must match
    the recorded full sources. This helper does not waive a failed FVU gate.
    """
    require(metadata.get("fit_recipe") == FIT_RECIPE and metadata.get("epochs") == 64
        and metadata.get("max_dev_fvu") == .5, "Unknown normalized recipe/epochs/fidelity threshold")
    require(metadata.get("method") == "per_mlp_topk_input_to_output_v1"
        and metadata.get("train_output_mse_semantics") == "full_raw_train_end_epoch_MSE_not_normalized_optimizer_loss"
        and metadata.get("test_data_used") is False, "Normalized checkpoint semantics changed")
    cfg = TranscoderConfig(**metadata["config"]); cfg.validate()
    if config is not None:
        require(metadata["config"] == asdict(config), "Normalized configuration differs")
    provenance = metadata["provenance"]; normalization = metadata["normalization"]
    require(normalization["fit_recipe"] == FIT_RECIPE and normalization["train_only"] is True
        and normalization["statistics_dtype"] == "torch.float64" and normalization["runtime_mean_dtype"] == "torch.float32"
        and normalization["input_space"] == "raw_captured_mlp_input_output"
        and normalization["export"]["dtype"] == "torch.float32"
        and normalization["export"]["arbitrary_input_FP32_pointwise_equivalence_claimed"] is False,
        "Normalization statistics/export contract changed")
    require(normalization["policy_fingerprint"] == provenance["policy_fingerprint"]
        and normalization["layer_path"] == provenance["layer_path"]
        and normalization["rows"] == provenance["rows"]["train"]
        and normalization["sources"] == metadata["source_files"]["train"], "Statistics used a different source")
    if policy_fingerprint is not None:
        require(provenance["policy_fingerprint"] == policy_fingerprint, "Normalized policy differs")
    if layer_path is not None:
        require(provenance["layer_path"] == layer_path, "Normalized layer differs")
    if expected_registration is not None:
        require(metadata["registration"] == expected_registration, "Normalized registration differs")
    for split, paths in (("train", train_paths), ("dev", dev_paths)):
        require(metadata["source_files"][split], "Missing normalized source files")
        if paths is not None:
            require(_files(paths) == metadata["source_files"][split], "Normalized input source bytes/order changed")
    stats = {"mean_x": torch.tensor(normalization["mean_x"], dtype=torch.float64),
        "mean_y": torch.tensor(normalization["mean_y"], dtype=torch.float64),
        "scale_x": torch.tensor(normalization["scale_x"], dtype=torch.float64),
        "scale_y": torch.tensor(normalization["scale_y"], dtype=torch.float64)}
    require(stats["mean_x"].shape == (cfg.input_dim,) and stats["mean_y"].shape == (cfg.output_dim,)
        and all(bool(torch.isfinite(value).all()) for value in stats.values())
        and stats["scale_x"] > 0 and stats["scale_y"] > 0
        and tensor_state_hash(stats) == normalization["train_statistics_tensor_hash"], "Normalization statistics hash/shape differs")
    history = metadata["history"]
    require(len(history) == 64, "Normalized history is not complete64")
    for index, row in enumerate(history):
        require(row["epoch"] == index and row["train_rows"] == provenance["rows"]["train"]
            and row["train_raw_end_epoch"]["rows"] == provenance["rows"]["train"]
            and row["dev"]["rows"] == provenance["rows"]["dev"]
            and row["train_output_mse"] == row["train_raw_end_epoch"]["output_mse"], "Normalized history omitted rows or mislabeled loss")
        require(math.isfinite(row["train_normalized_batch_mse"]) and row["train_normalized_batch_mse"] >= 0,
                "Invalid normalized optimizer loss")
        for metric in (row["train_raw_end_epoch"], row["dev"]):
            require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in metric.values()),
                    "Invalid raw reconstruction metric")
    best = min(range(64), key=lambda index: history[index]["dev"]["output_mse"])
    dev = history[best]["dev"]
    require(metadata["selected_epoch"] == best and metadata["dev"] == dev
        and metadata["fidelity_gate_passed"] == (not dev["fvu_undefined"] and dev["output_fvu"] <= .5),
        "Normalized checkpoint does not use original raw dev selection/fidelity")
    runtime = metadata["runtime"]
    require(runtime["parameter_dtype"] == "torch.float32" and runtime["torch_cpu_threads"] == 2
        and runtime["float32_matmul_precision"] == "highest" and runtime["cuda_tf32"] is False
        and runtime["cudnn_tf32"] is False and runtime["gpu_memory_bytes"] == 2 * 1024**3, "Normalized runtime changed")
    require(metadata["source_sha256"] == {name: file_sha256(Path(__file__).with_name(name)) for name in
        ("normalized_transcoders.py", "transcoders.py", "esopt.py")}, "Normalized fitting source changed")
    return metadata
