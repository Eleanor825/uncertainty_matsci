#!/usr/bin/env python3
"""Registered four-layer normalized64 GPU diagnostic; never a main-study gate."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import inspect
from pathlib import Path
import time
import traceback

import torch
import torch.nn.functional as F

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.transcoders import TopKTranscoder, TranscoderConfig, iter_activation_batches, reconstruction_metrics, validate_shard_splits
from matdiscovery.core_protocol import read_core
from matdiscovery.core_collection import completed_core_collections
from verify_affine_fold import train_statistics, fold_to_original_topk

torch.set_num_threads(2)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
LAYERS = [1, 12, 13, 16]
CAP = 2 * 1024**3


def require(value, message):
    if not value: raise RuntimeError(message)


def artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


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


def worker(task):
    output = Path(task["output"])
    require(not output.exists(), "Existing diagnostic output must never be replayed/overwritten")
    output.mkdir(parents=True)
    started = time.time()
    write_json_atomic(output / "started.json", {"task": task, "started_at": started, "complete": False})
    try:
        for item in task["sources"]:
            require(file_sha256(item["path"]) == item["sha256"], "Diagnostic source differs before training")
        torch.cuda.set_device(0)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(CAP / total, 0)
        torch.cuda.reset_peak_memory_stats(0)
        config = TranscoderConfig(2560, 2560, 5120, 64)
        provenance = validate_shard_splits(task["train_paths"], task["dev_paths"],
            policy_fingerprint=task["policy_fingerprint"], layer_path=task["layer_path"], config=config)
        require(provenance["rows"] == {"train": 4832, "dev": 1344}, "Diagnostic must retain all exact train/dev rows")
        stats = train_statistics(task["train_paths"])
        torch.save({"mean_x": stats["mean_x"], "mean_y": stats["mean_y"],
            "scale_x": stats["scale_x"], "scale_y": stats["scale_y"]}, output / "training_statistics.pt")
        stats_record = {key: stats[key] for key in ("rows", "scale_x", "scale_y", "sources", "policy_fingerprint", "layer_path")}
        stats_record["tensor_file"] = artifact(output / "training_statistics.pt")
        write_json_atomic(output / "training_statistics.json", stats_record)
        model = TopKTranscoder(config, seed=task["seed"]).cuda()
        raw = fold_to_original_topk(model, stats).cuda()
        mx, my = stats["mean_x"].cuda().float(), stats["mean_y"].cuda().float()
        sx, sy = stats["scale_x"], stats["scale_y"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.0)
        history, best, best_state, best_normalized, best_epoch = [], float("inf"), None, None, None
        max_fold_error = 0.
        for epoch in range(64):
            model.train(); loss_sum, rows = 0., 0
            for x, y in iter_activation_batches(task["train_paths"], 256, seed=task["seed"] + epoch, shuffle=True):
                x, y = x.cuda(), y.cuda()
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
                # This is a numerical transform check on an already consumed
                # training batch, not an extra training or evaluation example.
                validate_raw_mapping(model, raw, mx, my, sx, sy)
                fold_audit = audit_fold(model, raw, x, mx, my, sx, sy)
                fold_error = fold_audit["output_max_absolute_error"]
                max_fold_error = max(max_fold_error, fold_error)
                train_raw = reconstruction_metrics(raw, iter_activation_batches(task["train_paths"], 256, seed=task["seed"], shuffle=False))
                dev_raw = reconstruction_metrics(raw, iter_activation_batches(task["dev_paths"], 256, seed=task["seed"], shuffle=False))
            require(rows == 4832 and train_raw["rows"] == 4832 and dev_raw["rows"] == 1344, "An epoch omitted data")
            row = {"epoch": epoch, "train_rows": rows, "train_normalized_batch_mse": loss_sum / rows,
                "train_raw_end_epoch": train_raw, "dev_raw": dev_raw, "fold_max_absolute_error": fold_error,
                "fold_audit": fold_audit}
            history.append(row)
            with (output / "history.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n"); f.flush()
            if dev_raw["output_mse"] < best:
                best, best_epoch = dev_raw["output_mse"], epoch
                best_state = {key: value.detach().cpu().clone() for key, value in raw.state_dict().items()}
                best_normalized = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        raw.load_state_dict(best_state); raw.eval()
        frequency = torch.zeros(5120, dtype=torch.int64)
        zero_rows = 0
        with torch.no_grad():
            for x, _ in iter_activation_batches(task["train_paths"], 256, seed=task["seed"], shuffle=False):
                active = raw.encode(x.cuda()) > 0
                frequency += active.sum(0).cpu(); zero_rows += int((~active.any(1)).sum())
        torch.cuda.synchronize()
        dev = history[best_epoch]["dev_raw"]
        torch.save(frequency, output / "selected_train_feature_frequencies.pt")
        result = {"schema": "normalized64_four_layer_diagnostic_v2", "classification": "technical_diagnostic_not_main_acceptance",
            "complete": True, "task": task, "started_at": started, "finished_at": time.time(),
            "elapsed_seconds": time.time() - started, "epochs": 64, "selected_epoch": best_epoch,
            "dev_raw": dev, "fidelity_gate_passed": not dev["fvu_undefined"] and dev["output_fvu"] <= .5,
            "original_FVU_threshold": .5, "statistics": stats_record, "history": history,
            "selected_train_firing": {"rows": 4832, "ever_positive": int((frequency > 0).sum()),
                "never_positive": int((frequency == 0).sum()), "zero_feature_rows": zero_rows,
                "mean_active_features": float(frequency.sum()) / 4832},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0), "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "max_fold_error": max_fold_error, "new_policy_or_oracle_calls": 0, "test_data_used": False}
        result["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "threads": torch.get_num_threads(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32, "cap_bytes": CAP}
        torch.save({"state_dict": best_state, "metadata": {"config": task["config"], "normalization": stats_record,
            "seed": task["seed"], "epochs": 64, "selected_epoch": best_epoch, "dev_raw": dev,
            "classification": "diagnostic_folded_raw_weights_not_registered_main_bank"}}, output / "selected_folded.pt")
        torch.save({"state_dict": best_normalized, "config": task["config"]}, output / "selected_normalized.pt")
        result["weights"] = artifact(output / "selected_folded.pt")
        for item in task["sources"]:
            require(file_sha256(item["path"]) == item["sha256"], "Diagnostic source changed during run")
        write_json_atomic(output / "result.json", result)
        print(json.dumps({"layer": task["layer_index"], "FVU": dev["output_fvu"], "best_epoch": best_epoch,
            "passed": result["fidelity_gate_passed"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        return result
    except BaseException as exc:
        write_json_atomic(output / "failure.json", {"complete": False, "task": task,
            "error": repr(exc), "traceback": traceback.format_exc(), "new_policy_or_oracle_calls": 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--precompute64", type=Path, required=True)
    parser.add_argument("--selected-bank64", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Diagnostic registration/output already exists; never rerun silently")
    core = read_core(args.core); completed_core_collections(core)
    source = Path(__file__).resolve(); helper = source.with_name("verify_affine_fold.py")
    technical = source.parent.parent
    boundary = technical / "tc_fold_boundary_v1/output/result.json"
    boundary_report = json.loads(boundary.read_text())
    require(boundary_report["float64_refold_support_equal"] and boundary_report["float64_refold_output_max_error"] < 1e-10,
        "Missing independent diagnosis of the FP32 TopK cutoff discrepancy")
    prior_failed = technical / "tc_normalized64_diagnostic_v1"
    failed_evidence = [artifact(prior_failed / "registration.json")] + [artifact(prior_failed / f"layer_{i:02d}/failure.json") for i in LAYERS]
    tasks = []
    for index in LAYERS:
        prior_path = args.precompute64 / f"layer_{index:02d}/complete.json"
        prior = json.loads(prior_path.read_text()); c = prior["contract"]
        meta_path = args.selected_bank64 / f"layer_{index:02d}.pt.json"
        metadata = json.loads(meta_path.read_text())
        require(c["registration"]["core_fingerprint"] == core["fingerprint"] and c["epochs"] == 64
            and c["seed"] == 1729 + index and metadata["max_dev_fvu"] == .5, "Wrong baseline condition")
        require(prior["complete"] and c["config"] == {"input_dim": 2560, "output_dim": 2560, "feature_dim": 5120, "top_k": 64}
            and c["batch_size"] == 256 and c["learning_rate"] == .0004, "Unexpected baseline training contract")
        require(c["source"]["transcoders.py"] == file_sha256(inspect.getfile(TopKTranscoder)), "Wrong original numeric module")
        dev_sources = metadata["execution_provenance"]["development_files"]
        tasks.append({"core_fingerprint": core["fingerprint"], "layer_index": index, "layer_path": c["layer_path"], "config": c["config"],
            "policy_fingerprint": c["policy_fingerprint"], "seed": c["seed"], "train_paths": c["train_paths"],
            "dev_paths": [item["path"] for item in dev_sources], "output": str(args.output.resolve() / f"layer_{index:02d}"),
            "sources": [artifact(p) for p in c["train_paths"]] + dev_sources + [artifact(prior_path), artifact(meta_path), artifact(source), artifact(helper), artifact(inspect.getfile(TopKTranscoder)), artifact(boundary)] + failed_evidence})
    args.output.mkdir(parents=True)
    registration = {"schema": "registered_normalized64_diagnostic_v2", "core_fingerprint": core["fingerprint"],
        "layers": LAYERS, "epochs": 64, "workers": 4, "policy_or_oracle_calls": 0,
        "source": artifact(source), "affine_proof_source": artifact(helper), "tasks": tasks,
        "prior_failed_diagnostic": failed_evidence, "FP32_cutoff_diagnosis": artifact(boundary),
        "numerical_contract": "exact_finite_registered_parameter_mapping_required; scale_aware_FP32_forward_error_and_cutoff_switches_recorded; actual_folded_raw_space_dev_selects; native_gates_unchanged",
        "changes": ["all_train_coordinate_means_and_positive_global_centered_RMS", "train_normalized_MSE", "fold_every_epoch_into_original_raw_TopK_before_complete_dev_selection"],
        "unchanged": {"seed": "1729+layer", "optimizer": "AdamW", "lr": .0004, "weight_decay": 0.,
            "batch_size": 256, "top_k": 64, "feature_dim": 5120, "fvu_threshold": .5,
            "train_rows": 4832, "dev_rows": 1344, "cuda_tf32": False, "cudnn_tf32": False, "threads_per_worker": 2},
        "classification": "technical_diagnostic_not_main_acceptance", "created_at": time.time()}
    registration["fingerprint"] = fingerprint(registration)
    write_json_atomic(args.output / "registration.json", registration)
    free, total = torch.cuda.mem_get_info(0)
    require(free >= 16 * 1024**3, "Insufficient current free memory; no worker launched, retain registration")
    started = time.time()
    write_json_atomic(args.output / "started.json", {"started_at": started, "free_total_bytes": [free, total], "registration_fingerprint": registration["fingerprint"]})
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(worker, tasks))
    summary = {"complete": True, "classification": "technical_diagnostic_not_main_acceptance",
        "registration": artifact(args.output / "registration.json"), "started_at": started,
        "finished_at": time.time(), "elapsed_seconds": time.time() - started,
        "cases": [{"layer": row["task"]["layer_index"], "dev_FVU": row["dev_raw"]["output_fvu"],
            "best_epoch": row["selected_epoch"], "passed": row["fidelity_gate_passed"],
            "firing": row["selected_train_firing"], "peak_allocated_bytes": row["peak_allocated_bytes"],
            "peak_reserved_bytes": row["peak_reserved_bytes"], "elapsed_seconds": row["elapsed_seconds"]} for row in results],
        "all_four_passed": all(row["fidelity_gate_passed"] for row in results), "new_policy_or_oracle_calls": 0}
    write_json_atomic(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
