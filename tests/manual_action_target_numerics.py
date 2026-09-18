"""ZERO-ORACLE DIAGNOSTICS: BF16 finite differences and FP32 same-weight shadow.

This does not change the running study, install kernels, train transcoders, or
claim that a shadow precision validates the production BF16 attribution gate.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from matdiscovery.accounting import file_sha256, write_json_atomic
from matdiscovery.action_targets import build_action_targets
from matdiscovery.policy import QwenPolicyAdapter


def differences(first, second):
    a, b = first.detach().float().cpu().numpy(), second.detach().float().cpu().numpy()
    delta = np.abs(a - b).ravel()
    return {"n": int(delta.size), "max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
            "p50_abs": float(np.quantile(delta, 0.5)), "p95_abs": float(np.quantile(delta, 0.95)),
            "p99_abs": float(np.quantile(delta, 0.99)),
            "relative_l2": float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(a.ravel()), 1e-12))}


def profile(model, tokenizer, source, targets, mlp_paths, direction, dtype_name, base_captured=None,
            relative_specs=None):
    import transformers.models.qwen3_5.modeling_qwen3_5 as official
    ids = source.to(next(model.parameters()).device)
    target_positions = list(targets.prediction_positions)
    controls = sorted(set([min(15, target_positions[-1]), min(31, target_positions[-1]), min(63, target_positions[-1]), *target_positions]))
    selector = torch.tensor(controls, device=ids.device)
    target_offsets = torch.tensor([controls.index(x) for x in target_positions], device=ids.device)
    collected = []
    head_inputs = []
    head_handle = model.get_output_embeddings().register_forward_pre_hook(lambda module, arguments: head_inputs.append(arguments[0].detach().clone()))
    original_chunk = official.torch_chunk_gated_delta_rule
    def capture_chunk(query, key, value, *args, **kwargs):
        g = kwargs.get("g", args[0] if len(args) > 0 else None)
        beta = kwargs.get("beta", args[1] if len(args) > 1 else None)
        collected.append({"query": query.detach().cpu(), "key": key.detach().cpu(), "value": value.detach().cpu(),
                          "g": g.detach().cpu(), "beta": beta.detach().cpu()})
        return original_chunk(query, key, value, *args, **kwargs)
    official.torch_chunk_gated_delta_rule = capture_chunk
    try:
        with torch.no_grad():
            reference = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=selector).logits.detach()
    finally:
        official.torch_chunk_gated_delta_rule = original_chunk
        head_handle.remove()
    if len(collected) != 24 or len(head_inputs) != 1:
        raise RuntimeError("The diagnostic did not observe every expected native GDN layer and one LM head")
    with torch.no_grad():
        single_head_reference = torch.cat([model.get_output_embeddings()(head_inputs[0][:, i:i+1]) for i in range(len(controls))], 1)
        action_reference = model.get_output_embeddings()(head_inputs[0][:, target_offsets])
    report = {"precision": dtype_name, "model_device": str(ids.device), "control_prediction_positions": controls,
              "raw_action_mean_logprob": float(targets.mean_logprob(action_reference)),
              "control_batch_action_mean_logprob": float(targets.mean_logprob(reference[:, target_offsets])),
              "same_hidden_head_batch_vs_single": {str(pos): differences(reference[:, i], single_head_reference[:, i]) for i, pos in enumerate(controls)},
              "actual_captured_gdn_layers": len(collected)}
    print(json.dumps({"stage": "captured_reference", "precision": dtype_name, "gdn_layers": len(collected)}), flush=True)
    suffix = tokenizer.encode(" unrelated future text", add_special_tokens=False)
    variants = sorted(set([ids.shape[1], 80, 96, 128, 160]))
    length_rows = []
    with torch.no_grad():
        for length in variants:
            if length < ids.shape[1]:
                continue
            extra = (suffix * ((length - ids.shape[1]) // len(suffix) + 1))[:length - ids.shape[1]]
            longer = torch.cat([ids, torch.tensor([extra], device=ids.device, dtype=ids.dtype)], 1) if extra else ids
            logits = model(input_ids=longer, attention_mask=torch.ones_like(longer), use_cache=False, logits_to_keep=selector).logits
            length_rows.append({"length": length, "per_position": {str(pos): differences(reference[:, i], logits[:, i]) for i, pos in enumerate(controls)},
                                "action_logprob_difference_same_control_batch": float(targets.mean_logprob(logits[:, target_offsets])) - report["control_batch_action_mean_logprob"]})
        prefix_rows = []
        for index, position in enumerate(controls):
            logits = model(input_ids=ids[:, :position + 1], attention_mask=torch.ones_like(ids[:, :position + 1]), use_cache=False, logits_to_keep=1).logits
            prefix_rows.append({"position": position, "raw_logits_same_single_head_shape": differences(single_head_reference[:, index], logits[:, -1])})
        prompt = targets.prompt_token_count
        current = model(input_ids=ids[:, :prompt], attention_mask=torch.ones_like(ids[:, :prompt]), use_cache=True, logits_to_keep=1)
        cache = current.past_key_values
        cached = {}
        if prompt - 1 in target_positions:
            cached[prompt - 1] = current.logits[:, -1].detach()
        for pos in range(prompt, max(target_positions) + 1):
            current = model(input_ids=ids[:, pos:pos + 1], attention_mask=torch.ones_like(ids[:, :pos + 1]),
                            past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = current.past_key_values
            if pos in target_positions:
                cached[pos] = current.logits[:, -1].detach()
        cached_logits = torch.stack([cached[pos] for pos in target_positions], dim=1)
        report["full_vs_actual_incremental_cache"] = {
            "raw_logits": differences(action_reference, cached_logits),
            "action_mean_logprob_difference": float(targets.mean_logprob(cached_logits)) - report["raw_action_mean_logprob"],
        }
        del current, cache, cached, cached_logits
    report.update(future_length_comparisons=length_rows, independently_truncated_prefix_comparisons=prefix_rows)
    print(json.dumps({"stage": "length_and_cache_done", "precision": dtype_name}), flush=True)

    leaves, handles = {}, []
    flags = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in flags:
        p.requires_grad_(False)
    def leaf_hook(name):
        def hook(module, args, output):
            leaf = output.detach().requires_grad_(True)
            leaves[name] = leaf
            return leaf
        return hook
    try:
        handles.append(model.get_input_embeddings().register_forward_hook(leaf_hook("embedding")))
        for path in mlp_paths:
            handles.append(model.get_submodule(path).register_forward_hook(leaf_hook(path)))
        with torch.enable_grad():
            cut_logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=torch.tensor(target_positions, device=ids.device)).logits
            score = targets.mean_logprob(cut_logits)
            names = ["embedding", *mlp_paths]
            gradients = torch.autograd.grad(score, [leaves[name] for name in names], allow_unused=True)
        report["cut_forward_parity"] = differences(action_reference, cut_logits)
        report["future_gradient_max_abs"] = max(float(g[:, targets.causal_horizon + 1:].float().abs().max()) for g in gradients if g is not None and g[:, targets.causal_horizon + 1:].numel())
        selected_path, position, channel = direction["layer"], direction["position"], direction["channel"]
        analytic = float(gradients[names.index(selected_path)][0, position, channel])
        last_gradient = gradients[-1].detach().float().cpu()
        embedding_gradient = gradients[0].detach().float().cpu()
        baselines = {name: value.detach() for name, value in leaves.items()}
        value = baselines[selected_path][0, position, channel].detach().cpu()
        ulp = float(torch.nextafter(value, torch.tensor(float("inf"), dtype=value.dtype)) - value)
        for handle in handles:
            handle.remove()
        handles = []
        def perturbed_score(scale):
            temporary = []
            try:
                temporary.append(model.get_input_embeddings().register_forward_hook(lambda module, inputs, output: baselines["embedding"]))
                for path in mlp_paths:
                    def fixed(module, inputs, output, path=path):
                        if path != selected_path:
                            return baselines[path]
                        result = baselines[path].clone()
                        result[0, position, channel] += scale
                        return result
                    temporary.append(model.get_submodule(path).register_forward_hook(fixed))
                with torch.no_grad():
                    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=torch.tensor(target_positions, device=ids.device)).logits
                    return float(targets.mean_logprob(logits))
            finally:
                for handle in temporary:
                    handle.remove()
        sweep = []
        for epsilon in (0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            positive, negative = perturbed_score(epsilon), perturbed_score(-epsilon)
            displacement = float((value + epsilon).float() - (value - epsilon).float())
            numerical = (positive - negative) / (2 * epsilon)
            effective = (positive - negative) / displacement if displacement else None
            sweep.append({"epsilon": epsilon, "positive_score": positive, "negative_score": negative,
                          "nominal_numerical": numerical, "effective_displacement": displacement,
                          "effective_numerical": effective, "analytical": analytic,
                          "absolute_error": abs(numerical - analytic),
                          "relative_error": abs(numerical - analytic) / max(abs(analytic), 1e-12)})
        report["fixed_basis_finite_difference_sweep"] = {"direction": direction, "baseline_value": float(value),
                                                       "next_representable_spacing": ulp, "rows": sweep}
        if relative_specs is None:
            effects = {}
            for name, gradient in zip(names, gradients):
                if gradient is not None:
                    values = (gradient[0].float() * baselines[name][0].float()).sum(-1)[:targets.causal_horizon + 1]
                    index = int(values.abs().argmax())
                    effects[name] = (index, float(values[index]))
            best_mlp = max(mlp_paths, key=lambda name: abs(effects[name][1]))
            relative_specs = [{"source": "embedding", "position": effects["embedding"][0]},
                              {"source": best_mlp, "position": effects[best_mlp][0]}]
        relative_rows = []
        for spec in relative_specs:
            name, pos = spec["source"], spec["position"]
            vector = baselines[name][0, pos].detach()
            gradient = gradients[names.index(name)][0, pos].detach().float()
            analytic_relative = float((gradient * vector.float()).sum())
            def relative_value(scale):
                temporary = []
                def perturb(path):
                    if path != name:
                        return baselines[path]
                    changed = baselines[path].clone()
                    changed[0, pos] += vector * scale
                    return changed
                try:
                    temporary.append(model.get_input_embeddings().register_forward_hook(lambda module, inputs, output: perturb("embedding")))
                    for path in mlp_paths:
                        temporary.append(model.get_submodule(path).register_forward_hook(lambda module, inputs, output, path=path: perturb(path)))
                    with torch.no_grad():
                        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                       logits_to_keep=torch.tensor(target_positions, device=ids.device)).logits
                        return float(targets.mean_logprob(logits))
                finally:
                    for handle in temporary:
                        handle.remove()
            central = relative_value(0.0)
            relative_sweep = []
            for epsilon in (0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
                positive, negative = relative_value(epsilon), relative_value(-epsilon)
                plus = (vector + vector * epsilon).float()
                minus = (vector - vector * epsilon).float()
                displacement = plus - minus
                projection = float((displacement * vector.float()).sum() / vector.float().square().sum().clamp_min(1e-20))
                expected_score_change = float((gradient * displacement).sum())
                numerical = (positive - negative) / (2 * epsilon)
                relative_sweep.append({"epsilon": epsilon, "positive_score": positive, "negative_score": negative,
                    "numerical": numerical, "analytical": analytic_relative,
                    "relative_error": abs(numerical - analytic_relative) / max(abs(analytic_relative), 1e-12),
                    "input_components_changed_fraction": float((displacement != 0).float().mean()),
                    "actual_input_displacement_l2": float(displacement.norm()),
                    "effective_direction_coefficient": projection,
                    "score_delta": positive - negative, "linear_prediction_using_actual_rounded_displacement": expected_score_change,
                    "output_residual_beyond_input_rounding": positive - negative - expected_score_change,
                    "central_second_difference": positive + negative - 2 * central})
            relative_rows.append({**spec, "kind": "real_embedding_token_vector" if name == "embedding" else "real_mlp_output_token_vector_not_transcoder_feature",
                                  "direction_l2": float(vector.float().norm()), "analytical": analytic_relative,
                                  "central_score": central, "rows": relative_sweep})
        report["high_signal_relative_direction_sweeps"] = relative_rows
    finally:
        for handle in handles:
            handle.remove()
        for p, flag in flags:
            p.requires_grad_(flag)
    print(json.dumps({"stage": "finite_difference_sweep_done", "precision": dtype_name}), flush=True)
    # For strict kernel isolation, both arithmetic variants see identical captured
    # BF16 numbers promoted losslessly to FP32; no changed neural projections.
    operator_rows = []
    captured = collected if base_captured is None else base_captured
    with torch.no_grad():
        for ordinal, item in enumerate(captured):
            args = {name: value.to(device="cuda:0", dtype=torch.float32 if dtype_name == "fp32_same_weight_shadow" else value.dtype) for name, value in item.items()}
            options = {"output_final_state": True, "use_qk_l2norm_in_kernel": True}
            chunk, chunk_state = original_chunk(**args, **options)
            recurrent, recurrent_state = official.torch_recurrent_gated_delta_rule(**args, **options)
            cropped = {name: value[:, :targets.causal_horizon + 1] for name, value in args.items()}
            prefix, _ = original_chunk(**cropped, **options)
            operator_rows.append({"linear_layer_ordinal": ordinal, "captured_input_dtype": str(item["query"].dtype),
                                  "evaluation_input_dtype": str(args["query"].dtype),
                                  "operator_device": str(args["query"].device),
                                  "chunk_vs_recurrent_output": differences(chunk, recurrent),
                                  "chunk_vs_recurrent_final_state": differences(chunk_state, recurrent_state),
                                  "same_input_future_length_prefix": differences(chunk[:, :targets.causal_horizon + 1], prefix)})
    report["gdn_same_captured_input_operator_comparisons"] = operator_rows
    print(json.dumps({"stage": "gdn_operators_done", "precision": dtype_name, "layers": len(operator_rows)}), flush=True)
    del baselines, leaves, gradients, cut_logits, reference
    gc.collect()
    torch.cuda.empty_cache()
    return report, collected, {"last_mlp": last_gradient, "embedding": embedding_gradient}, relative_specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    previous = json.loads(args.verification.read_text())
    started = time.monotonic()
    result = {"classification": "technical_not_main_v2_numerical_diagnostics", "physical_oracle_calls": 0,
              "production_dtype_or_backend_changed": False, "transcoder_fidelity_tested": False,
              "production_bf16_gate_promoted_to_pass": False,
              "source_verification_sha256": file_sha256(args.verification), "script_sha256": file_sha256(__file__),
              "profiles": {}}
    try:
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(0.25)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        free, _ = torch.cuda.mem_get_info()
        result["gpu_free_bytes_before"] = free
        if free < 14 * 1024**3:
            raise RuntimeError("BF16 diagnostic needs at least 14 GiB currently free; no other task is modified")
        policy = QwenPolicyAdapter.from_verified_checkpoint(root / "configs/model_manifest.json", "qwen35_4b", root / "data/models/qwen35_4b")
        if policy.checkpoint_hash != previous["checkpoint_hash"]:
            raise RuntimeError("The diagnostic checkpoint differs from the recorded real action")
        generation = previous["generated"]
        ids = torch.tensor(generation["input_ids_with_completion"], dtype=torch.long)
        targets = build_action_targets(ids, prompt_token_count=generation["prompt_token_count"], tokenizer=policy.tokenizer,
                                        benchmark="crystalgym", parsed_action=generation["parsed_action"])
        direction = {key: previous["last_mlp_basis_finite_difference"][key] for key in ("layer", "position", "channel")}
        result.update(checkpoint_hash=policy.checkpoint_hash, action_targets=targets.to_dict(), fixed_direction=direction,
                      tf32_disabled_in_diagnostic_process_only=True)
        bf16, captured, bf16_gradients, relative_specs = profile(policy.model, policy.tokenizer, ids, targets, policy.mlp_paths, direction, "bf16_production_precision")
        result["profiles"]["bf16_production_precision"] = bf16
        write_json_atomic(args.output, result)
        # This exact widening cast is a separate numerical shadow, not the
        # verified adapter's production arithmetic. Do not use adapter guards or
        # present the shadow as a newly verified production checkpoint.
        samples = {name: parameter.detach().flatten()[::max(1, parameter.numel() // 16)].float().cpu() for name, parameter in policy.model.named_parameters()}
        policy.model.cpu()
        torch.cuda.empty_cache()
        torch.set_num_threads(8)
        policy.model.float()
        result["fp32_whole_model_shadow_device"] = "cpu; production GPU arithmetic unchanged"
        result["sampled_weight_values_preserved_under_exact_fp32_widening"] = all(torch.equal(value, parameter.detach().flatten()[::max(1, parameter.numel() // 16)].float().cpu()) for (name, parameter) in policy.model.named_parameters() for value in [samples[name]])
        fp32, _, fp32_gradients, _ = profile(policy.model, policy.tokenizer, ids, targets, policy.mlp_paths, direction, "fp32_same_weight_shadow", base_captured=captured, relative_specs=relative_specs)
        result["profiles"]["fp32_same_weight_shadow"] = fp32
        result["cross_precision_cut_gradient_comparisons"] = {name: differences(bf16_gradients[name], fp32_gradients[name]) for name in bf16_gradients}
        result["diagnostic_completed"] = True
    except Exception as exc:
        result.update(diagnostic_completed=False, error_type=type(exc).__name__, error=str(exc))
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["max_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        write_json_atomic(args.output, result)
    print(json.dumps({key: result.get(key) for key in ("classification", "diagnostic_completed", "physical_oracle_calls", "elapsed_seconds", "max_gpu_allocated_bytes", "error_type", "error")}))
    return 0 if result.get("diagnostic_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
