"""ZERO-ORACLE same-GPU tail precision and earliest length-divergence diagnostics."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from matdiscovery.accounting import file_sha256, write_json_atomic
from matdiscovery.action_targets import build_action_targets
from matdiscovery.policy import QwenPolicyAdapter
from manual_action_target_numerics import differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    source = json.loads(args.verification.read_text())
    report = {"classification": "technical_not_main_same_gpu_precision_isolation",
              "physical_oracle_calls": 0, "production_dtype_or_backend_changed": False,
              "production_bf16_gate_promoted_to_pass": False, "transcoder_fidelity_tested": False,
              "script_sha256": file_sha256(__file__), "source_verification_sha256": file_sha256(args.verification)}
    started = time.monotonic()
    old_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(0.25)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        free, _ = torch.cuda.mem_get_info()
        report["gpu_free_bytes_before"] = free
        if free < 14 * 1024**3:
            raise RuntimeError("Insufficient free GPU memory; no other process is changed")
        policy = QwenPolicyAdapter.from_verified_checkpoint(root / "configs/model_manifest.json", "qwen35_4b", root / "data/models/qwen35_4b")
        if policy.checkpoint_hash != source["checkpoint_hash"]:
            raise RuntimeError("Initial checkpoint changed")
        generation = source["generated"]
        ids = torch.tensor(generation["input_ids_with_completion"], device="cuda:0", dtype=torch.long)
        targets = build_action_targets(ids, prompt_token_count=generation["prompt_token_count"],
                                        tokenizer=policy.tokenizer, benchmark="crystalgym", parsed_action=generation["parsed_action"])
        selector = torch.tensor(targets.prediction_positions, device=ids.device)
        extra = policy.tokenizer.encode(" unrelated future text", add_special_tokens=False)
        suffix = (extra * (160 // len(extra) + 1))[:160 - ids.shape[1]]
        longer = torch.cat([ids, torch.tensor([suffix], device=ids.device)], 1)
        observed_paths = {
            "embedding": policy.model.get_input_embeddings(),
            "layer0_input_norm": policy.model.get_submodule("model.language_model.layers.0.input_layernorm"),
            "layer0_qkv_projection": policy.model.get_submodule("model.language_model.layers.0.linear_attn.in_proj_qkv"),
            "layer0_a_projection": policy.model.get_submodule("model.language_model.layers.0.linear_attn.in_proj_a"),
            "layer0_b_projection": policy.model.get_submodule("model.language_model.layers.0.linear_attn.in_proj_b"),
            "last_mlp": policy.model.get_submodule(policy.mlp_paths[-1]),
        }
        final_norm = policy.model.get_submodule("model.language_model.norm")
        last_norm = policy.model.get_submodule("model.language_model.layers.31.post_attention_layernorm")
        def collect(tokens):
            import transformers.models.qwen3_5.modeling_qwen3_5 as official
            saved, handles = {}, []
            original = official.torch_chunk_gated_delta_rule
            def gdn(query, key, value, *other, **kwargs):
                if "layer0_gdn_query" not in saved:
                    saved.update(layer0_gdn_query=query.detach().clone(), layer0_gdn_key=key.detach().clone(),
                                 layer0_gdn_value=value.detach().clone())
                return original(query, key, value, *other, **kwargs)
            official.torch_chunk_gated_delta_rule = gdn
            try:
                for name, module in observed_paths.items():
                    def record(module, arguments, output, name=name):
                        saved[name] = output.detach().clone()
                    handles.append(module.register_forward_hook(record))
                handles.append(final_norm.register_forward_pre_hook(lambda module, arguments: saved.__setitem__("final_pre_norm", arguments[0].detach().clone())))
                handles.append(last_norm.register_forward_pre_hook(lambda module, arguments: saved.__setitem__("last_residual", arguments[0].detach().clone())))
                with torch.no_grad():
                    saved["logits"] = policy.model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False, logits_to_keep=selector).logits.detach()
                return saved
            finally:
                official.torch_chunk_gated_delta_rule = original
                for handle in handles:
                    handle.remove()
        stages = []
        baselines = None
        for reduction in (old_reduction, False):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduction
            before, after = collect(ids), collect(longer)
            stages.append({"allow_bf16_reduced_precision_reduction": reduction,
                           "short_length": int(ids.shape[1]), "long_length": int(longer.shape[1]),
                           "per_stage_original_prefix": {name: differences(value, after[name][:, :value.shape[1]])
                               for name, value in before.items() if name != "logits"},
                           "selected_raw_logits": differences(before["logits"], after["logits"]),
                           "action_logprob_difference": float(targets.mean_logprob(after["logits"])) - float(targets.mean_logprob(before["logits"]))})
            if baselines is None:
                baselines = before
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduction
        report["shape_dependent_first_divergence"] = stages
        print(json.dumps({"stage": "same_gpu_length_localization_complete"}), flush=True)

        for parameter in policy.model.parameters():
            parameter.requires_grad_(False)
        norm_fp32 = copy.deepcopy(final_norm).float()
        head_fp32 = policy.model.get_output_embeddings().weight.detach().float()
        residual, mlp, anchor = baselines["last_residual"], baselines["last_mlp"], baselines["final_pre_norm"]
        report["recorded_last_residual_addition_exact"] = bool(torch.equal(residual + mlp, anchor))
        position = source["last_mlp_basis_finite_difference"]["position"]
        channel = source["last_mlp_basis_finite_difference"]["channel"]
        def forward_tail(variable, mode):
            if mode == "production_bf16_tail":
                hidden = final_norm(residual + variable)
                return policy.model.get_output_embeddings()(hidden[:, selector])
            if mode == "fp32_anchored_same_bf16_pre_norm_point":
                hidden = anchor.float() + (variable - mlp.float())
            else:
                hidden = residual.float() + variable
            return F.linear(norm_fp32(hidden)[:, selector], head_fp32)
        profiles = []
        for mode in ("production_bf16_tail", "fp32_anchored_same_bf16_pre_norm_point", "fp32_widened_same_residual_and_mlp_values"):
            base = mlp if mode == "production_bf16_tail" else mlp.float()
            variable = base.clone().requires_grad_(True)
            logits = forward_tail(variable, mode)
            score = targets.mean_logprob(logits)
            derivative = torch.autograd.grad(score, variable)[0]
            analytic = float(derivative[0, position, channel])
            rows = []
            for epsilon in (0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
                plus, minus = base.clone(), base.clone()
                plus[0, position, channel] += epsilon
                minus[0, position, channel] -= epsilon
                with torch.no_grad():
                    positive = float(targets.mean_logprob(forward_tail(plus, mode)))
                    negative = float(targets.mean_logprob(forward_tail(minus, mode)))
                numerical = (positive - negative) / (2 * epsilon)
                rows.append({"epsilon": epsilon, "positive_score": positive, "negative_score": negative,
                             "numerical": numerical, "analytical": analytic,
                             "relative_error": abs(numerical - analytic) / max(abs(analytic), 1e-12),
                             "central_second_difference": positive + negative - 2 * float(score.detach())})
            profiles.append({"mode": mode, "device": "cuda:0", "raw_action_mean_logprob": float(score.detach()),
                             "analytic_basis_derivative": analytic, "source_position": position, "source_channel": channel,
                             "difference_from_original_bf16_raw_logits": differences(baselines["logits"], logits), "rows": rows})
        report["same_gpu_tail_precision_profiles"] = profiles
        report["diagnostic_completed"] = True
    except Exception as exc:
        report.update(diagnostic_completed=False, error_type=type(exc).__name__, error=str(exc))
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduction
        report["elapsed_seconds"] = time.monotonic() - started
        report["max_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        write_json_atomic(args.output, report)
    print(json.dumps({key: report.get(key) for key in ("classification", "diagnostic_completed", "physical_oracle_calls", "elapsed_seconds", "max_gpu_allocated_bytes", "error_type", "error")}))
    return 0 if report.get("diagnostic_completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
