"""REAL-WEIGHT TECHNICAL CHECK ONLY: no physical oracle and no transcoder claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from matdiscovery.accounting import file_sha256, write_json_atomic
from matdiscovery.action_targets import build_action_targets
from matdiscovery.policy import DecodingConfig, QwenPolicyAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    started = time.monotonic()
    report = {
        "classification": "technical_not_main_real_qwen4b_action_target_check",
        "physical_oracle_calls": 0, "transcoder_fidelity_tested": False,
        "native_attributor_graph_validated": False,
        "scope": "Actual emitted action alignment, unwarped teacher forcing and same-valued MLP-cut gradient causality only",
        "source_code_sha256": {name: file_sha256(root / "src/matdiscovery" / name) for name in ("action_targets.py", "native_attribution.py", "policy.py")},
    }
    handles, flags = [], []
    try:
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(0.25)
        free, total = torch.cuda.mem_get_info()
        report["gpu_free_bytes_before"] = free
        if free < 14 * 1024**3:
            raise RuntimeError("Insufficient currently free GPU memory for the bounded 4B technical check")
        policy = QwenPolicyAdapter.from_verified_checkpoint(
            root / "configs/model_manifest.json", "qwen35_4b", root / "data/models/qwen35_4b",
            decoding=DecodingConfig(max_new_tokens=48, temperature=0.6, top_p=0.95, top_k=20),
            enable_thinking=False, max_input_tokens=256,
        )
        generated = policy.generate_action([
            {"role": "system", "content": "This is a synthetic API check, not a materials experiment. Pick one allowed element and return only one JSON object with the key action."},
            {"role": "user", "content": 'Allowed elements: Li, Na, K. Return only {"action":"ELEMENT"} using one allowed element.'},
        ], seed=92317, legal_schema={"type": "object", "required": ["action"], "properties": {
            "action": {"type": "string", "enum": ["Li", "Na", "K"]}}})
        report.update(model_id=policy.model_id, checkpoint_hash=policy.checkpoint_hash,
                      policy_configuration_fingerprint=policy.configuration_fingerprint,
                      generated=generated.to_record())
        if not generated.success:
            raise RuntimeError(f"Real policy failed the technical action contract: {generated.failure_code}")
        cpu_ids = generated.input_ids_with_completion
        targets = build_action_targets(cpu_ids, prompt_token_count=generated.prompt_token_count,
                                        tokenizer=policy.tokenizer, benchmark="crystalgym",
                                        parsed_action=generated.parsed_action)
        report["action_targets"] = targets.to_dict()
        ids, kwargs = policy.prepare_trace_inputs(cpu_ids, stamp=generated.model_stamp)
        kwargs["logits_to_keep"] = torch.tensor(targets.prediction_positions, device=ids.device)
        with torch.no_grad():
            reference = policy.model(input_ids=ids, **kwargs).logits.detach()
            reference_score = float(targets.mean_logprob(reference))
            prefix_differences = []
            prefix_score_differences = []
            prefix_parity = True
            for index, position in enumerate(targets.token_positions):
                prefix_logits = policy.model(input_ids=ids[:, :position], attention_mask=kwargs["attention_mask"][:, :position],
                                              use_cache=False, logits_to_keep=1).logits[:, -1]
                prefix_differences.append(float((prefix_logits.float() - reference[:, index].float()).abs().max()))
                full_lp = reference[:, index].float().log_softmax(-1)[0, targets.token_ids[index]]
                prefix_lp = prefix_logits.float().log_softmax(-1)[0, targets.token_ids[index]]
                prefix_score_differences.append(float((full_lp - prefix_lp).abs()))
                prefix_parity &= torch.allclose(prefix_logits.float(), reference[:, index].float(), rtol=0.02, atol=0.05)
            suffix_ids = policy.tokenizer.encode(" unrelated future text", add_special_tokens=False)
            longer = torch.cat([cpu_ids, torch.tensor([suffix_ids], dtype=cpu_ids.dtype)], dim=1)
            longer_targets = build_action_targets(longer, prompt_token_count=generated.prompt_token_count,
                                                  tokenizer=policy.tokenizer, benchmark="crystalgym",
                                                  parsed_action=generated.parsed_action)
            long_ids, long_kwargs = policy.prepare_trace_inputs(longer, stamp=policy.model_stamp)
            long_kwargs["logits_to_keep"] = torch.tensor(longer_targets.prediction_positions, device=ids.device)
            later_score = float(longer_targets.mean_logprob(policy.model(input_ids=long_ids, **long_kwargs).logits))
        report.update(raw_mean_logprob=reference_score,
                      full_vs_individual_prefix_max_logit_errors=prefix_differences,
                      full_vs_individual_prefix_logprob_errors=prefix_score_differences,
                      prefix_parity_tolerance={"rtol": 0.02, "atol": 0.05, "reason": "Explicit BF16 runtime tolerance; exact CPU tests run separately"},
                      full_vs_individual_prefix_parity=bool(prefix_parity),
                      future_append_target_spec_unchanged=targets.target_spec_hash == longer_targets.target_spec_hash,
                      future_append_mean_logprob_difference=abs(later_score - reference_score))

        leaves = {}
        flags = [(parameter, parameter.requires_grad) for parameter in policy.model.parameters()]
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        def hook_for(name):
            def hook(module, arguments, output):
                leaf = output.detach().requires_grad_(True)
                leaves[name] = leaf
                return leaf
            return hook
        handles.append(policy.model.get_input_embeddings().register_forward_hook(hook_for("embedding")))
        for path in policy.mlp_paths:
            handles.append(policy.model.get_submodule(path).register_forward_hook(hook_for(path)))
        with torch.enable_grad():
            cut_logits = policy.model(input_ids=ids, **kwargs).logits
            score = targets.mean_logprob(cut_logits)
            names = ["embedding", *policy.mlp_paths]
            gradients = torch.autograd.grad(score, [leaves[name] for name in names], allow_unused=True)
        cut_parity = float((cut_logits.detach().float() - reference.float()).abs().max())
        future_max, past_norm, finite = 0.0, 0.0, True
        for gradient in gradients:
            if gradient is None:
                continue
            finite &= bool(torch.isfinite(gradient).all())
            past_norm += float(gradient[:, :targets.causal_horizon + 1].float().square().sum())
            future = gradient[:, targets.causal_horizon + 1:]
            if future.numel():
                future_max = max(future_max, float(future.float().abs().max()))
        report.update(cut_forward_max_abs_error=cut_parity, finite_gradients=finite,
                      causal_source_gradient_squared_norm=past_norm,
                      future_source_gradient_max_abs=future_max, cut_layers=len(policy.mlp_paths))
        for handle in handles:
            handle.remove()
        handles = []
        # One direct final-MLP basis perturbation checks the real scalar's cut
        # derivative. No feature/transcoder reconstruction is invented here.
        last_name = policy.mlp_paths[-1]
        last_gradient = gradients[-1].detach().float()
        flat = int(last_gradient[:, :targets.causal_horizon + 1].abs().argmax())
        position, channel = divmod(flat, last_gradient.shape[-1])
        analytical = float(last_gradient[0, position, channel])
        baselines = {name: value.detach() for name, value in leaves.items()}
        epsilon = 0.125
        def cut_value(scale):
            temporary = []
            try:
                temporary.append(policy.model.get_input_embeddings().register_forward_hook(lambda module, inputs, output: baselines["embedding"]))
                for path in policy.mlp_paths:
                    def fixed(module, inputs, output, path=path):
                        if path != last_name:
                            return baselines[path]
                        result = baselines[path].clone()
                        result[0, position, channel] += scale
                        return result
                    temporary.append(policy.model.get_submodule(path).register_forward_hook(fixed))
                with torch.no_grad():
                    return float(targets.mean_logprob(policy.model(input_ids=ids, **kwargs).logits))
            finally:
                for handle in temporary:
                    handle.remove()
        numerical = (cut_value(epsilon) - cut_value(-epsilon)) / (2 * epsilon)
        fd_passed = abs(numerical - analytical) <= 0.002 + 0.25 * abs(analytical)
        report["last_mlp_basis_finite_difference"] = {
            "layer": last_name, "position": position, "channel": channel, "epsilon": epsilon,
            "analytical": analytical, "numerical": numerical,
            "absolute_error": abs(numerical - analytical), "atol": 0.002, "rtol": 0.25,
            "passed": fd_passed, "scope": "BF16 same-valued cut model only; not a transcoder fidelity/graph validation",
        }
        report["passed"] = bool(prefix_parity and cut_parity == 0 and finite and past_norm > 0
                                 and future_max <= 1e-7 and abs(later_score - reference_score) <= 0.02
                                 and targets.target_spec_hash == longer_targets.target_spec_hash and fd_passed)
    except Exception as exc:
        report.update(passed=False, error_type=type(exc).__name__, error=str(exc))
    finally:
        for handle in handles:
            handle.remove()
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        report["elapsed_seconds"] = time.monotonic() - started
        report["max_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        write_json_atomic(args.output, report)
    print(json.dumps({k: report.get(k) for k in ("classification", "passed", "physical_oracle_calls",
                     "cut_forward_max_abs_error", "future_source_gradient_max_abs", "future_append_mean_logprob_difference",
                     "elapsed_seconds", "max_gpu_allocated_bytes", "error_type", "error")}))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
