#!/usr/bin/env python3
"""Real-weight FP32 runtime check only; zero material-oracle calls.

Tests CPU vocabulary/GPU decoder placement, a 3072-token prompt, a forced
384-token cache stress run, and complete-prefix native MLP-cut VJP/FD. Random or
untrained transcoders are never used and no trained graph-fidelity pass is claimed.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time

import torch

from matdiscovery.accounting import file_sha256, write_json_atomic
from matdiscovery.action_targets import build_action_targets
from matdiscovery.policy import DecodingConfig, QwenPolicyAdapter


def assess_primary_runtime_gates(report):
    """Apply the original native epsilon=1e-3 gate; larger steps are diagnostics.

    A finite difference at a larger step includes real nonlinear curvature and
    is not the original local derivative gate. Preserve its failed row rather
    than changing the original tolerance or hiding the scan.
    """
    failures = []
    runtime = report.get("policy_runtime", {})
    if runtime.get("dtype") != "torch.float32" or runtime.get("all_parameters_materialized") is not True:
        failures.append("materialized_fp32_parameters")
    capacity = report.get("cache_stress", {})
    if any(capacity.get(key) != value for key, value in {"prompt_tokens": 3072, "new_tokens": 384, "score_tensors": 384}.items()):
        failures.append("full_3072_plus_384_cache_capacity")
    capture = report.get("capture", {})
    if capture.get("tokens") != 3456 or capture.get("layers") != 32 or capture.get("passed") is not True:
        failures.append("full_prefix_32_layer_capture")
    cut = report.get("same_full_prefix_cut_forward_passed")
    if cut is None:
        # An absolute maximum <= native atol proves the legacy receipt's
        # allclose gate without reconstructing missing full-vocabulary vectors.
        cut = report.get("native_cut_forward_max_abs", float("inf")) <= 1e-6
    if not cut:
        failures.append("same_prefix_native_cut_forward")
    if report.get("action_objective_forward_gates_passed") is not True:
        failures.append("actual_action_logprob_cache_and_length_parity")
    if report.get("future_source_gradient_max_abs") != 0:
        failures.append("causal_future_source_gradient")
    if report.get("parameters_unchanged") is not True:
        failures.append("policy_parameter_identity_and_values")
    expected = {"embedding", "model.language_model.layers.15.mlp", "model.language_model.layers.31.mlp"}
    original = [row for row in report.get("finite_differences", []) if row.get("epsilon") == .001]
    if len(original) != 3 or {r.get("source") for r in original} != expected:
        failures.append("original_native_fd_source_coverage")
    for row in original:
        if (not math.isfinite(row["analytical"]) or not math.isfinite(row["numerical"]) or abs(row["analytical"]) < 1e-3
                or abs(row["numerical"] - row["analytical"]) > 1e-4 + .05 * abs(row["analytical"])):
            failures.append("native_fd_1e-3:" + row["source"])
    return {"primary_gate_passed": not failures, "primary_gate_failures": failures,
            "original_native_fd_gate": {"epsilon": .001, "atol": 1e-4, "rtol": .05, "sources": sorted(expected)},
            "auxiliary_fd_scan_all_passed": all(row["passed"] for row in report.get("finite_differences", [])),
            "auxiliary_fd_failures": [row for row in report.get("finite_differences", []) if not row["passed"]],
            "auxiliary_full_vocab_absolute_gate_passed": report.get("auxiliary_full_vocab_absolute_gate_passed"),
            "scope": "Technical actual-runtime component acceptance only; original failed diagnostic receipts remain immutable; no material result or TC fidelity claim"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model-key", choices=("qwen35_4b", "qwen35_9b"), required=True)
    parser.add_argument("--attention", choices=("eager", "sdpa"), required=True)
    parser.add_argument("--sdpa-backend", choices=("auto", "math", "efficient"), default="auto")
    parser.add_argument("--attention-checkpointing", action="store_true")
    parser.add_argument("--cuda-memory-fraction", type=float, default=.44)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-cache-stress", action="store_true", help="Partial technical check; cannot earn full runtime acceptance")
    parser.add_argument("--cache-proof", type=Path, help="Reuse a hashed full-cache receipt only for identical production code, runtime, checkpoint and prompt IDs")
    args = parser.parse_args(argv)
    if args.skip_cache_stress and args.cache_proof:
        parser.error("Choose either an incomplete check or a verified reusable cache proof")
    root, output = args.project.resolve(), args.output.resolve()
    start = time.monotonic()
    report = {"classification": "technical_not_main_real_fp32_runtime", "model_key": args.model_key,
              "attention": args.attention, "sdpa_backend": args.sdpa_backend, "physical_oracle_calls": 0,
              "attention_checkpointing": args.attention_checkpointing,
              "transcoder_graph_fidelity_validated": False, "main_protocol_changed": False,
              "production_graph_gate_still_required": "Train real TCs and run NativeAttributor.validate_backend on this same runtime",
              "primary_gate_specification": {"same_full_prefix_cut_forward_atol": 1e-6, "same_full_prefix_cut_forward_rtol": 1e-5,
                  "native_fd_epsilon": .001, "native_fd_atol": 1e-4, "native_fd_rtol": .05,
                  "action_logprob_cross_length_atol": 1e-4, "action_logprob_cache_atol": 1e-4,
                  "future_source_gradient_abs": 0, "full_prompt_tokens": 3072, "full_generated_cache_tokens": 384},
              "auxiliary_nonblocking_diagnostic": "Cross-shape full-vocabulary max-logit absolute error <=2e-4; report every failure, but this is distinct from the unchanged native same-prefix and action-objective gates",
              "auxiliary_fd_scan_epsilons": [.001, .003, .01, .03],
              "source_code_sha256": {name: file_sha256(root / "src/matdiscovery" / name) for name in ("policy.py", "native_attribution.py", "action_targets.py")},
              "script_sha256": file_sha256(__file__), "passed": False}
    handles, flags = [], []
    policy = None
    def save(phase):
        report["phase"] = phase
        report["elapsed_seconds"] = time.monotonic() - start
        if torch.cuda.is_available():
            report["max_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["max_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json_atomic(output, report)
        print(json.dumps({key: report[key] for key in ("phase", "elapsed_seconds", "max_gpu_allocated_bytes") if key in report}), flush=True)
    try:
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        free, total = torch.cuda.mem_get_info()
        report.update(gpu_free_bytes_before=free, gpu_total_bytes=total, cuda_memory_fraction=args.cuda_memory_fraction)
        # This is an instantaneous check, never a reservation or permission to
        # evict a background process. All OOM failures are retained as failures.
        minimum = (22 if args.model_key == "qwen35_4b" else 36) * 1024**3
        if free < minimum:
            raise RuntimeError(f"Insufficient current free GPU memory: {free} < {minimum}")
        save("loading_verified_fp32_policy")
        policy = QwenPolicyAdapter.from_verified_checkpoint(root / "configs/model_manifest.json", args.model_key,
            root / "data/models" / args.model_key, device="cuda:0", dtype="float32",
            attn_implementation=args.attention, sdpa_backend=args.sdpa_backend,
            attention_checkpointing=args.attention_checkpointing,
            cpu_embedding_and_lm_head=True,
            decoding=DecodingConfig(max_new_tokens=384, temperature=.6, top_p=.95, top_k=20),
            enable_thinking=False, max_input_tokens=3072)
        report.update(policy_runtime=policy.runtime_precision_record(), checkpoint_hash=policy.checkpoint_hash,
                      configuration_fingerprint=policy.configuration_fingerprint, parameter_report=policy.parameter_report())
        before_versions = policy._parameter_versions()
        if not report["policy_runtime"]["all_parameters_materialized"] or report["policy_runtime"]["dtype"] != "torch.float32":
            raise RuntimeError("Runtime is not an entirely materialized FP32 policy")
        if policy.model.get_input_embeddings().weight.device.type != "cpu" or policy.model.get_output_embeddings().weight.device.type != "cpu":
            raise RuntimeError("Vocabulary parameters are not genuinely resident on CPU")
        if policy._input_device() != torch.device("cuda:0"):
            raise RuntimeError("Generation input IDs must remain on the decoder GPU")
        save("fp32_policy_loaded")
        system = "This is a synthetic API and numerical check, not a materials experiment. Return exactly one JSON object with action Li, Na, or K. Ignore padding."
        def messages(repeats):
            return [{"role": "system", "content": system}, {"role": "user", "content": "Padding:" + " x" * repeats + '\nPick one allowed element. Return {"action":"ELEMENT"} only.'}]
        def count(repeats):
            rendered = policy.tokenizer.apply_chat_template(messages(repeats), tokenize=True, add_generation_prompt=True,
                        enable_thinking=False, return_tensors="pt", return_dict=True, truncation=False)
            return rendered["input_ids"].shape[1]
        low, high = 0, 3072
        while low < high:
            middle = (low + high + 1) // 2
            if count(middle) <= 3072:
                low = middle
            else:
                high = middle - 1
        if count(low) != 3072:
            raise RuntimeError("Unable to form an exact untruncated 3072-token technical prompt")
        raw_generation_logits = []
        def observe_raw_generation(module, inputs, value):
            raw_generation_logits.append(value[:, -1].detach().float().cpu().clone())
            return None
        raw_hook = policy.model.lm_head.register_forward_hook(observe_raw_generation)
        try:
            generated = policy.generate_action(messages(low), seed=92317,
                legal_schema={"type": "object", "required": ["action"], "properties": {"action": {"type": "string", "enum": ["Li", "Na", "K"]}}})
        finally:
            raw_hook.remove()
        report["generation"] = generated.to_record()
        if not generated.success or generated.prompt_token_count != 3072:
            raise RuntimeError("Real FP32 generation did not satisfy the exact long-prompt action contract: " + str(generated.error))
        report["raw_cache_logit_capture_aligned"] = len(raw_generation_logits) == generated.completion_count
        save("long_prompt_action_generated")
        cache_capacity_verified = False
        if args.cache_proof:
            prior = json.loads(args.cache_proof.read_text())
            prior_ids = prior.get("generation", {}).get("input_ids_with_completion", [[]])[0][:3072]
            current_ids = generated.input_ids_with_completion[0, :3072].tolist()
            if (prior.get("source_code_sha256") != report["source_code_sha256"] or prior.get("policy_runtime") != report["policy_runtime"]
                    or prior.get("checkpoint_hash") != policy.checkpoint_hash or prior.get("configuration_fingerprint") != policy.configuration_fingerprint
                    or prior_ids != current_ids or prior.get("cache_stress", {}).get("new_tokens") != 384
                    or prior["cache_stress"].get("prompt_tokens") != 3072 or prior["cache_stress"].get("score_tensors") != 384):
                raise RuntimeError("Reusable cache proof differs from the exact production runtime/code/input contract")
            report["cache_stress"] = prior["cache_stress"]
            report["cache_proof_reuse"] = {"path": str(args.cache_proof.resolve()), "sha256": file_sha256(args.cache_proof),
                "same_production_code_runtime_checkpoint_and_prompt_verified": True}
            cache_capacity_verified = True
            save("full_cache_capacity_reused_from_verified_receipt")
        elif not args.skip_cache_stress:
            ids, mask = policy._render(messages(low))
            options = policy.decoding.generation_kwargs()
            options["min_new_tokens"] = 384
            with policy._observation(), torch.random.fork_rng(devices=[0]), torch.no_grad():
                torch.manual_seed(92317)
                stressed = policy.model.generate(input_ids=ids, attention_mask=mask, **options)
            report["cache_stress"] = {"prompt_tokens": ids.shape[1], "new_tokens": stressed.sequences.shape[1] - ids.shape[1],
                "score_tensors": len(stressed.scores), "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "technical_override": "min_new_tokens=384 solely to test full cache and score-storage capacity; not a main action trajectory"}
            if report["cache_stress"]["new_tokens"] != 384 or len(stressed.scores) != 384:
                raise RuntimeError("The full 3072+384 cache stress budget was not exercised")
            cache_capacity_verified = True
            del stressed, ids, mask
            gc.collect()
            torch.cuda.empty_cache()
            save("full_3072_plus_384_cache_verified")
        # Keep the genuine generated action and its whole prefix. Append neutral
        # future tokens solely to exercise the maximum full-forward length and
        # verify that later source gradients are exactly zero.
        original = generated.input_ids_with_completion
        fill = policy.tokenizer.encode(" neutral", add_special_tokens=False)
        remaining = 3456 - original.shape[1]
        if remaining < 0 or not fill:
            raise RuntimeError("Generated prefix exceeds the declared full-context capacity")
        suffix = (fill * (remaining // len(fill) + 1))[:remaining]
        probe = torch.cat([original, torch.tensor([suffix], dtype=torch.long)], dim=1)
        targets = build_action_targets(probe, prompt_token_count=generated.prompt_token_count, tokenizer=policy.tokenizer,
                                       benchmark="crystalgym", parsed_action=generated.parsed_action)
        report["action_targets"] = targets.to_dict()
        captured = policy.capture_prefix(probe, stamp=policy.model_stamp)
        report["capture"] = {"tokens": probe.shape[1], "layers": len(captured.mlp_inputs),
                             "parity_max_abs": captured.parity_max_abs, "passed": captured.parity_passed}
        del captured
        ids, kwargs = policy.prepare_trace_inputs(probe, stamp=policy.model_stamp)
        kwargs["logits_to_keep"] = torch.tensor(targets.prediction_positions, device=ids.device)
        with torch.no_grad():
            reference = policy.model(input_ids=ids, **kwargs).logits.detach()
            score_reference = float(targets.mean_logprob(reference))
            prefix_errors = []
            for index, position in enumerate(targets.token_positions):
                single = policy.model(input_ids=ids[:, :position], attention_mask=kwargs["attention_mask"][:, :position],
                                      use_cache=False, logits_to_keep=1).logits[:, -1]
                score = single.float().log_softmax(-1)[0, targets.token_ids[index]]
                whole = reference[:, index].float().log_softmax(-1)[0, targets.token_ids[index]]
                prefix_errors.append({"max_logit_error": float((single - reference[:, index]).abs().max()),
                                      "target_logprob_error": float((score - whole).abs()),
                                      "relative_logit_l2_error": float((single - reference[:, index]).norm() / reference[:, index].norm()),
                                      "native_tolerance_allclose": bool(torch.allclose(single, reference[:, index], atol=1e-6, rtol=1e-5)),
                                      "auxiliary_absolute_gate_passed": float((single - reference[:, index]).abs().max()) <= 2e-4})
            cached_errors = []
            try:
                if not report["raw_cache_logit_capture_aligned"]:
                    raise RuntimeError("Raw-logit capture did not align with every actual generation step")
                for index, position in enumerate(targets.token_positions):
                    step = position - generated.prompt_token_count
                    cached_score = raw_generation_logits[step].log_softmax(-1)[0, targets.token_ids[index]]
                    full_score = reference[:, index].float().log_softmax(-1)[0, targets.token_ids[index]].cpu()
                    cached_errors.append({"prediction_position": position - 1, "generation_step": step,
                                          "action_logprob_error": float((cached_score - full_score).abs()),
                                          "source": "read-only raw lm_head output from the actual native generation cache path"})
            except Exception as exc:
                report["cache_comparison_error"] = {"type": type(exc).__name__, "message": str(exc)}
            del raw_generation_logits
        report["full_vs_individual_prefix"] = prefix_errors
        report["cache_vs_full_prefix"] = cached_errors
        report["auxiliary_full_vocab_absolute_gate_passed"] = all(row["auxiliary_absolute_gate_passed"] for row in prefix_errors)
        report["action_objective_forward_gates_passed"] = (all(row["target_logprob_error"] <= 1e-4 for row in prefix_errors)
            and len(cached_errors) == len(targets.token_ids) and all(row["action_logprob_error"] <= 1e-4 for row in cached_errors))
        report["raw_action_mean_logprob"] = score_reference
        save("full_prefix_capture_and_causal_forward_verified")
        leaves = {}
        flags = [(parameter, parameter.requires_grad) for parameter in policy.model.parameters()]
        for parameter, _ in flags:
            parameter.requires_grad_(False)
        names = ["embedding", *policy.mlp_paths]
        modules = {"embedding": policy.model.get_input_embeddings(), **{path: policy.model.get_submodule(path) for path in policy.mlp_paths}}
        def leaf_hook(name):
            def hook(module, inputs, value):
                leaf = value.detach().requires_grad_(True)
                leaves[name] = leaf
                return leaf
            return hook
        for name in names:
            handles.append(modules[name].register_forward_hook(leaf_hook(name)))
        with torch.enable_grad():
            cut_logits = policy.model(input_ids=ids, **kwargs).logits
            scalar = targets.mean_logprob(cut_logits)
            gradients = torch.autograd.grad(scalar, [leaves[name] for name in names], allow_unused=True)
        parity = float((cut_logits.detach() - reference).abs().max())
        report["native_cut_forward_max_abs"] = parity
        report["same_full_prefix_cut_forward_passed"] = bool(torch.allclose(cut_logits.detach(), reference, atol=1e-6, rtol=1e-5))
        if not report["same_full_prefix_cut_forward_passed"]:
            raise RuntimeError("Same-value native MLP cuts changed the actual FP32 forward")
        norms, future = {}, 0.
        for name, gradient in zip(names, gradients, strict=True):
            if gradient is None or not torch.isfinite(gradient).all():
                raise RuntimeError("A native complete-prefix source has absent/nonfinite gradients: " + name)
            norms[name] = float(gradient[:, :targets.causal_horizon + 1].norm())
            later = gradient[:, targets.causal_horizon + 1:]
            if later.numel():
                future = max(future, float(later.abs().max()))
        report.update(native_source_gradient_norms=norms, future_source_gradient_max_abs=future)
        if future != 0 or norms["embedding"] <= 0 or norms[names[-1]] <= 0:
            raise RuntimeError("Causal source-gradient contract failed")
        for handle in handles:
            handle.remove()
        handles.clear()
        baselines = {name: value.detach() for name, value in leaves.items()}
        gradient_by_name = dict(zip(names, gradients, strict=True))
        del leaves, gradients, cut_logits, scalar, reference
        gc.collect()
        torch.cuda.empty_cache()
        save("native_full_prefix_vjp_verified")
        checks = []
        for source in ("embedding", policy.mlp_paths[15], policy.mlp_paths[-1]):
            gradient = gradient_by_name[source]
            direction = gradient.detach().clone()
            direction[:, targets.causal_horizon + 1:] = 0
            norm = direction.norm()
            direction /= norm
            analytical = float((gradient * direction).sum())
            if analytical < 1e-3:
                raise RuntimeError("Finite-difference probe has insufficient derivative signal")
            def score_at(scale):
                temporary = []
                try:
                    for name in names:
                        def fixed(module, inputs, value, name=name):
                            return baselines[name] + scale * direction if name == source else baselines[name]
                        temporary.append(modules[name].register_forward_hook(fixed))
                    with torch.no_grad():
                        return float(targets.mean_logprob(policy.model(input_ids=ids, **kwargs).logits))
                finally:
                    for handle in temporary:
                        handle.remove()
            for epsilon in (.001, .003, .01, .03):
                positive, negative = score_at(epsilon), score_at(-epsilon)
                numerical = (positive - negative) / (2 * epsilon)
                absolute = abs(numerical - analytical)
                checks.append({"source": source, "epsilon": epsilon, "analytical": analytical, "numerical": numerical,
                               "absolute_error": absolute, "relative_error": absolute / abs(analytical),
                               "atol": 1e-4, "rtol": .05, "passed": absolute <= 1e-4 + .05 * abs(analytical)})
                report["finite_differences"] = checks
                save("finite_difference_" + source + "_" + str(epsilon))
            del direction
        report["parameters_unchanged"] = before_versions == policy._parameter_versions()
        if policy._attention_checkpoint_handle:
            report["attention_recompute_counters"] = dict(policy._attention_checkpoint_handle.counters)
        report["primary_assessment"] = assess_primary_runtime_gates(report)
        report["passed"] = bool(cache_capacity_verified and report["primary_assessment"]["primary_gate_passed"])
        report["validated_scope"] = "Real FP32 materialized policy, CPU vocabulary/GPU decoder, exact 3072+384 capacity, 32 native cuts and VJP/FD; no learned-TC graph claim"
    except BaseException as exc:
        report.update(passed=False, error_type=type(exc).__name__, error=str(exc))
    finally:
        for handle in handles:
            handle.remove()
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        save("complete" if report.get("passed") else "failed")
    print(json.dumps({key: report.get(key) for key in ("passed", "model_key", "attention", "sdpa_backend", "elapsed_seconds", "max_gpu_allocated_bytes", "error_type", "error")}))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
