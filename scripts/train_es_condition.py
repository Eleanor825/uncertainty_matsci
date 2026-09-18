#!/usr/bin/env python3
"""Run one complete protocol-locked ES condition; no pilot or generation cap.

The output argument is a root: artifacts are stored under
MODEL/BENCHMARK/PROPERTY_OR_audc/METHOD/seed_N. --plan-only performs no model
loading and no scientific actions. Interrupted physical outcomes require a
verified completed receipt; --resume never retries an unknown trajectory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from matdiscovery.accounting import write_json_atomic
from matdiscovery.es_training import ESTrainingDriver, TrainingContractError
from matdiscovery.policy import declared_runtime_options
from matdiscovery.training_jobs import (
    METHODS, MODELS, PROPERTIES, TrainingCondition, TrainingJobCallbacks,
    cpu_clean_reload_validator, load_frozen_controllers,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model-key", choices=MODELS, required=True)
    parser.add_argument("--benchmark", choices=("made", "crystalgym"), required=True)
    parser.add_argument("--property", dest="property_name", choices=PROPERTIES)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--risk-checkpoint", type=Path,
                        help="Graph arm: required graph_risk.pt; ES-only: optional entropy_risk.pt for confidence logging")
    parser.add_argument("--transcoder-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Parent of the model/benchmark/property/method/seed hierarchy")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--device", help="Must match main_protocol.policy_runtime when declared")
    parser.add_argument("--cuda-memory-fraction", type=float,
                        help="Optional explicit allocator ceiling; must match the locked runtime")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.benchmark == "crystalgym" and args.property_name is None:
        parser.error("--property is required for CrystalGym")
    if args.benchmark == "made" and args.property_name is not None:
        parser.error("MADE uses AUDC; do not set --property")
    if args.method == "esopt_graph_risk" and (args.risk_checkpoint is None or args.transcoder_manifest is None):
        parser.error("esopt_graph_risk requires --risk-checkpoint and --transcoder-manifest")
    if args.method == "esopt" and args.transcoder_manifest is not None:
        parser.error("ES-only does not use --transcoder-manifest")
    if args.cuda_memory_fraction is not None and not 0 < args.cuda_memory_fraction <= 1:
        parser.error("--cuda-memory-fraction must be in (0, 1]")
    condition = TrainingCondition(args.project, args.model_key, args.benchmark, args.method, args.seed, args.property_name)
    protocol = condition.protocol()
    runtime = declared_runtime_options(protocol, device=args.device, cuda_memory_fraction=args.cuda_memory_fraction)
    config = condition.config()
    train_cases, dev_cases = condition.cases()
    destination = condition.output_directory(args.output)
    sweeps = sum(i % config.dev_every == 0 or i == config.generations for i in range(1, config.generations + 1))
    train_jobs = config.generations * config.population * config.cases_per_generation
    dev_jobs = sweeps * len(dev_cases)
    plan = {"condition": {**asdict(condition), "project": str(condition.project)}, "es_config": asdict(config),
            "train_case_count": len(train_cases), "dev_case_count": len(dev_cases),
            "train_cases": [asdict(case) for case in train_cases], "dev_cases": [asdict(case) for case in dev_cases],
            "training_evaluations": train_jobs, "development_evaluations": dev_jobs,
            "expected_physical_attempts": 50 * (train_jobs + dev_jobs) if args.benchmark == "made" else train_jobs + 2 * dev_jobs,
            "output": str(destination), "no_test_data_used": True,
            "complete_generation_schedule_required": True, "device": runtime["device"],
            "cuda_memory_fraction": runtime["cuda_memory_fraction"], "torch_cpu_threads": runtime["torch_cpu_threads"],
            "policy_runtime_declared": runtime,
            "risk_checkpoint": str(args.risk_checkpoint.resolve()) if args.risk_checkpoint else None,
            "transcoder_manifest": str(args.transcoder_manifest.resolve()) if args.transcoder_manifest else None,
            "hyperparameters_are": "prespecified research plan; not claimed official optimum"}
    if args.plan_only:
        print(json.dumps({"status": "plan_only_no_physical_execution", **plan}, indent=2))
        return 0
    import torch
    from matdiscovery.policy import DecodingConfig, QwenPolicyAdapter
    from matdiscovery.rollouts import RolloutSettings
    torch.set_num_threads(runtime["torch_cpu_threads"])
    if torch.device(runtime["device"]).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(runtime["cuda_memory_fraction"], device=runtime["device"])
    decode = protocol["decoding"]
    policy = QwenPolicyAdapter.from_verified_checkpoint(condition.project / "configs/model_manifest.json", args.model_key,
        condition.project / "data/models" / args.model_key,
        **{k: runtime[k] for k in ("device", "dtype", "attn_implementation", "sdpa_backend", "cpu_embedding_and_lm_head", "attention_checkpointing")},
        decoding=DecodingConfig(**{k: decode[k] for k in ("max_new_tokens", "temperature", "top_p", "top_k")}),
        enable_thinking=decode["enable_thinking"], max_input_tokens=decode["max_input_tokens"])
    risk, attributor, auxiliary_files = load_frozen_controllers(policy, model_key=args.model_key, benchmark=args.benchmark, method=args.method,
        risk_checkpoint=args.risk_checkpoint, transcoder_manifest=args.transcoder_manifest,
        graph_config=protocol["graph"], device=runtime["device"], failure_control=protocol.get("failure_control"))
    settings = RolloutSettings(max_generation_retries=protocol["risk_network"]["maximum_candidate_generations_per_decision"],
        risk_threshold=protocol["risk_network"]["threshold"], history_results=protocol["memory"]["scientific_history_per_episode"],
        recent_tools=protocol["memory"]["recent_tool_responses"],
        activation_tokens_per_decision=protocol["collection"]["activation_token_rows_per_complete_decision_prefix"],
        failure_aware_control=args.benchmark in protocol.get("failure_control", {}).get("enabled_benchmarks", []))
    callbacks = TrainingJobCallbacks(condition, policy, destination, settings=settings,
                                    risk_model=risk, attributor=attributor, auxiliary_files=auxiliary_files)
    # Keep this descriptor adjacent to the run: the driver owns its directory and
    # requires it empty on first execution. It freezes all callback source hashes.
    descriptor = destination.parent / (destination.name + ".execution.json")
    execution = {"plan": plan, "callback_identity": callbacks.callback_identity,
                 "callback_fingerprint": callbacks.fingerprint, "policy_configuration_fingerprint": policy.configuration_fingerprint}
    if descriptor.exists() and json.loads(descriptor.read_text()) != execution:
        raise TrainingContractError("Execution descriptor changed; this is not a compatible resume")
    if not descriptor.exists():
        write_json_atomic(descriptor, execution)
    driver = ESTrainingDriver(policy, config, train_cases=train_cases, dev_cases=dev_cases,
        job_factory=callbacks.job_factory, evaluate=callbacks.evaluate, result_verifier=callbacks.result_verifier,
        recover_result=callbacks.recover_result, clean_reload_validator=cpu_clean_reload_validator,
        output_dir=destination, callback_fingerprint=callbacks.fingerprint)
    summary = driver.run(resume=args.resume)
    from matdiscovery.checkpoint_retention import retire_completed_es_checkpoints
    summary["checkpoint_retention"] = retire_completed_es_checkpoints(destination)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
