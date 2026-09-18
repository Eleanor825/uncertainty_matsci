#!/usr/bin/env python3
"""Run every declared train/dev case, with verified per-job resume and no sample cap."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from matdiscovery.accounting import fingerprint
from matdiscovery.collection_provenance import (
    build_collection_plan, prepare_collection_resume,
    verify_collection_completion,
)
from matdiscovery.experiment_plan import collection_jobs
from matdiscovery.policy import QwenPolicyAdapter, DecodingConfig
from matdiscovery.rollouts import DiscoveryRollout, dump_json, file_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--model-key", required=True)
    p.add_argument("--benchmark", choices=["made", "crystalgym"], required=True)
    p.add_argument("--frozen-source", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--lineage", type=Path, help="Operator-recorded source lineage under condition/provenance; never auto-generated")
    p.add_argument("--lineage-sha256", help="Exact registered SHA256 of --lineage")
    args = p.parse_args()
    if (args.lineage is None) != (args.lineage_sha256 is None):
        p.error("--lineage and --lineage-sha256 are required together")
    if args.lineage is not None:
        args.lineage = args.lineage.resolve(strict=True)
    root = args.project.resolve()
    jobs = [j for j in collection_jobs(root) if j["model_key"] == args.model_key and j["benchmark"] == args.benchmark]
    if not jobs:
        raise ValueError("Requested collection condition is outside the complete protocol")
    destination = root / "experiments/collection" / args.model_key / args.benchmark
    destination.mkdir(parents=True, exist_ok=True)
    plan_path = destination / "plan.json"
    plan = build_collection_plan(root, jobs)
    frozen_inputs = plan["inputs"]
    if not args.frozen_source:
        frozen = root / "frozen_sources" / ("collection-" + plan["fingerprint"][:20])
        if not frozen.exists():
            frozen.mkdir(parents=True)
            shutil.copytree(root / "src", frozen / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
            shutil.copytree(root / "configs", frozen / "configs")
            (frozen / "scripts").mkdir()
            shutil.copy2(Path(__file__), frozen / "scripts/run_collection.py")
            for directory in ["data", "vendor", "environments", "experiments", "logs"]:
                (frozen / directory).symlink_to(root / directory, target_is_directory=True)
            dump_json(frozen / "source_manifest.json", {"original_project": str(root), "plan_fingerprint": plan["fingerprint"], "inputs": frozen_inputs})
        for name, digest in frozen_inputs.items():
            if file_hash(frozen / name) != digest:
                raise RuntimeError("Frozen execution source differs from the declared inputs")
        print(json.dumps({"frozen_execution_root": str(frozen), "plan_fingerprint": plan["fingerprint"]}), flush=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(frozen / "src")
        command = [sys.executable, str(frozen / "scripts/run_collection.py"), "--project", str(frozen), "--model-key", args.model_key, "--benchmark", args.benchmark, "--frozen-source"]
        if args.lineage is not None:
            command += ["--lineage", str(args.lineage), "--lineage-sha256", args.lineage_sha256]
        os.execve(sys.executable, command, env)
    admission = prepare_collection_resume(root, destination, plan, jobs,
        lineage_path=args.lineage, expected_lineage_sha256=args.lineage_sha256)
    # Scan ALL existing jobs before loading a model or dispatching any new work.
    # A partial job anywhere in the matrix blocks execution until separately archived.
    plan, context, verified = admission["plan"], admission["context"], admission["verified"]
    if admission["publish_plan"]:
        dump_json(plan_path, plan)
    protocol = json.loads((root / "configs/main_protocol.json").read_text())
    runtime, decode = protocol["policy_runtime"], protocol["decoding"]
    torch.set_num_threads(runtime["torch_cpu_threads"])
    torch.cuda.set_per_process_memory_fraction(runtime["cuda_memory_fraction"])
    policy = QwenPolicyAdapter.from_verified_checkpoint(root / "configs/model_manifest.json", args.model_key, root / "data/models" / args.model_key,
        device=runtime["device"], dtype=runtime["dtype"], attn_implementation=runtime["attn_implementation"],
        cpu_embedding_and_lm_head=runtime["cpu_embedding_and_lm_head"], sdpa_backend=runtime["sdpa_backend"],
        attention_checkpointing=runtime["attention_checkpointing"],
        decoding=DecodingConfig(**{k: decode[k] for k in ("max_new_tokens", "temperature", "top_p", "top_k")}),
        enable_thinking=decode["enable_thinking"], max_input_tokens=decode["max_input_tokens"])
    gates = json.loads((root / "configs/policy_runtime_gates.json").read_text())
    if gates["fingerprint"] != fingerprint({k: v for k, v in gates.items() if k != "fingerprint"}):
        raise RuntimeError("Runtime gate registry was modified")
    gate = gates["models"][args.model_key]
    if (not gate["assessment"]["primary_gate_passed"] or file_hash(Path(gate["source_report"])) != gate["source_report_sha256"]
            or gate["policy_runtime"] != policy.runtime_precision_record() or gate["checkpoint_hash"] != policy.checkpoint_hash
            or gate["configuration_fingerprint"] != policy.configuration_fingerprint):
        raise RuntimeError("Collection must use the actual numerically validated FP32 checkpoint/runtime")
    policy_identity = destination / "policy_configuration.json"
    identity = {"configuration_fingerprint": policy.configuration_fingerprint, "checkpoint_hash": policy.checkpoint_hash,
                "policy_runtime": policy.runtime_precision_record(), "policy_configuration": policy._configuration()}
    if policy_identity.exists() and json.loads(policy_identity.read_text()) != identity:
        raise RuntimeError("Cannot resume collection with a different policy configuration")
    if not policy_identity.exists():
        dump_json(policy_identity, identity)
    completed = []
    job_sources = {}

    def save_progress():
        dump_json(destination / "progress.json", {"expected_jobs": len(jobs), "completed_jobs": len(completed),
            "complete": len(completed) == len(jobs), "collection_manifests": completed,
            "active_plan_fingerprint": plan["fingerprint"], "lineage": context.lineage, "job_sources": job_sources})

    for job in jobs:
        if any(file_hash(root / name) != digest for name, digest in frozen_inputs.items()):
            raise RuntimeError("A protocol, dependency wrapper, or evaluator input changed during collection")
        if context.lineage is not None and file_hash(Path(context.lineage["path"])) != context.lineage["sha256"]:
            raise RuntimeError("Registered collection lineage changed during execution")
        output = destination / job["job_id"]
        receipt = output / "completion.json"
        if receipt.exists():
            proof = verified.get(job["job_id"])
            if proof is None or proof["receipt"]["policy_configuration_fingerprint"] != policy.configuration_fingerprint:
                raise RuntimeError("Completed job was not verified before physical execution")
            completed.append(str(output / "collection_manifest.json"))
            job_sources[job["job_id"]] = proof["source"]
            continue
        if output.exists():
            raise RuntimeError(f"Interrupted collection requires reconciliation, not blind physical replay: {output}")
        print(json.dumps({"event": "job_started", "job_id": job["job_id"], "task_id": job["task_id"], "split": job["split"], "seed": job["seed"]}), flush=True)
        rollout = DiscoveryRollout(root, policy, risk_model=None, attributor=None)
        if job["method"] != "baseline" or rollout.risk_model is not None or rollout.attributor is not None:
            raise RuntimeError("Fixed baseline collection cannot use a fitted controller or attributor")
        summaries = rollout.run(job, output, collection=True)
        expected = job["expected_counts"]
        if len(summaries) != expected["episodes"] or any(not x["complete"] for x in summaries):
            raise RuntimeError("Collection returned an incomplete episode matrix")
        for key in ["candidate_oracle_attempts", "dft_episode_attempts"]:
            if sum(x["costs"].get(key, 0) for x in summaries) != expected[key]:
                raise RuntimeError(f"Collection did not consume its prescribed physical budget: {key}")
        artifacts = [{"path": str(path), "sha256": file_hash(path)} for path in sorted(output.rglob("*")) if path.is_file() and path != receipt and not path.name.endswith(".partial")]
        dump_json(receipt, {"complete": True, "job_fingerprint": fingerprint(job), "plan_fingerprint": plan["fingerprint"], "policy_configuration_fingerprint": policy.configuration_fingerprint, "artifacts": artifacts, "completed_at": time.time(),
            "collection_execution": {"method": "baseline", "risk_model": None, "attributor": None,
                "source_manifest": context.source_manifest, "lineage": context.lineage, "made_execution": protocol.get("made_execution")}})
        proof = verify_collection_completion(output, job, plan, context, policy_configuration_fingerprint=policy.configuration_fingerprint)
        completed.append(str(output / "collection_manifest.json"))
        job_sources[job["job_id"]] = proof["source"]
        save_progress()
        print(json.dumps({"event": "job_completed", "job_id": job["job_id"], "completed_jobs": len(completed), "expected_jobs": len(jobs)}), flush=True)
    save_progress()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
