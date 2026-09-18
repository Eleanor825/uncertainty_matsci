#!/usr/bin/env python3
"""Freeze actual two-model numerical evidence; performs no model forward."""
import argparse
import importlib.util
import json
from pathlib import Path

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    assessor_path = root / "scripts/verify_fp32_policy_runtime.py"
    spec = importlib.util.spec_from_file_location("fp32_runtime_assessor", assessor_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gates = {}
    for key in ("qwen35_4b", "qwen35_9b"):
        source = root / "technical_not_main" / key / "fp32_cpu_vocab_eager_recompute_full_runtime.json"
        report = json.loads(source.read_text())
        assessed = module.assess_primary_runtime_gates(report)
        if report["model_key"] != key or report["physical_oracle_calls"] != 0 or not assessed["primary_gate_passed"]:
            raise RuntimeError("Both actual model runtimes must pass the original native gate")
        runtime = report["policy_runtime"]
        if (runtime["dtype"] != "torch.float32" or runtime["attention_implementation"] != "eager"
                or not runtime["cpu_embedding_and_lm_head"] or not runtime["attention_checkpointing"]["enabled"]
                or runtime["float32_matmul_precision"] != "highest" or runtime["cuda_matmul_allow_tf32"]
                or runtime["torch_cpu_threads"] != 2):
            raise RuntimeError("Evidence differs from the selected uniform FP32 runtime")
        gates[key] = {"source_report": str(source), "source_report_sha256": file_sha256(source),
            "source_original_overall_passed": report.get("passed"), "assessment": assessed,
            "checkpoint_hash": report["checkpoint_hash"], "configuration_fingerprint": report["configuration_fingerprint"],
            "policy_runtime": runtime, "max_gpu_allocated_bytes": report["max_gpu_allocated_bytes"],
            "physical_oracle_calls": 0, "learned_transcoder_graph_validated": False}
    payload = {"schema": "actual_fp32_runtime_gate_registry_v1", "models": gates,
        "assessment_source_sha256": file_sha256(assessor_path),
        "scope": "Actual policy numerical/capacity gate only; all trained transcoders still require their later fidelity/native graph gates"}
    payload["fingerprint"] = fingerprint(payload)
    write_json_atomic(root / "configs/policy_runtime_gates.json", payload)
    print(json.dumps({"runtime_gates_frozen": list(gates), "model_forwards": 0, "scientific_calls": 0}))


if __name__ == "__main__":
    main()
