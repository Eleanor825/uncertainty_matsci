#!/usr/bin/env python3
"""Declare full budgets, executing MADE before deferred CrystalGym work."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--phase", choices=("made", "all"), default="made",
                        help="MADE is the current user priority; all explicitly enables the later benchmark")
    parser.add_argument("--seal", action="store_true", help="Only after all stage implementations pass their required checks")
    args = parser.parse_args()
    root = args.project.resolve()
    alignment_path = root / "configs/method_alignment_gate.json"
    if args.seal and alignment_path.exists():
        alignment = json.loads(alignment_path.read_text())
        if alignment.get("implementation_matches_requested_method") is not True:
            raise RuntimeError("Requested method alignment is incomplete; full baseline collection may continue but downstream method stages cannot be sealed")
        evidence = alignment.get("validation")
        if not isinstance(evidence, dict) or not Path(evidence.get("path", "")).is_file():
            raise RuntimeError("Sealing the requested method requires its actual validation evidence")
        proof_path = Path(evidence["path"])
        if hashlib.sha256(proof_path.read_bytes()).hexdigest() != evidence["sha256"]:
            raise RuntimeError("Method validation evidence changed")
        proof = json.loads(proof_path.read_text())
        if proof.get("passed") is not True or not proof.get("source_sha256"):
            raise RuntimeError("Method implementation validation did not pass")
        for relative, expected in proof["source_sha256"].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError("A method source/config changed after validation: " + relative)
    path = root / "configs/stage_queue.json"
    queue = json.loads(path.read_text())
    status_path = root / "logs/pipeline_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"stages": {}}
    original = queue["stages"] + queue.get("declared_next_stages", []) + queue.get("deferred_stages", [])
    if len({s["id"] for s in original}) != len(original):
        raise RuntimeError("Duplicate stages across runnable, pending and deferred declarations")
    # Existing collection stages retain their exact identities/commands.
    stages = [s for s in original if s["id"].startswith("collect-")]
    if len(stages) != 4:
        raise RuntimeError("Exactly four complete model/benchmark collection conditions are required")
    python = str(root / "environments/policy-py312/bin/python")
    def stage(identifier, script, *arguments):
        if not (root / "scripts" / script).is_file():
            raise RuntimeError("Missing executable stage: " + script)
        stages.append({"id": identifier, "requires_sealed": True, "command": [python, str(root / "scripts/run_locked_stage.py"),
            "--project", str(root), "--entry", script, "--", *map(str, arguments)]})
    risk_root = root / "experiments/risk_models"
    tc_root = root / "experiments/transcoders"
    es_root = root / "experiments/es_training"
    for model in ("qwen35_4b", "qwen35_9b"):
        for benchmark in ("made", "crystalgym"):
            for kind in ("transcoders", "graphs", "risk"):
                stage(f"{kind}-{model}-{benchmark}", "run_representation_condition.py",
                    "--stage", kind, "--model-key", model, "--benchmark", benchmark)
    for model in ("qwen35_4b", "qwen35_9b"):
        for benchmark in ("made", "crystalgym"):
            for property_name in ([None] if benchmark == "made" else ["bm", "density", "band_gap"]):
                for method in ("esopt", "esopt_graph_risk"):
                    for seed in range(1, 6):
                        arguments = ["--model-key", model, "--benchmark", benchmark, "--method", method,
                            "--seed", seed, "--output", es_root, "--risk-checkpoint", risk_root / model / benchmark /
                            ("graph_risk.pt" if method == "esopt_graph_risk" else "entropy_risk.pt")]
                        if property_name:
                            arguments += ["--property", property_name]
                        if method == "esopt_graph_risk":
                            arguments += ["--transcoder-manifest", tc_root / model / benchmark / "transcoder_manifest.json"]
                        stage(f"es-{model}-{benchmark}-{property_name or 'audc'}-{method}-seed{seed}", "train_es_condition.py", *arguments)
    for model in ("qwen35_4b", "qwen35_9b"):
        for benchmark in ("made", "crystalgym"):
            resume = [] if (model, benchmark) == ("qwen35_4b", "made") else ["--resume"]
            stage(f"final-{model}-{benchmark}", "run_final_evaluation.py", "--model-key", model,
                "--benchmark", benchmark, "--output", root / "experiments/final_evaluation",
                "--risk-root", risk_root, "--transcoder-root", tc_root, "--es-root", es_root, *resume)
    report = root / "experiments/reports"
    stage("report-made-phase", "report_made_phase.py", "--output", report / "made_phase")
    stage("report-complete-study", "report_full_study.py", "--output", report)
    stage("finalize-complete-study", "finalize_stage_queue.py", "--acceptance", report / "acceptance.json")
    if len(stages) != 103 or len({s["id"] for s in stages}) != 103:
        raise RuntimeError("Expected 102 original stages plus one complete MADE report")
    def benchmark_of(item):
        command = item["command"]
        return command[command.index("--benchmark") + 1] if "--benchmark" in command else None
    made = [s for s in stages if benchmark_of(s) == "made"]
    crystal = [s for s in stages if benchmark_of(s) == "crystalgym"]
    reporting = {s["id"]: s for s in stages if benchmark_of(s) is None}
    if len(made) != 30 or len(crystal) != 70:
        raise RuntimeError("Each benchmark must retain all collection, representation, ES and final stages")
    made += [reporting.pop("report-made-phase")]
    tail = [reporting.pop("report-complete-study"), reporting.pop("finalize-complete-study")]
    if reporting:
        raise RuntimeError("Unclassified reporting stage")
    stages = made + crystal + tail
    old = {s["id"]: s for s in original}
    new = {s["id"]: s for s in stages}
    for identifier in status["stages"]:
        if identifier not in old or new.get(identifier) != old[identifier]:
            raise RuntimeError("An already-started stage may not change")
    started_ids = set(status["stages"])
    if args.phase == "made" and any(s["id"] in started_ids for s in crystal + tail):
        raise RuntimeError("A started later-benchmark stage cannot be silently deferred")
    # Only the two complete MADE collection conditions can run while unsealed.
    # A sealed MADE phase does not enable CrystalGym or global finalization.
    active = made if args.phase == "made" else stages
    runnable = active if args.seal else made[:2]
    if not started_ids.issubset({s["id"] for s in runnable}):
        raise RuntimeError("Cannot remove an already-started stage from the runnable queue")
    queue.update(stages=runnable, declared_next_stages=[] if args.seal else active[2:],
        deferred_stages=crystal + tail if args.phase == "made" else [],
        sealed=args.seal, final_acceptance_verified=False,
        active_phase=args.phase, benchmark_order=["made", "crystalgym"],
        completion_scope="full_current_two_model_MADE_phase" if args.phase == "made" else "current_two_model_both_benchmarks",
        latest_3_to_4_model_extension_complete=False,
        counts={"collection_conditions": 4, "representation_stages": 12, "ES_conditions": 80,
                "final_conditions": 4, "final_jobs": 2160, "stages": 103,
                "MADE_phase_stages": 31, "MADE_final_jobs": 1800,
                "runnable_stages": len(runnable), "deferred_stages": len(crystal + tail) if args.phase == "made" else 0})
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(queue, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps({"declared_stages": 103, "runnable_stages": len(queue["stages"]),
        "active_phase": args.phase, "deferred_stages": len(queue["deferred_stages"]),
        "sealed": args.seal, "study_complete": False}))


if __name__ == "__main__":
    main()
