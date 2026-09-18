#!/usr/bin/env python3
"""Consume one COMPLETE model/benchmark collection; no subsampling controls."""
import argparse
from dataclasses import fields
import json
from pathlib import Path

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.collection_provenance import load_collection_lineage, preflight_collection, read_json as read_collection_json
from matdiscovery.experiment_plan import collection_jobs
from matdiscovery.fit_risk import fit_risk_models
from matdiscovery.representation_training import (
    GraphStageConfig, TranscoderStageConfig, generate_graph_features,
    train_transcoders_from_collections,
)


def completed_collections(root, model_key, benchmark):
    expected = [j for j in collection_jobs(root) if j["model_key"] == model_key and j["benchmark"] == benchmark]
    base = root / "experiments/collection" / model_key / benchmark
    plan = read_collection_json(base / "plan.json")
    progress = read_collection_json(base / "progress.json")
    if (plan["jobs"] != expected or plan["fingerprint"] != fingerprint({"jobs": plan["jobs"], "inputs": plan["inputs"]})
            or plan["expected_jobs"] != len(expected) or progress.get("complete") is not True
            or progress["expected_jobs"] != len(expected) or progress["completed_jobs"] != len(expected)):
        raise RuntimeError("Representation training requires every declared collection job")
    lineage = load_collection_lineage(root, base, plan, expected)
    verified_jobs = preflight_collection(base, plan, lineage)
    if set(verified_jobs) != {job["job_id"] for job in expected}:
        raise RuntimeError("Representation training requires every unique completed collection job")
    manifests = []
    for job in expected:
        directory = base / job["job_id"]
        verified = verified_jobs[job["job_id"]]
        manifest = (directory / "collection_manifest.json").resolve()
        # All original raw files remain checksum-bound. Later graph files are
        # separate derived artifacts. Only explicitly registered source lineage
        # can accept an old plan; the immutable receipts are never rewritten.
        summaries = verified["episodes"]
        if len(summaries) != job["expected_counts"]["episodes"] or not all(s["complete"] for s in summaries):
            raise RuntimeError("An incomplete collection episode cannot enter fitting")
        if {s["episode_id"] for s in summaries} != set(job["episode_ids"]):
            raise RuntimeError("Collection episodes are duplicated or have wrong IDs")
        for summary in summaries:
            index = job["episode_ids"].index(summary["episode_id"])
            if any(summary.get(k) != job[k] for k in ("benchmark", "model_key", "method", "task_id", "seed")) or summary["environment_seed"] != job["environment_seeds"][index]:
                raise RuntimeError("Collection episode belongs to another task, method, or seed")
        for key in ("candidate_oracle_attempts", "dft_episode_attempts"):
            if sum(s["costs"].get(key, 0) for s in summaries) != job["expected_counts"][key]:
                raise RuntimeError("Collection physical budget is incomplete")
        manifests.append(manifest)
    if len(progress["collection_manifests"]) != len(manifests) or set(map(str, manifests)) != {str(Path(p).resolve()) for p in progress["collection_manifests"]}:
        raise RuntimeError("Progress omitted or duplicated a collection manifest")
    return manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model-key", choices=("qwen35_4b", "qwen35_9b"), required=True)
    parser.add_argument("--benchmark", choices=("made", "crystalgym"), required=True)
    parser.add_argument("--stage", choices=("transcoders", "graphs", "risk"), required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    manifests = completed_collections(root, args.model_key, args.benchmark)
    protocol = json.loads((root / "configs/main_protocol.json").read_text())
    outputs = root / "experiments"
    bank = outputs / "transcoders" / args.model_key / args.benchmark
    graph_report = outputs / "graph_reports" / args.model_key / args.benchmark / "report.json"
    risk_output = outputs / "risk_models" / args.model_key / args.benchmark
    import torch
    runtime = protocol["policy_runtime"]
    torch.set_num_threads(runtime["torch_cpu_threads"])
    if args.stage == "transcoders":
        tc = protocol["transcoder"]
        config = TranscoderStageConfig(feature_multiplier=tc["feature_expansion"], top_k=tc["top_k"],
            epochs=tc["epochs"], batch_size=tc["batch_size"], learning_rate=tc["learning_rate"],
            max_dev_fvu=tc["max_development_output_fvu"], device="cuda:0")
        torch.cuda.set_per_process_memory_fraction(.32)
        report = train_transcoders_from_collections(manifests, bank, model_key=args.model_key, config=config, resume=True)
        if report["status"] != "succeeded" or not report.get("ready_for_graphs"):
            raise RuntimeError("Full transcoder bank failed its declared development fidelity gate")
    elif args.stage == "graphs":
        torch.cuda.set_per_process_memory_fraction(runtime["cuda_memory_fraction"])
        allowed = {f.name for f in fields(GraphStageConfig)}
        options = {k: v for k, v in protocol["graph"].items() if k in allowed}
        options.update({k: runtime[k] for k in ("device", "dtype", "attn_implementation", "sdpa_backend", "cpu_embedding_and_lm_head", "attention_checkpointing")})
        options["transcoder_device"] = protocol["transcoder"]["attribution_device"]
        config = GraphStageConfig(**options)
        report = generate_graph_features(manifests, model_key=args.model_key,
            model_manifest=root / "configs/model_manifest.json", checkpoint_dir=root / "data/models" / args.model_key,
            transcoder_manifest=bank / "transcoder_manifest.json", report_path=graph_report, config=config, resume=True)
        if not report["complete"] or not report.get("successful_graphs", 0):
            raise RuntimeError("All decisions must be accounted and verified graphs must exist")
        # Failed graph mappings remain explicit missing rows under the declared
        # protocol. This never represents unavailable graphs as verified ones.
        write_json_atomic(graph_report.with_name("stage_gate.json"), {
            "model_key": args.model_key, "benchmark": args.benchmark,
            "all_decisions_accounted": True, "graph_status": report["status"],
            "successful_graphs": report["successful_graphs"], "unavailable_graphs": report["unavailable_graphs"],
            "missingness_policy": protocol["graph"]["missing_graphs"], "complete_graph_coverage": report["unavailable_graphs"] == 0,
            "report_sha256": file_sha256(graph_report),
            "transcoder_bank_fingerprint": report["transcoder_bank_fingerprint"],
            "graph_pipeline_fingerprint": report["graph_pipeline_fingerprint"],
            "expected_decisions": report["expected_decisions"], "completed_decisions": report["completed_decisions"],
            "graph_files": [{"path": str(path), "sha256": file_sha256(path)} for path in sorted({m.parent / "graph_features.jsonl" for m in manifests})],
            "collection_manifests": [{"path": str(m), "sha256": file_sha256(m)} for m in manifests]})
    else:
        gate = json.loads(graph_report.with_name("stage_gate.json").read_text())
        if not gate["all_decisions_accounted"] or not gate["successful_graphs"]:
            raise RuntimeError("Risk fitting requires the complete graph-accounting stage")
        if (gate["model_key"] != args.model_key or gate["benchmark"] != args.benchmark
                or gate["report_sha256"] != file_sha256(graph_report)
                or gate["expected_decisions"] != gate["completed_decisions"]
                or gate["collection_manifests"] != [{"path": str(m), "sha256": file_sha256(m)} for m in manifests]):
            raise RuntimeError("Graph-accounting gate is stale or belongs to another complete collection")
        for artifact in gate["graph_files"]:
            if file_sha256(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError("A graph feature file changed after the full graph stage")
        risk = protocol["risk_network"]
        options = {k: risk[k] for k in ("hidden_width", "epochs", "batch_size", "learning_rate", "weight_decay", "patience")}
        report = fit_risk_models(manifests, risk_output, model_key=args.model_key, hyperparameters={"seed": 1729, **options})
        if report["status"] != "succeeded":
            raise RuntimeError("One or more risk arms did not complete fitting/calibration")
        if args.benchmark in protocol.get("failure_control", {}).get("enabled_benchmarks", []):
            from matdiscovery.failure_fit import fit_failure_models
            type_report = fit_failure_models(manifests, risk_output,
                outputs / "failure_labels" / args.model_key / args.benchmark,
                model_key=args.model_key, hyperparameters={"seed": 1729, **options})
            if type_report["status"] != "succeeded":
                raise RuntimeError("One or more observed failure-type predictors failed fitting/calibration")
            report["failure_types"] = {"status": type_report["status"], "heads": type_report["heads"]}
    print(json.dumps({"stage": args.stage, "model_key": args.model_key, "benchmark": args.benchmark,
        "complete": True, "collection_jobs": len(manifests), "status": report["status"]}), flush=True)


if __name__ == "__main__":
    main()
