"""Core corpus adapter to the unchanged full-layer/native-graph/NN algorithms.

This module changes only registered corpus admission and output namespace. All
32 transcoders, all 16 epochs, original fidelity gates, every captured decision,
and existing fitted risk/type heads remain mandatory. No scientific calls occur
in any representation stage. Passed receipts are revalidated before reuse.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from .accounting import file_sha256, fingerprint, write_json_atomic
from .collection_provenance import read_json
from .core_collection import (completed_core_collections, publish_stage_receipt,
    stage_receipt_path, verify_stage_receipt)
from .core_protocol import artifact, corpus_contract, read_core, require, verify_artifact
from .representation_training import (GraphStageConfig, TranscoderStageConfig,
    generate_graph_features, read_collections, train_transcoders_from_collections,
    validate_transcoder_bank, _load_resume)


def paths_for(core):
    root = Path(core["workspace"]) / "experiments"
    bank = root / "transcoders/qwen35_4b/made"
    from .core_protocol import CAUSAL_GRAPH_REGISTRATIONS
    if core.get("registration") in CAUSAL_GRAPH_REGISTRATIONS:
        reference = core["passed_bank_reuse_contract"]; verify_artifact(reference)
        bank = Path(read_json(reference["path"])["source_bank_manifest"]["path"]).parent
    return {"bank": bank,
            "graph": root / "graph_reports/qwen35_4b/made/report.json",
            "risk": root / "risk_models/qwen35_4b/made",
            "labels": root / "failure_labels/qwen35_4b/made"}


def transcoder_config(core):
    tc = core["transcoder"]
    return TranscoderStageConfig(feature_multiplier=tc["feature_expansion"], top_k=tc["top_k"],
        epochs=tc["epochs"], batch_size=tc["batch_size"], learning_rate=tc["learning_rate"],
        max_dev_fvu=tc["max_development_output_fvu"], device="cuda:0")


def graph_config(core):
    allowed = {field.name for field in fields(GraphStageConfig)}
    options = {k: v for k, v in core["graph"].items() if k in allowed}
    options.update({key: core["policy_runtime"][key] for key in
        ("device", "dtype", "attn_implementation", "sdpa_backend", "cpu_embedding_and_lm_head", "attention_checkpointing", "torch_cpu_threads")})
    options["transcoder_device"] = core["transcoder"]["attribution_device"]
    return GraphStageConfig(**options)


def _bank(core, manifests, paths):
    from .core_protocol import CAUSAL_GRAPH_REGISTRATIONS
    if core.get("registration") in CAUSAL_GRAPH_REGISTRATIONS:
        from .graph_recovery import reused_bank_for_core
        bank, collection, proof = reused_bank_for_core(core, manifests)
        require(Path(paths["bank"]) / "transcoder_manifest.json" == Path(proof["source_bank_manifest"]["path"]),
                "Causal graph core points to another source bank")
        return bank, collection
    collection = read_collections(manifests, model_key=core["model_key"])
    bank = validate_transcoder_bank(paths["bank"] / "transcoder_manifest.json", model_key=core["model_key"],
        checkpoint_hash=collection["checkpoint_hash"], policy_runtime=collection["policy_runtime"],
        policy_configuration_fingerprint=collection["policy_configuration_fingerprint"])
    require(bank["collections"] == collection["collections"] and bank["collection_fingerprint"] == collection["collection_fingerprint"],
            "Core bank was fitted from a different collection corpus")
    from dataclasses import asdict
    require(bank["configuration"] == asdict(transcoder_config(core)), "Core TC settings changed")
    from .core_protocol import registered_transcoder_recipe
    if registered_transcoder_recipe(core) == "train_centered_scalar_rms_fp32_export_v1":
        from .parallel_normalized_bank import verify_bank_recipe
        verify_bank_recipe(bank, core_fingerprint=core["fingerprint"])
    return bank, collection


def _graphs(core, manifests, paths):
    bank, collection = _bank(core, manifests, paths)
    report = read_json(paths["graph"])
    provenance = read_json(paths["graph"].with_suffix(".provenance.json"))
    pipeline = provenance["pipeline"]
    from dataclasses import asdict
    require(pipeline["config"] == asdict(graph_config(core)) and pipeline["transcoder_bank_fingerprint"] == bank["bank_fingerprint"]
            and provenance["graph_pipeline_fingerprint"] == fingerprint(pipeline)
            and provenance["collections"] == collection["collections"], "Core graph pipeline/source changed")
    from .core_protocol import CAUSAL_GRAPH_REGISTRATIONS
    if core.get("registration") in CAUSAL_GRAPH_REGISTRATIONS:
        reference = core["passed_bank_reuse_contract"]
        contract = read_json(reference["path"])
        proof = pipeline.get("passed_bank_reuse", {})
        require(proof.get("schema") == "causal_core_passed_bank_adoption_v1" and proof.get("complete") is True
            and proof.get("contract") == reference
            and proof.get("source_bank_manifest") == contract["source_bank_manifest"]
            and proof.get("source_core_fingerprint") == contract["source_core_fp"]
            and proof.get("target_core_fingerprint") == core["fingerprint"]
            and proof.get("target_source_selection_rule") == core["graph"]["source_selection_rule"]
            and proof.get("source_evidence_files") == contract["source_artifacts"]
            and proof.get("new_transcoder_training_calls") == proof.get("new_policy_or_material_oracle_calls") == 0,
            "Core graph omitted or changed its explicit original-bank adoption proof")
    for name, digest in pipeline["source_code_sha256"].items():
        require(file_sha256(Path(__file__).with_name(name)) == digest, "Graph source code changed after extraction")
    completed = _load_resume(collection["records"], bank["bank_fingerprint"], provenance["graph_pipeline_fingerprint"], True)
    successful = sum(row["graph_status"] == "succeeded" for row in completed.values())
    require(report["complete"] is True and len(completed) == len(collection["records"])
            and report["expected_decisions"] == report["completed_decisions"] == len(completed)
            and report["successful_graphs"] == successful and successful > 0
            and report["transcoder_bank_fingerprint"] == bank["bank_fingerprint"]
            and report["graph_pipeline_fingerprint"] == provenance["graph_pipeline_fingerprint"], "Core graph stage is incomplete or has no verified graph")
    return report


def _primary(core, manifests, output):
    from .fit_risk import RISK_METHODS, load_risk_dataset
    from .uncertainty import CalibratedRiskModel
    dataset = load_risk_dataset(manifests, model_key=core["model_key"])
    report = read_json(output / "fit_report.json")
    require(report["status"] == "succeeded" and report["model_key"] == core["model_key"]
            and report["dataset_fingerprint"] == dataset.provenance["dataset_fingerprint"], "Incomplete/different core risk fit")
    models = {}
    for method in RISK_METHODS:
        fit = read_json(output / (method + ".fit.json")); checkpoint = output / (method + ".pt")
        require(fit["status"] == "succeeded" and fit["checkpoint_sha256"] == file_sha256(checkpoint)
                and report["methods"][method] == fit, "Core primary fit receipt/checkpoint differs")
        model = CalibratedRiskModel.load(checkpoint)
        require(model.provenance["collection_provenance"] == dataset.provenance, "Core predictor trained from another corpus")
        models[method] = model
    return dataset, models


def _typed(core, manifests, output, contract):
    from .failure_labels import load_failure_labels
    from .failure_risk import FailureAwareRiskModel
    dataset, models = _primary(core, manifests, output)
    inventory = read_json(output / "failure_label_inventory.json")
    require(inventory.get("corpus_contract") == contract and inventory["expected_collection_jobs"] == len(manifests)
            and inventory["fingerprint"] == fingerprint({k: v for k, v in inventory.items() if k != "fingerprint"}), "Core typed corpus inventory differs")
    report = read_json(output / "failure_fit_report.json")
    require(report["status"] == "succeeded", "Core type prediction fit failed its original control readiness gate")
    require(len(inventory["sources"]) == len(manifests)
            and {x["source_manifest"]["path"] for x in inventory["sources"]} == {str(p) for p in manifests}, "Typed labels omitted a core collection")
    for item in inventory["sources"]:
        verify_artifact(item)
        load_failure_labels(item["path"], manifest_path=item["source_manifest"]["path"])
    for method, primary in models.items():
        fit = read_json(output / (method + ".failure_heads.fit.json"))
        require(fit["status"] == "succeeded" and fit == report["methods"][method], "Incomplete typed fit receipt")
        model = FailureAwareRiskModel.load(output / (method + ".failure_heads.pt"), primary=primary,
            primary_checkpoint=output / (method + ".pt"), expected_sha256=fit["checkpoint_sha256"],
            expected_primary_sha256=file_sha256(output / (method + ".pt")), expected_policy_runtime=dataset.provenance["policy_runtime"],
            expected_provenance=dataset.provenance, require_control_ready=True)
        require(model.failure_provenance["label_inventory"] == inventory, "Typed checkpoint/inventory disagree")
    return report


def run_stage(core, stage):
    require(stage in {"transcoders", "graphs", "risk"}, "Unknown core representation stage")
    verify_stage_receipt(core, "collect")
    manifests = completed_core_collections(core)
    contract = corpus_contract(core, manifests); paths = paths_for(core)
    previous = stage_receipt_path(core, stage)
    if previous.exists():
        receipt = verify_stage_receipt(core, stage)
        require(receipt["corpus_contract"] == contract, "Representation receipt uses another corpus")
        if stage == "transcoders":
            _bank(core, manifests, paths)
        elif stage == "graphs":
            _graphs(core, manifests, paths)
        else:
            _typed(core, manifests, paths["risk"], contract)
        return receipt
    import torch
    torch.set_num_threads(core["policy_runtime"]["torch_cpu_threads"])
    if stage == "transcoders":
        from .core_protocol import CAUSAL_GRAPH_REGISTRATIONS
        if core.get("registration") in CAUSAL_GRAPH_REGISTRATIONS:
            from .graph_recovery import reused_bank_for_core
            bank, collection, proof = reused_bank_for_core(core, manifests)
            evidence_path = Path(core["workspace"]) / "experiments/bank_reuse_verification.json"
            require(not evidence_path.exists(), "Partial bank adoption needs reconciliation; do not overwrite")
            write_json_atomic(evidence_path, proof)
            return publish_stage_receipt(core, stage, [evidence_path, Path(core["passed_bank_reuse_contract"]["path"]),
                paths["bank"] / "transcoder_manifest.json"], corpus_contract=contract, status="succeeded",
                reused_complete_bank=True, source_core_fingerprint=proof["source_core_fingerprint"],
                new_transcoder_training_calls=0, new_physical_calls=0)
        else:
            torch.cuda.set_per_process_memory_fraction(.32)
            report = train_transcoders_from_collections(manifests, paths["bank"], model_key=core["model_key"],
                config=transcoder_config(core), resume=True)
            require(report["status"] == "succeeded" and report.get("ready_for_graphs"), "Full TC bank did not pass original fidelity gate")
            _bank(core, manifests, paths)
            files = [p for p in paths["bank"].rglob("*") if p.is_file()]
    elif stage == "graphs":
        verify_stage_receipt(core, "transcoders")
        torch.cuda.set_per_process_memory_fraction(core["policy_runtime"]["cuda_memory_fraction"])
        report = generate_graph_features(manifests, model_key=core["model_key"],
            model_manifest=Path(core["workspace"]) / "configs/model_manifest.json",
            checkpoint_dir=Path(core["workspace"]) / "data/models" / core["model_key"],
            transcoder_manifest=paths["bank"] / "transcoder_manifest.json", report_path=paths["graph"],
            config=graph_config(core), resume=True, corpus_contract=contract)
        _graphs(core, manifests, paths)
        files = [paths["graph"], paths["graph"].with_suffix(".provenance.json")] + [p.parent / "graph_features.jsonl" for p in manifests]
    else:
        verify_stage_receipt(core, "graphs"); _graphs(core, manifests, paths)
        from .fit_risk import fit_risk_models
        from .failure_fit import fit_failure_models
        output = paths["risk"]
        options = {key: core["risk_network"][key] for key in ("hidden_width", "epochs", "batch_size", "learning_rate", "weight_decay", "patience")}
        if (output / "fit_report.json").exists():
            _primary(core, manifests, output)
        else:
            report = fit_risk_models(manifests, output, model_key=core["model_key"], hyperparameters={"seed": 1729, **options})
            require(report["status"] == "succeeded", "Core overall risk fit failed; no retry/relaxed gate")
            _primary(core, manifests, output)
        if not (output / "failure_fit_report.json").exists():
            fit_failure_models(manifests, output, paths["labels"], model_key=core["model_key"],
                hyperparameters={"seed": 1729, **options}, corpus_contract=contract)
        report = _typed(core, manifests, output, contract)
        files = [p for directory in (output, paths["labels"]) for p in directory.rglob("*") if p.is_file()]
    return publish_stage_receipt(core, stage, files, corpus_contract=contract, status=report["status"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--stage", choices=("transcoders", "graphs", "risk"), required=True)
    args = parser.parse_args(); result = run_stage(read_core(args.project), args.stage)
    print(json.dumps({"stage": args.stage, "complete": result["complete"]}), flush=True)


if __name__ == "__main__":
    main()
