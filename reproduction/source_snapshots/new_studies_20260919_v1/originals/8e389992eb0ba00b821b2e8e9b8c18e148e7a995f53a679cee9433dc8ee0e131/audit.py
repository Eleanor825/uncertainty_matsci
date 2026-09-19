#!/usr/bin/env python3
"""Offline all-prefix/all-layer SnAr fidelity audit. No generation, oracle, or fit.

The only diagnostic change to frozen NativeAttributor.capture is suppression of
its first failed-FVU *exception*. Its condition and arithmetic remain intact;
every failed layer is recorded and this output cannot certify a native graph.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
import textwrap
import time
import traceback

ROOT = Path(__file__).resolve().parent
SCHEMA = "snar_offline_all_prefix_fvu_and_causal128_v1"
EPISODES = {"train": (1101, 1102, 1103), "dev": (2101, 2102)}
RULE = "qwen_final_mlp_causal_activation_v1"
ROWS = 128


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": sha(p)}


def check(ref):
    if sha(ref["path"]) != ref["sha256"]:
        raise ValueError("Source/artifact changed: " + ref["path"])
    return Path(ref["path"])


def publish(path, value, *, exclusive=True):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if exclusive:
        with p.open("xb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
    else:
        tmp = p.with_name(p.name + ".tmp." + str(os.getpid()))
        with tmp.open("xb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, p)


def transform_capture(source):
    """One auditable AST edit, no threshold or numerical operation replacement."""
    original = ast.parse(textwrap.dedent(source))
    functions = [x for x in original.body if isinstance(x, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "capture":
        raise ValueError("Expected exactly the frozen NativeAttributor.capture")
    expected_test = ast.dump(ast.parse("not fvu_defined or not np.isfinite(fvu) or fvu > threshold", mode="eval").body)
    matches = [node for node in ast.walk(original) if isinstance(node, ast.If)
               and ast.dump(node.test) == expected_test]
    if len(matches) != 1:
        raise ValueError("Frozen first-FVU guard changed")
    guard = matches[0]
    if (guard.orelse or len(guard.body) != 1 or not isinstance(guard.body[0], ast.Raise)
            or not isinstance(guard.body[0].exc, ast.Call)
            or not isinstance(guard.body[0].exc.func, ast.Name)
            or guard.body[0].exc.func.id != "UnsupportedAttribution"):
        raise ValueError("FVU exception has an unexpected body")
    prior_raise = ast.unparse(guard.body[0])
    guard.body = [ast.copy_location(ast.Pass(), guard.body[0])]
    ast.fix_missing_locations(original)
    return ast.unparse(original) + "\n", {
        "schema": "diagnostic_capture_suppress_only_first_fvu_exception_v1",
        "changed_ast_nodes": 1, "original_raise": prior_raise,
        "original_source_sha256": hashlib.sha256(textwrap.dedent(source).encode()).hexdigest(),
        "original_fvu_condition": "not fvu_defined or not np.isfinite(fvu) or fvu > threshold",
        "threshold_changed": False, "all_other_exceptions_unchanged": True,
        "graph_admission": False, "finite_difference_performed": False}


def capture_source(path):
    text = Path(path).read_text()
    tree = ast.parse(text)
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "NativeAttributor")
    method = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "capture")
    lines = text.splitlines(keepends=True)
    return textwrap.dedent("".join(lines[method.lineno-1:method.end_lineno]))


def init_sources(original, runtime):
    # Deliberately do not import pipeline/driver/OracleClient.
    sys.path.insert(0, runtime["method_source_root"])
    sys.path.insert(0, str(original))
    sys.dont_write_bytecode = True


def catalog(original, output, protocol, base):
    from matdiscovery.esopt import tensor_state_hash
    import torch
    selected, evidence, train_hashes = [], [], set()
    expected_names = {f"collection_{split}_{seed}" for split, seeds in EPISODES.items() for seed in seeds}
    actual = {p.name for p in (output/"episodes").iterdir() if p.is_dir()}
    if actual != expected_names:
        raise ValueError("Only the five original collection episodes may precede this audit")
    if protocol["risk"]["train_seeds"] != list(EPISODES["train"]) or protocol["risk"]["dev_seeds"] != list(EPISODES["dev"]):
        raise ValueError("Original train/development partition changed")
    for split, seeds in EPISODES.items():
        for seed in seeds:
            name = f"collection_{split}_{seed}"
            directory = output/"episodes"/name
            header = read(directory/"episode.json")
            if (header["name"], header["arm"], header["seed"], header["budget"], header["weight_hash"]) != (name, "qwen_base", seed, 30, base["base_state_hash"]):
                raise ValueError("Original episode identity differs")
            if header["protocol_fingerprint"] != digest(protocol):
                raise ValueError("Original episode protocol differs")
            if not (directory/"summary.json").is_file():
                raise ValueError("Original episode is incomplete")
            evidence += [artifact(directory/"episode.json"), artifact(directory/"summary.json")]
            query_paths = sorted(directory.glob("query[0-9][0-9][0-9].json"))
            if [p.name for p in query_paths] != [f"query{i:03d}.json" for i in range(30)]:
                raise ValueError("Original collection is not exactly 30 closed queries")
            for i, path in enumerate(query_paths):
                query = read(path); evidence.append(artifact(path))
                if query["query_id"] != name + f"/q{i:03d}":
                    raise ValueError("Query identity drift")
                if i < 5:
                    if query.get("chosen_generation"):
                        raise ValueError("Initial designs must not become generated training prefixes")
                    continue
                generation_path = Path(query["chosen_generation"]).resolve()
                if not generation_path.is_relative_to(directory.resolve()) or generation_path.name != "generation.json":
                    raise ValueError("Chosen generation leaves its original episode")
                saved = read(generation_path); g = saved["generation"]
                if saved["weight_hash"] != base["base_state_hash"] or g["success"] is not True:
                    raise ValueError("Source is not a successful original base generation")
                stamp = g["model_stamp"]
                if (stamp["generation"], stamp["perturbation_seed"], stamp["perturbation_sigma"]) != (0, None, None):
                    raise ValueError("Only original theta0 tokens are eligible")
                if (stamp["checkpoint_hash"] != base["checkpoint_hash"]
                        or g["configuration_fingerprint"] != base["configuration_fingerprint"]
                        or g["policy_runtime"] != base["runtime"]):
                    raise ValueError("Generation runtime/configuration/checkpoint differs")
                ids = torch.tensor(g["input_ids_with_completion"], dtype=torch.long)
                if ids.ndim != 2 or ids.shape[0] != 1 or not 1 < ids.shape[1] <= 2048:
                    raise ValueError("Original full-prefix context differs")
                if tensor_state_hash({"input_ids": ids}) != g["prefix_hash"]:
                    raise ValueError("Original token hash differs")
                if query["parameters"] != g["parsed_action"] or query["no_hvi"] not in (0, 1):
                    raise ValueError("Original chosen action/label differs")
                if split == "train": train_hashes.add(g["prefix_hash"])
                elif g["prefix_hash"] in train_hashes:
                    raise ValueError("Exact train/dev prefix overlap")
                graph_path = generation_path.parent/"risk_graph.json"
                graph = read(graph_path)
                selected.append({"index": len(selected), "split": split, "episode": name,
                    "query_id": query["query_id"], "query": artifact(path), "generation": artifact(generation_path),
                    "old_graph": artifact(graph_path), "old_graph_available": graph["available"],
                    "label": query["no_hvi"], "original_generation_prefix_hash": g["prefix_hash"],
                    "prefix_tokens": ids.shape[1]})
    if len(selected) != 125 or sum(x["split"] == "train" for x in selected) != 75:
        raise ValueError("Exact original 75 train / 50 dev prefix catalog required")
    return selected, evidence


def prepare(original, registration_path):
    original = Path(original).resolve(); out = original/"runs/v1"
    registration_path = Path(registration_path).resolve()
    if not registration_path.is_relative_to(ROOT) or registration_path.exists():
        raise ValueError("Registration must be a new artifact in the independent audit namespace")
    runtime = read(original/"runtime_config.json")
    prior = read(out/"registration.json")
    for ref in prior["sources"]: check(ref)
    protocol = read(original/"protocol.json"); base = read(out/"base_policy.json")
    if sha(original/"protocol.json") != runtime["protocol_sha256"]:
        raise ValueError("Original protocol hash differs")
    if (protocol["graph"]["source_selection_rule"] != RULE or protocol["representation"]["max_dev_fvu"] != .5
            or protocol["risk"]["minimum_valid_graph_fraction"] != .9
            or protocol["model"]["gpu_allocator_fraction"] != .35):
        raise ValueError("Original scientific gates/runtime changed")
    closure = read(original/"logs/main_v1/completion.json")
    if closure.get("status") != "command_failed_no_automatic_replay" or closure.get("complete") is not False:
        raise ValueError("Preserve the closed original failed invocation")
    init_sources(original, runtime)
    rows, evidence = catalog(original, out, protocol, base)
    selected = read(out/"representation_selected.json")
    bank_path = Path(selected["path"]).resolve()
    if bank_path != out/"snar_transcoders/transcoder_manifest.json":
        raise ValueError("Audit requires the actual complete fresh SnAr bank")
    bank = read(bank_path); transfer = read(out/"fresh_snar_transfer_fidelity.json")
    if (not bank["complete"] or not bank["ready_for_graphs"] or bank["passed_layers"] != 32
            or bank["bank_fingerprint"] != selected["bank_fingerprint"]
            or not transfer["passed"] or transfer["fingerprint"] != selected["fidelity_fingerprint"]):
        raise ValueError("Original complete 32-layer representation evidence differs")
    old_filter = read(out/"representation_dev_prefix_filter.json")
    if old_filter != {"rejected_proposal_directories": [], "test_used": False}:
        raise ValueError("This exact 125-prefix audit requires the original empty dev exclusion set")
    native = Path(runtime["method_source_root"])/"matdiscovery/native_attribution.py"
    diagnostic_source, transformation = transform_capture(capture_source(native))
    generated = ROOT/"diagnostic_capture.py"
    with generated.open("x") as f:
        f.write(diagnostic_source); f.flush(); os.fsync(f.fileno())
    fixed = [original/"protocol.json", original/"runtime_config.json", out/"registration.json",
        out/"base_policy.json", out/"collection_complete.json", out/"representation_selected.json",
        out/"representation_dev_prefix_filter.json", out/"fresh_snar_transfer_fidelity.json",
        original/"logs/main_v1/completion.json", bank_path, Path(runtime["model_manifest"])]
    bank_artifacts = []
    for layer in bank["layers"]:
        cp = Path(layer["checkpoint"])
        if not cp.is_absolute(): cp = bank_path.parent/cp
        ref = artifact(cp)
        if ref["sha256"] != layer["checkpoint_sha256"]:
            raise ValueError("Original TC checkpoint changed")
        bank_artifacts.extend([ref, artifact(str(cp)+".json")])
    own_sources = [artifact(ROOT/name) for name in ("audit.py", "test_audit.py", "diagnostic_capture.py",
        "runtime/resource_guard.py", "runtime/supervise.py", "runtime/deployment_facts.json")]
    reg = {"schema": SCHEMA, "classification": "offline_diagnostic_not_main_acceptance",
        "original": str(original), "original_output": str(out), "output": str(ROOT/"output"),
        "source_selection_rule": RULE, "prefixes": rows, "counts": {"prefixes": 125, "train": 75, "dev": 50, "layers": 32},
        "token_selection": {"rule": "uniform_linspace_integer_unique_over_original_causal_horizon", "rows": ROWS,
            "full_forward_prefix": True, "horizon": "original_action_targets.max_prediction_position"},
        "native_capture_transformation": transformation, "native_source": artifact(native),
        "original_sources": prior["sources"], "own_sources": own_sources,
        "fixed_evidence": [artifact(p) for p in fixed] + evidence,
        "bank": artifact(bank_path), "bank_artifacts": bank_artifacts,
        "source_bank_fingerprint": selected["bank_fingerprint"],
        "runtime_contract": {"dtype": "float32", "attention": "eager", "allocator_fraction": .35,
             "threads": 2, "transcoder_device": "cpu", "fvu_reduction_device": "cuda:0"},
        "no_generation": True, "new_oracle_calls": 0, "test_used": False, "training_performed": False,
        "FD_performed": False, "FVU_threshold": .5, "native_graph_minimum_availability": .9,
        "registered_at_unix": time.time()}
    reg["fingerprint"] = digest(reg)
    publish(registration_path, reg)
    return reg


def validate_registration(path):
    reg = read(path)
    body = {k: v for k, v in reg.items() if k != "fingerprint"}
    if reg.get("schema") != SCHEMA or digest(body) != reg.get("fingerprint"):
        raise ValueError("Audit registration seal changed")
    if (reg["counts"] != {"prefixes": 125, "train": 75, "dev": 50, "layers": 32}
            or reg["token_selection"]["rows"] != ROWS or reg["source_selection_rule"] != RULE
            or reg["FVU_threshold"] != .5 or reg["native_graph_minimum_availability"] != .9
            or reg["no_generation"] is not True or reg["new_oracle_calls"] != 0
            or reg["training_performed"] or reg["test_used"] or reg["FD_performed"]):
        raise ValueError("Audit scope changed")
    for refs in (reg["own_sources"], reg["original_sources"], reg["fixed_evidence"], reg["bank_artifacts"]):
        for ref in refs: check(ref)
    for row in reg["prefixes"]:
        for key in ("query", "generation", "old_graph"): check(row[key])
    check(reg["native_source"])
    expected, transform = transform_capture(capture_source(reg["native_source"]["path"]))
    if expected != (ROOT/"diagnostic_capture.py").read_text() or transform != reg["native_capture_transformation"]:
        raise ValueError("Diagnostic is not the single registered exception edit")
    if Path(reg["output"]).resolve() != ROOT/"output":
        raise ValueError("Audit cannot write outside its own fixed output")
    return reg


def causal_positions(length, horizon, count=ROWS):
    import torch
    if type(horizon) is not int or type(length) is not int or not 0 <= horizon < length or count != ROWS:
        raise ValueError("Original causal horizon and fixed 128-row policy required")
    return torch.linspace(0, horizon, min(count, horizon+1)).long().unique(sorted=True)


def metrics_rows(fidelity, bindings):
    rows = []
    if set(fidelity) != {b.module_path for b in bindings} or len(bindings) != 32:
        raise ValueError("All 32 real layers must be retained, including failed gates")
    for index, binding in enumerate(bindings):
        value = dict(fidelity[binding.module_path])
        threshold = float(binding.training_metadata["max_dev_fvu"])
        if threshold != .5 or any(not math.isfinite(float(x)) for x in value.values()):
            raise ValueError("Nonfinite diagnostic value or changed gate")
        rows.append({"layer_index": index, "layer_path": binding.module_path, **value,
            "threshold": threshold, "passed": value["fvu_undefined"] == 0 and value["output_fvu"] <= threshold})
    return rows


def save_prefix(reg, row, trace, targets, bindings, replay):
    import torch
    from matdiscovery.transcoders import write_activation_shard
    folder = Path(reg["output"])/"prefixes"/f"{row['index']:03d}_{row['episode']}_q{row['query_id'].rsplit('q',1)[1]}"
    folder.mkdir(parents=True, exist_ok=False)
    full_length = trace.input_ids.shape[1]
    if full_length != row["prefix_tokens"] or targets.source_prefix_hash != trace.prefix_hash:
        raise ValueError("Diagnostic trace differs from original complete tokens/targets")
    positions = causal_positions(full_length, targets.causal_horizon)
    if len(positions) != ROWS:
        raise ValueError("This registration requires all 128 distinct causal positions")
    matrix = metrics_rows(trace.fidelity, bindings)
    shards = []
    for index, binding in enumerate(bindings):
        path = binding.module_path
        x = trace.mlp_inputs[path][0, positions.to(trace.mlp_inputs[path].device)].detach().cpu().float()
        y = trace.mlp_leaves[path][0, positions.to(trace.mlp_leaves[path].device)].detach().cpu().float()
        target = folder/f"layer{index:02d}.pt"
        metadata = write_activation_shard(target, x, y, group_ids=[row["episode"]]*len(positions),
            prefix_hashes=[trace.prefix_hash]*len(positions), split=row["split"],
            policy_fingerprint=trace.stamp.checkpoint_hash, layer_path=path)
        shards.append({"layer_index": index, "layer_path": path, "shard": artifact(target),
                       "sidecar": artifact(str(target)+".json"), "tensor_hash": metadata["tensor_hash"], "rows": len(positions)})
    receipt = {"schema": "snar_causal128_prefix_fvu_receipt_v1", "complete": True,
        "registration_fingerprint": reg["fingerprint"], "source": row, "replay": replay,
        "full_prefix_hash": trace.prefix_hash, "complete_prefix_tokens": full_length,
        "target": targets.to_dict(), "selected_token_positions": positions.tolist(),
        "capture_parity_max_abs": trace.parity_max_abs, "capture_parity_passed": True,
        "all_layer_fvu": matrix, "all_layer_fvu_passed": all(x["passed"] for x in matrix),
        "shards": shards, "scientific_oracle_calls": 0, "generation_calls": 0,
        "native_graph_available": None, "finite_difference_passed": None,
        "not_a_native_graph_admission": True}
    publish(folder/"receipt.json", receipt)
    return receipt, artifact(folder/"receipt.json")


def run(registration_path):
    # A one-shot claim precedes all model loading. Unknown/failed work is preserved.
    reg = validate_registration(registration_path)
    output = Path(reg["output"])
    output.mkdir(exist_ok=False)
    started = time.time()
    publish(output/"invocation.json", {"registration": artifact(registration_path), "pid": os.getpid(),
        "hostname": socket.gethostname(), "started_at_unix": started, "status": "admitted_no_model_yet"})
    current = None
    try:
        original = Path(reg["original"]); runtime = read(original/"runtime_config.json")
        init_sources(original, runtime)
        # New copied guard only; importing original runtime could use a stale namespace.
        sys.path.insert(0, str(ROOT/"runtime"))
        from resource_guard import apply_torch_contract
        if os.environ.get("SNAR_RESOURCE_GUARD") != "1":
            raise RuntimeError("Audit must launch beneath its private resource supervisor")
        contract = apply_torch_contract()
        import torch
        from matdiscovery.esopt import tensor_state_hash
        import matdiscovery.native_attribution as native
        from method.snar_policy import SnArPolicyAdapter, SnArPolicyConfig, replay_base_generation
        from method.features import build_snar_action_targets, load_candidate_bank
        from matdiscovery.representation_training import GraphStageConfig, QWEN_MLP_PATHS
        protocol = read(original/"protocol.json"); base = read(Path(reg["original_output"])/"base_policy.json")
        cfg = protocol["policy"]
        pcfg = SnArPolicyConfig(**{k:cfg[k] for k in ("context_budget","max_input_tokens","max_new_tokens","history_window","temperature","top_p","top_k")})
        objective = ("Seek the Pareto frontier: maximize sty and minimize e_factor. "
            "The fixed normalized hypervolume has coordinates sty/13000 and (1000-e_factor)/1000 "
            "with reference (0,0), without clipping sty. Explore and improve observed tradeoffs.")
        adapter = SnArPolicyAdapter.from_verified_checkpoint(runtime["model_manifest"], runtime["model_directory"],
            objective_description=objective, config=pcfg, device="cuda:0")
        policy = adapter.policy
        if (policy.checkpoint_hash != base["checkpoint_hash"] or policy.configuration_fingerprint != base["configuration_fingerprint"]
                or policy.runtime_precision_record() != base["runtime"]
                or tensor_state_hash(dict(policy.model.state_dict())) != base["base_state_hash"]
                or tuple(policy.mlp_paths) != tuple(QWEN_MLP_PATHS)):
            raise ValueError("Loaded model is not exact original SnAr theta0/runtime")
        bank = load_candidate_bank(reg["bank"]["path"], policy, device="cpu")
        if bank.bank_fingerprint != reg["source_bank_fingerprint"]:
            raise ValueError("Loaded bank differs from original selected bank")
        graph = GraphStageConfig(**protocol["graph"]); graph.validate()
        attributor = native.NativeAttributor(policy.model, bank.bindings, state_id_getter=policy.get_state_id,
            architecture_review=policy.architecture_review, max_nodes=graph.max_nodes,
            max_feature_nodes=graph.max_feature_nodes, max_backward_targets=graph.max_backward_targets,
            max_logits=graph.max_logits, constant_storage_device=graph.constant_storage_device,
            target_mode="action_logprob_v2", source_selection_rule=graph.source_selection_rule)
        namespace = dict(vars(native))
        code = (ROOT/"diagnostic_capture.py").read_text()
        exec(compile(code, str(ROOT/"diagnostic_capture.py"), "exec"), namespace)
        capture = namespace["capture"]
        publish(output/"model_admission.json", {"complete": True, "base_state_hash": base["base_state_hash"],
            "checkpoint_hash": policy.checkpoint_hash, "policy_configuration_fingerprint": policy.configuration_fingerprint,
            "policy_runtime": policy.runtime_precision_record(), "contract": contract,
            "bank_fingerprint": bank.bank_fingerprint, "source_selection_rule": RULE,
            "source_selection_proof": attributor.source_selection_proof, "loaded_at_unix": time.time()})
        torch.cuda.reset_peak_memory_stats(0)
        receipts, matrix, by_split = [], [], {"train": [], "dev": []}
        for row in reg["prefixes"]:
            current = row["index"]; start = time.time()
            check(row["generation"]); saved = read(row["generation"]["path"])
            generation, replay = replay_base_generation(saved["generation"], policy)
            targets = build_snar_action_targets(generation, policy.tokenizer)
            ids, kwargs = policy.prepare_trace_inputs(generation.input_ids_with_completion, stamp=generation.model_stamp)
            trace = capture(attributor, ids, generation.model_stamp, model_kwargs=kwargs, action_targets=targets)
            receipt, ref = save_prefix(reg, row, trace, targets, bank.bindings, replay)
            del trace, generation, ids, kwargs
            torch.cuda.synchronize()
            elapsed = time.time()-start
            timing = {"prefix_index": current, "split": row["split"], "elapsed_seconds": elapsed,
                "finished_at_unix": time.time(), "failed_layers": [x["layer_index"] for x in receipt["all_layer_fvu"] if not x["passed"]],
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}
            publish(Path(ref["path"]).with_name("timing.json"), timing)
            receipts.append(ref); matrix.append([x["output_fvu"] for x in receipt["all_layer_fvu"]])
            by_split[row["split"]].append(receipt["all_layer_fvu_passed"])
            publish(output/"status.json", {"status": "auditing", "completed_prefixes": len(receipts),
                "expected_prefixes": 125, **timing}, exclusive=False)
            print(json.dumps({"event": "prefix_completed", "completed": len(receipts), **timing}), flush=True)
        # Structural/source audit is rerun after computation; no old evidence is written.
        validate_registration(registration_path)
        if tensor_state_hash(dict(policy.model.state_dict())) != base["base_state_hash"]:
            raise ValueError("Offline audit changed model parameters")
        final = {"schema": SCHEMA, "complete": True, "classification": reg["classification"],
            "registration": artifact(registration_path), "registration_fingerprint": reg["fingerprint"],
            "counts": reg["counts"], "fvu_matrix": matrix, "prefix_receipts": receipts,
            "per_split": {s: {"prefixes": len(v), "all32_fvu_passed": sum(v), "all32_fvu_failed": len(v)-sum(v)} for s,v in by_split.items()},
            "shards": {"total": 4000, "rows_per_prefix_per_layer": 128, "train_rows_per_layer": 9600, "dev_rows_per_layer": 6400},
            "started_at_unix": started, "finished_at_unix": time.time(), "elapsed_seconds": time.time()-started,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0), "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "new_generation_calls": 0, "new_oracle_calls": 0, "training_performed": False,
            "FD_performed": False, "native_graph_available": None, "main_experiment_complete": False,
            "source_files_reverified": True, "original_model_weights_unchanged": True}
        publish(output/"summary.json", final)
        print(json.dumps({"event": "audit_complete", "summary": artifact(output/"summary.json"), "per_split": final["per_split"]}), flush=True)
    except BaseException as exc:
        publish(output/"failure.json", {"complete": False, "prefix_index": current, "error_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(), "failed_at_unix": time.time(),
            "elapsed_seconds": time.time()-started, "new_generation_calls": 0, "new_oracle_calls": 0,
            "no_automatic_replay": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--original")
    parser.add_argument("--registration", default=str(ROOT/"registration.json"))
    args = parser.parse_args()
    if args.prepare:
        if not args.original: parser.error("--prepare requires --original")
        reg = prepare(args.original, args.registration)
        print(json.dumps({"prepared": True, "fingerprint": reg["fingerprint"], "counts": reg["counts"],
                          "registration": artifact(args.registration), "new_oracle_calls": 0}))
    else:
        run(args.registration)


if __name__ == "__main__":
    main()
