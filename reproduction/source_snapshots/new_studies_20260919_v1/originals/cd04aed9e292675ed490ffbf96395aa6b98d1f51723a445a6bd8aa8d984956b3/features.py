"""SnAr token targets, actual activations, and gated pretrained TC transfer.

Uses public mathematical components only. No MADE core/ledger/controller loader
is involved. A MADE bank is a candidate representation, never a SnAr fit result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Mapping, Sequence
from weakref import WeakKeyDictionary

import torch

from matdiscovery.accounting import file_sha256, fingerprint, write_json_atomic
from matdiscovery.action_targets import ActionTargetError, ActionTargets, TARGET_SCHEMA
from matdiscovery.esopt import tensor_state_hash
from matdiscovery.graph_features import extract_graph_features
from matdiscovery.native_attribution import NativeAttributor, TranscoderBinding
from matdiscovery.representation_training import BANK_SCHEMA, GraphStageConfig, QWEN_MLP_PATHS, validate_transcoder_bank
from matdiscovery.transcoders import (iter_activation_batches, load_transcoder,
                                     reconstruction_metrics, validate_shard_splits, TranscoderConfig)

from .snar_policy import ACTION_NAMES, parse_action_json, validate_action


SNAR_TARGET_MAPPING = "snar_four_numeric_values_original_tokens_v1"
_VALIDATED_STATES = WeakKeyDictionary()
_NUMBER = r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_PARAMETER = re.compile(r'"(?P<key>tau|equiv_pldn|conc_dfnb|temperature)"\s*:\s*(?P<number>' + _NUMBER + r')')


def build_snar_action_targets(generation, tokenizer, *, attention_mask=None) -> ActionTargets:
    """Score only emitted numeric values, with t predicted at t-1.

    The sink remains the original mean raw LM log probability. Field-name and
    punctuation-only tokens, rationale, EOS and other special tokens are excluded.
    An inseparable token containing both a number and a field name is rejected.
    Nothing is re-tokenized, clamped, or substituted with a canonical JSON string.
    """
    if not generation.success:
        raise ActionTargetError("Invalid proposals have no SnAr action target")
    action = validate_action(generation.parsed_action)
    ids = generation.input_ids_with_completion
    start = generation.prompt_token_count
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ActionTargetError("One complete original integer token sequence is required")
    if type(start) is not int or not 1 <= start < ids.shape[1]:
        raise ActionTargetError("Missing original prompt token count")
    mask = torch.ones_like(ids) if attention_mask is None else attention_mask
    if mask.shape != ids.shape or not bool((mask == 1).all()):
        raise ActionTargetError("Original complete all-one mask is required")
    tokens = ids[0, start:].detach().cpu().tolist()
    special = tuple(sorted(set(int(t) for t in tokenizer.all_special_ids)))
    decode = lambda value: tokenizer.decode(value, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    text = decode(tokens)
    ends = [0]
    for end in range(1, len(tokens)+1):
        prefix = decode(tokens[:end])
        ends.append(len(prefix) if text.startswith(prefix) else None)
    spans = [(a, b) if a is not None and b is not None and a <= b else None
             for a, b in zip(ends, ends[1:])]
    visible = list(text)
    for token, span in zip(tokens, spans):
        if token in special:
            if span is None:
                raise ActionTargetError("Unmapped special-token boundary")
            visible[span[0]:span[1]] = " " * (span[1]-span[0])
    visible = "".join(visible)
    if parse_action_json(visible) != action:
        raise ActionTargetError("Original emitted action differs from the parsed SnAr action")
    matches = list(_PARAMETER.finditer(visible))
    if len(matches) != 4 or {m.group("key") for m in matches} != set(ACTION_NAMES):
        raise ActionTargetError("Ambiguous numeric JSON value mapping")
    wanted = {p for match in matches for p in range(*match.span("number"))}
    forbidden = {p for match in matches for p in range(*match.span("key"))}
    positions, texts, covered = [], [], set()
    for offset, (token, span) in enumerate(zip(tokens, spans)):
        if token in special or span is None:
            continue
        chars = set(range(*span))
        overlap = chars & wanted
        if not overlap:
            continue
        if chars & forbidden:
            raise ActionTargetError("A token inseparably mixes an action number and a field name")
        positions.append(start+offset); texts.append(text[slice(*span)]); covered.update(overlap)
    if not positions or covered != wanted:
        raise ActionTargetError("Numeric values are not completely covered by original token boundaries")
    values = dict(benchmark="summit_snar", prompt_token_count=start,
                  token_positions=tuple(positions), prediction_positions=tuple(t-1 for t in positions),
                  token_ids=tuple(int(ids[0,t]) for t in positions), token_texts=tuple(texts),
                  semantic_paths=tuple("/"+m.group("key") for m in matches), special_token_ids=special,
                  source_prefix_hash=tensor_state_hash({"input_ids": ids, "attention_mask": mask}),
                  source_length=ids.shape[1], tokenizer_class=type(tokenizer).__module__+"."+type(tokenizer).__name__,
                  schema=TARGET_SCHEMA)
    spec = {k:v for k,v in values.items() if k not in ("source_prefix_hash", "source_length")}
    result = ActionTargets(**values, target_spec_hash=fingerprint(spec))
    result.validate(ids, mask)
    return result


def capture_features(policy, generation) -> tuple[object, dict, dict]:
    """Actual full-prefix observation; no outcome/label argument is accepted."""
    capture = policy.capture_prefix(generation.input_ids_with_completion, stamp=generation.model_stamp)
    features = {}
    for index, path in enumerate(policy.mlp_paths):
        for name in ("input_mean", "output_mean", "input_rms", "output_rms"):
            features[f"hidden.layer_{index}.{name}"] = float(capture.hidden_summaries[path][name])
    if generation.entropy:
        features["sampling.mean_entropy"] = sum(generation.entropy)/len(generation.entropy)
    if generation.logprobs:
        features["sampling.mean_logprob"] = sum(generation.logprobs)/len(generation.logprobs)
        features["sampling.min_logprob"] = min(generation.logprobs)
    features["sampling.completion_count"] = generation.completion_count
    if not all(math.isfinite(v) for v in features.values()):
        raise ValueError("Actual activation/sampling features must be finite")
    provenance = {"schema": "snar_actual_full_prefix_features_v1", "prefix_hash": capture.prefix_hash,
                  "policy_stamp": asdict(generation.model_stamp), "policy_runtime": policy.runtime_precision_record(),
                  "configuration_fingerprint": generation.configuration_fingerprint,
                  "capture_contract": capture.contract, "parity_passed": capture.parity_passed,
                  "sampling_score_semantics": generation.score_semantics,
                  "future_outcomes_used": False}
    return capture, features, provenance


def activation_rows(capture, *, token_rows=16) -> dict:
    """Uniform original prefix positions for train/dev shard writers, all layers."""
    if type(token_rows) is not int or token_rows < 1:
        raise ValueError("Positive explicit token-row count required")
    length = capture.input_ids.shape[1]
    positions = torch.linspace(0, length-1, min(token_rows,length)).long().unique(sorted=True)
    if set(capture.mlp_inputs) != set(capture.mlp_outputs):
        raise ValueError("MLP capture inputs/outputs differ")
    return {path: {"inputs": capture.mlp_inputs[path][0,positions].detach().cpu().float(),
                   "outputs": capture.mlp_outputs[path][0,positions].detach().cpu().float(),
                   "token_positions": positions.tolist(), "prefix_hash": capture.prefix_hash}
            for path in capture.mlp_inputs}


@dataclass
class CandidateBank:
    bindings: tuple[TranscoderBinding, ...]
    source: dict
    bank_fingerprint: str
    policy_fingerprint: str
    transfer_report: dict | None = None


def load_candidate_bank(path, policy, *, device="cpu") -> CandidateBank:
    """Load verified original raw TopK exports, without claiming SnAr fidelity."""
    stamp = policy.model_stamp
    if stamp.generation != 0 or stamp.perturbation_seed is not None or stamp.perturbation_sigma is not None:
        raise ValueError("Transfer admission must start from the original base policy, not MADE G2")
    path = Path(path).resolve()
    bank = validate_transcoder_bank(path, model_key="qwen35_4b", checkpoint_hash=policy.checkpoint_hash)
    if tuple(policy.mlp_paths) != tuple(QWEN_MLP_PATHS):
        raise ValueError("All original 32 Qwen MLP boundaries are required")
    bindings = []
    artifacts = []
    for layer in bank["layers"]:
        checkpoint = Path(layer["checkpoint"])
        if not checkpoint.is_absolute(): checkpoint = path.parent/checkpoint
        model, metadata = load_transcoder(checkpoint, expected_policy_fingerprint=policy.checkpoint_hash,
                                          expected_layer_path=layer["layer_path"])
        if metadata["transcoder_hash"] != layer["metadata"]["transcoder_hash"]:
            raise ValueError("Bank and actual TC metadata differ")
        if metadata.get("fit_recipe"):
            from matdiscovery.normalized_transcoders import verify_normalized_metadata
            verify_normalized_metadata(metadata, policy_fingerprint=policy.checkpoint_hash,
                                       layer_path=layer["layer_path"])
        model.to(device=device, dtype=torch.float32).eval()
        bindings.append(TranscoderBinding(layer["layer_path"], model, metadata))
        artifacts.append({"path": str(checkpoint.resolve()), "sha256": layer["checkpoint_sha256"],
                          "transcoder_hash": model.checkpoint_hash()})
    source = {"manifest": {"path": str(path), "sha256": file_sha256(path)}, "checkpoints": artifacts,
              "role": "fresh_snar_bank_pending_transfer_recheck" if bank.get("snar_training_performed") is True else "pretrained_representation_candidate_only",
              "source_policy_runtime": bank["policy_runtime"],
              "target_policy_runtime": policy.runtime_precision_record(), "source_is_snar_fit": bank.get("snar_training_performed") is True}
    return CandidateBank(tuple(bindings), source, bank["bank_fingerprint"], policy.checkpoint_hash)


def validate_transfer_on_shards(candidate: CandidateBank, train_by_layer: Mapping,
                                dev_by_layer: Mapping, *, max_dev_fvu=.5, batch_size=256) -> dict:
    """Measure original TC output FVU on SnAr only; never inspect test shards.

    No optimization or selection occurs here. Both train/dev data are checked
    for group and exact-prefix isolation by the original validator. A failed or
    undefined layer keeps the bank unavailable; its old MADE gate is insufficient.
    """
    if max_dev_fvu != .5:
        raise ValueError("The existing FVU<=0.5 threshold is retained")
    paths = [b.module_path for b in candidate.bindings]
    if not paths or set(train_by_layer) != set(paths) or set(dev_by_layer) != set(paths):
        raise ValueError("Transfer data must cover every bound layer")
    candidate.transfer_report = None
    layers = []
    population = None
    for binding in candidate.bindings:
        path, tc = binding.module_path, binding.transcoder
        train, dev = train_by_layer[path], dev_by_layer[path]
        source = validate_shard_splits(train, dev, policy_fingerprint=candidate.policy_fingerprint,
                                      layer_path=path, config=tc.config)
        this_population = {"rows": source["rows"], "groups": source["groups"],
                           "prefixes": {split: sorted({p for item in source["shards"] if item["split"] == split
                                                        for p in item["prefixes"]}) for split in ("train", "dev")}}
        if population is not None and population != this_population:
            raise ValueError("All SnAr layers must use the same complete train/dev population")
        population = this_population
        files = {split: [{"path": str(Path(p).resolve()), "sha256": file_sha256(p)} for p in ps]
                 for split,ps in (("train",train),("dev",dev))}
        metrics = {split: reconstruction_metrics(tc, iter_activation_batches(ps, batch_size, seed=0, shuffle=False))
                   for split,ps in (("train",train),("dev",dev))}
        passed = metrics["dev"]["fvu_undefined"] == 0 and math.isfinite(metrics["dev"]["output_fvu"]) and metrics["dev"]["output_fvu"] <= max_dev_fvu
        layers.append({"layer_path": path, "transcoder_hash": tc.checkpoint_hash(), "source": source,
                       "source_files": files, **metrics, "passed": passed})
    report = {"schema": "snar_pretrained_bank_transfer_fidelity_v1", "source_bank": candidate.source,
              "bank_fingerprint": candidate.bank_fingerprint, "policy_fingerprint": candidate.policy_fingerprint,
              "max_dev_fvu": max_dev_fvu, "layers": layers, "passed": all(r["passed"] for r in layers),
              "test_data_used": False, "transcoder_training_performed": False,
              "metric": "original reconstruction_metrics output FVU on raw MLP output"}
    report["fingerprint"] = fingerprint(report)
    candidate.transfer_report = report
    return report


def make_attributor(policy, candidate: CandidateBank, *, config: GraphStageConfig) -> NativeAttributor:
    config.validate()
    if config.source_selection_rule != "qwen_final_mlp_causal_activation_v1":
        raise ValueError("Use the already reviewed Qwen final-MLP causal source rule")
    report = candidate.transfer_report
    if not report or report.get("passed") is not True or report.get("fingerprint") != fingerprint({k:v for k,v in report.items() if k!='fingerprint'}):
        raise ValueError("An actual passed SnAr train/dev transfer gate is required")
    if tuple(b.module_path for b in candidate.bindings) != tuple(policy.mlp_paths):
        raise ValueError("TC bindings differ from the actual complete policy MLP inventory")
    if report["bank_fingerprint"] != candidate.bank_fingerprint or report["policy_fingerprint"] != policy.checkpoint_hash:
        raise ValueError("Transfer gate belongs to another bank or policy")
    if len(report["layers"]) != len(candidate.bindings) or report["max_dev_fvu"] != .5 or report["test_data_used"] is not False:
        raise ValueError("Incomplete or changed SnAr transfer gate")
    for binding, layer in zip(candidate.bindings, report["layers"]):
        if (binding.module_path != layer["layer_path"] or binding.transcoder.checkpoint_hash() != layer["transcoder_hash"]
                or layer["passed"] is not True or layer["dev"]["fvu_undefined"] != 0
                or not math.isfinite(layer["dev"]["output_fvu"]) or layer["dev"]["output_fvu"] > .5):
            raise ValueError("TC weights changed after transfer validation")
        for files in layer["source_files"].values():
            for item in files:
                if file_sha256(item["path"]) != item["sha256"]:
                    raise ValueError("SnAr transfer data changed after validation")
    if config.dtype != "float32" or any(p.is_floating_point() and p.dtype != torch.float32 for p in policy.model.parameters()):
        raise ValueError("SnAr white-box attribution requires the verified FP32 policy")
    return NativeAttributor(policy.model, candidate.bindings, state_id_getter=policy.get_state_id,
        architecture_review=policy.architecture_review, max_nodes=config.max_nodes,
        max_feature_nodes=config.max_feature_nodes, max_backward_targets=config.max_backward_targets,
        max_logits=config.max_logits, constant_storage_device=config.constant_storage_device,
        target_mode="action_logprob_v2", source_selection_rule=config.source_selection_rule)


def validate_snar_backend(attributor, policy, generation, *, config: GraphStageConfig):
    _VALIDATED_STATES.pop(attributor, None)
    config.validate()
    if (config.validation_epsilon, config.validation_rtol, config.validation_atol, config.validation_max_edges) != (1e-3, .05, 1e-4, 4):
        raise ValueError("Retain the existing four-edge finite-difference gate without relaxing tolerances")
    target = build_snar_action_targets(generation, policy.tokenizer)
    ids, kwargs = policy.prepare_trace_inputs(generation.input_ids_with_completion, stamp=generation.model_stamp)
    report = attributor.validate_backend(ids, generation.model_stamp, model_kwargs=kwargs, action_targets=target,
        epsilon=config.validation_epsilon, rtol=config.validation_rtol, atol=config.validation_atol,
        max_edges=config.validation_max_edges)
    _VALIDATED_STATES[attributor] = (generation.model_stamp.state_id, fingerprint(asdict(config)))
    return report


def attribute_features(attributor, policy, generation, *, config: GraphStageConfig):
    if _VALIDATED_STATES.get(attributor) != (generation.model_stamp.state_id, fingerprint(asdict(config))):
        raise ValueError("Run SnAr backend validation for this exact policy state and graph configuration first")
    target = build_snar_action_targets(generation, policy.tokenizer)
    ids, kwargs = policy.prepare_trace_inputs(generation.input_ids_with_completion, stamp=generation.model_stamp)
    result = attributor.attribute(ids, generation.model_stamp, model_kwargs=kwargs, action_targets=target)
    extracted = extract_graph_features(result.graph, max_path_sources=config.max_path_sources,
                                        max_path_edge_visits=config.max_path_edge_visits)
    values = {"graph."+name: value for name,value in extracted.values.items()}
    # Mean raw action log probability is audit metadata, never an extra graph NN input.
    metadata = {"target_mapping": SNAR_TARGET_MAPPING, "target": target.to_dict(),
                "native": result.metadata, "graph_diagnostics": extracted.metadata,
                "graph_configuration": asdict(config),
                "future_outcomes_used": False}
    return values, metadata, result


def train_snar_bank(policy, train_by_layer: Mapping, dev_by_layer: Mapping, *,
                    output_directory, registration: Mapping, template: CandidateBank | None = None,
                    policy_configuration: Mapping | None = None,
                    seed=1729, batch_size=256, learning_rate=4e-4,
                    training_backend="cpu", expected_gpu_uuid=None) -> Path:
    """Fresh independent normalized64 training, original trainer/math/gates.

    Only a previously COMPLETE identical request can resume; any partial output
    requires explicit reconciliation. Never silently redo a completed/failed fit.
    This function does not mutate the policy or source bank and calls no oracle.
    """
    from matdiscovery.normalized_transcoders import train_normalized_layer
    if training_backend not in ("cpu", "isolated_cuda_2gib"):
        raise ValueError("Explicit registered TC training backend required")
    if training_backend == "isolated_cuda_2gib" and not isinstance(expected_gpu_uuid,str):
        raise ValueError("Isolated TC child requires the audited physical GPU UUID")
    training_device="cuda:0" if training_backend=="isolated_cuda_2gib" else "cpu"
    stamp = policy.model_stamp
    if stamp.generation != 0 or stamp.perturbation_seed is not None or stamp.perturbation_sigma is not None:
        raise ValueError("Fit SnAr transcoders only on the verified base policy")
    if tuple(policy.mlp_paths) != tuple(QWEN_MLP_PATHS):
        raise ValueError("Independent SnAr bank must cover all32 original Qwen layers")
    if template is not None:
        if (tuple(b.module_path for b in template.bindings) != tuple(QWEN_MLP_PATHS)
                or template.policy_fingerprint != policy.checkpoint_hash):
            raise ValueError("Architecture template differs from the full verified base checkpoint")
        layer_configs = [(b.module_path, b.transcoder.config) for b in template.bindings]
    else:
        width = int(policy.model.config.text_config.hidden_size)
        layer_configs = [(path,TranscoderConfig(width,width,2*width,64)) for path in QWEN_MLP_PATHS]
    for _, config in layer_configs: config.validate()
    if set(train_by_layer) != set(QWEN_MLP_PATHS) or set(dev_by_layer) != set(QWEN_MLP_PATHS):
        raise ValueError("Every SnAr layer requires complete train and development inputs")
    if not isinstance(registration, Mapping) or not registration:
        raise ValueError("Explicit frozen SnAr registration is required")
    # Read-only original adapter record; unlike reconstructing a guessed config,
    # this includes tokenizer/default/runtime fields exactly as fingerprinted.
    if policy_configuration is None:
        policy_configuration = policy._configuration()
    configuration = json.loads(json.dumps(dict(policy_configuration), allow_nan=False))
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if config_hash != policy.configuration_fingerprint or configuration.get("policy_runtime") != policy.runtime_precision_record():
        raise ValueError("Original policy configuration record does not match its verified fingerprint/runtime")
    if type(seed) is not int or seed < 0:
        raise ValueError("Explicit nonnegative initialization seed required")
    inputs = {layer: {split: [{"path": str(Path(p).resolve()), "sha256": file_sha256(p)} for p in data[layer]]
                     for split,data in (("train",train_by_layer),("dev",dev_by_layer))} for layer in QWEN_MLP_PATHS}
    request = {"schema": "snar_fresh_normalized_bank_request_v1", "checkpoint_hash": policy.checkpoint_hash,
               "policy_configuration_fingerprint": config_hash, "registration": dict(registration),
               "inputs": inputs, "seed": seed, "epochs": 64, "batch_size": batch_size,
               "learning_rate": learning_rate, "device": training_device, "max_dev_fvu": .5,
               "training_backend":training_backend,"expected_gpu_uuid":expected_gpu_uuid,
               "worker_sha256":file_sha256(Path(__file__).with_name("tc_worker.py")) if training_backend=="isolated_cuda_2gib" else None,
               "template_architectures": {path: asdict(config) for path,config in layer_configs},
               "initialization": "fresh original seeded TopK; no pretrained weights copied",
               "adapter_source_sha256": file_sha256(__file__),
               "trainer_sources": {name: file_sha256(Path(__import__('matdiscovery.normalized_transcoders', fromlist=['x']).__file__).with_name(name))
                                   for name in ('normalized_transcoders.py','transcoders.py','esopt.py')}}
    request_fp = fingerprint(request)
    output = Path(output_directory).resolve(); manifest = output/'transcoder_manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get("snar_request_fingerprint") != request_fp:
            raise ValueError("Existing SnAr bank was fitted with different sources/configuration")
        validate_transcoder_bank(manifest, model_key="qwen35_4b", checkpoint_hash=policy.checkpoint_hash,
                                  policy_runtime=policy.runtime_precision_record(), policy_configuration_fingerprint=config_hash)
        return manifest
    if output.exists() and any(output.iterdir()):
        raise ValueError("Partial/existing SnAr training output requires reconciliation; no blind replay")
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive administrative claim precedes any fitting; this is not a query.
    with (output/'training_claim.json').open('x') as stream:
        json.dump({"request_fingerprint": request_fp, "request": request, "scientific_oracle_calls": 0}, stream, indent=2)
    started = time.time()
    bank = {"schema": BANK_SCHEMA, "status": "running", "complete": False, "ready_for_graphs": False,
            "model_key": "qwen35_4b", "model_id": policy.model_id, "checkpoint_hash": policy.checkpoint_hash,
            "policy_runtime": policy.runtime_precision_record(), "policy_configuration": configuration,
            "policy_configuration_fingerprint": config_hash, "expected_layer_paths": list(QWEN_MLP_PATHS),
            "expected_layers": 32, "completed_layers": 0, "passed_layers": 0,
            "configuration": {"epochs":64,"batch_size":batch_size,"learning_rate":learning_rate,"device":training_device,"max_dev_fvu":.5},
            "snar_request_fingerprint":request_fp,"snar_request":request,"snar_training_performed":True,
            "scientific_oracle_calls":0,"test_used_for_training_or_fidelity":False,"layers":[],"started_at":started}
    population = None
    for index,(path,config) in enumerate(layer_configs):
        source = validate_shard_splits(train_by_layer[path], dev_by_layer[path], policy_fingerprint=policy.checkpoint_hash,
                                      layer_path=path, config=config)
        identity = {"rows":source['rows'],"groups":source['groups'],"prefixes":{split:sorted({p for s in source['shards'] if s['split']==split for p in s['prefixes']}) for split in ('train','dev')}}
        if population is not None and identity != population:
            raise ValueError("Independent SnAr bank cannot mix different layer populations")
        population = identity
        checkpoint = output/f'layer_{index:02d}.pt'
        layer_start = time.time()
        trainer=train_normalized_layer; worker_options={}
        if training_backend=="isolated_cuda_2gib":
            from .tc_worker import run_layer_subprocess
            trainer=run_layer_subprocess
            worker_options={"worker_directory":output.parent/(output.name+"_workers")/f"layer{index:02d}",
                            "expected_gpu_uuid":expected_gpu_uuid}
        trained,metadata = trainer(config,train_by_layer[path],dev_by_layer[path],
            policy_fingerprint=policy.checkpoint_hash,layer_path=path,output_path=checkpoint,seed=seed+index,
            epochs=64,batch_size=batch_size,learning_rate=learning_rate,device=training_device,max_dev_fvu=.5,
            cpu_threads=2,gpu_memory_bytes=2*1024**3,registration=dict(registration),**worker_options)
        del trained
        if metadata['source_files'] != inputs[path]:
            raise ValueError("SnAr training inputs changed after the request was sealed")
        bank['layers'].append({'layer_index':index,'layer_path':path,'checkpoint':str(checkpoint),
            'checkpoint_sha256':file_sha256(checkpoint),'metadata_sha256':file_sha256(str(checkpoint)+'.json'),
            'metadata':metadata,'transcoder_hash':metadata['transcoder_hash'],
            'status':'succeeded' if metadata['fidelity_gate_passed'] else 'failed_fidelity',
            'elapsed_seconds':time.time()-layer_start})
        bank['completed_layers']=len(bank['layers']);bank['passed_layers']=sum(r['status']=='succeeded' for r in bank['layers'])
        bank['bank_fingerprint']=fingerprint({k:v for k,v in bank.items() if k!='bank_fingerprint'})
        write_json_atomic(manifest,bank)
    bank.update(complete=True,ready_for_graphs=bank['passed_layers']==32,
                status='succeeded' if bank['passed_layers']==32 else 'failed_fidelity',
                finished_at=time.time(),elapsed_seconds=time.time()-started)
    bank['bank_fingerprint']=fingerprint({k:v for k,v in bank.items() if k!='bank_fingerprint'})
    write_json_atomic(manifest,bank)
    if bank['ready_for_graphs']:
        validate_transcoder_bank(manifest,model_key='qwen35_4b',checkpoint_hash=policy.checkpoint_hash,
            policy_runtime=policy.runtime_precision_record(),policy_configuration_fingerprint=config_hash)
    return manifest
