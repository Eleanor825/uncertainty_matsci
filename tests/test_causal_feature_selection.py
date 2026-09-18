"""Official-HF tiny CPU contracts, not pretrained fidelity or materials results."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from matdiscovery.accounting import fingerprint, file_sha256, write_json_atomic
from matdiscovery.action_targets import ActionTargets
from matdiscovery.esopt import tensor_state_hash
from matdiscovery.native_attribution import NativeAttributor, PolicyStamp, TranscoderBinding, UnsupportedAttribution
from matdiscovery.representation_training import GraphStageConfig, QWEN_MLP_PATHS, RepresentationError
from matdiscovery.transcoders import TopKTranscoder, TranscoderConfig
from torch_runtime_fixture import restore_torch_runtime

RULE = "qwen_final_mlp_causal_activation_v1"


@pytest.fixture(autouse=True)
def one_cpu_thread(restore_torch_runtime):
    torch.set_num_threads(1)


@pytest.fixture
def official_case():
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=16, intermediate_size=24,
        num_hidden_layers=32, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 8,
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]},
        attention_dropout=0., max_position_embeddings=32)
    vision = Qwen3_5VisionConfig(depth=1, hidden_size=16, intermediate_size=24, num_heads=2,
                               out_hidden_size=16, num_position_embeddings=16)
    full = Qwen3_5Config(text_config=config, vision_config=vision,
                        image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    full._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(371)
        policy = Qwen3_5ForConditionalGeneration(full).eval().cpu()
    ids = torch.tensor([[3, 4, 5]])
    captures, handles = {}, []
    for path in QWEN_MLP_PATHS:
        def hook(module, args, output, path=path):
            captures[path] = (args[0][0].detach().clone(), output[0].detach().clone())
        handles.append(policy.get_submodule(path).register_forward_hook(hook))
    try:
        with torch.no_grad(): policy(input_ids=ids, use_cache=False)
    finally:
        for handle in handles: handle.remove()
    bindings = []
    for layer, path in enumerate(QWEN_MLP_PATHS):
        x, y = captures[path]
        tc = TopKTranscoder(TranscoderConfig(16, 16, 64, 64), seed=layer)
        # A finite-fixture interpolant; no learned/global fidelity is claimed.
        matrix = torch.linalg.lstsq(x.double(), y.double()).solution.T.float()
        norms = matrix.norm(dim=0).clamp_min(1e-10)
        with torch.no_grad():
            tc.encoder.weight.zero_(); tc.encoder.bias.zero_()
            tc.decoder.weight.zero_(); tc.decoder.bias.zero_()
            tc.encoder.weight[:16].copy_(torch.diag(norms))
            tc.encoder.weight[16:32].copy_(-torch.diag(norms))
            tc.decoder.weight[:, :16].copy_(matrix / norms)
            tc.decoder.weight[:, 16:32].copy_(-matrix / norms)
            # Equal-and-opposite decoder pairs cancel in reconstruction. Their
            # large activation at position0 reproduces the real selection bug.
            if layer == 31:
                weight = torch.linalg.lstsq(x.double(), torch.tensor([10., 1., 20.], dtype=torch.float64)).solution.float()
                direction = torch.arange(1, 17).float(); direction /= direction.norm()
                for feature in range(32, 64, 2):
                    tc.encoder.weight[feature:feature + 2].copy_(weight.expand(2, -1))
                    tc.decoder.weight[:, feature].copy_(direction)
                    tc.decoder.weight[:, feature + 1].copy_(-direction)
        bindings.append(TranscoderBinding(path, tc, {"fidelity_gate_passed": True, "max_dev_fvu": .5,
            "transcoder_hash": tc.checkpoint_hash(), "unit_only": "finite fixture interpolant, not fitted research bank",
            "provenance": {"layer_path": path}}))
    stamp = PolicyStamp("tiny-unit-state", tensor_state_hash(policy.state_dict()), "official-tiny-unit-only", 0)
    review = {"verified": True, "source": "official transformers 5.17.0 Qwen3_5 source; random tiny CPU fixture",
              "model_type": "qwen3_5", "layers": 32, "mlp_paths": list(QWEN_MLP_PATHS)}
    target = ActionTargets("crystalgym", 1, (2,), (1,), (5,), ("unit",), ("/action",), (),
        tensor_state_hash({"input_ids": ids, "attention_mask": torch.ones_like(ids)}), 3, "", "unit-token-fixture")
    target = replace(target, target_spec_hash=fingerprint(target.specification()))
    return policy, bindings, stamp, review, ids, target


def attributor(case, rule="global_activation", **kwargs):
    policy, bindings, stamp, review, _, _ = case
    return NativeAttributor(policy, bindings, state_id_getter=lambda: stamp.state_id, architecture_review=review,
        source_selection_rule=rule, max_nodes=512, max_feature_nodes=32, max_backward_targets=42, **kwargs)


def test_unreachable_last_layer_activation_old_fails_new_passes_original_fd(official_case):
    _, _, stamp, _, ids, target = official_case
    old = attributor(official_case)
    with pytest.raises(UnsupportedAttribution, match="no nonzero feature edge"):
        old.validate_backend(ids, stamp, action_targets=target)
    new = attributor(official_case, RULE)
    report = new.validate_backend(ids, stamp, action_targets=target)
    assert report["passed"] and (report["epsilon"], report["rtol"], report["atol"]) == (.001, .05, .0001)
    assert report["required_direct_action_sink_source_kinds"] == ["token", "feature"]
    assert {check["source"]["kind"] for check in report["checks"]} >= {"token", "feature"}
    assert all(check["source"]["position"] == 1 for check in report["checks"] if check["source"]["kind"] == "feature")
    assert new._backend_identity != old._backend_identity and report["source_selection_rule"] == RULE
    assert report["source_selection_proof"]["final_mlp_path"] == QWEN_MLP_PATHS[-1]


def test_production_and_preflight_share_filter_full_prefix_and_original_coverage(official_case):
    _, _, stamp, _, ids, target = official_case
    old, new = attributor(official_case), attributor(official_case, RULE)
    trace = new.capture(ids, stamp, action_targets=target)
    original = old._assemble(trace)
    assert all(node["layer"] == 31 and node["position"] == 0 for node in original.node_records if node["kind"] == "feature")
    last_gradient = torch.autograd.grad(target.mean_logprob(trace.logits), trace.mlp_leaves[QWEN_MLP_PATHS[-1]], retain_graph=True)[0]
    assert torch.count_nonzero(last_gradient[:, (0, 2)]) == 0 and torch.count_nonzero(last_gradient[:, 1]) > 0
    old_source = next(i for i, node in enumerate(original.node_records) if node["kind"] == "feature")
    old_sink = next(i for i, node in enumerate(original.node_records) if node["kind"] == "score")
    assert old.intervention_value(trace, original, old_source, old_sink, .001) == old.intervention_value(trace, original, old_source, old_sink, -.001)
    excluded = trace.activations[QWEN_MLP_PATHS[-1]][0, 0]
    for cap in (8, 32):
        result = new._assemble(trace, feature_cap=cap)
        features = [node for node in result.node_records if node["kind"] == "feature"]
        assert len(features) == cap and all(node["position"] == 1 for node in features if node["layer"] == 31)
        assert result.metadata["full_prefix_tokens"] == 3 and result.metadata["excluded_future_positions"] == 1
        assert result.metadata["backward_targets"] == cap + 1 <= 42
        assert result.metadata["total_active_features"] == original.metadata["total_active_features"]
        assert result.metadata["feature_count_coverage"] == cap / original.metadata["total_active_features"]
        mass = sum(float(value[0, :2].sum()) for value in trace.activations.values())
        assert result.metadata["activation_mass_coverage"] == pytest.approx(sum(node["activation"] for node in features) / mass)
        assert result.metadata["structurally_excluded_active_features"] == int((excluded > 0).sum()) > 0
        assert result.metadata["structurally_excluded_activation_mass"] == float(excluded.sum()) > 0
        assert result.metadata["eligible_active_features"] == original.metadata["total_active_features"] - int((excluded > 0).sum())
        assert result.metadata["future_source_gradient_max_abs"] == 0
        assert sum(node["kind"] in {"reconstruction_error", "omitted_features", "decoder_bias"} for node in result.node_records) == 96
        assert all(node.get("position", 0) <= 1 for node in result.node_records)
        sink = next(i for i, node in enumerate(result.node_records) if node["kind"] == "score")
        assert any(result.node_records[int(source)]["kind"] == "feature" and int(destination) == sink
                   for source, destination in zip(result.graph.sources, result.graph.targets))
    assert torch.equal(trace.input_ids, ids)


def test_legacy_uses_last_position_and_unknown_or_partial_architectures_rejected(official_case):
    _, bindings, stamp, _, ids, _ = official_case
    new = attributor(official_case, RULE, target_mode="after_step_next_token_v1", max_logits=1)
    result = new._assemble(new.capture(ids, stamp))
    assert all(node["position"] == 2 for node in result.node_records if node["kind"] == "feature" and node["layer"] == 31)
    with pytest.raises(UnsupportedAttribution, match="reviewed complete 32-layer"):
        attributor((torch.nn.Linear(2, 2).eval(), *official_case[1:]), RULE)
    with pytest.raises(UnsupportedAttribution, match="reviewed complete 32-layer"):
        attributor((official_case[0], bindings[:-1], *official_case[2:]), RULE)
    official_case[0].model.language_model.norm = torch.nn.LayerNorm(16)
    with pytest.raises(UnsupportedAttribution, match="reviewed complete 32-layer"):
        attributor(official_case, RULE)


def test_rule_validation_and_online_offline_factories_forward_same_opt_in(tmp_path, monkeypatch):
    from matdiscovery import native_attribution as native, representation_training as offline, training_jobs as online
    from matdiscovery import policy as policy_module, transcoders, uncertainty
    assert GraphStageConfig().source_selection_rule == "global_activation"
    with pytest.raises(RepresentationError, match="source-selection"):
        GraphStageConfig(source_selection_rule="unknown").validate()
    seen = []
    class ReachedConstructor(Exception): pass
    def constructor(*args, **kwargs):
        seen.append(kwargs["source_selection_rule"]); raise ReachedConstructor
    monkeypatch.setattr(native, "NativeAttributor", constructor)
    runtime = {"torch_cpu_threads": 1}
    stamp = PolicyStamp("unit", "base", "unit-model", 0)
    policy = SimpleNamespace(model=object(), model_stamp=stamp, checkpoint_hash="base", model_id="unit-model",
        mlp_paths=QWEN_MLP_PATHS, architecture_review={"unit_only": True}, configuration_fingerprint="cfg",
        get_state_id=lambda: "unit", runtime_precision_record=lambda: runtime)
    monkeypatch.setattr(policy_module.QwenPolicyAdapter, "from_verified_checkpoint", lambda *a, **kw: policy)
    tc = TopKTranscoder(TranscoderConfig(2, 2, 2, 2))
    metadata = {"unit_only": True}
    monkeypatch.setattr(transcoders, "load_transcoder", lambda *a, **kw: (tc, metadata))
    source_manifest = tmp_path / "source.json"; source_manifest.write_text("{}")
    bank = {"layers": [{"checkpoint": str(tmp_path / "tc.pt"), "layer_path": QWEN_MLP_PATHS[0],
                       "metadata": metadata, "transcoder_hash": tc.checkpoint_hash()}],
            "collections": [{"path": str(source_manifest), "sha256": file_sha256(source_manifest), "job_id": "collection-made-unit"}],
            "test_used_for_training_or_fidelity": False}
    monkeypatch.setattr(offline, "validate_transcoder_bank", lambda *a, **kw: bank)
    monkeypatch.setattr(offline, "read_collections", lambda *a, **kw: {
        "checkpoint_hash": "base", "model_id": "unit-model", "policy_runtime": runtime,
        "policy_configuration_fingerprint": "cfg"})
    checkpoint = tmp_path / "graph_risk.pt"; checkpoint.write_text("unit loader boundary")
    schema = {"feature_names": ["unit-feature"]}
    write_json_atomic(checkpoint.with_suffix(".fit.json"), {"status": "succeeded", "method": "graph_risk", "checkpoint_sha256": file_sha256(checkpoint)})
    write_json_atomic(checkpoint.with_suffix(".schema.json"), schema)
    collection = {"model_key": "qwen35_4b", "benchmarks": ["made"], "policy_runtime": runtime,
        "policy_configuration_fingerprint": "cfg", "checkpoint_hash": "base", "test_used_for_fit": False,
        "test_used_for_threshold_selection": False, "label_kind": "future_failure"}
    risk = SimpleNamespace(provenance={"method": "graph_risk", "collection_provenance": collection,
        "test_used_for_fit": False, "feature_schema": schema}, features=SimpleNamespace(names=["unit-feature"]), model=torch.nn.Linear(1, 1))
    monkeypatch.setattr(uncertainty.CalibratedRiskModel, "load", lambda *a, **kw: risk)
    for rule in ("global_activation", RULE):
        config = GraphStageConfig(device="cpu", source_selection_rule=rule)
        with pytest.raises(ReachedConstructor):
            offline.generate_graph_features([source_manifest], model_key="qwen35_4b", model_manifest=tmp_path / "models.json",
                checkpoint_dir=tmp_path, transcoder_manifest=tmp_path / "bank.json", report_path=tmp_path / "report.json", config=config)
        with pytest.raises(ReachedConstructor):
            online.load_frozen_controllers(policy, model_key="qwen35_4b", benchmark="made", method="esopt_graph_risk",
                risk_checkpoint=checkpoint, transcoder_manifest=tmp_path / "bank.json", graph_config=asdict(config), device="cpu")
        assert seen[-2:] == [rule, rule]


def test_graph_cli_explicit_rule_and_legacy_default_without_model(tmp_path):
    import runpy
    from pathlib import Path
    main = runpy.run_path(str(Path(__file__).parents[1] / "scripts/train_representations.py"))["main"]
    observed = []
    main.__globals__["generate_graph_features"] = lambda *a, **kw: (observed.append(kw["config"].source_selection_rule) or {"status": "succeeded"})
    args = ["graph-features", "--manifest", "unit.json", "--model-key", "qwen35_4b", "--model-manifest", "unit-model.json",
            "--checkpoint-dir", "unit", "--transcoder-manifest", "unit-bank.json", "--report", str(tmp_path / "never-written.json")]
    assert main(args) == 0 and observed[-1] == "global_activation"
    assert main(args + ["--source-selection-rule", RULE]) == 0 and observed[-1] == RULE
    assert not (tmp_path / "never-written.json").exists()
