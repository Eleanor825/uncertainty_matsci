"""CPU contracts only. Tiny networks never substitute for real SnAr evidence."""
import ast
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("snar_fvu_audit", Path(__file__).with_name("audit.py"))
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
import matdiscovery.native_attribution as native
from matdiscovery.esopt import tensor_state_hash
from matdiscovery.transcoders import TopKTranscoder, TranscoderConfig, _load_shard


def diagnostic():
    source, proof = audit.transform_capture(inspect.getsource(native.NativeAttributor.capture))
    namespace = dict(vars(native))
    exec(compile(source, "unit_diagnostic_capture", "exec"), namespace)
    return namespace["capture"], source, proof


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.mlp = torch.nn.Linear(1, 1, bias=False)
        self.mlp.weight.data.fill_(.02)
    def forward(self, x):
        return x + self.mlp(x)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="cpu_fixture", _attn_implementation="eager")
        self.embed_tokens = torch.nn.Embedding(4, 1)
        self.embed_tokens.weight.data.copy_(torch.arange(1., 5.)[:, None])
        self.layers = torch.nn.ModuleList(Layer() for _ in range(32))
        self.lm_head = torch.nn.Linear(1, 2, bias=False)
        self.lm_head.weight.data.copy_(torch.tensor([[1.], [-1.]]))
        self.eval()
    def get_input_embeddings(self): return self.embed_tokens
    def get_output_embeddings(self): return self.lm_head
    def forward(self, input_ids, **kwargs):
        assert kwargs["use_cache"] is False
        x = self.embed_tokens(input_ids)
        for layer in self.layers: x = layer(x)
        return SimpleNamespace(logits=self.lm_head(x))


def fixture():
    model = Tiny(); before = tensor_state_hash(dict(model.state_dict()))
    stamp = native.PolicyStamp("cpu-only", before, "tiny_fixture", 0)
    bindings = []
    for i in range(32):
        tc = TopKTranscoder(TranscoderConfig(1, 1, 1, 1))
        with torch.no_grad():
            tc.encoder.weight.fill_(.02); tc.encoder.bias.zero_()
            tc.decoder.weight.fill_(1); tc.decoder.bias.zero_()
        path = f"layers.{i}.mlp"
        bindings.append(native.TranscoderBinding(path, tc, {"fidelity_gate_passed": True,
            "max_dev_fvu": .5, "provenance": {"layer_path": path}, "fixture_only": True}))
    attr = native.NativeAttributor(model, bindings, state_id_getter=lambda: stamp.state_id,
        architecture_review={"verified": True, "source": "tiny CPU fixture only"},
        target_mode="after_step_next_token_v1", constant_storage_device="cpu")
    ids = torch.tensor([[0, 1, 2, 3]*40])
    return model, attr, bindings, stamp, ids


def test_only_one_ast_raise_changes_no_threshold_or_arithmetic():
    _, source, proof = diagnostic()
    old = ast.parse(__import__('textwrap').dedent(inspect.getsource(native.NativeAttributor.capture)))
    new = ast.parse(source)
    old_guard = next(n for n in ast.walk(old) if isinstance(n, ast.If) and "fvu_defined" in ast.unparse(n.test))
    assert isinstance(old_guard.body[0], ast.Raise)
    old_guard.body = [ast.Pass()]
    assert ast.dump(old) == ast.dump(new)
    assert proof["changed_ast_nodes"] == 1 and proof["threshold_changed"] is False
    bad = inspect.getsource(native.NativeAttributor.capture).replace("fvu > threshold", "fvu > 1000")
    with pytest.raises(ValueError, match="guard changed"): audit.transform_capture(bad)


def test_original_vs_diagnostic_exact_32_layers_and_weights_unchanged():
    model, attr, bindings, stamp, ids = fixture()
    before = tensor_state_hash(dict(model.state_dict()))
    reference = attr.capture(ids, stamp)
    capture, _, _ = diagnostic(); actual = capture(attr, ids, stamp)
    assert actual.fidelity == reference.fidelity
    for path in actual.fidelity:
        assert torch.equal(actual.mlp_inputs[path], reference.mlp_inputs[path])
        assert torch.equal(actual.mlp_leaves[path], reference.mlp_leaves[path])
        assert torch.equal(actual.activations[path], reference.activations[path])
    assert tensor_state_hash(dict(model.state_dict())) == before
    assert len(audit.metrics_rows(actual.fidelity, bindings)) == 32


def test_first_and_last_layer_failures_both_recorded_no_native_admission():
    model, attr, bindings, stamp, ids = fixture()
    with torch.no_grad():
        bindings[0].transcoder.decoder.weight.mul_(10)
        bindings[-1].transcoder.decoder.weight.mul_(20)
    with pytest.raises(native.UnsupportedAttribution, match="fidelity gate"):
        attr.capture(ids, stamp)
    capture, _, _ = diagnostic(); actual = capture(attr, ids, stamp)
    rows = audit.metrics_rows(actual.fidelity, bindings)
    assert [r["layer_index"] for r in rows if not r["passed"]] == [0, 31]
    assert attr._validation is None
    with pytest.raises(native.UnsupportedAttribution, match="validate_backend"):
        attr.attribute(ids, stamp)
    with pytest.raises(native.UnsupportedAttribution, match="cache"):
        capture(attr, ids, stamp, model_kwargs={"use_cache": True})
    with pytest.raises(native.UnsupportedAttribution, match="version changed"):
        capture(attr, ids, native.PolicyStamp("other", stamp.checkpoint_hash, stamp.model_id, 0))


def test_uniform_rows_are_within_original_causal_horizon():
    positions = audit.causal_positions(1000, 899)
    assert len(positions) == 128 and positions[0] == 0 and positions[-1] == 899
    assert torch.equal(positions, torch.linspace(0, 899, 128).long())
    with pytest.raises(ValueError): audit.causal_positions(1000, 1000)
    with pytest.raises(ValueError): audit.causal_positions(1000, 899, 16)


def test_saved_real_tiny_shards_keep_split_tokens_hash_and_no_old_writes(tmp_path):
    model, attr, bindings, stamp, ids = fixture()
    capture, _, _ = diagnostic(); trace = capture(attr, ids, stamp)
    targets = SimpleNamespace(causal_horizon=149, source_prefix_hash=trace.prefix_hash,
                             to_dict=lambda: {"causal_horizon": 149, "source_prefix_hash": trace.prefix_hash})
    old = tmp_path/"old.json"; old.write_text('{"immutable":true}')
    identity = (audit.sha(old), old.stat().st_mtime_ns)
    row = {"index": 0, "episode": "collection_dev_2101", "split": "dev", "query_id": "collection_dev_2101/q005",
           "prefix_tokens": ids.shape[1], "generation": audit.artifact(old)}
    reg = {"output": str(tmp_path/"new"), "fingerprint": "cpu-unit-only"}
    receipt, ref = audit.save_prefix(reg, row, trace, targets, bindings, {"scientific_oracle_calls": 0})
    assert receipt["complete"] and receipt["not_a_native_graph_admission"]
    assert receipt["native_graph_available"] is None and receipt["finite_difference_passed"] is None
    assert len(receipt["shards"]) == 32
    for item in receipt["shards"]:
        shard = _load_shard(item["shard"]["path"])
        assert shard["metadata"]["split"] == "dev"
        assert shard["metadata"]["rows"] == 128
        assert set(shard["metadata"]["group_ids"]) == {"collection_dev_2101"}
        assert set(shard["metadata"]["prefix_hashes"]) == {trace.prefix_hash}
        assert torch.equal(shard["outputs"], trace.mlp_leaves[item["layer_path"]][0, receipt["selected_token_positions"]])
    assert (audit.sha(old), old.stat().st_mtime_ns) == identity
    with pytest.raises(FileExistsError): audit.save_prefix(reg, row, trace, targets, bindings, {})
    Path(receipt["shards"][0]["shard"]["path"]).write_bytes(b"tamper")
    with pytest.raises(ValueError, match="changed"): audit.check(receipt["shards"][0]["shard"])


def test_matrix_rejects_missing_layer_nonfinite_and_relaxed_threshold():
    model, attr, bindings, stamp, ids = fixture()
    trace = attr.capture(ids, stamp)
    first = bindings[0].module_path
    with pytest.raises(ValueError): audit.metrics_rows({k:v for k,v in trace.fidelity.items() if k != first}, bindings)
    trace.fidelity[first]["output_fvu"] = float('nan')
    with pytest.raises(ValueError, match="Nonfinite"): audit.metrics_rows(trace.fidelity, bindings)
    trace.fidelity[first]["output_fvu"] = 0.
    bindings[0].training_metadata["max_dev_fvu"] = .6
    with pytest.raises(ValueError, match="changed gate"): audit.metrics_rows(trace.fidelity, bindings)


def test_no_generate_or_oracle_or_fit_calls_in_audit_ast():
    tree = ast.parse(Path(audit.__file__).read_text())
    names = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not names & {"generate", "generate_action", "evaluate", "fit", "backward", "step", "attribute", "validate_backend"}


def make_catalog(tmp_path):
    out = tmp_path/"original"; out.mkdir()
    base = {"base_state_hash": "unit_weights", "checkpoint_hash": "unit_checkpoint",
            "configuration_fingerprint": "unit_config", "runtime": {"unit": True}}
    protocol = {"risk": {"train_seeds": [1101,1102,1103], "dev_seeds": [2101,2102]}}
    for split, seeds in audit.EPISODES.items():
        for seed in seeds:
            name = f"collection_{split}_{seed}"; d = out/"episodes"/name; d.mkdir(parents=True)
            audit.publish(d/"episode.json", {"name": name,"arm":"qwen_base","seed":seed,"budget":30,
                "weight_hash":"unit_weights","protocol_fingerprint":audit.digest(protocol)})
            audit.publish(d/"summary.json", {"unit": True})
            for i in range(30):
                q={"query_id":name+f"/q{i:03d}","chosen_generation":None}
                if i>=5:
                    gd=d/f"query{i:03d}_proposal0"; gd.mkdir()
                    ids=torch.tensor([[seed]+list(range(i,i+139))])
                    g={"input_ids_with_completion":ids.tolist(),"success":True,"model_stamp":{
                        "generation":0,"perturbation_seed":None,"perturbation_sigma":None,"checkpoint_hash":"unit_checkpoint"},
                        "configuration_fingerprint":"unit_config","policy_runtime":{"unit":True},
                        "prefix_hash":tensor_state_hash({"input_ids":ids}),"parsed_action":{"unit":i}}
                    audit.publish(gd/"generation.json",{"weight_hash":"unit_weights","generation":g})
                    audit.publish(gd/"risk_graph.json",{"available":False})
                    q.update(chosen_generation=str(gd/"generation.json"),parameters={"unit":i},no_hvi=i%2)
                audit.publish(d/f"query{i:03d}.json",q)
    return out, protocol, base


def test_catalog_exact125_train75_dev50_readonly_and_original_ids(tmp_path):
    out, protocol, base=make_catalog(tmp_path)
    before={str(p):(audit.sha(p),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    rows,evidence=audit.catalog(out,out,protocol,base)
    assert len(rows)==125 and sum(r['split']=='train' for r in rows)==75
    assert {r['episode'] for r in rows if r['split']=='dev'}=={'collection_dev_2101','collection_dev_2102'}
    assert all(r['query_id'].endswith(tuple(f'q{i:03d}' for i in range(5,30))) for r in rows)
    assert before=={str(p):(audit.sha(p),p.stat().st_mtime_ns) for p in out.rglob('*') if p.is_file()}
    p=Path(rows[0]['generation']['path']); v=audit.read(p);v['generation']['input_ids_with_completion'][0][0]+=1
    p.write_text(json.dumps(v))
    with pytest.raises(ValueError,match='token hash'):audit.catalog(out,out,protocol,base)


def test_catalog_rejects_test_rows_missing_queries_or_other_policy(tmp_path):
    out, protocol, base=make_catalog(tmp_path)
    (out/'episodes'/'test_qwen_base_5101').mkdir()
    with pytest.raises(ValueError,match='five original'):audit.catalog(out,out,protocol,base)
    (out/'episodes'/'test_qwen_base_5101').rmdir()
    p=out/'episodes'/'collection_train_1101'/'query029.json';p.unlink()
    with pytest.raises(ValueError,match='exactly 30'):audit.catalog(out,out,protocol,base)
