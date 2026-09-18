"""CPU contract tests with synthetic weights, never scientific model results."""
from dataclasses import replace
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from matdiscovery.policy import (
    DecodingConfig, PolicyContextError, PolicyIntegrityError, PolicyStateError,
    QwenPolicyAdapter, SUPPORTED_CHECKPOINTS, parse_last_json, verify_checkpoint,
)


class TinyMLP(nn.Module):
    def __init__(self, width=4, intermediate=8):
        super().__init__()
        self.gate_proj = nn.Linear(width, intermediate, bias=False)
        self.up_proj = nn.Linear(width, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, width, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLayer(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.mlp = TinyMLP()
        setattr(self, "self_attn" if index % 4 == 3 else "linear_attn", nn.Identity())

    def forward(self, x):
        return x + 0.05 * self.mlp(x)


class TinyTokenizer:
    chat_template = "synthetic chat template"

    def __init__(self):
        self.prompt_ids = [1, 2, 3]
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        ids = torch.tensor([self.prompt_ids], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, tokens, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "".join(chr(token) for token in tokens)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(256, 4)
        self.model.language_model.layers = nn.ModuleList([TinyLayer(i) for i in range(32)])
        self.model.visual = nn.Linear(3, 3)
        self.lm_head = nn.Linear(4, 256, bias=False)
        self.config = SimpleNamespace(
            model_type="qwen3_5", _attn_implementation="eager",
            text_config=SimpleNamespace(num_hidden_layers=32, hidden_size=4, intermediate_size=8,
                                        max_position_embeddings=1024,
                                        layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(32)]),
        )
        self.generation_config = SimpleNamespace(eos_token_id=255, pad_token_id=0)
        self.response = '{"action": 2, "rationale": "check"}'
        self.forward_calls = []
        self.generate_calls = []
        self.fail_parity = False
        self.duplicate_mlp = False
        self.bad_prefix = False

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=0):
        self.forward_calls.append({"ids": input_ids.clone(), "mask": attention_mask.clone(), "use_cache": use_cache})
        hidden = self.get_input_embeddings()(input_ids)
        for layer in self.model.language_model.layers:
            hidden = layer(hidden)
        if self.duplicate_mlp:
            self.model.language_model.layers[0].mlp(hidden)
        logits = self.lm_head(hidden)
        if self.fail_parity and self.model.language_model.layers[0].mlp._forward_hooks:
            logits = logits + 1
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)

    def generate(self, input_ids, attention_mask, **kwargs):
        self.generate_calls.append({"ids": input_ids.clone(), "attention_mask": attention_mask.clone(), **kwargs})
        completion = torch.tensor([[ord(character) for character in self.response]], device=input_ids.device)
        sequence = torch.cat([input_ids, completion], dim=1)
        if self.bad_prefix:
            sequence[0, 0] += 1
        scores = tuple(torch.randn(1, 256, device=input_ids.device) for _ in range(completion.shape[1]))
        return SimpleNamespace(sequences=sequence, scores=scores)


@pytest.fixture
def assets(tmp_path):
    """The IDs identify the loader contract; these bytes are tiny test fixtures."""
    model_id = "Qwen/Qwen3.5-4B"
    revision = SUPPORTED_CHECKPOINTS[model_id]
    payloads = {"model.safetensors": b"synthetic weights for CPU contract test only",
                "config.json": b'{"test_fixture":true}', "chat_template.jinja": b"synthetic chat template"}
    files = []
    records = []
    for name, content in payloads.items():
        (tmp_path / name).write_bytes(content)
        item = {"path": name, "size_bytes": len(content)}
        if name == "chat_template.jinja":
            item["git_blob_sha1"] = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        else:
            item["sha256"] = hashlib.sha256(content).hexdigest()
        files.append(item)
        records.append({"path": name, "status": "downloaded_verified", "hash": item.get("sha256", item.get("git_blob_sha1"))})
    entry = {"key": "qwen35_4b", "model_id": model_id, "revision": revision,
             "model_class": "Qwen3_5ForConditionalGeneration", "metadata_integrity_status": "fixed_revision_official_hashes_verified",
             "weight_files": files[:1], "metadata_files": files[1:],
             "architecture": {"num_hidden_layers": 32, "hidden_size": 4}, "safetensors_parameter_count": 123}
    manifest = {"models": [entry]}
    receipt = {"status": "complete_verified", "scope": "complete_checkpoint", "model_id": model_id,
               "revision": revision, "files": records}
    (tmp_path / "download_receipt.json").write_text(json.dumps(receipt))
    return manifest, tmp_path, receipt


@pytest.fixture
def adapter(assets):
    manifest, directory, _ = assets
    checkpoint = verify_checkpoint(manifest, "qwen35_4b", directory)
    return QwenPolicyAdapter(TinyPolicy(), TinyTokenizer(), checkpoint,
                             decoding=DecodingConfig(max_new_tokens=80), max_input_tokens=100)


def test_receipt_and_file_content_required_before_loading(assets):
    manifest, directory, receipt = assets
    original = verify_checkpoint(manifest, "qwen35_4b", directory)
    assert original.checkpoint_hash == verify_checkpoint(manifest, "qwen35_4b", directory).checkpoint_hash
    receipt["status"] = "metadata_verified"
    (directory / "download_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(PolicyIntegrityError, match="complete checkpoint"):
        verify_checkpoint(manifest, "qwen35_4b", directory)
    receipt["status"] = "complete_verified"
    receipt["revision"] = "0" * 40
    (directory / "download_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(PolicyIntegrityError, match="exact revision"):
        verify_checkpoint(manifest, "qwen35_4b", directory)
    receipt["revision"] = manifest["models"][0]["revision"]
    (directory / "download_receipt.json").write_text(json.dumps(receipt))
    weights = directory / "model.safetensors"
    weights.write_bytes(b"x" * weights.stat().st_size)
    with pytest.raises(PolicyIntegrityError, match="content changed"):
        verify_checkpoint(manifest, "qwen35_4b", directory)


def test_native_loader_is_local_bf16_and_preserves_lm_head(assets, monkeypatch):
    manifest, directory, _ = assets
    calls = []
    native_class = type("Qwen3_5ForConditionalGeneration", (TinyPolicy,), {})
    def tokenizer_loader(path, **kwargs):
        calls.append(("tokenizer", path, kwargs))
        return TinyTokenizer()
    def model_loader(path, **kwargs):
        calls.append(("model", path, kwargs))
        return native_class().to(dtype=kwargs["dtype"])
    fake = SimpleNamespace(__version__="5.17.0", AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
                           AutoModelForImageTextToText=SimpleNamespace(from_pretrained=model_loader))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    policy = QwenPolicyAdapter.from_verified_checkpoint(manifest, "qwen35_4b", directory, device="cpu")
    assert calls[1][2]["dtype"] is torch.bfloat16
    assert all(call[2]["local_files_only"] and call[2]["trust_remote_code"] is False for call in calls)
    assert policy.model.get_output_embeddings() is policy.model.lm_head
    assert policy.model.get_input_embeddings().weight.dtype is torch.bfloat16
    assert len(policy.mlp_paths) == 32


def test_explicit_fp32_cpu_vocabulary_loader_keeps_tied_parameters_and_gradients(assets, monkeypatch):
    manifest, directory, _ = assets
    native_class = type("Qwen3_5ForConditionalGeneration", (TinyPolicy,), {})
    calls = []
    def load(path, **kwargs):
        calls.append(kwargs)
        model = native_class().to(dtype=kwargs["dtype"])
        model.lm_head.weight = model.get_input_embeddings().weight
        model.config._attn_implementation = kwargs["attn_implementation"]
        return model
    fake = SimpleNamespace(__version__="5.17.0", AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: TinyTokenizer()),
                           AutoModelForImageTextToText=SimpleNamespace(from_pretrained=load))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    policy = QwenPolicyAdapter.from_verified_checkpoint(manifest, "qwen35_4b", directory, dtype="float32", device="cpu",
                                                        cpu_embedding_and_lm_head=True)
    assert calls[0]["dtype"] is torch.float32
    assert policy.model.lm_head.weight is policy.model.get_input_embeddings().weight
    runtime = policy.runtime_precision_record()
    assert runtime["cpu_embedding_and_lm_head"] and runtime["embedding_head_tied"]
    assert runtime["dtype"] == "torch.float32" and runtime["all_parameters_materialized"]
    assert runtime["float32_matmul_precision"] == "highest" and not runtime["cuda_matmul_allow_tf32"] and not runtime["cudnn_allow_tf32"]
    assert sum(v["parameters"] for v in runtime["materialized_parameter_inventory"].values()) == sum(p.numel() for p in policy.model.parameters())
    ids, kwargs = policy.prepare_trace_inputs(torch.tensor([[1, 2, 3]]))
    logits = policy.model(input_ids=ids, **kwargs).logits
    gradients = torch.autograd.grad(logits.square().sum(), [policy.model.lm_head.weight, policy.model.get_submodule(policy.mlp_paths[0]).up_proj.weight])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    observation = policy.capture_prefix(ids)
    assert observation.parity_passed and observation.parity_max_abs == 0
    generated = policy.generate_action([{"role": "user", "content": "choose"}], seed=1)
    assert generated.policy_runtime == runtime and generated.to_record()["policy_runtime"] == runtime


def test_runtime_identity_excludes_transient_grad_flags_and_guard_catches_precision_change(adapter):
    before = adapter.runtime_precision_record()
    flags = [p.requires_grad for p in adapter.model.parameters()]
    adapter.model.requires_grad_(False)
    assert adapter.runtime_precision_record() == before
    for parameter, flag in zip(adapter.model.parameters(), flags):
        parameter.requires_grad_(flag)
    old = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("medium" if old != "medium" else "highest")
        with pytest.raises(PolicyStateError, match="changed"):
            adapter.generate_action([{"role": "user", "content": "choose"}], seed=1)
    finally:
        torch.set_float32_matmul_precision(old)


def test_explicit_execution_input_device_is_not_inferred_from_cpu_embedding(assets):
    manifest, directory, _ = assets
    policy = QwenPolicyAdapter(TinyPolicy(), TinyTokenizer(), verify_checkpoint(manifest, "qwen35_4b", directory),
                               cpu_embedding_and_lm_head=True, execution_device="cuda:0")
    # No CUDA allocation is performed by this CPU contract test. Real device
    # transport and backward are verified by the separate hardware diagnostic.
    assert policy.model.get_input_embeddings().weight.device.type == "cpu"
    assert policy._input_device() == torch.device("cuda:0")
    assert policy.runtime_precision_record()["execution_input_device"] == "cuda:0"


def test_declared_runtime_options_reject_silent_cli_precision_or_memory_changes():
    from matdiscovery.policy import declared_runtime_options
    protocol = {"policy_runtime": {"dtype": "float32", "float32_matmul_precision": "highest",
        "attn_implementation": "eager", "sdpa_backend": "auto", "cpu_embedding_and_lm_head": True,
        "attention_checkpointing": False,
        "device": "cuda:0", "cuda_memory_fraction": .44, "torch_cpu_threads": 2}}
    resolved = declared_runtime_options(protocol)
    assert resolved == protocol["policy_runtime"]
    with pytest.raises(PolicyIntegrityError, match="differs"):
        declared_runtime_options(protocol, cuda_memory_fraction=.40)
    with pytest.raises(PolicyIntegrityError, match="differs"):
        declared_runtime_options(protocol, device="cpu")
    protocol["policy_runtime"]["float32_matmul_precision"] = "high"
    with pytest.raises(PolicyIntegrityError, match="highest"):
        declared_runtime_options(protocol)


def test_generation_exact_tokens_scores_seed_and_no_silent_truncation(adapter):
    torch.manual_seed(413)
    rng_before = torch.random.get_rng_state().clone()
    first = adapter.generate_action([{"role": "user", "content": "Choose"}], seed=17)
    second = adapter.generate_action([{"role": "user", "content": "Choose"}], seed=17)
    assert first.success and first.parsed_action == {"action": 2, "rationale": "check"}
    assert first.raw_text == adapter.model.response
    assert first.logprobs == second.logprobs and first.entropy == second.entropy
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert first.prompt_token_count == 3
    assert first.completion_count == len(adapter.model.response) == len(first.logprobs)
    assert first.input_ids_with_completion.tolist() == [[1, 2, 3] + [ord(c) for c in adapter.model.response]]
    assert first.input_ids_with_completion.device.type == "cpu"
    assert first.score_semantics == "hf_processed_generation_distribution"
    _, render = adapter.tokenizer.calls[0]
    assert render["truncation"] is False and render["enable_thinking"] is False
    assert render["add_generation_prompt"] and render["tokenize"]
    assert adapter.model.generate_calls[0]["use_cache"] is True
    json.dumps(first.to_record(), allow_nan=False)


def test_format_and_legality_failures_do_not_invent_actions(adapter):
    adapter.model.response = "I cannot emit an action"
    result = adapter.generate_action([{"role": "user", "content": "Choose"}], seed=1)
    assert not result.success and result.failure_code == "invalid_json" and result.parsed_action is None
    assert result.raw_text == adapter.model.response and result.completion_count > 0
    adapter.model.response = '{"action": 10}'
    schema = {"type": "object", "required": ["action"], "additionalProperties": False,
              "properties": {"action": {"type": "integer", "minimum": 0, "maximum": 3}}}
    result = adapter.generate_action([{"role": "user", "content": "Choose"}], seed=1, legal_schema=schema)
    assert not result.success and result.failure_code == "illegal_action" and result.parsed_action == {"action": 10}
    adapter.model.response = '{"action": 2}'
    result = adapter.generate_action([{"role": "user", "content": "Choose"}], seed=1,
                                     action_validator=lambda _: "element unavailable")
    assert not result.success and result.error == "element unavailable"
    with pytest.raises(ValueError, match="Unsupported legal schema"):
        adapter.generate_action([{"role": "user", "content": "Choose"}], seed=1,
                                legal_schema={"$ref": "#/definitions/action"})


@pytest.mark.parametrize("raw,expected", [
    ('example {"action":1} final ```json\n{"action":2,"nested":{"x":3}}\n```', {"action": 2, "nested": {"x": 3}}),
    ('<think>{"action":0}</think>{"action":2}', {"action": 2}),
    ('{"value":"a } bracket [ inside string"}', {"value": "a } bracket [ inside string"}),
    ('2', 2),
])
def test_last_complete_json(raw, expected):
    assert parse_last_json(raw) == expected


@pytest.mark.parametrize("raw", ['<think>{"action":1}', '{"x":{"action":1}',
                                 '{"action":1,"action":2}', '{"action":NaN}', '{"action":1e999}'])
def test_incomplete_nested_thinking_or_non_json_numbers_fail(raw):
    with pytest.raises(ValueError):
        parse_last_json(raw)


def test_context_overflow_is_explicit_and_does_not_call_generate(adapter):
    adapter.tokenizer.prompt_ids = list(range(101))
    with pytest.raises(PolicyContextError, match="max_input_tokens"):
        adapter.generate_action([{"role": "user", "content": "long history"}], seed=1)
    assert not adapter.model.generate_calls
    with pytest.raises(ValueError, match="text-only"):
        adapter.generate_action([{"role": "user", "content": [{"type": "image"}]}], seed=1)


def test_every_state_transition_invalidates_prior_stamp(adapter):
    previous = adapter.model_stamp
    seen = {previous.state_id}
    for reason in ("perturb", "update", "restore", "reload"):
        current = adapter.mark_state(reason, generation=1, perturbation_seed=12 if reason == "perturb" else None,
                                     perturbation_sigma=0.01 if reason == "perturb" else None)
        assert current.state_id not in seen and current.checkpoint_hash == previous.checkpoint_hash
        seen.add(current.state_id)
        with pytest.raises(PolicyStateError, match="stale"):
            adapter.capture_prefix(torch.tensor([[1, 2]]), stamp=previous)
        previous = current
    with torch.no_grad():
        adapter.model.lm_head.weight.add_(0.1)
    with pytest.raises(PolicyStateError, match="without mark_state"):
        adapter.generate_action([{"role": "user", "content": "Choose"}], seed=2)
    adapter.mark_state("update", generation=2)
    assert adapter.generate_action([{"role": "user", "content": "Choose"}], seed=2).success


def test_settings_are_fingerprinted_and_later_mutation_rejected(adapter, assets):
    checkpoint = verify_checkpoint(assets[0], "qwen35_4b", assets[1])
    changed = QwenPolicyAdapter(TinyPolicy(), TinyTokenizer(), checkpoint, enable_thinking=True,
                                decoding=adapter.decoding, max_input_tokens=100)
    assert changed.configuration_fingerprint != adapter.configuration_fingerprint
    adapter.decoding = replace(adapter.decoding, temperature=0.9)
    with pytest.raises(PolicyStateError, match="settings changed"):
        adapter.generate_action([{"role": "user", "content": "Choose"}], seed=2)


def test_capture_replays_only_complete_prefix_and_does_not_modify_logits(adapter):
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    flags = [p.requires_grad for p in adapter.model.parameters()]
    capture = adapter.capture_prefix(ids, stamp=adapter.model_stamp)
    trace_ids, trace_kwargs = adapter.prepare_trace_inputs(ids, stamp=adapter.model_stamp)
    assert torch.equal(trace_ids, ids) and trace_ids.data_ptr() != ids.data_ptr()
    assert trace_kwargs["use_cache"] is False and trace_kwargs["logits_to_keep"] == 1
    assert capture.parity_passed and capture.parity_max_abs == 0
    assert list(capture.mlp_inputs) == list(adapter.mlp_paths) and len(capture.mlp_outputs) == 32
    assert all(value.shape == (1, 5, 4) and not value.requires_grad for value in capture.mlp_inputs.values())
    assert len(adapter.model.forward_calls) == 2
    assert all(torch.equal(call["ids"], ids) and call["use_cache"] is False for call in adapter.model.forward_calls)
    assert flags == [p.requires_grad for p in adapter.model.parameters()]
    assert all(not adapter.model.get_submodule(path)._forward_hooks for path in adapter.mlp_paths)
    with pytest.raises(TypeError):
        adapter.capture_prefix(ids, labels=torch.tensor([[99]]))
    report = adapter.parameter_report()
    assert report["gradient_trainable_visual_parameters"] == 0
    assert report["unused_visual_parameters"] > 0 and report["es_scope_must_be_declared_separately"]
    assert report["runtime_total_parameters"] == report["text_including_lm_head_parameters"] + report["unused_visual_parameters"]


@pytest.mark.parametrize("mode", ["fail_parity", "duplicate_mlp"])
def test_capture_fails_closed_and_removes_hooks(adapter, mode):
    setattr(adapter.model, mode, True)
    with pytest.raises(PolicyIntegrityError):
        adapter.capture_prefix(torch.tensor([[1, 2, 3]]))
    assert all(not adapter.model.get_submodule(path)._forward_hooks for path in adapter.mlp_paths)


def test_wrong_runtime_layer_shape_and_generated_prefix_fail(adapter, assets):
    bad = TinyPolicy()
    bad.model.language_model.layers[7].mlp.down_proj = nn.Linear(8, 5, bias=False)
    with pytest.raises(PolicyIntegrityError, match="projection shape"):
        QwenPolicyAdapter(bad, TinyTokenizer(), verify_checkpoint(assets[0], "qwen35_4b", assets[1]))
    adapter.model.bad_prefix = True
    result = adapter.generate_action([{"role": "user", "content": "Choose"}], seed=1)
    assert not result.success and result.failure_code == "generation_contract_failed"
