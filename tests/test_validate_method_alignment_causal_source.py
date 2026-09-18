"""Read-only CPU source/AST contracts; no full suite, model, GPU or oracle run."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("unit_method_alignment_source_gate", PROJECT / "scripts/validate_method_alignment.py")
alignment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(alignment)
OLD_NATIVE_SHA = "6b5eabe8a974a961bce12095b779a130092a1b88310e675238357a96fc6445fc"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def original_native_bytes():
    """Reconstruct the known old file from reviewed additions, then check its
    independently fixed SHA. This keeps tests portable to frozen trees without
    .git and avoids copying a second 50KB native implementation into fixtures.
    """
    text = (PROJECT / alignment.NATIVE_RELATIVE).read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == alignment.CAUSAL_NATIVE_SHA256
    tree = ast.parse(text)
    removed = set()
    def remove(node): removed.update(range(node.lineno, node.end_lineno + 1))
    statements = tuple(alignment._INIT_ADDITIONS) + tuple(alignment._ASSEMBLE_ADDITIONS) + (
        "import inspect", "from pathlib import Path",
        'SOURCE_SELECTION_RULES = ("global_activation", "qwen_final_mlp_causal_activation_v1")',
        'QWEN_TOKENWISE_TAIL_SOURCE_SHA256 = "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"',
    )
    patterns = {alignment._statement(value) for value in statements}
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and alignment._ast(node) in patterns: remove(node)
        if isinstance(node, ast.FunctionDef) and node.name == "_qwen_final_mlp_source_proof":
            remove(node); removed.update((node.end_lineno + 1, node.end_lineno + 2))
        if isinstance(node, ast.arg) and node.arg == "source_selection_rule": remove(node)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NativeAttributor")
    fields = {"source_selection_rule", "source_selection_proof", "structurally_excluded_active_features",
              "structurally_excluded_activation_mass", "eligible_active_features", "coverage_denominator"}
    for method in owner.body:
        if isinstance(method, ast.FunctionDef) and method.name in {"_assemble", "_backend_signature", "validate_backend"}:
            for node in ast.walk(method):
                if isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if isinstance(key, ast.Constant) and key.value in fields:
                            removed.update(range(key.lineno, value.end_lineno + 1))
    previous = "".join(line for index, line in enumerate(text.splitlines(keepends=True), 1) if index not in removed)
    assert hashlib.sha256(previous.encode()).hexdigest() == OLD_NATIVE_SHA
    return previous.encode(), text.encode()


@pytest.fixture
def source_fixture(tmp_path):
    original, current = original_native_bytes()
    root, runtime, core, staged = [tmp_path / name for name in ("project", "validated-runtime", "normalized-source", "staged-delta")]
    for name in alignment.NATIVE_FILES:
        relative = "src/matdiscovery/" + name
        raw = original if name == "native_attribution.py" else (PROJECT / relative).read_bytes()
        for destination in (runtime, core, staged, root):
            path = destination / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    (root / alignment.NATIVE_RELATIVE).write_bytes(current)
    (staged / alignment.NATIVE_RELATIVE).write_bytes(current)
    inputs = alignment._inventory(runtime)
    write(runtime / "source_manifest.json", {"classification": "unit_runtime_source_fixture", "inputs": inputs})
    core_value = {"classification": "unit_only_not_real_protocol", "workspace": str(core)}
    core_value["fingerprint"] = core_value["core_fingerprint"] = alignment.fingerprint(core_value)
    write(core / "configs/deadline_core_protocol.json", core_value)
    (staged / "configs").mkdir()
    shutil.copyfile(core / "configs/deadline_core_protocol.json", staged / "configs/deadline_core_protocol.json")
    files = {key: value for key, value in alignment._inventory(core).items() if key != "configs/deadline_core_protocol.json"}
    write(core / "source_manifest.json", {"original_project": str(core), "files": files, "source_fingerprint": alignment.fingerprint(files)})
    proof = {"schema": "technical_only_native_source_delta_v1", "complete": True, "source_core": str(core),
        "source_core_fingerprint": core_value["fingerprint"], "staged_source": str(staged),
        "delta": {alignment.NATIVE_RELATIVE: {"old": OLD_NATIVE_SHA, "new": alignment.CAUSAL_NATIVE_SHA256}},
        "old_files": alignment._inventory(core), "new_files": alignment._inventory(staged),
        "core_source_manifest_sha256": alignment.sha(core / "source_manifest.json"),
        "all_other_source_bytes_equal": True, "activation_or_model_files_copied": 0, "new_GPU_or_oracle_calls": 0}
    proof_path = tmp_path / "source_delta.json"; write(proof_path, proof)
    return {"root": root, "runtime": runtime, "core": core, "staged": staged,
            "proof": proof, "proof_path": proof_path, "old": original, "new": current}


def test_default_gate_keeps_exact_four_file_identity(source_fixture):
    case = source_fixture
    with pytest.raises(RuntimeError, match="explicit reviewed"):
        alignment.verify_native_sources(case["root"], case["runtime"])
    (case["root"] / alignment.NATIVE_RELATIVE).write_bytes(case["old"])
    result = alignment.verify_native_sources(case["root"], case["runtime"])
    assert result["native_all_source_bytes_unchanged"] and len(result["native_numerical_sources_unchanged"]) == 4
    assert result["source_identity_policy"] == "exact_validated_native_bytes"
    assert result["validated_scope"] == "current_two_model_MADE_implementation_not_completed_empirical_study"


def test_exact_reviewed_delta_proves_numerical_ast_and_only_4b_scope(source_fixture):
    case = source_fixture
    result = alignment.verify_native_sources(case["root"], case["runtime"], causal_source_proof=case["proof_path"])
    assert not result["native_all_source_bytes_unchanged"]
    assert len(result["native_numerical_sources_unchanged"]) == 3
    assert alignment.NATIVE_RELATIVE not in result["native_numerical_sources_unchanged"]
    assert result["native_source_sha256"][alignment.NATIVE_RELATIVE] == alignment.CAUSAL_NATIVE_SHA256
    assert result["source_bound_real_gpu_fd_required"] and not result["source_bound_real_gpu_fd_passed"]
    assert "qwen35_4b" in result["validated_scope"] and "two_model" not in result["validated_scope"]
    assert "qwen35_9b" not in result["validated_scope"]
    assert result["validated_native_source"] == str(case["runtime"])
    assert result["causal_source_selection_proof"] == {"path": str(case["proof_path"]), "sha256": alignment.sha(case["proof_path"])}
    assert set(result["native_numerical_path_ast_unchanged"]) == {"NativeAttributor." + name for name in
        ("capture", "_assemble", "intervention_value", "validate_backend", "attribute", "_guard", "_constant_device")}


@pytest.mark.parametrize("mutation", ["policy", "attention_checkpointing", "action_targets", "new_native", "validated_native",
    "extra_delta", "missing_file", "changed_stage", "omitted_stage_inventory", "core_manifest", "core_fingerprint", "false_complete"])
def test_unknown_or_unbound_source_delta_is_rejected(source_fixture, mutation):
    case = source_fixture; proof = case["proof"]
    if mutation in {"policy", "attention_checkpointing", "action_targets"}:
        path = case["root"] / f"src/matdiscovery/{mutation}.py"; path.write_bytes(path.read_bytes() + b"\n# extra change\n")
    elif mutation == "new_native":
        path = case["root"] / alignment.NATIVE_RELATIVE; path.write_bytes(path.read_bytes() + b"\n# unreviewed\n")
    elif mutation == "validated_native":
        path = case["runtime"] / alignment.NATIVE_RELATIVE; path.write_bytes(path.read_bytes() + b"\n# changed old source\n")
    elif mutation == "extra_delta": proof["delta"]["src/matdiscovery/policy.py"] = {"old": "a" * 64, "new": "b" * 64}
    elif mutation == "missing_file": (case["core"] / "src/matdiscovery/policy.py").unlink()
    elif mutation == "changed_stage":
        path = case["staged"] / alignment.NATIVE_RELATIVE; path.write_bytes(path.read_bytes() + b"\n# stage mutation\n")
    elif mutation == "omitted_stage_inventory":
        for key in ("old_files", "new_files"): proof[key].pop("src/matdiscovery/action_targets.py")
    elif mutation == "core_manifest": proof["core_source_manifest_sha256"] = "0" * 64
    elif mutation == "core_fingerprint": proof["source_core_fingerprint"] = "0" * 64
    else: proof["complete"] = False
    write(case["proof_path"], proof)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        alignment.verify_native_sources(case["root"], case["runtime"], causal_source_proof=case["proof_path"])


@pytest.mark.parametrize("replacement", [
    ("retain_graph=True", "retain_graph=False"),
    ("epsilon: float = 1e-3", "epsilon: float = 1e-2"),
    ("scalar = action_targets.mean_logprob(trace.logits)", "scalar = 2 * action_targets.mean_logprob(trace.logits)"),
    ("active = active[eligible]", "active = active[~eligible]"),
])
def test_ast_projection_independently_rejects_cut_vjp_fd_or_filter_math_changes(source_fixture, replacement):
    old, new = source_fixture["old"].decode(), source_fixture["new"].decode()
    before, after = replacement
    assert before in new
    with pytest.raises(RuntimeError, match="AST|additions"):
        alignment.numerical_ast_equivalence(old, new.replace(before, after, 1))


def test_technical_stage_cannot_replace_original_validated_runtime(source_fixture):
    case = source_fixture
    with pytest.raises(RuntimeError, match="original runtime"):
        alignment.verify_native_sources(case["root"], case["staged"], causal_source_proof=case["proof_path"])


def test_cli_advertises_opt_in_without_running_any_suite():
    result = subprocess.run([sys.executable, str(PROJECT / "scripts/validate_method_alignment.py"), "--help"],
                            capture_output=True, text=True, check=True)
    assert "--causal-source-proof" in result.stdout
    assert "separate source-bound real GPU FD" in " ".join(result.stdout.split())


def test_duplicate_proof_json_key_rejected(tmp_path):
    path = tmp_path / "bad.json"; path.write_text('{"complete":true,"complete":false}')
    with pytest.raises(RuntimeError, match="Duplicate"):
        alignment.read_json(path)
