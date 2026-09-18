#!/usr/bin/env python3
"""Run the full CPU regression suite before enabling the implemented MADE method.

This proves implementation contracts, not fitted-head quality or scientific gain.
Native numerical source identity is checked against the validated FP32 snapshot.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import textwrap
import xml.etree.ElementTree as ET


NATIVE_FILES = ("policy.py", "attention_checkpointing.py", "action_targets.py", "native_attribution.py")
CAUSAL_NATIVE_SHA256 = "a6d4b6bf985b5a9ac56a211a676ae5b8fb006c1b75ff69999d91d9cf7a7e7185"
CAUSAL_RULE = "qwen_final_mlp_causal_activation_v1"
NATIVE_RELATIVE = "src/matdiscovery/native_attribution.py"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "Duplicate source-proof JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    def invalid(value):
        raise RuntimeError("Nonfinite source-proof JSON value: " + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs, parse_constant=invalid)


def _ast(node):
    return ast.dump(node, include_attributes=False)


def _statement(text):
    return _ast(ast.parse(textwrap.dedent(text)).body[0])


_INIT_ADDITIONS = (
    """if source_selection_rule not in SOURCE_SELECTION_RULES:
        raise ValueError("Unknown attribution source-selection rule")""",
    "self.source_selection_rule = source_selection_rule",
    """self.source_selection_proof = (_qwen_final_mlp_source_proof(policy, self.bindings, self.architecture_review)
        if source_selection_rule == "qwen_final_mlp_causal_activation_v1" else None)""",
)
_ASSEMBLE_ADDITIONS = (
    "excluded_active, excluded_activation_mass = 0, 0.0",
    """if self.source_selection_rule == "qwen_final_mlp_causal_activation_v1" and layer == layers - 1:
        allowed = torch.tensor(action_targets.prediction_positions if action_targets is not None else (positions - 1,),
                               dtype=torch.long, device=active.device)
        eligible = (active[:, 0, None] == allowed[None, :]).any(dim=1)
        excluded = active[~eligible]
        excluded_active += len(excluded)
        excluded_activation_mass += float(z[excluded[:, 0], excluded[:, 1]].sum())
        active = active[eligible]""",
)


def _remove_statements(node, statements):
    expected = {_statement(text): 0 for text in statements}
    class Remove(ast.NodeTransformer):
        def visit(self, value):
            if isinstance(value, ast.stmt) and _ast(value) in expected:
                expected[_ast(value)] += 1
                return None
            return super().visit(value)
    result = Remove().visit(node)
    require(all(count == 1 for count in expected.values()), "Reviewed AST additions are missing, duplicated or changed")
    return result


def _remove_metadata(function, assignment_name, keys):
    matches = [node for node in ast.walk(function) if isinstance(node, ast.Assign)
               and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id == assignment_name and isinstance(node.value, ast.Dict)]
    require(len(matches) == 1, "Ambiguous reviewed metadata assignment: " + assignment_name)
    value = matches[0].value
    pairs = list(zip(value.keys, value.values))
    removed = [key.value for key, _ in pairs if isinstance(key, ast.Constant) and key.value in keys]
    require(len(removed) == len(keys) and set(removed) == set(keys), "Reviewed metadata fields differ")
    retained = [(key, item) for key, item in pairs if not isinstance(key, ast.Constant) or key.value not in keys]
    value.keys, value.values = [x[0] for x in retained], [x[1] for x in retained]


def numerical_ast_equivalence(old_source, new_source):
    """Remove only reviewed selection/metadata additions and compare full AST.

    This is not permission to alter these regions: the caller also requires the
    single reviewed full-file SHA. The projection makes the unchanged numerical
    capture, cut/VJP, intervention and finite-difference paths reviewable.
    """
    old, new = ast.parse(old_source), ast.parse(new_source)
    projected = deepcopy(new)
    added = [node for node in projected.body if isinstance(node, ast.FunctionDef) and node.name == "_qwen_final_mlp_source_proof"]
    require(len(added) == 1, "Missing reviewed Qwen tokenwise-tail source proof")
    projected.body.remove(added[0])
    projected = _remove_statements(projected, (
        "import inspect", "from pathlib import Path",
        'SOURCE_SELECTION_RULES = ("global_activation", "qwen_final_mlp_causal_activation_v1")',
        'QWEN_TOKENWISE_TAIL_SOURCE_SHA256 = "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"',
    ))
    def methods(tree):
        owners = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NativeAttributor"]
        require(len(owners) == 1, "NativeAttributor class is missing or duplicated")
        result = {node.name: node for node in owners[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        require(len(result) == sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in owners[0].body), "Duplicate native method")
        return result
    current = methods(projected)
    constructor = current["__init__"]
    require(constructor.args.kwonlyargs[-1].arg == "source_selection_rule"
            and _ast(constructor.args.kwonlyargs[-1].annotation) == _ast(ast.Name(id="str", ctx=ast.Load()))
            and _ast(constructor.args.kw_defaults[-1]) == _ast(ast.Constant(value="global_activation")),
            "Source selection changed a constructor argument/default")
    constructor.args.kwonlyargs.pop(); constructor.args.kw_defaults.pop()
    _remove_statements(constructor, _INIT_ADDITIONS)
    _remove_statements(current["_assemble"], _ASSEMBLE_ADDITIONS)
    _remove_metadata(current["_backend_signature"], "record", {"source_selection_rule", "source_selection_proof"})
    _remove_metadata(current["validate_backend"], "report", {"source_selection_rule", "source_selection_proof"})
    _remove_metadata(current["_assemble"], "metadata", {"source_selection_rule", "source_selection_proof",
        "structurally_excluded_active_features", "structurally_excluded_activation_mass", "eligible_active_features", "coverage_denominator"})
    require(_ast(old) == _ast(projected), "Unreviewed native AST change outside exact candidate filtering/metadata additions")
    before = methods(old)
    required = ("capture", "_assemble", "intervention_value", "validate_backend", "attribute", "_guard", "_constant_device")
    require(all(name in before for name in required), "A required numerical path is absent")
    return {"NativeAttributor." + name: hashlib.sha256(_ast(before[name]).encode()).hexdigest() for name in required}


def _inventory(root):
    return {str(path.relative_to(root)): sha(path) for directory in ("src", "scripts", "configs")
            for path in sorted((root / directory).rglob("*")) if path.is_file()
            and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def verify_native_sources(root, original, *, causal_source_proof=None):
    """Pure read-only source gate. It never runs tests, imports a model, or grants GPU acceptance."""
    root, original = Path(root).resolve(), Path(original).resolve()
    require((original / "source_manifest.json").is_file(), "Validated source must contain the original runtime input snapshot")
    prior = read_json(original / "source_manifest.json")
    expected = prior.get("inputs", {})
    require(isinstance(expected, dict) and all("src/matdiscovery/" + name in expected for name in NATIVE_FILES),
            "Validated source must be the original runtime input snapshot, not a staged technical copy")
    unchanged, current_native = {}, {}
    for name in NATIVE_FILES:
        relative = "src/matdiscovery/" + name
        require(sha(original / relative) == expected[relative], "Validated FP32 source bytes changed: " + name)
        current_native[relative] = sha(root / relative)
        if current_native[relative] == expected[relative]: unchanged[relative] = expected[relative]
    result = {"native_numerical_sources_unchanged": unchanged, "validated_native_source": str(original),
        "native_source_sha256": current_native, "native_all_source_bytes_unchanged": len(unchanged) == 4,
        "source_identity_policy": "exact_validated_native_bytes", "causal_source_selection_proof": None,
        "native_source_delta": {}, "native_numerical_path_ast_unchanged": {},
        "source_bound_real_gpu_fd_required": True, "source_bound_real_gpu_fd_passed": False,
        "validated_scope": "current_two_model_MADE_implementation_not_completed_empirical_study"}
    if causal_source_proof is None:
        require(len(unchanged) == 4, "Native numerical code changed since validated FP32 collection; explicit reviewed causal source proof required")
        return result
    path = Path(causal_source_proof).resolve()
    proof = read_json(path)
    source_root, staged = Path(proof["source_core"]).resolve(), Path(proof["staged_source"]).resolve()
    require(original not in {root, source_root, staged}, "Validated native source must remain the original runtime snapshot")
    delta = {NATIVE_RELATIVE: {"old": expected[NATIVE_RELATIVE], "new": CAUSAL_NATIVE_SHA256}}
    require(proof.get("schema") == "technical_only_native_source_delta_v1" and proof.get("complete") is True
            and proof.get("all_other_source_bytes_equal") is True and proof.get("delta") == delta
            and proof.get("activation_or_model_files_copied") == 0 and proof.get("new_GPU_or_oracle_calls") == 0,
            "Only the reviewed single native causal-source delta is admitted")
    old_files, new_files = proof["old_files"], proof["new_files"]
    require(isinstance(old_files, dict) and set(old_files) == set(new_files), "Source proof omitted/added files")
    actual_delta = {name: {"old": old_files[name], "new": new_files[name]} for name in old_files if old_files[name] != new_files[name]}
    require(actual_delta == delta and old_files == _inventory(source_root) and new_files == _inventory(staged),
            "Source proof file inventory/delta does not match frozen bytes")
    source_manifest_path = source_root / "source_manifest.json"
    require(sha(source_manifest_path) == proof["core_source_manifest_sha256"], "Original core source manifest changed")
    source_manifest = read_json(source_manifest_path)
    files = source_manifest.get("files", {})
    require(files and source_manifest.get("source_fingerprint") == fingerprint(files)
            and source_manifest.get("original_project") == str(source_root)
            and all(old_files.get(name) == digest for name, digest in files.items()), "Causal source proof lost its original snapshot binding")
    source_core = read_json(source_root / "configs/deadline_core_protocol.json")
    require(source_core.get("fingerprint") == source_core.get("core_fingerprint") == proof["source_core_fingerprint"]
            == fingerprint({k: v for k, v in source_core.items() if k not in {"fingerprint", "core_fingerprint"}}),
            "Causal source core fingerprint changed")
    for name in NATIVE_FILES:
        relative = "src/matdiscovery/" + name
        require(old_files[relative] == expected[relative] and new_files[relative] == current_native[relative],
                "Current/native validated source is outside the single reviewed delta: " + name)
    require(current_native[NATIVE_RELATIVE] == CAUSAL_NATIVE_SHA256 and len(unchanged) == 3,
            "The causal implementation must be the one frozen reviewed SHA; other native files stay identical")
    ast_proof = numerical_ast_equivalence((source_root / NATIVE_RELATIVE).read_text(), (root / NATIVE_RELATIVE).read_text())
    result.update(source_identity_policy="reviewed_single_native_source_delta_with_unchanged_cut_vjp_fd_ast",
        causal_source_selection_proof={"path": str(path), "sha256": sha(path)}, native_source_delta=delta,
        native_numerical_path_ast_unchanged=ast_proof,
        validated_scope="qwen35_4b_MADE_qwen_final_mlp_causal_activation_v1_implementation_only",
        source_selection_rule=CAUSAL_RULE,
        source_core_fingerprint=proof["source_core_fingerprint"],
        causal_source_proof_scope="candidate eligibility and metadata only; all remaining module AST is identical; requires separate real GPU FD")
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--validated-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--causal-source-proof", type=Path,
                        help="Explicit reviewed native-only source delta; CPU acceptance still requires separate source-bound real GPU FD")
    args = parser.parse_args()
    root, original, output = args.project.resolve(), args.validated_source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from matdiscovery.failure_controller import HEADS, HEAD_WEIGHTS, MAX_FEEDBACK_CHARS
    protocol = json.loads((root / "configs/main_protocol.json").read_text())
    control = protocol["failure_control"]
    if (control["enabled_benchmarks"] != ["made"] or control["heads"] != list(HEADS)
            or control["head_weights"] != [HEAD_WEIGHTS[head] for head in HEADS]
            or control["threshold"] != protocol["risk_network"]["threshold"]
            or control["maximum_candidate_generations_per_decision"] != 2
            or control["maximum_candidate_generations_per_decision"] != protocol["risk_network"]["maximum_candidate_generations_per_decision"]
            or control["retry_feedback_max_characters"] != MAX_FEEDBACK_CHARS):
        raise RuntimeError("Declared type-control parameters differ from the implemented controller")
    native_identity = verify_native_sources(root, original, causal_source_proof=args.causal_source_proof)
    files = sorted(path for directory in ("src", "scripts", "tests") for path in (root / directory).rglob("*.py")
                   if "__pycache__" not in path.parts)
    files += [root / "configs/main_protocol.json", root / "configs/policy_runtime_gates.json"]
    sources = {str(path.relative_to(root)): sha(path) for path in files}
    write(output / "source_hashes_before.json", sources)
    command = [sys.executable, "-m", "pytest", "tests", "-q", "--junitxml=" + str(output / "pytest.xml")]
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(root / "src"), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONNOUSERSITE="1")
    started = time.time()
    with (output / "pytest.log").open("w") as log:
        result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
    after = {str(path.relative_to(root)): sha(path) for path in files}
    if sources != after:
        raise RuntimeError("Source changed while method regression was running")
    if verify_native_sources(root, original, causal_source_proof=args.causal_source_proof) != native_identity:
        raise RuntimeError("Native source proof changed during method regression")
    required = ("test_failure_labels", "test_failure_risk", "test_failure_controller",
                "test_failure_rollout_integration", "test_failure_fit_integration", "test_failure_reporting")
    if args.causal_source_proof is not None:
        required += ("test_causal_feature_selection", "test_validate_method_alignment_causal_source",
                     "test_passed_bank_reuse", "test_graph_bank_reuse_adapters")
    cases = ET.parse(output / "pytest.xml").getroot().findall(".//testcase")
    critical = {name: [] for name in required}
    counts = {"tests": len(cases), "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for case in cases:
        state = "failed" if case.find("failure") is not None else "errors" if case.find("error") is not None else "skipped" if case.find("skipped") is not None else "passed"
        counts[state] += 1
        for name in required:
            if name in case.get("classname", ""):
                critical[name].append(state)
    passed = (result.returncode == 0 and counts["tests"] > 0 and counts["failed"] == counts["errors"] == 0
              and all(states and all(state == "passed" for state in states) for states in critical.values()))
    report = {"schema": "MADE_method_implementation_validation_v1", "passed": passed,
        "started_at": started, "finished_at": time.time(), "returncode": result.returncode,
        "counts": counts, "critical_suites": {name: len(states) for name, states in critical.items()},
        "source_sha256": sources, **native_identity, "log_sha256": sha(output / "pytest.log"),
        "junit_sha256": sha(output / "pytest.xml"), "new_scientific_oracle_calls": 0,
        "real_corpus_fitting_completed": False, "empirical_improvement_established": False}
    write(output / "validation.json", report)
    if not passed:
        raise RuntimeError("Full method implementation regression did not pass; gate remains closed")
    gate_path = root / "configs/method_alignment_gate.json"
    gate = json.loads(gate_path.read_text())
    gate.update(implementation_matches_requested_method=True,
        current_implemented_method="native_graph_features_overall_and_masked_failure_type_NNs_type_conditioned_control_full_parameter_ESOpt",
        missing_behaviors=[], validation={"path": str(output / "validation.json"), "sha256": sha(output / "validation.json")},
        validated_scope=native_identity["validated_scope"],
        source_identity_policy=native_identity["source_identity_policy"],
        causal_source_selection_proof=native_identity["causal_source_selection_proof"],
        source_bound_real_gpu_fd_required=True, source_bound_real_gpu_fd_passed=False,
        real_corpus_fidelity_fit_and_full_final_results_still_required=True,
        CrystalGym_type_control_validated=False, latest_three_to_four_model_extension_complete=False)
    write(gate_path, gate)
    print(json.dumps({"method_implementation_verified": True, "counts": counts, "validation": str(output / "validation.json"),
                      "main_experiments_complete": False}), flush=True)


if __name__ == "__main__":
    main()
