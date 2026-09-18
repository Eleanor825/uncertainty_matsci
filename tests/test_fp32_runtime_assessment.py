"""Technical gate-accounting fixtures; no model or scientific evaluator runs."""
import importlib.util
from pathlib import Path


def assessor():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_fp32_policy_runtime.py"
    spec = importlib.util.spec_from_file_location("runtime_assessment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.assess_primary_runtime_gates


def fixture():
    rows = [{"source": source, "epsilon": .001, "analytical": 1., "numerical": 1.001, "passed": True}
            for source in ("embedding", "model.language_model.layers.15.mlp", "model.language_model.layers.31.mlp")]
    rows.append({"source": "embedding", "epsilon": .03, "analytical": 1., "numerical": 1.1, "passed": False})
    return {"policy_runtime": {"dtype": "torch.float32", "all_parameters_materialized": True},
            "cache_stress": {"prompt_tokens": 3072, "new_tokens": 384, "score_tensors": 384},
            "capture": {"tokens": 3456, "layers": 32, "passed": True}, "native_cut_forward_max_abs": 0,
            "action_objective_forward_gates_passed": True, "future_source_gradient_max_abs": 0,
            "parameters_unchanged": True, "finite_differences": rows, "auxiliary_full_vocab_absolute_gate_passed": False}


def test_original_native_gate_is_distinct_from_large_step_curvature_diagnostics():
    report = assessor()(fixture())
    assert report["primary_gate_passed"]
    assert not report["auxiliary_fd_scan_all_passed"] and not report["auxiliary_full_vocab_absolute_gate_passed"]
    assert report["auxiliary_fd_failures"][0]["epsilon"] == .03


def test_failed_original_epsilon_cannot_be_replaced_by_passing_other_steps():
    data = fixture()
    data["finite_differences"][0]["numerical"] = 1.2
    data["finite_differences"][-1].update(numerical=1., passed=True)
    report = assessor()(data)
    assert not report["primary_gate_passed"] and "native_fd_1e-3:embedding" in report["primary_gate_failures"]


def test_missing_capacity_future_gradient_and_nonfinite_derivatives_fail_closed():
    data = fixture()
    data["cache_stress"]["new_tokens"] = 20
    data["future_source_gradient_max_abs"] = .001
    data["finite_differences"][0]["numerical"] = float("nan")
    failures = assessor()(data)["primary_gate_failures"]
    assert "full_3072_plus_384_cache_capacity" in failures
    assert "causal_future_source_gradient" in failures
    assert "native_fd_1e-3:embedding" in failures
