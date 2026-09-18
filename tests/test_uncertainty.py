import numpy as np
import pytest

from matdiscovery.uncertainty import TrainOnlyFeatures, assert_group_disjoint, risk_metrics


def test_rejects_same_material_family_across_splits():
    with pytest.raises(ValueError, match="leakage"):
        assert_group_disjoint(["Al-Li-V"], ["Al-Li-V"])


def test_preprocessing_and_error_reference_do_not_change_at_inference():
    x = np.array([[1., np.nan], [3., 8.], [2., 4.]])
    features = TrainOnlyFeatures().fit(x, np.array([0, 1, 0]), ["a", "b"], ["a", "b", "c"])
    before = features.state()
    result = features.transform(np.array([[1e6, np.nan]]), ["a", "b"])
    assert np.isfinite(result).all()
    assert features.state() == before
    with pytest.raises(ValueError, match="schema"):
        features.transform(x, ["b", "a"])


def test_metrics_keep_confident_errors_in_the_denominator():
    result = risk_metrics([1, 0, 1, 0], [.01, .02, .9, .1])
    assert result["n"] == 4
    assert result["overconfident_error_rate_p_le_0_1"] == .25
    assert result["brier"] > 0
