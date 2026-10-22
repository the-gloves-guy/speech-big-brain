import numpy as np
import pytest
from scipy.special import softmax

from lid.metrics import evaluate, fit_temperature, reliability


def test_filtering_logits_equals_conditional_distribution():
    logits = np.array([[2.0, -1.0, 5.0, 3.0], [0.0, 2.0, -2.0, 1.0]])
    original = softmax(logits, axis=1)[:, [0, 3]]
    conditional = original / original.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(softmax(logits[:, [0, 3]], axis=1), conditional)


def test_ece_includes_confidence_one_and_bin_boundaries():
    probs = np.array([[1.0, 0.0], [0.5, 0.5], [0.25, 0.75]])
    ece, bins = reliability(probs, np.array([0, 1, 1]), n_bins=2)
    assert sum(row["count"] for row in bins) == 3
    assert ece == pytest.approx(abs(2 / 3 - 0.75))


def test_calibration_lowers_validation_nll_without_changing_predictions():
    logits = np.tile([[8.0, 0.0]], (10, 1))
    targets = np.array([0] * 7 + [1] * 3)
    temperature = fit_temperature(logits, targets)
    before = evaluate(logits, targets, ["a", "b"])
    after = evaluate(logits, targets, ["a", "b"], temperature)
    assert after["nll"] < before["nll"]
    assert after["accuracy"] == before["accuracy"]
    assert after["ece"] < before["ece"]


def test_baseline_counts_outside_set_guesses_as_errors():
    report = evaluate(
        np.array([[5.0, 1.0, 0.0], [0.0, 1.0, 5.0]]),
        np.array([0, 1]),
        ["a", "b"],
        report_indices=[0, 1],
    )
    assert report["accuracy"] == 0.5
    assert report["outside_set_rate"] == 0.5
    assert report["per_language"]["b"]["recall"] == 0
    assert sum(map(sum, report["confusion_matrix"])) == 2


def test_invalid_metrics_inputs():
    with pytest.raises(ValueError):
        evaluate(np.zeros((0, 2)), np.array([], dtype=int), ["a", "b"])
    with pytest.raises(ValueError):
        evaluate(np.array([[np.nan, 0]]), np.array([0]), ["a", "b"])
