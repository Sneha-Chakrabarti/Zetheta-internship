import numpy as np
import pytest

from src.models.ensemble.labels import (
    collapse_probs_5_to_3, collapse_labels_5_to_3, to_display_label, LABELS_5, LABELS_3,
)
from src.models.ensemble.combine import bma_weights, bma_combine, fit_stacking_weights, mean_log_likelihood


def test_collapse_probs_preserves_total_probability_mass():
    rng = np.random.default_rng(0)
    raw = rng.random((50, 5))
    probs5 = raw / raw.sum(axis=1, keepdims=True)
    probs3 = collapse_probs_5_to_3(probs5, LABELS_5)
    np.testing.assert_allclose(probs3.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(probs3.sum(), probs5.sum(), atol=1e-8)


def test_collapse_probs_matches_known_mapping():
    probs5 = np.array([[0.5, 0.1, 0.1, 0.2, 0.1]])  # order: RiskOn, LateCycle, Transitional, PostShock, RiskOff
    probs3 = collapse_probs_5_to_3(probs5, LABELS_5)
    # Risk-On absorbs Post-Shock: 0.5 + 0.2 = 0.7
    # Transitional absorbs Late-Cycle: 0.1 + 0.1 = 0.2
    # Risk-Off: 0.1
    np.testing.assert_allclose(probs3[0], [0.7, 0.2, 0.1])


def test_collapse_probs_handles_arbitrary_label_order():
    """The mapping must key off the label NAME passed in, not column
    position, so callers whose member happens to order classes
    differently still collapse correctly."""
    shuffled_order = ["Post-Shock", "Risk-On", "Risk-Off", "Late-Cycle", "Transitional"]
    probs5 = np.array([[0.2, 0.5, 0.1, 0.1, 0.1]])  # matches shuffled_order
    probs3 = collapse_probs_5_to_3(probs5, shuffled_order)
    np.testing.assert_allclose(probs3[0], [0.7, 0.2, 0.1])


def test_to_display_label_is_idempotent_and_handles_snake_case():
    assert to_display_label("risk_on") == "Risk-On"
    assert to_display_label("late_cycle") == "Late-Cycle"
    assert to_display_label("Risk-On") == "Risk-On"
    for l in LABELS_5:
        assert to_display_label(l) == l


def test_collapse_labels_5_to_3_both_naming_conventions_agree():
    snake = ["risk_on", "post_shock", "late_cycle", "transitional", "risk_off"]
    display = ["Risk-On", "Post-Shock", "Late-Cycle", "Transitional", "Risk-Off"]
    a = collapse_labels_5_to_3(snake)
    b = collapse_labels_5_to_3(display)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, ["Risk-On", "Risk-On", "Transitional", "Transitional", "Risk-Off"])


def test_bma_weights_sum_to_one_and_favor_higher_likelihood():
    w = bma_weights(np.array([-30.0, -20.0, -10.0]))
    assert w.sum() == pytest.approx(1.0)
    assert w[2] > w[1] > w[0]


def test_bma_weights_uniform_when_likelihoods_tied():
    w = bma_weights(np.array([-5.0, -5.0, -5.0]))
    np.testing.assert_allclose(w, 1 / 3, atol=1e-10)


def test_bma_combine_matches_manual_weighted_average():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    weights = np.array([0.3, 0.7])
    combined = bma_combine(probs, weights)
    expected = 0.3 * probs[0] + 0.7 * probs[1]
    np.testing.assert_allclose(combined, expected)


def test_fit_stacking_weights_is_a_simplex():
    rng = np.random.default_rng(0)
    base = rng.dirichlet(np.ones(4), size=(3, 100)).transpose(0, 1, 2)
    y = rng.integers(0, 4, 100)
    w = fit_stacking_weights(base, y)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-9).all()


def test_fit_stacking_weights_recovers_the_perfect_member():
    rng = np.random.default_rng(1)
    N, K = 300, 4
    y = rng.integers(0, K, N)
    perfect = np.eye(K)[y] * 0.97 + 0.01
    noise = rng.dirichlet(np.ones(K), size=N)
    base = np.stack([perfect, noise])
    w = fit_stacking_weights(base, y, n_classes=K)
    assert w[0] > 0.9


def test_fit_stacking_weights_rejects_mismatched_n_classes():
    base = np.full((2, 10, 3), 1 / 3)
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        fit_stacking_weights(base, y, n_classes=5)


def test_mean_log_likelihood_perfect_beats_uniform():
    K, N = 3, 50
    y = np.zeros(N, dtype=int)
    perfect = np.tile([0.98, 0.01, 0.01], (N, 1))
    uniform = np.full((N, K), 1 / K)
    assert mean_log_likelihood(perfect, y) > mean_log_likelihood(uniform, y)


def test_mean_log_likelihood_matches_hand_computation():
    y = np.array([0, 1])
    probs = np.array([[0.5, 0.5], [0.25, 0.75]])
    expected = (np.log(0.5) + np.log(0.75)) / 2
    assert mean_log_likelihood(probs, y) == pytest.approx(expected)
