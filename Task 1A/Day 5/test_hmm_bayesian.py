import numpy as np
import pytest

from src.models.hmm.bayesian import smoothed_state_probs, label_regimes_from_means, REGIME_LABELS_5


def test_label_regimes_from_means_rejects_non_five_states():
    with pytest.raises(ValueError):
        label_regimes_from_means(np.zeros(3), np.ones(3))


def test_label_regimes_from_means_matches_frequentist_convention():
    # Descending mean order should map to Risk-On..Risk-Off, same
    # convention as src.models.hmm.frequentist.label_regimes.
    mu = np.array([0.001, 0.003, -0.002, -0.0005, 0.0005])
    sigma = np.array([0.01, 0.01, 0.02, 0.015, 0.012])
    mapping = label_regimes_from_means(mu, sigma)
    assert set(mapping.values()) == set(REGIME_LABELS_5)
    # index 1 has the highest mean -> Risk-On
    assert mapping[1] == "Risk-On"
    # index 2 has the lowest mean -> Risk-Off
    assert mapping[2] == "Risk-Off"


def test_smoothed_state_probs_rows_sum_to_one():
    rng = np.random.default_rng(0)
    K, T = 3, 50
    P = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    pi = np.ones(K) / K
    mu = np.array([-0.01, 0.0, 0.01])
    sigma = np.array([0.01, 0.01, 0.01])
    y = rng.normal(0, 0.01, T)

    gamma = smoothed_state_probs(P, pi, mu, sigma, y)
    assert gamma.shape == (T, K)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-8)
    assert (gamma >= 0).all()


def test_smoothed_state_probs_recovers_obvious_regime_switch():
    """A sanity check with an unambiguous, well-separated two-block
    series: smoothing should assign each block confidently to the right
    state, not just produce valid probabilities."""
    K = 2
    P = np.array([[0.98, 0.02], [0.02, 0.98]])
    pi = np.array([0.5, 0.5])
    mu = np.array([0.05, -0.05])   # very well separated relative to sigma
    sigma = np.array([0.001, 0.001])
    y = np.concatenate([np.full(30, 0.05), np.full(30, -0.05)])

    gamma = smoothed_state_probs(P, pi, mu, sigma, y)
    assert gamma[:30, 0].mean() > 0.95   # confidently state 0 in the first block
    assert gamma[30:, 1].mean() > 0.95   # confidently state 1 in the second block
