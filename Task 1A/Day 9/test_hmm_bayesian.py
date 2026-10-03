import numpy as np
import pytest

from src.models.hmm.bayesian import smoothed_state_probs, label_regimes_from_means, REGIME_LABELS_5, filtered_state_probs, pointwise_loglik, _forward_pass_numpy


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


def test_filtered_matches_smoothed_only_at_the_final_timestep():
    """Filtering and smoothing must agree exactly at t=T-1 (nothing left
    to look ahead to) and disagree before that on data with real
    ambiguity - otherwise filtered_state_probs is silently just calling
    the smoother."""
    rng = np.random.default_rng(0)
    K, T = 3, 60
    P = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    pi = np.full(K, 1 / K)
    mu = np.array([-0.02, 0.0, 0.02])
    sigma = np.full(K, 0.01)
    y = rng.normal(0.01, 0.01, T)

    filt = filtered_state_probs(P, pi, mu, sigma, y)
    smooth = smoothed_state_probs(P, pi, mu, sigma, y)
    np.testing.assert_allclose(filt[-1], smooth[-1])
    assert not np.allclose(filt[10], smooth[10])
    np.testing.assert_allclose(filt.sum(axis=1), 1.0, atol=1e-8)


def test_pointwise_loglik_sums_to_the_forward_algorithms_total():
    rng = np.random.default_rng(1)
    K, T = 3, 50
    P = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    pi = np.full(K, 1 / K)
    mu = np.array([-0.02, 0.0, 0.02])
    sigma = np.full(K, 0.01)
    y = rng.normal(0.0, 0.01, T)

    pll = pointwise_loglik(P, pi, mu, sigma, y)
    assert pll.shape == (T,)
    _, c = _forward_pass_numpy(P, pi, mu, sigma, y)
    assert np.isclose(pll.sum(), np.log(c).sum())
