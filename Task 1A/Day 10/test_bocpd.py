import numpy as np
import pytest

from src.models.sequential.bocpd import (
    bocpd, calibrate_normal_gamma_prior, changepoint_probability,
    map_run_length, short_run_length_probability, detected_changepoints,
)


def _three_block_series(seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal(0.0, 0.01, 100),
        rng.normal(0.05, 0.01, 100),
        rng.normal(-0.03, 0.01, 100),
    ])


def test_r_matrix_is_a_valid_run_length_posterior():
    data = _three_block_series()
    R = bocpd(data, hazard=1 / 100)
    assert R.shape == (len(data) + 1, len(data) + 1)
    col_sums = R[:, 1:].sum(axis=0)
    np.testing.assert_allclose(col_sums, 1.0, atol=1e-8)
    assert (R >= 0).all()


def test_changepoint_probability_is_exactly_the_hazard_rate_always():
    """The central correctness finding for this module: R[0, t] is a pure
    algebraic identity of the recursion (proved in the module docstring),
    not a data-dependent quantity. True for ANY data, not just the toy
    series it was originally found on."""
    for seed in range(5):
        data = np.random.default_rng(seed).normal(0, 1, 50)
        R = bocpd(data, hazard=0.03)
        np.testing.assert_allclose(changepoint_probability(R), 0.03, atol=1e-10)


def test_uncalibrated_default_prior_fails_to_detect_an_obvious_shift():
    """Regression test for the found-and-fixed problem: the spec's literal
    defaults (implied prior variance 1.0) on financial-returns-scale data
    (variance ~1e-4) should fail to produce any confident short-run-length
    signal at the true break, because the predictive density is too wide
    to register the shift as surprising."""
    data = _three_block_series()
    R = bocpd(data, hazard=1 / 100)  # mu0=0, kappa0=1, alpha0=1, beta0=1 (the spec's defaults)
    sig = short_run_length_probability(R, k=3)
    assert sig[100] < 0.3   # should NOT spike at the true break with the mismatched prior


def test_calibrated_prior_detects_both_true_changepoints_and_nothing_else():
    data = _three_block_series()
    prior = calibrate_normal_gamma_prior(data, calibration_window=60)
    R = bocpd(data, hazard=1 / 100, **prior)
    detected = detected_changepoints(R, k=3, threshold=0.5, warmup=5)
    # Allow a short lag (BOCPD needs a couple of observations to confirm
    # a break statistically): require a detection within 5 steps of each
    # true break, and nothing more than 5 steps from either.
    assert any(abs(d - 100) <= 5 for d in detected)
    assert any(abs(d - 200) <= 5 for d in detected)
    assert all(min(abs(d - 100), abs(d - 200)) <= 5 for d in detected)


def test_map_run_length_collapses_and_regrows_at_each_true_changepoint():
    data = _three_block_series()
    prior = calibrate_normal_gamma_prior(data, calibration_window=60)
    R = bocpd(data, hazard=1 / 100, **prior)
    mrl = map_run_length(R)
    # Just before each break, the MAP run length should be large (tracking
    # the stable regime); just after, it should drop sharply.
    assert mrl[99] > 50
    assert mrl[101] < 10
    assert mrl[199] > 50
    assert mrl[201] < 10


def test_calibrate_prior_matches_the_calibration_windows_variance():
    rng = np.random.default_rng(0)
    data = rng.normal(0.002, 0.015, 200)
    prior = calibrate_normal_gamma_prior(data, calibration_window=60, alpha0=5.0)
    implied_variance = prior["beta0"] / prior["alpha0"]
    np.testing.assert_allclose(implied_variance, np.var(data[:60]), rtol=1e-6)
    assert prior["mu0"] == pytest.approx(np.mean(data[:60]))


def test_calibrate_prior_falls_back_when_window_is_degenerate():
    """A calibration window with zero variance (e.g. a run of identical
    values) must not produce a zero or non-finite prior variance."""
    data = np.concatenate([np.full(60, 0.01), np.random.default_rng(0).normal(0, 0.02, 100)])
    prior = calibrate_normal_gamma_prior(data, calibration_window=60)
    assert prior["beta0"] > 0 and np.isfinite(prior["beta0"])


def test_detected_changepoints_excludes_the_uninformative_warmup_period():
    """At t < k, run length cannot exceed t, so short_run_length_probability
    is trivially high regardless of data - must not be reported as a
    detection."""
    data = _three_block_series()
    prior = calibrate_normal_gamma_prior(data, calibration_window=60)
    R = bocpd(data, hazard=1 / 100, **prior)
    detected_with_warmup = detected_changepoints(R, k=3, threshold=0.5, warmup=0)
    detected_no_warmup = detected_changepoints(R, k=3, threshold=0.5, warmup=5)
    assert 0 in detected_with_warmup or 1 in detected_with_warmup or 2 in detected_with_warmup
    assert not any(d < 5 for d in detected_no_warmup)
