"""Bayesian Online Changepoint Detection, Section A9.2 (Adams & MacKay,
2007). `bocpd` matches the spec's function exactly (Normal-Gamma
conjugate model, Student-t predictive). Two things needed fixing before
this did anything useful on financial-returns-scale data, both found by
tracing the algorithm on a toy series with an obvious mean shift and
verified before being relied on - see `docs/day10_sequential_notes.md`
for the full trace.

1. The spec's default priors (mu0=0, kappa0=1, alpha0=1, beta0=1) imply
   a prior variance of 1.0 - four orders of magnitude above a typical
   daily return series' variance (~1e-4). Left as given, the predictive
   density stays so wide that a real 5-percentage-point mean shift reads
   as under half a (wrongly inflated) standard deviation, and detection
   never fires. `calibrate_normal_gamma_prior` sets the prior's implied
   variance from the data's own short-run scale instead.

2. `R[0, t]` - the run-length-0 row, the naive reading of "probability a
   changepoint just occurred" - is NOT informative. It is a pure
   algebraic identity of the recursion, exactly equal to the hazard rate
   at every single t regardless of the data (proved directly: for any
   nonnegative column R[:, t] and any nonnegative `pred`, splitting a
   total mass S into `hazard * S` and `(1 - hazard) * S` and normalising
   recovers `hazard` at the first coordinate no matter what S or pred
   are - verified both algebraically and against five random inputs).
   The actual signal is in how probability is distributed among the
   OTHER run lengths. `map_run_length` (the posterior mode) and
   `short_run_length_probability` (aggregated mass on run lengths at or
   below a small threshold) are the corrected extraction functions;
   `changepoint_probability` is kept, unmodified, only as a named
   regression check that this identity holds (see its docstring and
   `tests/test_bocpd.py`) - it must never be used as a detection signal.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bocpd(data: np.ndarray, hazard: float = 1 / 100, mu0: float = 0.0,
          kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0) -> np.ndarray:
    """Bayesian online changepoint detection with a Normal-Gamma model.
    Returns the run-length posterior matrix R (T+1, T+1): R[r, t] is
    P(run length = r | data up to and including step t-1), column 0
    being the prior before any data."""
    T = len(data)
    R = np.zeros((T + 1, T + 1))
    R[0, 0] = 1.0
    mu, kappa, alpha, beta = [mu0], [kappa0], [alpha0], [beta0]
    for t, x in enumerate(data):
        scale = np.sqrt(np.array(beta) * (np.array(kappa) + 1) / (np.array(alpha) * np.array(kappa)))
        pred = norm.pdf(x, loc=np.array(mu), scale=scale)
        R[1:t + 2, t + 1] = R[0:t + 1, t] * pred * (1 - hazard)
        R[0, t + 1] = np.sum(R[0:t + 1, t] * pred * hazard)
        R[:, t + 1] /= R[:, t + 1].sum()
        mu_new = (np.array(kappa) * np.array(mu) + x) / (np.array(kappa) + 1)
        kappa_new = np.array(kappa) + 1
        alpha_new = np.array(alpha) + 0.5
        beta_new = np.array(beta) + (np.array(kappa) * (x - np.array(mu)) ** 2) / (2 * (np.array(kappa) + 1))
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa_new])
        alpha = np.concatenate([[alpha0], alpha_new])
        beta = np.concatenate([[beta0], beta_new])
    return R


def calibrate_normal_gamma_prior(data: np.ndarray, calibration_window: int = 60,
                                  alpha0: float = 5.0) -> dict:
    """Pick (mu0, kappa0, alpha0, beta0) so the prior's implied variance
    (beta0 / alpha0) matches this data's own scale. See module docstring
    for why the spec's literal defaults silently defeat detection on
    financial-returns-scale data.

    Uses the variance of the first `calibration_window` points, not the
    whole series: using the whole series would conflate within-regime
    noise with any mean shift the series contains, which is exactly the
    effect being calibrated away from. `alpha0=5.0` (the spec's example
    uses 1.0) gives the prior enough weight that the predictive scale
    does not swing wildly over the first few observations while
    kappa/alpha are still tiny.
    """
    window = data[:calibration_window]
    variance = float(np.var(window))
    if variance <= 0 or not np.isfinite(variance):
        variance = float(np.var(data)) or 1.0
    return {
        "mu0": float(np.mean(window)),
        "kappa0": 1.0,
        "alpha0": alpha0,
        "beta0": alpha0 * variance,
    }


def changepoint_probability(R: np.ndarray) -> np.ndarray:
    """R[0, 1:] - kept as a named regression check, NOT a detection
    signal. Provably constant and exactly equal to the hazard rate at
    every t regardless of the data (see module docstring); a test
    asserts this. Use `map_run_length` or
    `short_run_length_probability` instead."""
    return R[0, 1:]


def map_run_length(R: np.ndarray) -> np.ndarray:
    """The posterior mode run length at each step, length T (excluding
    the t=0 prior column). The informative signal: a sharp drop (not
    necessarily to exactly 0) followed by a steady climb is what a real
    changepoint looks like, confirmed against a known three-block toy
    series in `docs/day10_sequential_notes.md`."""
    return np.argmax(R[:, 1:], axis=0)


def short_run_length_probability(R: np.ndarray, k: int = 3) -> np.ndarray:
    """P(run length <= k | data up to t), length T. Unlike `R[0, :]`
    alone, this DOES depend on the data: it aggregates the (data-
    dependent) relative mass across run lengths 0..k, not the row that
    is fixed at the hazard rate by construction. The practical
    changepoint-detection signal: compare against a threshold, or watch
    for a spike."""
    return R[0:k + 1, 1:].sum(axis=0)


def detected_changepoints(R: np.ndarray, k: int = 3, threshold: float = 0.5,
                           warmup: int = 5) -> np.ndarray:
    """Indices (into the original data, 0-based) where
    `short_run_length_probability` exceeds `threshold`. A simple
    thresholding rule, not the only reasonable one; kept separate so a
    caller can apply their own rule to the raw series instead.

    `warmup` drops the first few indices from consideration: run length
    cannot exceed the number of observations seen so far, so for t < k
    essentially all probability mass is necessarily at short run lengths
    regardless of the data - a guaranteed, uninformative "detection" at
    the very start of any series, confirmed empirically (indices 0-2
    fire at k=3 on a toy series with no change there at all), not a
    sign anything is wrong with the data at that point.
    """
    sig = short_run_length_probability(R, k=k)
    idx = np.where(sig > threshold)[0]
    return idx[idx >= warmup]
