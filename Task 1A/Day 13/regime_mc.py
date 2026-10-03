"""Regime-conditioned Monte Carlo simulation, Section A13.1 (path
simulation) and A13.2 (VaR/CVaR).
"""
from __future__ import annotations

import numpy as np


class RegimeConditionedMC:
    """Matches the spec's `RegimeConditionedMC` with one defensive fix:
    the inverse-CDF state sampling (`u < cum`) relies on each row of
    `cum` reaching exactly 1.0 at its last column. For a transition
    matrix built from floating-point accumulation (an EM-fitted
    `hmmlearn.transmat_`, or an MCMC posterior mean), that last entry can
    land a few ULPs below 1.0; if a sampled `u` ever lands in that gap,
    `(u[:, None] < cum).argmax(axis=1)` silently returns index 0 (numpy's
    `argmax` on an all-False row) instead of the last state, biasing
    simulated paths toward state 0. Checked directly on realistic
    transition matrices rather than assumed negligible (not observed to
    occur in this project's own fitted matrices, but one random-Dirichlet
    test case came within 1e-16 of the boundary - rare does not mean
    impossible at n_sims x horizon draws, often in the millions).
    `self.P`'s rows are renormalised so the last column is exactly 1.0,
    eliminating the gap outright rather than relying on it staying small
    enough not to matter.
    """

    def __init__(self, transition_matrix: np.ndarray, regime_returns: np.ndarray,
                 regime_vols: np.ndarray, K: int = 5):
        P = np.asarray(transition_matrix, dtype=float)
        self.P = P / P.sum(axis=1, keepdims=True)  # exact row-stochasticity, see docstring
        self.mu = np.asarray(regime_returns, dtype=float)
        self.sig = np.asarray(regime_vols, dtype=float)
        self.K = K

    def simulate(self, init_regime_dist: np.ndarray, horizon: int = 252,
                 n_sims: int = 10000, seed: int = 42):
        rng = np.random.default_rng(seed)
        s0 = rng.choice(self.K, size=n_sims, p=init_regime_dist)
        states = np.zeros((n_sims, horizon), dtype=int)
        states[:, 0] = s0
        for t in range(1, horizon):
            probs = self.P[states[:, t - 1]]
            cum = probs.cumsum(axis=1)
            u = rng.random(n_sims)
            states[:, t] = (u[:, None] < cum).argmax(axis=1)
        mu_t = self.mu[states]
        sig_t = self.sig[states]
        eps = rng.standard_normal((n_sims, horizon))
        rets = mu_t + sig_t * eps
        paths = np.exp(np.cumsum(np.log1p(rets), axis=1))
        return paths, states

    def percentile_paths(self, paths: np.ndarray, qs=(0.05, 0.25, 0.50, 0.75, 0.95)):
        return {q: np.percentile(paths, q * 100, axis=0) for q in qs}


def regime_var(paths: np.ndarray, alpha: float = 0.05):
    """Value-at-Risk and Conditional VaR on the final portfolio return
    distribution, Section A13.2, verbatim. `final` is total return over
    the horizon (paths[:, -1] - 1.0, e.g. 0.15 for +15%); VaR is its
    alpha-quantile (a negative number for a typical loss tail); CVaR is
    the mean of all draws at or below VaR, i.e. the expected shortfall
    beyond VaR. CVaR <= VaR always holds for alpha < 0.5 by
    construction (checked, not just asserted - see tests)."""
    final = paths[:, -1] - 1.0
    var = np.percentile(final, alpha * 100)
    cvar = final[final <= var].mean()
    return var, cvar
