"""Streaming conjugate posterior updates and the online/batch
reconciliation diagnostic, Section A9.3.

Dirichlet-categorical and Beta-Bernoulli conjugacy mean the posterior
after observing data one point at a time is mathematically IDENTICAL to
the posterior from observing the same data all at once: the posterior
depends only on total counts, not the order or batching of updates. Both
classes below are checked against this property directly
(`tests/test_streaming.py`), not just implemented and assumed correct.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class StreamingDirichletTransitions:
    """Tracks a K-regime transition-count matrix incrementally and
    exposes the Dirichlet-posterior-mean transition matrix at any point,
    without ever re-reading past transitions - the "streaming" part of
    the nightly-batch / intraday-online design: the intraday side only
    needs O(K^2) state, not the full history.
    """

    def __init__(self, K: int, alpha_prior: np.ndarray | float = 1.0):
        self.K = K
        self.alpha_prior = np.full((K, K), alpha_prior) if np.isscalar(alpha_prior) else np.asarray(alpha_prior)
        self.counts = np.zeros((K, K))

    def update(self, prev_state: int, curr_state: int) -> None:
        self.counts[prev_state, curr_state] += 1

    def update_sequence(self, states: np.ndarray) -> None:
        for prev, curr in zip(states[:-1], states[1:]):
            self.update(int(prev), int(curr))

    def posterior_mean(self) -> np.ndarray:
        """Row-stochastic transition matrix: each row's Dirichlet
        posterior mean, (alpha_prior + counts) normalised."""
        posterior_alpha = self.alpha_prior + self.counts
        return posterior_alpha / posterior_alpha.sum(axis=1, keepdims=True)

    def posterior_credible_interval(self, row: int, col: int, width: float = 0.9) -> tuple[float, float]:
        """A `width` credible interval for one transition PROBABILITY
        (row -> col), using the marginal Beta distribution of a single
        Dirichlet component: Beta(alpha_row_col, sum(alpha_row) - alpha_row_col)."""
        from scipy.stats import beta as beta_dist

        posterior_alpha = self.alpha_prior + self.counts
        a = posterior_alpha[row, col]
        b = posterior_alpha[row].sum() - a
        lo = (1 - width) / 2
        return float(beta_dist.ppf(lo, a, b)), float(beta_dist.ppf(1 - lo, a, b))


class StreamingBetaBernoulli:
    """Tracks a single Beta-Bernoulli posterior incrementally (e.g. "did
    a regime change happen today: yes/no"), for the same streaming/batch
    equivalence reason as `StreamingDirichletTransitions`."""

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.n_success = 0
        self.n_total = 0

    def update(self, outcome: bool) -> None:
        self.n_total += 1
        if outcome:
            self.n_success += 1

    def update_sequence(self, outcomes) -> None:
        for o in outcomes:
            self.update(bool(o))

    def posterior_mean(self) -> float:
        a = self.alpha_prior + self.n_success
        b = self.beta_prior + (self.n_total - self.n_success)
        return a / (a + b)

    def posterior_credible_interval(self, width: float = 0.9) -> tuple[float, float]:
        from scipy.stats import beta as beta_dist

        a = self.alpha_prior + self.n_success
        b = self.beta_prior + (self.n_total - self.n_success)
        lo = (1 - width) / 2
        return float(beta_dist.ppf(lo, a, b)), float(beta_dist.ppf(1 - lo, a, b))


def reconciliation_diagnostic(online_probs: np.ndarray, batch_probs: np.ndarray,
                               dates, days_since_batch: np.ndarray) -> pd.DataFrame:
    """The online/batch reconciliation check Section A9.3 calls a
    mandatory model-health diagnostic: does the fast online posterior
    (e.g. the particle filter, updated every day) agree with the last
    nightly-batch posterior (e.g. a frozen HMM/RS-VAR refit) on the
    overlap window between refits?

    online_probs, batch_probs: (N, K) regime-probability arrays over the
    same N days, same K regimes, same column order.
    days_since_batch: (N,) integer, days elapsed since the batch model
    was last refit as of each day - the expected axis of degradation
    (the online estimate should track the batch estimate closely right
    after a refit and drift further from it the longer the batch
    reference goes stale).

    Returns a per-day DataFrame with total-variation distance and the
    days-since-refit, for plotting the expected "staleness" relationship
    rather than asserting it holds without checking.
    """
    tv = 0.5 * np.abs(online_probs - batch_probs).sum(axis=1)
    return pd.DataFrame({"date": dates, "days_since_batch_refit": days_since_batch, "tv_distance": tv})
