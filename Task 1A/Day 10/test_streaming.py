import numpy as np
import pandas as pd
import pytest

from src.models.sequential.streaming import (
    StreamingDirichletTransitions, StreamingBetaBernoulli, reconciliation_diagnostic,
)


def test_dirichlet_streaming_matches_batch_update_sequence():
    rng = np.random.default_rng(0)
    K = 4
    states = rng.integers(0, K, 500)

    s1 = StreamingDirichletTransitions(K, alpha_prior=1.0)
    for prev, curr in zip(states[:-1], states[1:]):
        s1.update(int(prev), int(curr))

    s2 = StreamingDirichletTransitions(K, alpha_prior=1.0)
    s2.update_sequence(states)

    np.testing.assert_allclose(s1.posterior_mean(), s2.posterior_mean())


def test_dirichlet_streaming_matches_independent_manual_computation():
    rng = np.random.default_rng(1)
    K = 3
    states = rng.integers(0, K, 300)
    s = StreamingDirichletTransitions(K, alpha_prior=1.0)
    s.update_sequence(states)

    counts = np.zeros((K, K))
    for prev, curr in zip(states[:-1], states[1:]):
        counts[prev, curr] += 1
    expected = (1.0 + counts) / (1.0 + counts).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(s.posterior_mean(), expected)


def test_dirichlet_posterior_is_order_invariant():
    """Conjugacy means the posterior depends only on total counts, not
    the sequence in which updates arrive - checked, not assumed."""
    rng = np.random.default_rng(2)
    K = 4
    states = rng.integers(0, K, 200)
    pairs = list(zip(states[:-1], states[1:]))

    s_original = StreamingDirichletTransitions(K)
    for prev, curr in pairs:
        s_original.update(int(prev), int(curr))

    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    s_shuffled = StreamingDirichletTransitions(K)
    for prev, curr in shuffled:
        s_shuffled.update(int(prev), int(curr))

    np.testing.assert_allclose(s_original.posterior_mean(), s_shuffled.posterior_mean())


def test_dirichlet_posterior_mean_rows_sum_to_one():
    s = StreamingDirichletTransitions(3, alpha_prior=2.0)
    s.update_sequence(np.array([0, 1, 2, 0, 1, 1, 2, 0]))
    np.testing.assert_allclose(s.posterior_mean().sum(axis=1), 1.0)


def test_dirichlet_credible_interval_contains_posterior_mean():
    s = StreamingDirichletTransitions(3, alpha_prior=1.0)
    s.update_sequence(np.array([0, 0, 0, 1, 0, 0, 2, 0]))
    mean = s.posterior_mean()[0, 0]
    lo, hi = s.posterior_credible_interval(0, 0, width=0.9)
    assert lo <= mean <= hi


def test_beta_bernoulli_streaming_matches_batch_and_manual():
    rng = np.random.default_rng(3)
    outcomes = rng.random(300) < 0.3

    b_stream = StreamingBetaBernoulli(1.0, 1.0)
    for o in outcomes:
        b_stream.update(bool(o))

    b_batch = StreamingBetaBernoulli(1.0, 1.0)
    b_batch.update_sequence(outcomes)

    n_success = int(outcomes.sum())
    n_total = len(outcomes)
    manual_mean = (1.0 + n_success) / (1.0 + n_success + 1.0 + n_total - n_success)

    assert b_stream.posterior_mean() == pytest.approx(b_batch.posterior_mean())
    assert b_stream.posterior_mean() == pytest.approx(manual_mean)


def test_beta_bernoulli_credible_interval_narrows_with_more_data():
    rng = np.random.default_rng(4)
    b_few = StreamingBetaBernoulli(1.0, 1.0)
    b_few.update_sequence(rng.random(10) < 0.4)
    b_many = StreamingBetaBernoulli(1.0, 1.0)
    b_many.update_sequence(rng.random(1000) < 0.4)

    lo_few, hi_few = b_few.posterior_credible_interval(0.9)
    lo_many, hi_many = b_many.posterior_credible_interval(0.9)
    assert (hi_many - lo_many) < (hi_few - lo_few)


def test_reconciliation_diagnostic_zero_when_online_equals_batch():
    probs = np.tile([0.7, 0.2, 0.1], (10, 1))
    dates = pd.date_range("2024-01-01", periods=10)
    days_since = np.arange(10)
    df = reconciliation_diagnostic(probs, probs, dates, days_since)
    assert (df["tv_distance"] == 0).all()
    assert list(df.columns) == ["date", "days_since_batch_refit", "tv_distance"]


def test_reconciliation_diagnostic_matches_hand_computed_tv_distance():
    online = np.array([[0.9, 0.1], [0.5, 0.5]])
    batch = np.array([[0.5, 0.5], [0.5, 0.5]])
    df = reconciliation_diagnostic(online, batch, ["d0", "d1"], [0, 1])
    np.testing.assert_allclose(df["tv_distance"].values, [0.4, 0.0])
