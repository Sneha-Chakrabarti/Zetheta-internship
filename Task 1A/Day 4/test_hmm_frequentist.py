import warnings

import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataConfig, load_market_data
from src.models.hmm.frequentist import (
    fit_regime_hmm, fit_regime_hmm_robust, label_regimes,
    regime_duration_stats, transition_matrix_stats, compare_by_bic,
    REGIME_LABELS_5,
)

warnings.filterwarnings("ignore")  # hmmlearn's own convergence/degeneracy warnings

DATA = load_market_data(DataConfig(backend="synthetic", n_years=15.0, seed=7))
RETURNS = DATA["nifty50"]["close"].pct_change().dropna()


def test_fit_regime_hmm_runs_and_shapes_are_consistent():
    model, states, probs = fit_regime_hmm(RETURNS, n_states=5, seed=42)
    assert len(states) == len(RETURNS)
    assert probs.shape == (len(RETURNS), 5)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_label_regimes_rejects_non_five_states():
    model, _, _ = fit_regime_hmm(RETURNS, n_states=3, seed=42)
    with pytest.raises(ValueError):
        label_regimes(model, K=3)


def test_label_regimes_returns_all_five_labels_exactly_once():
    model, _, _ = fit_regime_hmm(RETURNS, n_states=5, seed=42)
    mapping = label_regimes(model, K=5)
    assert set(mapping.values()) == set(REGIME_LABELS_5)
    assert len(mapping) == 5


def test_fit_regime_hmm_robust_never_returns_a_collapsed_state():
    """Regression test for the degeneracy found on Day 4: a single-seed
    fit at seed=42 leaves a state with zero observations. The robust
    multi-restart fit must not."""
    model, states, probs, seed_used, n_valid = fit_regime_hmm_robust(
        RETURNS, n_states=5, n_restarts=20, base_seed=42
    )
    counts = np.bincount(states, minlength=5)
    assert (counts > 0).all()
    assert n_valid >= 1
    assert np.all(np.isfinite(model.means_))
    assert np.all(np.isfinite(model.covars_))


def test_fit_regime_hmm_robust_raises_when_nothing_survives():
    with pytest.raises(RuntimeError):
        fit_regime_hmm_robust(RETURNS, n_states=5, n_restarts=2,
                               base_seed=42, min_state_share=0.49)


def test_regime_duration_stats_run_lengths_sum_to_series_length():
    states = np.array([0, 0, 0, 1, 1, 2, 0, 0])
    stats = regime_duration_stats(states)
    # Runs in order: (0, len 3), (1, len 2), (2, len 1), (0, len 2) ->
    # state 0 has 2 runs (lengths 3 and 2), state 1 has 1 run, state 2 has 1 run.
    assert stats.loc[0, "count"] == 2
    assert stats.loc[1, "count"] == 1
    assert stats.loc[2, "count"] == 1
    total_days = (stats["count"] * stats["mean"]).sum()
    assert total_days == len(states)


def test_transition_matrix_stats_rows_sum_to_one_or_zero():
    model, states, _, _, _ = fit_regime_hmm_robust(RETURNS, n_states=5, n_restarts=20, base_seed=42)
    label_map = label_regimes(model, K=5)
    trans_df, stick_df = transition_matrix_stats(model, label_map)
    row_sums = trans_df.sum(axis=1)
    # hmmlearn leaves a row at all-zero (not 1) if a state was never
    # visited in decoding, so allow either.
    assert ((row_sums.round(6) == 1.0) | (row_sums.round(6) == 0.0)).all()
    assert (stick_df["self_transition_prob"] >= 0).all()
    assert (stick_df["self_transition_prob"] < 1).all()


def test_compare_by_bic_returns_a_row_per_state_count_even_when_one_fails():
    result = compare_by_bic(RETURNS, state_counts=(3, 5), n_restarts=10, base_seed=1)
    assert list(result.index) == [3, 5]
    assert {"log_likelihood", "aic", "bic", "valid_restarts"}.issubset(result.columns)
