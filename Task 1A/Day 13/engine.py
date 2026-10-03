"""Backtest engine: apply an allocation overlay's daily equity weights
to realised returns, and compute the performance metrics Day 13 asks
for (Information Ratio, tracking error, regime-conditioned drawdowns).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def overlay_portfolio_returns(asset_returns: pd.Series, equity_weights: pd.Series,
                               cash_return: float = 0.0) -> pd.Series:
    """Daily portfolio return: equity_weight * asset_return +
    (1 - equity_weight) * cash_return. `equity_weights` must be
    LAGGED relative to `asset_returns` by the caller (the weight set at
    the end of day t-1 earns day t's return) - not done here, since the
    right lag convention depends on when the regime signal is actually
    available, a decision the caller should make explicitly rather than
    have silently baked into this function."""
    weights = equity_weights.reindex(asset_returns.index)
    return weights * asset_returns + (1 - weights) * cash_return


def cumulative_from_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    cum = cumulative_from_returns(returns)
    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    return float(drawdown.min())


def information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series,
                       periods_per_year: int = 252) -> float:
    """Annualised mean active return / annualised tracking error. NaN if
    tracking error is zero (strategy == benchmark every day), rather than
    raising or silently returning inf."""
    active = strategy_returns - benchmark_returns
    te = active.std() * np.sqrt(periods_per_year)
    if te == 0:
        return float("nan")
    return float((active.mean() * periods_per_year) / te)


def tracking_error(strategy_returns: pd.Series, benchmark_returns: pd.Series,
                    periods_per_year: int = 252) -> float:
    active = strategy_returns - benchmark_returns
    return float(active.std() * np.sqrt(periods_per_year))


def regime_conditioned_drawdowns(returns: pd.Series, regime_labels: pd.Series) -> pd.DataFrame:
    """Max drawdown of `returns`, computed separately within each
    contiguous run of a single regime label (not within each label
    pooled across all its occurrences - a drawdown spanning two
    non-adjacent Risk-Off episodes is not one drawdown). Returns one row
    per contiguous run, with its regime label, start/end dates, and that
    run's own max drawdown."""
    labels = regime_labels.reindex(returns.index)
    run_id = (labels != labels.shift()).cumsum()
    rows = []
    for rid, idx in returns.groupby(run_id).groups.items():
        sub_returns = returns.loc[idx]
        rows.append({
            "regime": labels.loc[idx[0]],
            "start": idx[0], "end": idx[-1], "n_days": len(idx),
            "max_drawdown": max_drawdown(sub_returns),
            "total_return": float(cumulative_from_returns(sub_returns).iloc[-1] - 1.0),
        })
    return pd.DataFrame(rows)
