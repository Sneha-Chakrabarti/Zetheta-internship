"""Data quality checks shared by every day's ingestion work.

Kept generic (operates on any OHLC or single-column DataFrame with a
DatetimeIndex) so the same functions run against the synthetic panel now
and against real NSE/RBI/AMFI data later, with identical output shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def missing_value_report(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count NaNs per column for each named frame."""
    rows = []
    for name, df in frames.items():
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            rows.append({"series": name, "column": col, "n_missing": n_missing,
                         "pct_missing": round(100 * n_missing / len(df), 3)})
    return pd.DataFrame(rows)


def business_day_gap_report(index: pd.DatetimeIndex, max_ordinary_gap_days: int = 4) -> pd.DataFrame:
    """Find gaps between consecutive index dates longer than an ordinary
    weekend (allowing a little slack for a single Monday holiday).
    Gaps beyond `max_ordinary_gap_days` are flagged for review — they may
    be legitimate multi-day market holidays (Diwali, a long weekend) or a
    genuine data hole, and this check does not distinguish the two.
    """
    index = pd.DatetimeIndex(sorted(index))
    deltas = index.to_series().diff().dt.days.dropna()
    flagged = deltas[deltas > max_ordinary_gap_days]
    return pd.DataFrame({
        "gap_start": [index[i - 1] for i in flagged.index.map(index.get_loc)],
        "gap_end": flagged.index,
        "gap_days": flagged.values,
    }).reset_index(drop=True)


def large_jump_report(close: pd.Series, threshold: float = 0.08) -> pd.DataFrame:
    """Flag days where the log return exceeds `threshold` in absolute value.

    This is a proxy for corporate actions (splits, bonus issues) and data
    errors alike, not a classifier of which — each flagged date needs a
    human or a corporate-action calendar to disposition it. For an index
    (rather than a single stock) a flagged day more often reflects a
    genuine market shock than a corporate action, since index divisors
    already adjust for constituent-level actions.
    """
    log_ret = np.log(close).diff()
    flagged = log_ret[log_ret.abs() > threshold]
    return pd.DataFrame({"date": flagged.index, "log_return": flagged.values})


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    """Rolling Pearson correlation between two aligned return series."""
    a, b = a.align(b, join="inner")
    return a.rolling(window).corr(b)
