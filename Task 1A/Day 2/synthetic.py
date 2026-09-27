"""Synthetic Indian-market-like dataset with a known regime path.

Purpose: let every downstream module (features, HMM, RS-VAR, BDL, conformal,
backtest) be built and unit-tested before real NSE/RBI/AMFI data is wired
in, and give the changepoint/BOCPD code in Day 10 a ground-truth regime
path to validate against. This is a development and testing aid, not a
market model: parameters are illustrative (see regimes.py docstring), not
fitted, and must never be presented as empirical Indian-market statistics
in the report or model card.

Schema mirrors what src/data/loader.py expects from the CSV backend, so
switching between "synthetic" and "csv" is a one-line config change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .regimes import REGIME_NAMES, DEFAULT_REGIME_SPECS, sample_regime_path

TRADING_DAYS_PER_YEAR = 252


def _ohlc_from_close(close: np.ndarray, daily_vol: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Build a plausible OHLC frame from a close series plus a per-day vol
    estimate, scaling the intraday range with vol so high-vol regimes show
    visibly wider bars."""
    n = len(close)
    prev_close = np.empty(n)
    prev_close[0] = close[0] / (1 + rng.normal(0, 0.001))
    prev_close[1:] = close[:-1]
    open_ = prev_close * (1 + rng.normal(0, daily_vol * 0.15, n))
    intraday_range = np.abs(rng.normal(0, daily_vol * 0.6, n))
    high = np.maximum(open_, close) * (1 + intraday_range)
    low = np.minimum(open_, close) * (1 - intraday_range)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def generate_synthetic_market(
    n_years: float = 15.0,
    seed: int = 7,
    start_date: str = "2011-01-03",
    start_regime: str = "risk_on",
) -> dict[str, pd.DataFrame]:
    """Generate the full multi-asset panel described in Section A11/D2 of
    the task document, plus the hidden regime path (for validation only;
    downstream models must not consume `regime_path` directly).

    Returns a dict of DataFrames, each indexed by business-day DatetimeIndex
    except `sip_totals` (monthly) and `regime_path` (daily, ground truth):
      nifty50, nifty_midcap100, nifty_smallcap100   -> OHLC
      india_vix, usdinr, gilt_10y, aaa_gilt_spread   -> single 'value' column
      fii_dii_flows                                   -> 'fii_cr', 'dii_cr'
      sip_totals                                      -> monthly 'sip_cr'
      regime_path                                     -> 'regime' (string label)
    """
    rng = np.random.default_rng(seed)
    n_days = int(n_years * TRADING_DAYS_PER_YEAR)
    dates = pd.bdate_range(start=start_date, periods=n_days)

    regime_idx = sample_regime_path(n_days, rng, start_regime=start_regime)
    regime_labels = np.array(REGIME_NAMES)[regime_idx]
    specs = DEFAULT_REGIME_SPECS

    # Per-day parameter arrays, looked up by the day's regime.
    drift = np.array([specs[REGIME_NAMES[i]].nifty_drift for i in regime_idx]) / TRADING_DAYS_PER_YEAR
    vol = np.array([specs[REGIME_NAMES[i]].nifty_vol for i in regime_idx]) / np.sqrt(TRADING_DAYS_PER_YEAR)
    mid_beta = np.array([specs[REGIME_NAMES[i]].midcap_beta for i in regime_idx])
    mid_extra = np.array([specs[REGIME_NAMES[i]].midcap_extra_vol for i in regime_idx]) / np.sqrt(TRADING_DAYS_PER_YEAR)
    small_beta = np.array([specs[REGIME_NAMES[i]].smallcap_beta for i in regime_idx])
    small_extra = np.array([specs[REGIME_NAMES[i]].smallcap_extra_vol for i in regime_idx]) / np.sqrt(TRADING_DAYS_PER_YEAR)

    # Common market factor shared across cap segments, plus idiosyncratic
    # noise per segment, so large/mid/small are correlated but not identical.
    common_shock = rng.normal(0, 1, n_days)
    large_idio = rng.normal(0, 1, n_days)
    mid_idio = rng.normal(0, 1, n_days)
    small_idio = rng.normal(0, 1, n_days)

    # Total volatility per segment is beta-scaled Nifty vol plus idiosyncratic
    # extra vol, added in quadrature (independent components). The mixing
    # weights below only shape *correlation* with the common factor (a unit
    # variance combination of common_shock and idio); they must not also be
    # used to scale the overall magnitude, or beta and extra_vol stop doing
    # what their names say - a bug an earlier version of this function had:
    # scaling the mixed shock by the correlation weight (e.g. 0.75x) instead
    # of by beta shrank total mid/small-cap vol below the large-cap value.
    large_vol = vol
    mid_vol = np.sqrt((mid_beta * vol) ** 2 + mid_extra**2)
    small_vol = np.sqrt((small_beta * vol) ** 2 + small_extra**2)

    large_ret = drift + large_vol * (0.85 * common_shock + 0.15 * large_idio) / np.sqrt(0.85**2 + 0.15**2)
    mid_ret = mid_beta * drift + mid_vol * (0.75 * common_shock + 0.25 * mid_idio) / np.sqrt(0.75**2 + 0.25**2)
    small_ret = small_beta * drift + small_vol * (0.65 * common_shock + 0.35 * small_idio) / np.sqrt(0.65**2 + 0.35**2)

    nifty_close = 5000.0 * np.exp(np.cumsum(large_ret))
    midcap_close = 8000.0 * np.exp(np.cumsum(mid_ret))
    smallcap_close = 3000.0 * np.exp(np.cumsum(small_ret))

    nifty50 = _ohlc_from_close(nifty_close, large_vol, rng)
    nifty_midcap100 = _ohlc_from_close(midcap_close, mid_vol, rng)
    nifty_smallcap100 = _ohlc_from_close(smallcap_close, small_vol, rng)
    for df in (nifty50, nifty_midcap100, nifty_smallcap100):
        df.index = dates

    # India VIX: mean-reverting (OU) around the regime's level, in vol
    # points, with a same-day spike overlay whenever the regime just
    # changed, so changepoint detectors have something real to catch.
    vix_level = np.array([specs[REGIME_NAMES[i]].vix_level for i in regime_idx])
    vix_vol = np.array([specs[REGIME_NAMES[i]].vix_vol for i in regime_idx])
    changed = np.concatenate([[False], regime_idx[1:] != regime_idx[:-1]])
    vix = np.empty(n_days)
    vix[0] = vix_level[0]
    kappa = 0.08
    for t in range(1, n_days):
        spike = 6.0 if changed[t] else 0.0
        vix[t] = vix[t - 1] + kappa * (vix_level[t] - vix[t - 1]) + rng.normal(0, vix_vol[t]) + spike
    vix = np.clip(vix, 8.0, 90.0)
    india_vix = pd.DataFrame({"value": vix}, index=dates)

    # USD/INR: random walk with regime-dependent drift.
    inr_drift = np.array([specs[REGIME_NAMES[i]].inr_drift for i in regime_idx]) / TRADING_DAYS_PER_YEAR
    inr_vol = np.array([specs[REGIME_NAMES[i]].inr_vol for i in regime_idx]) / np.sqrt(TRADING_DAYS_PER_YEAR)
    inr = 45.0 * np.exp(np.cumsum(inr_drift + inr_vol * rng.normal(0, 1, n_days)))
    usdinr = pd.DataFrame({"value": inr}, index=dates)

    # 10Y Gilt yield: OU around regime level.
    gilt_level = np.array([specs[REGIME_NAMES[i]].gilt_level for i in regime_idx])
    gilt = np.empty(n_days)
    gilt[0] = gilt_level[0]
    for t in range(1, n_days):
        gilt[t] = gilt[t - 1] + 0.05 * (gilt_level[t] - gilt[t - 1]) + rng.normal(0, 0.0008)
    gilt_10y = pd.DataFrame({"value": gilt}, index=dates)

    spread_level = np.array([specs[REGIME_NAMES[i]].aaa_gilt_spread for i in regime_idx])
    spread = np.empty(n_days)
    spread[0] = spread_level[0]
    for t in range(1, n_days):
        spread[t] = spread[t - 1] + 0.1 * (spread_level[t] - spread[t - 1]) + rng.normal(0, 0.0003)
    aaa_gilt_spread = pd.DataFrame({"value": np.clip(spread, 0.002, 0.05)}, index=dates)

    # FII/DII flows: FII mean-reverts to a regime-dependent level; DII is
    # partially counter-cyclical (buys what FII sells), plus the
    # SIP-driven structural inflow.
    fii_mean = np.array([specs[REGIME_NAMES[i]].fii_flow_mean for i in regime_idx])
    fii_vol = np.array([specs[REGIME_NAMES[i]].fii_flow_vol for i in regime_idx])
    fii = np.empty(n_days)
    fii[0] = fii_mean[0]
    for t in range(1, n_days):
        fii[t] = fii[t - 1] + 0.2 * (fii_mean[t] - fii[t - 1]) + rng.normal(0, fii_vol[t])
    dii = -0.5 * fii + 600.0 + rng.normal(0, 900.0, n_days)
    fii_dii_flows = pd.DataFrame({"fii_cr": fii, "dii_cr": dii}, index=dates)

    # SIP totals: monthly, slow structural uptrend, close to regime-insensitive
    # (this is the point made in Section A1: SIP flow is largely price-insensitive).
    month_starts = pd.date_range(start=dates[0], end=dates[-1], freq="MS")
    months_elapsed = np.arange(len(month_starts))
    sip = 8000.0 * (1 + 0.012) ** months_elapsed + rng.normal(0, 150.0, len(month_starts))
    sip_totals = pd.DataFrame({"sip_cr": np.clip(sip, 0, None)}, index=month_starts)

    regime_path = pd.DataFrame({"regime": regime_labels}, index=dates)

    return {
        "nifty50": nifty50,
        "nifty_midcap100": nifty_midcap100,
        "nifty_smallcap100": nifty_smallcap100,
        "india_vix": india_vix,
        "usdinr": usdinr,
        "gilt_10y": gilt_10y,
        "aaa_gilt_spread": aaa_gilt_spread,
        "fii_dii_flows": fii_dii_flows,
        "sip_totals": sip_totals,
        "regime_path": regime_path,
    }
