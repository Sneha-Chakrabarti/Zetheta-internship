"""Feature engineering for regime classification: Section A4.5 plus the
Day 3 additions (Section D2) - cap-segmented, flow, and macro features.

Section A4.5's illustrative code assumes one merged DataFrame with columns
like 'IndiaVIX', 'Advances', 'AAA10Y'. Our actual schema (`src/data/
loader.py`) is a dict of separate DataFrames, one per series, because
that is how the data actually arrives (index OHLC, single-column macro
series, two-column flows). `engineer_regime_features` below takes that
dict directly rather than requiring a pre-merge step.

Three items the task document asks for are NOT implemented here, and are
left out rather than faked, because the required data panel (Section A1)
does not contain what they need:

- Breadth features (advance/decline ratio, % above 50DMA, new highs minus
  new lows) need per-constituent stock data for all Nifty 50 members.
  The panel has only three index-level OHLC series, no constituent data.
- Real interest rates need an inflation series (CPI). Not in the panel.
- INR "DXY-orthogonalised" needs a US Dollar Index series. Not in the
  panel.

Two related items ARE implemented, but as clearly labelled proxies for
data we do not have:

- `rbi_stance_proxy`: no policy-rate or MPC-decision series is in the
  panel, so this proxies monetary stance with the 63-day change in the
  10Y gilt yield (yields drift up ahead of a hawkish hold, down ahead of
  a cut). This is a proxy, not a replacement for actual repo-rate data.
- `midcap_large_ratio_z` / `smallcap_large_ratio_z`: the task document
  calls for a "mid-cap valuation z-score", which properly needs P/E or
  P/B data (not in the panel). These use a rolling z-score of the
  price RATIO between segments instead - a relative-price proxy, not a
  fundamentals-based valuation measure.

See `docs/day3_feature_engineering_notes.md` for the full rationale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _log_ret(close: pd.Series, periods: int = 1) -> pd.Series:
    return np.log(close).diff(periods)


def _rolling_z(series: pd.Series, window: int = 252) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def engineer_regime_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the tabular feature matrix for Indian equity regime
    classification from the dict returned by `load_market_data`.

    Returns a DataFrame indexed on the Nifty 50 trading calendar (the
    other daily series are expected to share that calendar; SIP totals,
    being monthly, are forward-filled onto it). Rows with any NaN from
    warm-up windows (the 252-day rolling z-scores need a year of history)
    are dropped, matching A4.5's own `.dropna()` convention.
    """
    nifty = data["nifty50"]["close"]
    midcap = data["nifty_midcap100"]["close"]
    smallcap = data["nifty_smallcap100"]["close"]
    vix = data["india_vix"]["value"]
    usdinr = data["usdinr"]["value"]
    gilt = data["gilt_10y"]["value"]
    spread = data["aaa_gilt_spread"]["value"]
    fii = data["fii_dii_flows"]["fii_cr"]
    dii = data["fii_dii_flows"]["dii_cr"]
    sip = data["sip_totals"]["sip_cr"]

    f = pd.DataFrame(index=nifty.index)

    # --- Return features (A4.5) ---
    f["ret_1d"] = nifty.pct_change()
    f["ret_5d"] = nifty.pct_change(5)
    f["ret_21d"] = nifty.pct_change(21)
    f["ret_63d"] = nifty.pct_change(63)

    # --- Trend features (A4.5) ---
    f["ma_50_200"] = nifty.rolling(50).mean() / nifty.rolling(200).mean() - 1
    f["above_200dma"] = (nifty > nifty.rolling(200).mean()).astype(int)

    # --- Volatility features (A4.5) ---
    f["vol_21d"] = nifty.pct_change().rolling(21).std() * np.sqrt(252)
    f["vol_63d"] = nifty.pct_change().rolling(63).std() * np.sqrt(252)
    f["vol_ratio"] = f["vol_21d"] / f["vol_63d"]
    # Day 3 addition: is realised vol itself in an unusual regime relative
    # to its own trailing year, distinct from the level-vs-level vol_ratio.
    f["vol_21d_z"] = _rolling_z(f["vol_21d"], 252)

    # --- India VIX features (A4.5) ---
    f["vix_level"] = vix.reindex(f.index)
    f["vix_change_5d"] = vix.pct_change(5).reindex(f.index)
    f["vix_z"] = _rolling_z(vix, 252).reindex(f.index)

    # --- Breadth features: NOT IMPLEMENTED, see module docstring ---

    # --- Macro features (A4.5 + Day 3) ---
    f["gilt_10y_change_21d"] = gilt.diff(21).reindex(f.index)
    f["inr_change_21d"] = usdinr.pct_change(21).reindex(f.index)
    # Our schema already carries the AAA-gilt spread as its own series
    # (not separate AAA and gilt yield levels to subtract), so credit_spread
    # here is that series directly, unlike A4.5's `AAA10Y - Gilt10Y` form.
    f["credit_spread"] = spread.reindex(f.index)
    f["spread_change_21d"] = spread.diff(21).reindex(f.index)
    # Day 3: monetary-stance proxy (see module docstring for the caveat).
    f["rbi_stance_proxy"] = gilt.diff(63).reindex(f.index)
    # Day 3: currency-stress vol, distinct from the level-change feature above.
    f["usdinr_vol_21d"] = (usdinr.pct_change().rolling(21).std() * np.sqrt(252)).reindex(f.index)
    # real_rate and inr_dxy_orthogonalised: NOT IMPLEMENTED, see module docstring.

    # --- Flow features (A4.5 + Day 3) ---
    f["fii_eq_5d"] = fii.rolling(5).sum().reindex(f.index)
    f["dii_eq_5d"] = dii.rolling(5).sum().reindex(f.index)
    f["flow_balance"] = f["dii_eq_5d"] / (f["fii_eq_5d"].abs() + 1e-9)
    f["fii_z"] = _rolling_z(fii, 252).reindex(f.index)
    f["dii_z"] = _rolling_z(dii, 252).reindex(f.index)
    # SIP is monthly; the month-over-month growth rate is the "SIP
    # momentum" the task asks for, forward-filled onto the daily calendar
    # between month-starts (a level change only exists once a month).
    sip_mom_monthly = sip.pct_change()
    f["sip_momentum"] = sip_mom_monthly.reindex(f.index, method="ffill")

    # --- Cap-segmented features (Day 3) ---
    mid_ret_21d = midcap.pct_change(21)
    small_ret_21d = smallcap.pct_change(21)
    mid_ret_63d = midcap.pct_change(63)
    small_ret_63d = smallcap.pct_change(63)
    f["midcap_rel_perf_21d"] = (mid_ret_21d - f["ret_21d"]).reindex(f.index)
    f["smallcap_rel_perf_21d"] = (small_ret_21d - f["ret_21d"]).reindex(f.index)
    f["midcap_rel_perf_63d"] = (mid_ret_63d - f["ret_63d"]).reindex(f.index)
    f["smallcap_rel_perf_63d"] = (small_ret_63d - f["ret_63d"]).reindex(f.index)
    # "Valuation z-score" proxy - see module docstring caveat.
    mid_large_ratio = (midcap / nifty).reindex(f.index)
    small_large_ratio = (smallcap / nifty).reindex(f.index)
    f["midcap_large_ratio_z"] = _rolling_z(mid_large_ratio, 252)
    f["smallcap_large_ratio_z"] = _rolling_z(small_large_ratio, 252)

    return f.dropna()


FEATURE_GROUPS: dict[str, list[str]] = {
    "return": ["ret_1d", "ret_5d", "ret_21d", "ret_63d"],
    "trend": ["ma_50_200", "above_200dma"],
    "volatility": ["vol_21d", "vol_63d", "vol_ratio", "vol_21d_z"],
    "vix": ["vix_level", "vix_change_5d", "vix_z"],
    "macro": ["gilt_10y_change_21d", "inr_change_21d", "credit_spread",
              "spread_change_21d", "rbi_stance_proxy", "usdinr_vol_21d"],
    "flow": ["fii_eq_5d", "dii_eq_5d", "flow_balance", "fii_z", "dii_z", "sip_momentum"],
    "cap_segment": ["midcap_rel_perf_21d", "smallcap_rel_perf_21d",
                     "midcap_rel_perf_63d", "smallcap_rel_perf_63d",
                     "midcap_large_ratio_z", "smallcap_large_ratio_z"],
}
