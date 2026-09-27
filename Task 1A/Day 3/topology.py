"""Persistent homology features from rolling correlation matrices,
Section A7.2 of the task document.

The task document's example applies this to a correlation matrix over
"the universe of stocks/sectors". We have neither (see
`docs/day3_feature_engineering_notes.md`): the panel has nine series
total (three cap-segment indices plus six single-series macro/flow
inputs). We build the rolling correlation matrix over those nine, which
is enough for the Vietoris-Rips/persistence-landscape machinery to be
demonstrated correctly and to produce genuine structure (the point cloud
has real, non-trivial topology - VIX and gilt yields do not move with
the cap-segment indices the way large/mid/small move with each other),
but it is not the "universe of stocks" the illustrative example assumes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceLandscape

# The nine return/change series that make up the correlation-matrix point
# cloud. Order matters only for reproducibility of column order below.
TOPOLOGY_SERIES = [
    "nifty_ret", "midcap_ret", "smallcap_ret", "vix_chg",
    "usdinr_ret", "gilt_chg", "spread_chg", "fii_flow", "dii_flow",
]


def build_return_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble the nine-series daily change panel used for both the
    rolling correlation matrices here and, incidentally, a sane basis for
    a sector/asset graph in `src/features/sector_gnn.py`."""
    panel = pd.DataFrame({
        "nifty_ret": data["nifty50"]["close"].pct_change(),
        "midcap_ret": data["nifty_midcap100"]["close"].pct_change(),
        "smallcap_ret": data["nifty_smallcap100"]["close"].pct_change(),
        "vix_chg": data["india_vix"]["value"].diff(),
        "usdinr_ret": data["usdinr"]["value"].pct_change(),
        "gilt_chg": data["gilt_10y"]["value"].diff(),
        "spread_chg": data["aaa_gilt_spread"]["value"].diff(),
        "fii_flow": data["fii_dii_flows"]["fii_cr"],
        "dii_flow": data["fii_dii_flows"]["dii_cr"],
    })
    return panel.dropna()


def rolling_correlation_matrices(panel: pd.DataFrame, window: int = 63, step: int = 5) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Rolling correlation matrices over `panel`, sampled every `step`
    days rather than daily. Persistent-homology computation cost grows
    with the number of matrices, and adjacent daily windows overlap by
    window-1 days, so the matrices barely move day to day - sampling every
    `step` days keeps the run fast without losing anything a daily
    cadence would show that this one does not.
    """
    dates = panel.index[window - 1 :: step]
    matrices = np.empty((len(dates), panel.shape[1], panel.shape[1]))
    for i, end_date in enumerate(dates):
        window_data = panel.loc[:end_date].tail(window)
        matrices[i] = window_data.corr().values
    return matrices, pd.DatetimeIndex(dates)


def topology_features(corr_matrices_window: np.ndarray) -> np.ndarray:
    """Extract persistence landscape features from rolling correlation
    matrices, exactly Section A7.2's recipe: correlation converted to a
    distance via d = sqrt(2*(1-rho)), Vietoris-Rips persistence in
    homology dimensions 0 and 1, summarised as a persistence landscape
    and flattened to one feature vector per input matrix.

    Note on the output: the earliest bins of the homology-dimension-0
    portion of the landscape are identical (to floating-point precision)
    across every window, for every dataset - not a bug here. A dimension-0
    bar always starts at filtration value 0 (a connected component exists
    from the start), so the landscape function near t=0 is a fixed
    slope-1 ramp regardless of the correlation matrix; the differences
    between windows only appear later in the landscape, near each bar's
    death time. Use the full flattened vector (or a summary like its L2
    norm) as a feature, not any individual early bin in isolation.

    corr_matrices_window: (T, N, N) array of correlation matrices.
    Returns: (T, n_layers * n_bins * n_homology_dims) feature array.
    """
    distances = np.sqrt(np.clip(2 * (1 - corr_matrices_window), 0, None))
    vr = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1], n_jobs=-1)
    diagrams = vr.fit_transform(distances)
    pl = PersistenceLandscape(n_layers=5, n_bins=100)
    landscapes = pl.fit_transform(diagrams)
    return landscapes.reshape(landscapes.shape[0], -1)


def topology_feature_frame(data: dict[str, pd.DataFrame], window: int = 63, step: int = 5) -> pd.DataFrame:
    """End-to-end: build the return panel, roll correlation matrices,
    extract persistence-landscape features, and return them indexed by
    the date each window ends on. Columns are unnamed (`pl_0000`
    style) since a flattened landscape has no individually meaningful
    dimension; use the whole vector as classifier input, not any single
    column.
    """
    panel = build_return_panel(data)
    matrices, dates = rolling_correlation_matrices(panel, window=window, step=step)
    features = topology_features(matrices)
    columns = [f"pl_{i:04d}" for i in range(features.shape[1])]
    return pd.DataFrame(features, index=dates, columns=columns)
