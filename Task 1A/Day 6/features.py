"""Feature panel for the multivariate RS-VAR, Section A8.1/A8.3.

The spec's feature list is "Nifty return, vol, breadth, FII, DII, INR,
gilt" (7 series). Breadth is not built here: it needs per-constituent
stock data not in the required panel (same documented gap as Day 3's
`src/features/engineer.py`). The six series that ARE available differ by
orders of magnitude in scale (returns ~0.01, FII/DII flows in hundreds to
thousands of crore, gilt yield changes ~0.001) - fed raw into a VAR, this
would make the `A` coefficient matrices and their Normal(0, 0.3) prior in
Section A8.3 meaningless (a coefficient of 0.3 on a raw FII-flow input
means something totally different from 0.3 on a raw return input).
Every column here is standardised (z-scored on its own training-window
mean/std) before being handed to any model; `RSVARPanel` keeps the
scaling factors so results can be converted back to original units.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_NAMES = ["nifty_ret", "vol_21d", "fii_flow", "dii_flow", "inr_ret", "gilt_chg"]


@dataclass
class RSVARPanel:
    Y: np.ndarray          # (T, d) standardised feature matrix
    index: pd.DatetimeIndex
    feature_names: list
    means: np.ndarray       # (d,) - original-units mean used to standardise
    stds: np.ndarray        # (d,) - original-units std used to standardise

    def to_original_units(self, standardized_values: np.ndarray) -> np.ndarray:
        """Convert a (..., d) array of standardised values back to
        original units - e.g. to report an impulse response in actual
        return/flow/yield terms rather than standard deviations."""
        return standardized_values * self.stds + self.means


def build_rsvar_panel(data: dict[str, pd.DataFrame], tail_days: int | None = None) -> RSVARPanel:
    """Build the standardised (T, 6) feature panel from the data dict
    returned by `load_market_data`. `tail_days`, if given, restricts to
    the most recent N trading days (used the same way Day 5 restricted
    its window, for compute-budget reasons - see docs/day6_rsvar_notes.md)."""
    nifty = data["nifty50"]["close"]
    usdinr = data["usdinr"]["value"]
    gilt = data["gilt_10y"]["value"]
    fii = data["fii_dii_flows"]["fii_cr"]
    dii = data["fii_dii_flows"]["dii_cr"]

    nifty_ret = nifty.pct_change()
    vol_21d = nifty_ret.rolling(21).std() * np.sqrt(252)
    inr_ret = usdinr.pct_change()
    gilt_chg = gilt.diff()

    panel = pd.DataFrame({
        "nifty_ret": nifty_ret,
        "vol_21d": vol_21d,
        "fii_flow": fii,
        "dii_flow": dii,
        "inr_ret": inr_ret,
        "gilt_chg": gilt_chg,
    }).dropna()

    if tail_days is not None:
        panel = panel.tail(tail_days)

    means = panel.mean().values
    stds = panel.std().values
    standardized = (panel.values - means) / stds

    return RSVARPanel(Y=standardized, index=panel.index, feature_names=FEATURE_NAMES,
                       means=means, stds=stds)
