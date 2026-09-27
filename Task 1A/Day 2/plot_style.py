"""Shared figure style for every plot in this project.

Conventions (apply everywhere, not just here):
- Single plot with all curves on it, not a grid of subplots, unless the
  series genuinely cannot share an axis.
- Clean lines: no fill-between/shaded shadow bands around a mean line.
- White backgrounds.
- When curves overlap or differ by orders of magnitude, rescale a series
  and say so in the legend label (e.g. "FII flow (x1e-3)") rather than
  using a second y-axis or a log scale by default.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.1,
        "font.size": 10,
        "figure.figsize": (9, 4.5),
        "legend.frameon": False,
    })


def new_figure():
    """Return (fig, ax) for a single-panel plot with the house style applied."""
    apply_style()
    fig, ax = plt.subplots()
    return fig, ax
