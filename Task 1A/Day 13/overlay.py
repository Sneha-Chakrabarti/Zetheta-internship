"""Allocation overlay: tilt rules conditioned on regime probabilities
and conviction.

A tilt rule needs a view on what each regime is actually worth. This
project's own synthetic regime taxonomy (`src/data/regimes.py`) gives
one directly, and it contains a real subtlety worth stating rather than
assuming away: Post-Shock has the HIGHEST annualised drift of any
regime (0.25, against Risk-On's 0.15), reflecting a recovery rally, not
continued stress. A tilt rule built on regime NAMES alone ("Post-Shock"
sounds bad, de-risk) would get this backwards; a tilt rule built on each
regime's own drift/vol characteristics gets it right by construction.
Both are implemented below so the difference is visible, not just
asserted.
"""
from __future__ import annotations

import numpy as np


def regime_sharpe_proxy(regime_drift: np.ndarray, regime_vol: np.ndarray) -> np.ndarray:
    """Annualised drift / vol per regime - a simple risk-adjusted-return
    score, not a claim of a true Sharpe ratio (no risk-free rate here)."""
    return regime_drift / regime_vol


def tilt_probability_weighted(regime_probs: np.ndarray, regime_drift: np.ndarray,
                               regime_vol: np.ndarray, base_weight: float = 0.7,
                               sensitivity: float = 0.3, min_weight: float = 0.3,
                               max_weight: float = 1.0) -> np.ndarray:
    """Equity weight as a function of the PROBABILITY-WEIGHTED blend of
    every regime's own risk-adjusted-return score - naturally moderates
    toward `base_weight` under genuine multi-regime uncertainty, since a
    diffuse probability vector blends several regimes' scores together
    rather than committing to one. regime_probs: (N, K) or (K,).
    """
    regime_probs = np.atleast_2d(regime_probs)
    sharpe = regime_sharpe_proxy(regime_drift, regime_vol)
    expected_sharpe = regime_probs @ sharpe
    weight = base_weight + sensitivity * expected_sharpe
    return np.clip(weight, min_weight, max_weight)


def tilt_conviction_scaled(regime_probs: np.ndarray, regime_drift: np.ndarray,
                            regime_vol: np.ndarray, base_weight: float = 0.7,
                            min_weight: float = 0.3, max_weight: float = 1.0) -> np.ndarray:
    """Equity weight interpolated between `base_weight` (no conviction)
    and the DOMINANT regime's own target weight (full conviction),
    explicitly separating "which regime" from "how sure" - the two
    factors A13's tilt rule asks to condition on. Conviction is the
    dominant class' own probability (1/K at total uncertainty, 1.0 when
    certain); the interpolation collapses to `base_weight` as conviction
    approaches 1/K and to the dominant regime's own target weight as
    conviction approaches 1.

    regime_probs: (N, K) or (K,).
    """
    regime_probs = np.atleast_2d(regime_probs)
    K = regime_probs.shape[1]
    sharpe = regime_sharpe_proxy(regime_drift, regime_vol)
    dominant_weight = np.clip(base_weight + sharpe, min_weight, max_weight)  # per-regime target if fully convicted

    dominant_idx = regime_probs.argmax(axis=1)
    conviction = regime_probs.max(axis=1)
    conviction_excess = np.clip((conviction - 1.0 / K) / (1.0 - 1.0 / K), 0.0, 1.0)

    target = dominant_weight[dominant_idx]
    weight = base_weight + conviction_excess * (target - base_weight)
    return np.clip(weight, min_weight, max_weight)


def tilt_naive_name_based(regime_probs: np.ndarray, risk_off_like_idx: list[int],
                           base_weight: float = 0.7, reduction: float = 0.4,
                           min_weight: float = 0.3) -> np.ndarray:
    """A deliberately simplistic comparison rule: reduce equity weight
    in proportion to the combined probability of regimes whose NAMES
    sound risk-off (Transitional, Post-Shock, Risk-Off), with no
    reference to their actual drift/vol. Kept only to make the Post-Shock
    mislabelling concrete in a backtest, not recommended for use - see
    the module docstring and docs/day13_monte_carlo_notes.md.
    """
    regime_probs = np.atleast_2d(regime_probs)
    risk_off_like_prob = regime_probs[:, risk_off_like_idx].sum(axis=1)
    weight = base_weight - reduction * risk_off_like_prob
    return np.clip(weight, min_weight, base_weight)
