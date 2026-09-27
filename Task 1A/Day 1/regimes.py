"""Canonical regime labels and a sticky Markov chain sampler.

The five states match Part A of the task document: risk-on, transitional,
late-cycle, post-shock, risk-off. Kept here (not inside the synthetic
generator) because model code in src/models/ also needs a shared,
authoritative label set and transition-matrix convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REGIME_NAMES = ["risk_on", "transitional", "late_cycle", "post_shock", "risk_off"]
N_REGIMES = len(REGIME_NAMES)
REGIME_INDEX = {name: i for i, name in enumerate(REGIME_NAMES)}


@dataclass
class RegimeSpec:
    """Per-regime parameters for the synthetic market generator.

    All rates are annualised; the generator converts to daily internally.
    `mean_duration_days` sets the diagonal of the transition matrix
    (self-transition probability = 1 - 1/mean_duration_days), which is the
    standard way to parameterise regime persistence directly rather than
    guessing transition probabilities.
    """

    name: str
    nifty_drift: float
    nifty_vol: float
    midcap_beta: float
    midcap_extra_vol: float
    smallcap_beta: float
    smallcap_extra_vol: float
    vix_level: float
    vix_vol: float
    inr_drift: float
    inr_vol: float
    gilt_level: float
    aaa_gilt_spread: float
    fii_flow_mean: float
    fii_flow_vol: float
    mean_duration_days: int


# Defaults are illustrative, calibrated loosely to the Part C case studies
# (2020 COVID, 2018 IL&FS, 2013 taper tantrum, 2017-18 mid-cap correction,
# 2024 election gap) so the synthetic path exercises HMM/changepoint code
# against episodes with roughly the right shape. They are NOT fitted to
# data and must not be quoted as empirical figures in the report.
DEFAULT_REGIME_SPECS: dict[str, RegimeSpec] = {
    "risk_on": RegimeSpec(
        name="risk_on", nifty_drift=0.15, nifty_vol=0.12,
        midcap_beta=1.15, midcap_extra_vol=0.03,
        smallcap_beta=1.30, smallcap_extra_vol=0.06,
        vix_level=13.0, vix_vol=1.5,
        inr_drift=0.01, inr_vol=0.04,
        gilt_level=0.068, aaa_gilt_spread=0.006,
        fii_flow_mean=800.0, fii_flow_vol=1500.0,
        mean_duration_days=180,
    ),
    "transitional": RegimeSpec(
        name="transitional", nifty_drift=0.02, nifty_vol=0.18,
        midcap_beta=1.05, midcap_extra_vol=0.05,
        smallcap_beta=1.10, smallcap_extra_vol=0.08,
        vix_level=18.0, vix_vol=3.0,
        inr_drift=0.02, inr_vol=0.06,
        gilt_level=0.070, aaa_gilt_spread=0.008,
        fii_flow_mean=0.0, fii_flow_vol=2000.0,
        mean_duration_days=45,
    ),
    "late_cycle": RegimeSpec(
        name="late_cycle", nifty_drift=0.06, nifty_vol=0.16,
        midcap_beta=0.70, midcap_extra_vol=0.10,
        smallcap_beta=0.55, smallcap_extra_vol=0.14,
        vix_level=16.0, vix_vol=2.5,
        inr_drift=0.03, inr_vol=0.05,
        gilt_level=0.072, aaa_gilt_spread=0.010,
        fii_flow_mean=-200.0, fii_flow_vol=1800.0,
        mean_duration_days=120,
    ),
    "post_shock": RegimeSpec(
        name="post_shock", nifty_drift=0.25, nifty_vol=0.22,
        midcap_beta=1.20, midcap_extra_vol=0.08,
        smallcap_beta=1.35, smallcap_extra_vol=0.12,
        vix_level=22.0, vix_vol=4.0,
        inr_drift=-0.01, inr_vol=0.07,
        gilt_level=0.065, aaa_gilt_spread=0.012,
        fii_flow_mean=600.0, fii_flow_vol=2500.0,
        mean_duration_days=60,
    ),
    "risk_off": RegimeSpec(
        name="risk_off", nifty_drift=-0.30, nifty_vol=0.30,
        midcap_beta=1.25, midcap_extra_vol=0.10,
        smallcap_beta=1.45, smallcap_extra_vol=0.16,
        vix_level=32.0, vix_vol=6.0,
        inr_drift=0.08, inr_vol=0.10,
        gilt_level=0.062, aaa_gilt_spread=0.020,
        fii_flow_mean=-2500.0, fii_flow_vol=3000.0,
        mean_duration_days=25,
    ),
}

# Off-diagonal transition preference: which regimes a state is more likely
# to move into next, before renormalising. Encodes a rough business-cycle
# ordering (risk_on -> transitional -> late_cycle -> risk_off/post_shock ->
# risk_on) rather than a uniform random walk between states.
_TRANSITION_BIAS = {
    "risk_on": {"transitional": 0.6, "late_cycle": 0.25, "post_shock": 0.1, "risk_off": 0.05},
    "transitional": {"risk_on": 0.35, "late_cycle": 0.35, "risk_off": 0.2, "post_shock": 0.1},
    "late_cycle": {"transitional": 0.3, "risk_off": 0.45, "post_shock": 0.05, "risk_on": 0.2},
    "post_shock": {"risk_on": 0.7, "transitional": 0.2, "late_cycle": 0.05, "risk_off": 0.05},
    "risk_off": {"post_shock": 0.55, "transitional": 0.3, "risk_on": 0.1, "late_cycle": 0.05},
}


def build_transition_matrix(specs: dict[str, RegimeSpec] = DEFAULT_REGIME_SPECS) -> np.ndarray:
    """Assemble a row-stochastic transition matrix from per-regime durations
    and the off-diagonal bias table above."""
    n = N_REGIMES
    P = np.zeros((n, n))
    for name, spec in specs.items():
        i = REGIME_INDEX[name]
        p_stay = 1.0 - 1.0 / spec.mean_duration_days
        P[i, i] = p_stay
        remaining = 1.0 - p_stay
        bias = _TRANSITION_BIAS[name]
        for other_name, weight in bias.items():
            j = REGIME_INDEX[other_name]
            P[i, j] = remaining * weight
    # Guard against rounding drift so every row sums to exactly 1.
    P = P / P.sum(axis=1, keepdims=True)
    return P


def sample_regime_path(n_steps: int, rng: np.random.Generator,
                        start_regime: str = "risk_on",
                        specs: dict[str, RegimeSpec] = DEFAULT_REGIME_SPECS) -> np.ndarray:
    """Sample a length-n_steps sequence of regime indices from the Markov
    chain implied by `specs`."""
    P = build_transition_matrix(specs)
    path = np.empty(n_steps, dtype=int)
    path[0] = REGIME_INDEX[start_regime]
    for t in range(1, n_steps):
        path[t] = rng.choice(N_REGIMES, p=P[path[t - 1]])
    return path
