"""Generic synthetic corpus for pretraining the Day 8 foundation-style
models.

Real foundation models (Chronos, TimesFM) are pretrained on huge real
and synthetic corpora that cannot be downloaded here (Hugging Face is
blocked). These small models are instead pretrained on series drawn
on the fly from nine generic families with randomised parameters and
scales, so the corpus is effectively unlimited and never repeats.

Honest limits, repeated in docs/day8_foundation_notes.md:

- Family 4 (Markov regime switching) is the same family as the project's
  synthetic market, though with independently randomised parameters. A
  real foundation corpus also contains regime shifts, but this overlap
  means transfer results here are friendlier than a real foundation model
  would show on real Indian data.
- Nothing here answers whether a REAL pretrained foundation model
  transfers to Indian equities. It tests the hybrid pipeline and whether
  pretraining on generic series helps a regime head at all.

All generators return float32 arrays of shape (n, length) and are
vectorised over the batch.
"""
from __future__ import annotations

import numpy as np

FAMILIES = [
    "ar",               # stationary AR(1)/AR(2)
    "rw_drift",         # random walk increments with drift
    "garch",            # volatility clustering
    "regime_switch",    # sticky Markov regime switching in mean and vol
    "student_t",        # heavy-tailed iid or AR noise
    "seasonal",         # sinusoid plus noise
    "trend",            # linear trend plus noise
    "jumps",            # Gaussian noise plus Poisson jumps
    "ou_level_shift",   # mean reversion with occasional level shifts
]


def _scale(rng, n):
    """Log-uniform overall scale so nothing depends on absolute units."""
    return np.exp(rng.uniform(np.log(1e-3), np.log(1e2), size=(n, 1)))


def _ar(rng, n, length):
    phi1 = rng.uniform(-0.6, 0.9, size=(n, 1))
    phi2 = rng.uniform(-0.3, 0.3, size=(n, 1)) * (np.abs(phi1) < 0.6)
    e = rng.normal(size=(n, length + 2))
    x = np.zeros((n, length + 2))
    for t in range(2, length + 2):
        x[:, t] = phi1[:, 0] * x[:, t - 1] + phi2[:, 0] * x[:, t - 2] + e[:, t]
    return x[:, 2:]


def _rw_drift(rng, n, length):
    drift = rng.normal(0, 0.15, size=(n, 1))
    return drift + rng.normal(size=(n, length))


def _garch(rng, n, length):
    omega = rng.uniform(0.02, 0.2, size=n)
    alpha = rng.uniform(0.05, 0.25, size=n)
    beta = rng.uniform(0.6, 0.93, size=n)
    beta = np.minimum(beta, 0.98 - alpha)
    var = omega / np.maximum(1 - alpha - beta, 0.02)
    x = np.zeros((n, length))
    for t in range(length):
        z = rng.normal(size=n)
        x[:, t] = np.sqrt(var) * z
        var = omega + alpha * x[:, t] ** 2 + beta * var
    return x


def _regime_switch(rng, n, length):
    k = rng.integers(2, 5, size=n)
    x = np.zeros((n, length))
    for i in range(n):
        K = int(k[i])
        mu = rng.normal(0, 0.4, size=K)
        sd = np.exp(rng.uniform(np.log(0.3), np.log(3.0), size=K))
        stay = rng.uniform(0.9, 0.995)
        s = rng.integers(0, K)
        for t in range(length):
            if rng.random() > stay:
                s = rng.integers(0, K)
            x[i, t] = mu[s] + sd[s] * rng.normal()
    return x


def _student_t(rng, n, length):
    df = rng.uniform(2.5, 8.0, size=(n, 1))
    e = rng.standard_t(df, size=(n, length))
    phi = rng.uniform(-0.3, 0.5, size=(n, 1))
    x = np.zeros_like(e)
    for t in range(1, length):
        x[:, t] = phi[:, 0] * x[:, t - 1] + e[:, t]
    return x


def _seasonal(rng, n, length):
    t = np.arange(length)[None, :]
    period = rng.uniform(5, 60, size=(n, 1))
    amp = rng.uniform(0.3, 3.0, size=(n, 1))
    phase = rng.uniform(0, 2 * np.pi, size=(n, 1))
    return amp * np.sin(2 * np.pi * t / period + phase) + rng.normal(size=(n, length))


def _trend(rng, n, length):
    t = np.linspace(-1, 1, length)[None, :]
    slope = rng.normal(0, 3.0, size=(n, 1))
    return slope * t + rng.normal(size=(n, length))


def _jumps(rng, n, length):
    rate = rng.uniform(0.005, 0.05, size=(n, 1))
    jump = (rng.random((n, length)) < rate) * rng.normal(0, 6.0, size=(n, length))
    return rng.normal(size=(n, length)) + jump


def _ou_level_shift(rng, n, length):
    kappa = rng.uniform(0.02, 0.4, size=n)
    level = np.zeros(n)
    x = np.zeros((n, length))
    cur = np.zeros(n)
    for t in range(length):
        shift = rng.random(n) < 0.01
        level = np.where(shift, level + rng.normal(0, 3.0, size=n), level)
        cur = cur + kappa * (level - cur) + rng.normal(size=n)
        x[:, t] = cur
    return x


_GEN = {
    "ar": _ar, "rw_drift": _rw_drift, "garch": _garch, "regime_switch": _regime_switch,
    "student_t": _student_t, "seasonal": _seasonal, "trend": _trend, "jumps": _jumps,
    "ou_level_shift": _ou_level_shift,
}


def generate_family(name: str, rng: np.random.Generator, n: int, length: int = 252) -> np.ndarray:
    x = _GEN[name](rng, n, length)
    return (x * _scale(rng, n)).astype("float32")


def generate_corpus_batch(rng: np.random.Generator, batch: int, length: int = 252,
                           families=None) -> tuple[np.ndarray, np.ndarray]:
    """A batch drawn uniformly over families. Returns (series, family_index)."""
    families = families or FAMILIES
    fam_idx = rng.integers(0, len(families), size=batch)
    out = np.empty((batch, length), dtype="float32")
    for j, name in enumerate(families):
        sel = np.where(fam_idx == j)[0]
        if len(sel):
            out[sel] = generate_family(name, rng, len(sel), length)
    return out, fam_idx
