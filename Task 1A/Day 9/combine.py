"""Model combination, Section A10.1 (Bayesian Model Averaging) and A10.2
(constrained stacking).
"""
from __future__ import annotations

import numpy as np


def bma_weights(log_predictive_likelihoods: np.ndarray) -> np.ndarray:
    """Section A10.1, verbatim. log_predictive_likelihoods: (M,)
    out-of-sample log-likelihood per model. Softmax over the log-liks:
    the model with the highest held-out likelihood gets the most weight,
    exponentially so."""
    w = np.exp(log_predictive_likelihoods - log_predictive_likelihoods.max())
    return w / w.sum()


def bma_combine(model_probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Section A10.1, verbatim. model_probs: (M, K) or (M, N, K); weights:
    (M,). Returns (K,) or (N, K): the weighted average regime
    probability."""
    return np.tensordot(weights, model_probs, axes=(0, 0))


def fit_stacking_weights(base_probs: np.ndarray, y_true: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Section A10.2, verbatim except accepting `n_classes` explicitly (the
    spec infers K from `base_probs.shape[2]`, which is only correct if
    every class the labels can take is actually present in this fold's
    `y_true` - with a rare class this is not guaranteed, and
    `np.eye(K)[y_true]` would then silently one-hot against the wrong K
    if K were inferred from `y_true.max()+1` instead of the model output
    width. Passing `n_classes` removes that ambiguity; if omitted it falls
    back to the spec's own inference from `base_probs`.

    base_probs: (M, N, K) out-of-fold probs; y_true: (N,) integer labels.
    Returns simplex weights (M,) minimising mean cross-entropy.
    """
    from scipy.optimize import minimize

    M, N, K = base_probs.shape
    if n_classes is not None and n_classes != K:
        raise ValueError(f"base_probs has K={K} classes but n_classes={n_classes} was given.")
    onehot = np.eye(K)[y_true]

    def neg_loglik(w):
        w = np.clip(w, 0, None)
        w = w / w.sum()
        combined = np.tensordot(w, base_probs, axes=(0, 0))  # (N, K)
        combined = np.clip(combined, 1e-9, 1.0)
        return -np.mean(np.sum(onehot * np.log(combined), axis=1))

    w0 = np.full(M, 1.0 / M)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    bnds = [(0, 1)] * M
    res = minimize(neg_loglik, w0, bounds=bnds, constraints=cons)
    return res.x / res.x.sum()


def mean_log_likelihood(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean log p(y_true | probs) - the "out-of-sample calibrated
    log-likelihood" Day 9 asks the ensemble to beat every individual
    member on (A10.4). Higher is better (less negative)."""
    clipped = np.clip(probs, 1e-9, 1.0)
    return float(np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))
