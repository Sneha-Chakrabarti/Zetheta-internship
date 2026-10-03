"""Split-conformal classification (A6.2) and Adaptive Prediction Sets
(A6.3). Both take already-computed probability arrays rather than a
`model` object with a `.predict` method (the spec's signature): every
model in this project already has its calibration/test-window
probabilities saved from earlier days, so there is nothing to call a
model to get - passing the arrays directly avoids re-deriving a
model-specific prediction call for each of five different model types.
"""
from __future__ import annotations

import numpy as np


def split_conformal_classifier(cal_probs: np.ndarray, y_cal: np.ndarray,
                                test_probs: np.ndarray, alpha: float = 0.1):
    """Section A6.2, verbatim except for taking probability arrays
    directly. Returns (pred_sets, q_hat): pred_sets is a boolean (N_test,
    K) mask of classes in each test point's conformal set."""
    cal_scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]
    n = len(cal_scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = np.quantile(cal_scores, q_level, method="higher")
    pred_sets = test_probs >= (1 - q_hat)
    return pred_sets, q_hat


def adaptive_prediction_sets(cal_probs: np.ndarray, y_cal: np.ndarray,
                              test_probs: np.ndarray, alpha: float = 0.1):
    """Section A6.3, verbatim except for taking probability arrays
    directly. Returns (in_set, sorted_idx_t): in_set is a boolean
    (N_test, K) mask ALIGNED TO sorted_idx_t's rank order (column 0 is
    each test point's own top class, not necessarily class 0) - callers
    must use `sorted_idx_t` to map back to original class indices (see
    `aps_set_as_class_mask` below), a step the spec's own return value
    requires but does not name."""
    sorted_idx = np.argsort(-cal_probs, axis=1)
    sorted_probs = np.take_along_axis(cal_probs, sorted_idx, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    rank_of_true = np.array([np.where(sorted_idx[i] == y_cal[i])[0][0] for i in range(len(y_cal))])
    cal_scores = cumsum[np.arange(len(y_cal)), rank_of_true]
    n = len(cal_scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = np.quantile(cal_scores, q_level, method="higher")

    sorted_idx_t = np.argsort(-test_probs, axis=1)
    sorted_probs_t = np.take_along_axis(test_probs, sorted_idx_t, axis=1)
    cumsum_t = np.cumsum(sorted_probs_t, axis=1)
    in_set = cumsum_t <= q_hat
    in_set[:, 0] = True
    return in_set, sorted_idx_t


def aps_set_as_class_mask(in_set: np.ndarray, sorted_idx_t: np.ndarray, K: int) -> np.ndarray:
    """Convert APS's rank-ordered (in_set, sorted_idx_t) pair into a
    boolean (N, K) mask in ORIGINAL class-index order - the same shape
    `split_conformal_classifier` returns, so both methods' outputs can
    be compared and scored with the same downstream code (coverage,
    set-size) without the caller re-deriving this mapping each time."""
    N = in_set.shape[0]
    mask = np.zeros((N, K), dtype=bool)
    rows = np.repeat(np.arange(N), K)
    mask[rows, sorted_idx_t.ravel()] = in_set.ravel()
    return mask


def set_contains_true_label(pred_sets: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(N,) boolean: did each prediction set (boolean N,K mask) contain
    the true label? The empirical coverage A6.2's guarantee is about."""
    return pred_sets[np.arange(len(y)), y]


def mean_set_size(pred_sets: np.ndarray) -> float:
    """Mean number of classes per prediction set - the efficiency
    measure that matters alongside coverage: a set containing every
    class trivially achieves 100% coverage but says nothing."""
    return float(pred_sets.sum(axis=1).mean())
