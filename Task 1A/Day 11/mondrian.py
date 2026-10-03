"""Mondrian / class-conditional conformal prediction, Section A6.5.

Chosen as this project's distribution-shift-robust method specifically
because it targets a problem already established since Day 7: the HMM
pseudo-labels are ~88.6% one class. Pooled split-conformal (A6.2)
computes ONE threshold from calibration scores dominated by the majority
class, so its marginal coverage guarantee can hold in aggregate while
coverage for a rare class is far from the target - exactly the kind of
average-vs-per-group gap this project has repeatedly found elsewhere
(Day 7's accuracy-vs-balanced-accuracy gap, Day 9's pooled-vs-per-class
ensemble results). Mondrian conformal computes a SEPARATE threshold per
TRUE class from calibration data, so the guarantee holds conditional on
each class, not just on average across them - at the cost of each
threshold being estimated from a much smaller calibration sample for a
rare class, which is checked below, not assumed away.
"""
from __future__ import annotations

import numpy as np


def mondrian_conformal_classifier(cal_probs: np.ndarray, y_cal: np.ndarray,
                                   test_probs: np.ndarray, K: int, alpha: float = 0.1):
    """Per-class non-conformity thresholds from calibration data grouped
    by TRUE class, then a test point's prediction set is {k : its score
    for class k is below class k's OWN threshold q_hat[k]} - this is
    what gives the per-class (not just marginal) coverage guarantee:
    when the true label is k, the relevant threshold was estimated from
    other class-k examples.

    Returns (pred_sets, q_hat, n_cal_per_class). q_hat[k] is NaN (and
    class k is always included, the safe default when there is no
    information) when calibration contains zero examples of class k.
    """
    q_hat = np.full(K, np.nan)
    n_cal_per_class = np.zeros(K, dtype=int)
    for k in range(K):
        mask = y_cal == k
        n_k = int(mask.sum())
        n_cal_per_class[k] = n_k
        if n_k == 0:
            continue
        scores_k = 1 - cal_probs[mask, k]
        q_level = min(np.ceil((n_k + 1) * (1 - alpha)) / n_k, 1.0)
        q_hat[k] = np.quantile(scores_k, q_level, method="higher")

    pred_sets = np.zeros((test_probs.shape[0], K), dtype=bool)
    for k in range(K):
        if np.isnan(q_hat[k]):
            pred_sets[:, k] = True   # no calibration data for k: cannot exclude it
        else:
            pred_sets[:, k] = (1 - test_probs[:, k]) <= q_hat[k]
    return pred_sets, q_hat, n_cal_per_class


def per_class_coverage(pred_sets: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    """(K,) empirical coverage conditional on each TRUE class - the
    quantity Mondrian conformal targets directly, and the one pooled
    split-conformal's guarantee says nothing about. NaN for a class with
    zero test examples."""
    cov = np.full(K, np.nan)
    for k in range(K):
        mask = y == k
        if mask.sum() == 0:
            continue
        cov[k] = pred_sets[mask, k].mean()
    return cov
