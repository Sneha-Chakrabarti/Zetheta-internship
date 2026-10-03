"""Reliability diagrams and Expected Calibration Error, Section A6.6.
Matches the spec's `reliability_diagram` and `expected_calibration_error`
functions exactly.
"""
from __future__ import annotations

import numpy as np


def reliability_diagram(probs_max: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_lowers, bin_uppers = bins[:-1], bins[1:]
    accuracies, confidences, counts = [], [], []
    for lo, hi in zip(bin_lowers, bin_uppers):
        in_bin = (probs_max > lo) & (probs_max <= hi)
        if in_bin.sum() > 0:
            accuracies.append(correct[in_bin].mean())
            confidences.append(probs_max[in_bin].mean())
            counts.append(in_bin.sum())
        else:
            accuracies.append(0)
            confidences.append((lo + hi) / 2)
            counts.append(0)
    return np.array(accuracies), np.array(confidences), np.array(counts)


def expected_calibration_error(probs_max: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    acc, conf, cnt = reliability_diagram(probs_max, correct, n_bins)
    return float(np.sum(cnt / cnt.sum() * np.abs(acc - conf)))


def probs_max_and_correct(probs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: from a (N, K) probability array and (N,) true labels,
    the two 1-D arrays `reliability_diagram`/`expected_calibration_error`
    need - the top predicted class's own probability, and whether that
    top class was actually correct."""
    pred = probs.argmax(axis=1)
    probs_max = probs.max(axis=1)
    correct = (pred == y).astype(float)
    return probs_max, correct
