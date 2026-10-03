"""Evaluation helpers for the Bayesian deep learning models.

Why not just accuracy: the HMM pseudo-labels these classifiers train on
are ~88.6% one class ("Risk-On", inherited from Day 4's weakly
identified 5-state HMM), so a model that predicts Risk-On for every day
scores ~87-89% accuracy while being useless as a regime detector -
verified directly on this project's data (recall 99.5% on Risk-On, 0% on
Late-Cycle and Transitional). Balanced accuracy, macro-F1, per-class
recall and the majority-class baseline are reported alongside accuracy
everywhere so that failure mode cannot hide.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int = 5) -> dict:
    """Accuracy, balanced accuracy, macro-F1, NLL and per-class recall
    (NaN for classes absent from y_true) from predicted probabilities."""
    y_pred = probs.argmax(axis=1)
    per_class_recall = []
    for c in range(n_classes):
        mask = y_true == c
        per_class_recall.append(float((y_pred[mask] == c).mean()) if mask.any() else float("nan"))
    clipped = np.clip(probs, 1e-7, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "nll": float(log_loss(y_true, clipped, labels=list(range(n_classes)))),
        "per_class_recall": per_class_recall,
    }


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Accuracy and balanced accuracy of always predicting the training
    set's most common class - the number every model must beat to be
    doing anything at all."""
    majority = np.bincount(y_train).argmax()
    y_pred = np.full_like(y_test, majority)
    return {
        "accuracy": float((y_pred == y_test).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    }


def entropy(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def entropy_decomposition(preds: np.ndarray) -> dict:
    """Information-theoretic uncertainty decomposition for a stack of
    predictive distributions, preds shape (n_members_or_samples, batch,
    n_classes):

      total     = H[ mean_m p_m ]         entropy of the averaged prediction
      aleatoric = mean_m H[ p_m ]         average per-member entropy
      epistemic = total - aleatoric       mutual information between the
                                          label and the member (>= 0)

    A cross-check on Section A4.4's std / p(1-p) decomposition: the two
    are not identical quantities, but should rank days similarly.
    Returns per-day arrays of length `batch`.
    """
    mean_probs = preds.mean(axis=0)
    total = entropy(mean_probs)
    aleatoric = entropy(preds).mean(axis=0)
    epistemic = np.maximum(total - aleatoric, 0.0)
    return {"total": total, "aleatoric": aleatoric, "epistemic": epistemic}
