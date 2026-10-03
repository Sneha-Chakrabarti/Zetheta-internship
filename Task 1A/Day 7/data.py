"""Load the Day 3 feature matrix + Day 4 HMM pseudo-labels bridge files
(generated in the main environment, where hmmlearn/features live) for
use by the TensorFlow-based models here (which live in .venv-bayesian
for the numpy>=2 reasons documented in PROJECT_PLAN.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_ORDER = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]


def load_bdl_data(features_path: str = "artifacts_data/day7_features.csv",
                   labels_path: str = "artifacts_data/day7_labels.csv",
                   standardize: bool = True):
    """Returns (X, y_int, y_onehot, feature_names, class_counts).

    standardize=True z-scores X using the mean/std of the WHOLE series
    (neural nets need standardised input; Day 3's feature matrix mixes
    raw z-scores, percentages and ratios on very different scales).
    That is fine for a single exploratory fit but leaks future
    statistics into any time-series cross-validation fold; use
    standardize=False there and scale per fold with `fit_scaler` /
    `apply_scaler` on the training slice only."""
    features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    labels = pd.read_csv(labels_path, index_col=0, parse_dates=True)["regime"]

    if standardize:
        X = ((features - features.mean()) / features.std()).values.astype("float32")
    else:
        X = features.values.astype("float32")

    class_to_int = {name: i for i, name in enumerate(REGIME_ORDER)}
    y_int = labels.map(class_to_int).values
    y_onehot = np.eye(len(REGIME_ORDER), dtype="float32")[y_int]

    class_counts = labels.value_counts().reindex(REGIME_ORDER).fillna(0).astype(int)
    return X, y_int, y_onehot, list(features.columns), class_counts


def fit_scaler(X_train: np.ndarray):
    """Mean/std from a training slice only (no look-ahead)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # constant column in a short early fold
    return mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype("float32")
