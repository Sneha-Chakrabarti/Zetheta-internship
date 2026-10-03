"""Monte Carlo Dropout regime classifier, Section A4.2 (Gal & Ghahramani,
2016): approximate Bayesian inference via repeated stochastic forward
passes with dropout left active at inference time.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np


def build_mc_dropout_regime_classifier(input_dim: int, n_regimes: int = 5,
                                        dropout_rate: float = 0.3):
    """Matches Section A4.2 exactly: three Dense+Dropout blocks
    (128/64/32 units), dropout left `training=True` permanently so it
    stays active at inference time, softmax output."""
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x, training=True)
    outputs = layers.Dense(n_regimes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def mc_predict(model, X, n_samples: int = 200):
    """Predictive distribution via repeated stochastic forward passes.
    Returns (mean, std, preds); preds shape (n_samples, batch, n_regimes)."""
    preds = np.stack([model(X, training=True).numpy() for _ in range(n_samples)])
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std, preds
