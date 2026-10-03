"""Hybrid architecture, Section A5.4: foundation-model embedding on the
input side, Bayesian classification head on the output side.

The head is the same variational (DenseFlipout) design as Day 7's BNN but
with a fixed small size (64, 32) for EVERY input type, so that the
sample-efficiency comparison isolates the input representation rather
than head capacity.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import math
import numpy as np


def build_variational_head(input_dim: int, n_classes: int = 5, hidden=(64, 32),
                            train_size: int = 1000):
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd, tfpl = tfp.distributions, tfp.layers
    kl = 1.0 / max(train_size, 1)
    div = lambda q, p, _: kl * tfd.kl_divergence(q, p)
    inp = tf.keras.Input(shape=(input_dim,))
    x = inp
    for h in hidden:
        x = tfpl.DenseFlipout(h, activation="relu", kernel_divergence_fn=div)(x)
    logits = tfpl.DenseFlipout(n_classes, kernel_divergence_fn=div)(x)
    out = tfpl.OneHotCategorical(n_classes)(logits)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss=lambda y, rv: -rv.log_prob(y))
    return model


def fit_head(X_train: np.ndarray, y_train: np.ndarray, seed: int = 0, steps: int = 600,
             batch: int = 64, n_classes: int = 5):
    """Train the head for a fixed number of gradient steps (not epochs), so
    small and large training sets get the same optimisation budget."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    model = build_variational_head(X_train.shape[1], n_classes, train_size=len(X_train))
    y1h = np.eye(n_classes, dtype="float32")[y_train]
    per_epoch = max(1, math.ceil(len(X_train) / batch))
    model.fit(X_train.astype("float32"), y1h, epochs=math.ceil(steps / per_epoch),
              batch_size=batch, verbose=0)
    return model


def predict_head(model, X: np.ndarray, n_mc: int = 30):
    """Mean predictive probabilities and the stack of stochastic passes."""
    X = X.astype("float32")
    preds = np.stack([model(X).mean().numpy() for _ in range(n_mc)])
    return preds.mean(axis=0), preds


class HybridRegimeModel:
    """Section A5.4's wrapper. `embed_fn` maps a batch of raw windows to
    embeddings (e.g. a pretrained encoder); the head is trained on the
    standardised embeddings."""

    def __init__(self, embed_fn, n_classes: int = 5):
        self.embed_fn, self.n_classes = embed_fn, n_classes
        self.mean_ = self.std_ = self.head_ = None

    def encode(self, windows: np.ndarray) -> np.ndarray:
        Z = np.asarray(self.embed_fn(windows), dtype="float64")
        if self.mean_ is None:
            return Z
        return ((Z - self.mean_) / self.std_).astype("float32")

    def fit(self, windows, y, seed: int = 0, steps: int = 600):
        Z = np.asarray(self.embed_fn(windows), dtype="float64")
        self.mean_, self.std_ = Z.mean(0), np.where(Z.std(0) < 1e-8, 1.0, Z.std(0))
        self.head_ = fit_head(self.encode(windows), y, seed=seed, steps=steps, n_classes=self.n_classes)
        return self

    def predict_with_uncertainty(self, windows, n_mc: int = 200):
        mean, preds = predict_head(self.head_, self.encode(windows), n_mc=n_mc)
        return mean, preds.std(axis=0)
