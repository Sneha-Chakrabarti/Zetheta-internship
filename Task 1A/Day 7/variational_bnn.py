"""Variational Bayesian neural network for regime classification,
Section A4.3: mean-field VI via DenseFlipout layers, weights replaced by
variational distributions optimised with stochastic-gradient VI.
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def build_variational_regime_classifier(input_dim: int, n_regimes: int = 5,
                                         train_size: int = 2000):
    """Matches Section A4.3: three DenseFlipout layers (128/64/n_regimes),
    KL term scaled by 1/train_size (so its weight shrinks as more data
    would ordinarily make the prior matter less), OneHotCategorical
    output, negative-log-likelihood loss."""
    import tensorflow as tf
    import tensorflow_probability as tfp

    tfd = tfp.distributions
    tfpl = tfp.layers

    kl_weight = 1.0 / train_size
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tfpl.DenseFlipout(128, activation="relu",
                           kernel_divergence_fn=lambda q, p, _: kl_weight * tfd.kl_divergence(q, p))(inputs)
    x = tfpl.DenseFlipout(64, activation="relu",
                           kernel_divergence_fn=lambda q, p, _: kl_weight * tfd.kl_divergence(q, p))(x)
    logits = tfpl.DenseFlipout(n_regimes,
                                kernel_divergence_fn=lambda q, p, _: kl_weight * tfd.kl_divergence(q, p))(x)
    outputs = tfpl.OneHotCategorical(n_regimes)(logits)
    model = tf.keras.Model(inputs, outputs)

    def nll(y, rv_y):
        return -rv_y.log_prob(y)

    model.compile(optimizer="adam", loss=nll, metrics=["accuracy"])
    return model


def vi_predict(model, X, n_samples: int = 200):
    """Predictive distribution via repeated stochastic forward passes -
    each call samples fresh weights from the variational posterior, the
    VI analogue of MC Dropout's repeated stochastic passes. One forward
    pass per sample: `model(X)` returns a OneHotCategorical distribution
    whose `.mean()` is that pass's class-probability vector. (An earlier
    draft called `model(X)` twice per iteration, which would have drawn
    two different weight samples and mixed them - caught before use.)
    Returns (mean, std, preds); preds shape (n_samples, batch, n_regimes)."""
    import numpy as np

    preds = np.stack([model(X).mean().numpy() for _ in range(n_samples)])
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, std, preds
