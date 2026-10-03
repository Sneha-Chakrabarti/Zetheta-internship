"""Deep ensemble for regime classification, Section A4.4: train M
independent networks from different initialisations; the predictive
distribution is the mixture across members. Separates epistemic
uncertainty (disagreement across members - reducible with more data)
from aleatoric uncertainty (irreducible per-member noise).
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np


def build_deterministic_regime_classifier(input_dim: int, n_regimes: int = 5):
    """Same 128/64/32 architecture as the MC Dropout classifier but with
    NO dropout, so each member's prediction is a deterministic function of
    its weights.

    Why a separate builder: a deep ensemble's epistemic uncertainty is the
    disagreement between independently-initialised members. If members are
    built with `build_mc_dropout_regime_classifier` (dropout permanently
    active), each member's `predict` is itself random, and the cross-member
    spread mixes initialisation disagreement with single-pass dropout
    noise - not the quantity Section A4.4 describes. Section A4.6's
    pipeline passes an arbitrary `build_fn`; this is the right one for an
    ensemble.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(n_regimes, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_deep_ensemble(build_fn, X_train, y_train, M: int = 10, epochs: int = 50,
                         verbose: int = 0, validation_split: float = 0.0,
                         patience: int = 3):
    """Matches Section A4.4: M independent models, seeded
    deterministically (m*17+3) so the ensemble is reproducible.

    `validation_split` > 0 enables early stopping (restore best weights)
    on the LAST `validation_split` fraction of the training arrays.
    Keras takes that slice from the end without shuffling, so for
    time-ordered data it is a chronological hold-out drawn from the
    training window only - no test-period information. Added because the
    spec's fixed 50 epochs badly overfits on this project's data: on CV
    fold 3, ensemble NLL rose from 0.61 (3 epochs) to 0.78 (10) to 1.33
    (30) while accuracy stayed flat, i.e. overconfidence, not skill
    (docs/day7_bdl_notes.md).
    """
    import tensorflow as tf

    callbacks = []
    fit_kwargs = {}
    if validation_split > 0:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True))
        fit_kwargs = {"validation_split": validation_split, "callbacks": callbacks}

    ensemble = []
    for m in range(M):
        tf.random.set_seed(m * 17 + 3)
        model = build_fn()
        model.fit(X_train, y_train, epochs=epochs, batch_size=64, verbose=verbose, **fit_kwargs)
        ensemble.append(model)
    return ensemble


def ensemble_predict(ensemble, X):
    """Matches Section A4.4's decomposition exactly: epistemic =
    cross-member std (disagreement), aleatoric = mean per-member
    Bernoulli-variance p(1-p) (irreducible noise each member itself
    reports)."""
    preds = np.stack([m.predict(X, verbose=0) for m in ensemble])
    mean = preds.mean(axis=0)
    epistemic = preds.std(axis=0)
    aleatoric = (preds * (1 - preds)).mean(axis=0)
    return mean, epistemic, aleatoric
