import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow lives in .venv-bayesian, not the main environment")

from src.models.bdl.mc_dropout import build_mc_dropout_regime_classifier, mc_predict
from src.models.bdl.variational_bnn import build_variational_regime_classifier, vi_predict
from src.models.bdl.deep_ensemble import (
    build_deterministic_regime_classifier, train_deep_ensemble, ensemble_predict,
)


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 6)).astype("float32")
    y = rng.integers(0, 5, size=120)
    return X, np.eye(5, dtype="float32")[y]


def test_mc_dropout_is_stochastic_at_inference(toy):
    X, _ = toy
    m = build_mc_dropout_regime_classifier(6)
    a, b = m(X[:10]).numpy(), m(X[:10]).numpy()
    assert not np.allclose(a, b)                      # dropout stays on by design (A4.2)


def test_deterministic_member_is_repeatable(toy):
    """Design point behind the ensemble: members must be deterministic so
    cross-member spread is initialisation disagreement, not dropout noise."""
    X, _ = toy
    m = build_deterministic_regime_classifier(6)
    np.testing.assert_allclose(m.predict(X[:10], verbose=0), m.predict(X[:10], verbose=0))


def test_mc_predict_shapes_and_valid_probabilities(toy):
    X, Y = toy
    m = build_mc_dropout_regime_classifier(6)
    m.fit(X, Y, epochs=2, verbose=0)
    mean, std, preds = mc_predict(m, X[:15], n_samples=8)
    assert preds.shape == (8, 15, 5) and mean.shape == (15, 5)
    np.testing.assert_allclose(mean.sum(axis=1), 1.0, atol=1e-4)
    assert (std >= 0).all()


def test_vi_predict_draws_fresh_weights_each_pass(toy):
    X, Y = toy
    m = build_variational_regime_classifier(6, train_size=len(X))
    m.fit(X, Y, epochs=2, verbose=0)
    mean, std, preds = vi_predict(m, X[:12], n_samples=6)
    assert preds.shape == (6, 12, 5)
    np.testing.assert_allclose(preds.sum(axis=2), 1.0, atol=1e-4)
    assert std.max() > 0                              # passes differ: weights are resampled each call


def test_train_deep_ensemble_with_early_stopping_returns_m_members(toy):
    X, Y = toy
    ens = train_deep_ensemble(lambda: build_deterministic_regime_classifier(6), X, Y,
                               M=3, epochs=20, validation_split=0.2, patience=1)
    assert len(ens) == 3
    mean, epi, ale = ensemble_predict(ens, X[:10])
    assert mean.shape == epi.shape == ale.shape == (10, 5)
    np.testing.assert_allclose(mean.sum(axis=1), 1.0, atol=1e-4)
    assert (epi >= 0).all() and (ale >= 0).all()


def test_ensemble_members_differ_because_of_seeding(toy):
    X, Y = toy
    ens = train_deep_ensemble(lambda: build_deterministic_regime_classifier(6), X, Y, M=2, epochs=2)
    p0, p1 = ens[0].predict(X[:10], verbose=0), ens[1].predict(X[:10], verbose=0)
    assert not np.allclose(p0, p1)                    # different seeds -> different members
