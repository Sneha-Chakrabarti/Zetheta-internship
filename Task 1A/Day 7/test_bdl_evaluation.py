import numpy as np
import pandas as pd
import pytest

from src.models.bdl.evaluation import (
    classification_metrics, majority_baseline, entropy, entropy_decomposition,
)
from src.models.bdl.data import fit_scaler, apply_scaler, load_bdl_data


def test_classification_metrics_perfect_predictions():
    y = np.array([0, 1, 2, 3, 4, 0])
    probs = np.eye(5)[y] * 0.999 + 0.0002
    m = classification_metrics(y, probs)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["nll"] < 0.01
    assert all(r == 1.0 for r in m["per_class_recall"])


def test_per_class_recall_is_nan_for_absent_classes():
    y = np.array([0, 0, 1, 1])
    probs = np.eye(5)[y]
    m = classification_metrics(y, probs)
    assert m["per_class_recall"][0] == 1.0 and m["per_class_recall"][1] == 1.0
    assert all(np.isnan(m["per_class_recall"][c]) for c in (2, 3, 4))


def test_always_majority_predictor_looks_good_on_accuracy_but_not_balanced_accuracy():
    """Regression test for the failure mode that shaped Day 7's reporting:
    ~89% of pseudo-labels are one class, so predicting it always scores
    high accuracy while balanced accuracy collapses to 1/n_present."""
    rng = np.random.default_rng(0)
    y = rng.choice(5, size=1000, p=[0.886, 0.011, 0.024, 0.041, 0.038])
    probs = np.tile([0.97, 0.0075, 0.0075, 0.0075, 0.0075], (len(y), 1))
    m = classification_metrics(y, probs)
    assert m["accuracy"] > 0.85
    assert m["balanced_accuracy"] == pytest.approx(0.2, abs=1e-9)
    assert m["per_class_recall"][0] == 1.0
    assert all(m["per_class_recall"][c] == 0.0 for c in range(1, 5))


def test_majority_baseline_matches_training_majority():
    y_train = np.array([0] * 90 + [1] * 10)
    y_test = np.array([0] * 80 + [1] * 10 + [2] * 10)
    b = majority_baseline(y_train, y_test)
    assert b["accuracy"] == pytest.approx(0.8)
    assert b["balanced_accuracy"] == pytest.approx(1 / 3)   # 3 classes present in y_test


def test_entropy_of_uniform_and_one_hot():
    assert entropy(np.array([0.25, 0.25, 0.25, 0.25])) == pytest.approx(np.log(4))
    assert entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-9)


def test_entropy_decomposition_identical_members_have_zero_epistemic():
    member = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    preds = np.stack([member] * 5)
    d = entropy_decomposition(preds)
    np.testing.assert_allclose(d["epistemic"], 0.0, atol=1e-12)
    np.testing.assert_allclose(d["total"], d["aleatoric"], atol=1e-12)


def test_entropy_decomposition_confident_disagreement_is_purely_epistemic():
    """Two members, each certain but of opposite classes: the mean is
    50/50 (total = ln 2) yet each member's own entropy is 0 (aleatoric =
    0), so all the uncertainty is disagreement, i.e. epistemic."""
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    d = entropy_decomposition(np.stack([a, b]))
    assert d["total"][0] == pytest.approx(np.log(2))
    assert d["aleatoric"][0] == pytest.approx(0.0, abs=1e-9)
    assert d["epistemic"][0] == pytest.approx(np.log(2))


def test_entropy_decomposition_is_nonnegative_and_additive():
    rng = np.random.default_rng(1)
    raw = rng.random((7, 40, 5))
    preds = raw / raw.sum(axis=-1, keepdims=True)
    d = entropy_decomposition(preds)
    assert (d["epistemic"] >= 0).all() and (d["aleatoric"] >= 0).all()
    np.testing.assert_allclose(d["total"], d["aleatoric"] + d["epistemic"], atol=1e-9)


def test_fit_scaler_handles_constant_column():
    X = np.column_stack([np.arange(10, dtype=float), np.full(10, 3.0)])
    mean, std = fit_scaler(X)
    assert std[1] == 1.0                      # constant column must not divide by ~0
    Z = apply_scaler(X, mean, std)
    assert np.isfinite(Z).all() and Z.dtype == np.float32


def test_scaler_uses_training_slice_only():
    """No look-ahead: statistics come from the training slice, so a test
    slice drawn from a shifted distribution is NOT re-centred."""
    train = np.random.default_rng(0).normal(0, 1, size=(500, 3))
    test = np.random.default_rng(1).normal(5, 1, size=(100, 3))
    mean, std = fit_scaler(train)
    assert np.abs(apply_scaler(train, mean, std).mean(axis=0)).max() < 1e-5
    assert (apply_scaler(test, mean, std).mean(axis=0) > 3).all()


def test_load_bdl_data_standardize_false_returns_raw_values(tmp_path):
    idx = pd.date_range("2020-01-01", periods=6)
    feats = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6], "b": [10.0, 10, 10, 20, 20, 20]}, index=idx)
    labs = pd.DataFrame({"regime": ["Risk-On", "Risk-On", "Late-Cycle", "Risk-On", "Risk-Off", "Risk-On"]}, index=idx)
    fp, lp = tmp_path / "f.csv", tmp_path / "l.csv"
    feats.to_csv(fp); labs.to_csv(lp)
    X, y_int, y_onehot, names, counts = load_bdl_data(str(fp), str(lp), standardize=False)
    np.testing.assert_allclose(X[:, 0], feats["a"].values)
    Xs, *_ = load_bdl_data(str(fp), str(lp), standardize=True)
    assert abs(Xs[:, 0].mean()) < 1e-6
    assert y_onehot.shape == (6, 5) and y_onehot.sum() == 6
    assert counts["Risk-On"] == 4 and counts["Late-Cycle"] == 1 and counts["Risk-Off"] == 1
