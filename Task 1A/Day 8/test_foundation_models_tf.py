import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow lives in .venv-bayesian, not the main environment")

from src.models.foundation import chronos_style as cs, patch_model as pm
from src.models.foundation.hybrid import HybridRegimeModel, build_variational_head


@pytest.fixture(scope="module")
def windows():
    return np.random.default_rng(0).normal(0, 0.01, size=(6, 252)).astype("float32")


@pytest.fixture(scope="module")
def chronos():
    return cs.build_mini_chronos(seed=0)


@pytest.fixture(scope="module")
def patch():
    return pm.build_mini_timesfm(seed=0)


def test_tokenizer_range_and_scale_invariance(windows):
    tok = cs.MeanScaleTokenizer()
    t1, s1 = tok.encode(windows)
    t2, s2 = tok.encode(windows * 37.0)
    assert t1.min() >= cs.N_SPECIAL and t1.max() < cs.N_SPECIAL + tok.n_bins
    assert np.array_equal(t1, t2)                      # mean-abs scaling removes units
    assert np.allclose(s2 / s1, 37.0, rtol=1e-4)


def test_chronos_style_is_causal(chronos, windows):
    """Changing a LATER token must not change earlier hidden states."""
    _, enc = chronos
    tokens, _ = cs.MeanScaleTokenizer().encode(windows)
    changed = tokens.copy(); changed[:, 200] = (changed[:, 200] + 50) % 256 + cs.N_SPECIAL
    h0 = enc(tf.constant(tokens), training=False).numpy()
    h1 = enc(tf.constant(changed), training=False).numpy()
    np.testing.assert_allclose(h0[:, :200], h1[:, :200], atol=1e-5)
    assert not np.allclose(h0[:, 200:], h1[:, 200:])


def test_patch_model_is_causal(patch, windows):
    _, enc = patch
    p = pm.to_patches(pm.normalize_windows(windows)[0], 12)
    changed = p.copy(); changed[:, 15] += 3.0
    h0, h1 = enc(tf.constant(p), training=False).numpy(), enc(tf.constant(changed), training=False).numpy()
    np.testing.assert_allclose(h0[:, :15], h1[:, :15], atol=1e-5)
    assert not np.allclose(h0[:, 15:], h1[:, 15:])


def test_forecast_next_patch_cannot_see_the_target(patch, windows):
    """Leakage guard for the pretraining evaluation: the forecast of the
    last 12 days must be identical however those 12 days actually go."""
    fm, _ = patch
    a, tgt_a = pm.forecast_next_patch(fm, windows)
    perturbed = windows.copy(); perturbed[:, -12:] = 5.0
    b, tgt_b = pm.forecast_next_patch(fm, perturbed)
    np.testing.assert_allclose(a, b, atol=1e-6)
    assert not np.allclose(tgt_a, tgt_b)
    assert (np.diff(a, axis=1) >= -1e-9).all()         # quantiles sorted, no crossing


def test_normalize_and_patchify(windows):
    z, mean, std = pm.normalize_windows(windows)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-5) and np.allclose(z.std(axis=1), 1, atol=1e-4)
    assert pm.to_patches(z, 12).shape == (6, 21, 12)
    with pytest.raises(AssertionError):
        pm.to_patches(z[:, :250], 12)


def test_embeddings_have_expected_shape_and_are_deterministic(chronos, patch, windows):
    _, cenc = chronos; _, penc = patch
    tokens, _ = cs.MeanScaleTokenizer().encode(windows)
    e1, e2 = cs.embed_windows(cenc, tokens), cs.embed_windows(cenc, tokens)
    assert e1.shape == (6, 64) and np.allclose(e1, e2)
    p = pm.to_patches(pm.normalize_windows(windows)[0], 12)
    assert pm.embed_windows(penc, p).shape == (6, 64)


def test_untrained_control_seed_gives_a_different_model(windows):
    tokens, _ = cs.MeanScaleTokenizer().encode(windows)
    _, e0 = cs.build_mini_chronos(seed=0); _, e1 = cs.build_mini_chronos(seed=123)
    assert not np.allclose(cs.embed_windows(e0, tokens), cs.embed_windows(e1, tokens))


def test_hybrid_model_end_to_end(windows):
    rng = np.random.default_rng(1)
    embed = lambda w: np.stack([w.std(axis=1), np.abs(w).mean(axis=1), w[:, -21:].std(axis=1)], axis=1)
    big = rng.normal(0, 0.01, size=(120, 252)).astype("float32")
    y = rng.integers(0, 5, size=120)
    m = HybridRegimeModel(embed, n_classes=5).fit(big, y, steps=20)
    mean, std = m.predict_with_uncertainty(windows, n_mc=8)
    assert mean.shape == std.shape == (6, 5)
    np.testing.assert_allclose(mean.sum(axis=1), 1.0, atol=1e-4)
    assert build_variational_head(7).output is not None
