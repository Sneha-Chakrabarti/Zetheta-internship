import numpy as np
import pytest

from src.models.foundation.corpus import FAMILIES, generate_corpus_batch, generate_family


def test_batch_shapes_dtype_and_finiteness():
    x, fam = generate_corpus_batch(np.random.default_rng(0), 64, length=252)
    assert x.shape == (64, 252) and x.dtype == np.float32
    assert np.isfinite(x).all()
    assert fam.shape == (64,) and fam.min() >= 0 and fam.max() < len(FAMILIES)


def test_batch_is_deterministic_given_the_generator_seed():
    a, fa = generate_corpus_batch(np.random.default_rng(5), 32)
    b, fb = generate_corpus_batch(np.random.default_rng(5), 32)
    assert np.array_equal(a, b) and np.array_equal(fa, fb)


def test_every_family_generates_nonconstant_finite_series():
    for name in FAMILIES:
        x = generate_family(name, np.random.default_rng(1), 20, 252)
        assert x.shape == (20, 252) and np.isfinite(x).all(), name
        assert (x.std(axis=1) > 0).all(), f"{name} produced a constant series"


def test_family_restriction_excludes_regime_switching():
    """The control experiment depends on this: with the regime-switching
    family removed, no drawn series may come from it."""
    keep = [f for f in FAMILIES if f != "regime_switch"]
    _, idx = generate_corpus_batch(np.random.default_rng(2), 500, families=keep)
    assert idx.min() >= 0 and idx.max() < len(keep)
    assert "regime_switch" not in keep and len(keep) == len(FAMILIES) - 1


def test_scales_span_orders_of_magnitude():
    """Series are rescaled log-uniformly so nothing depends on units."""
    x, _ = generate_corpus_batch(np.random.default_rng(3), 400)
    s = x.std(axis=1)
    assert s.max() / s.min() > 100


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        generate_family("not_a_family", np.random.default_rng(0), 4, 252)
