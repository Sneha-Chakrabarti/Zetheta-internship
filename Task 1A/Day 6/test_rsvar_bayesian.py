import numpy as np
import pytest

from src.models.rsvar.bayesian_rsvar import unpack_cholesky, impulse_response


def test_unpack_cholesky_matches_reference_construction():
    """Regression test: verified directly against pm.math.expand_packed_triangular
    when this was written (see commit message); pinned here without the
    pymc dependency so it can run in the main environment."""
    d = 4
    packed = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, -0.8, 0.9, 1.0])
    L = unpack_cholesky(packed, d)
    assert L.shape == (d, d)
    # Upper triangle (excluding diagonal) must be exactly zero.
    assert np.allclose(np.triu(L, k=1), 0.0)
    # Row-major packing: L[0,0], L[1,0], L[1,1], L[2,0], L[2,1], L[2,2], ...
    assert L[0, 0] == 0.1
    assert L[1, 0] == -0.2
    assert L[1, 1] == 0.3
    assert L[3, 3] == 1.0


def test_unpack_cholesky_batched():
    d = 3
    n_packed = d * (d + 1) // 2
    batch = np.random.default_rng(0).normal(size=(5, 7, n_packed))
    L = unpack_cholesky(batch, d)
    assert L.shape == (5, 7, d, d)
    assert np.allclose(np.triu(L, k=1), 0.0)


def test_impulse_response_shape_and_initial_value():
    A = np.array([[0.5, 0.1], [0.0, 0.3]])
    shock = np.array([1.0, 0.0])
    irf = impulse_response(A, shock, horizon=5)
    assert irf.shape == (6, 2)
    np.testing.assert_allclose(irf[0], shock)


def test_impulse_response_decays_for_stable_matrix():
    """A VAR(1) with all eigenvalues inside the unit circle should decay
    to zero, not blow up - a basic sanity check on the recursion."""
    A = np.diag([0.5, 0.3])
    shock = np.array([1.0, 1.0])
    irf = impulse_response(A, shock, horizon=20)
    assert np.abs(irf[-1]).max() < 0.01
    assert np.abs(irf[-1]).max() < np.abs(irf[0]).max()


def test_impulse_response_matches_closed_form_for_diagonal_A():
    A = np.diag([0.5, 0.5])
    shock = np.array([2.0, 2.0])
    irf = impulse_response(A, shock, horizon=3)
    expected = np.array([2.0 * 0.5**h for h in range(4)])
    np.testing.assert_allclose(irf[:, 0], expected)
    np.testing.assert_allclose(irf[:, 1], expected)
