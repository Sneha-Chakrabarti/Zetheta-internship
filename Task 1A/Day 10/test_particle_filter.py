import numpy as np
import pytest

from src.models.sequential.particle_filter import RegimeParticleFilter
from src.models.hmm.bayesian import _forward_pass_numpy


def test_propagation_matches_transition_matrix_rows():
    """Each particle's next state should be drawn from its OWN row of P,
    not some averaged or shared distribution."""
    P = np.array([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4], [0.5, 0.25, 0.25]])
    pf = RegimeParticleFilter(P, mu=np.zeros(3), sigma=np.ones(3), n_particles=60000, seed=1)
    particles = np.repeat([0, 1, 2], 20000)
    pf.particles = particles.copy()
    nxt = pf._propagate_vectorised()
    for s in range(3):
        empirical = np.bincount(nxt[particles == s], minlength=3) / (particles == s).sum()
        np.testing.assert_allclose(empirical, P[s], atol=0.02)


def test_step_returns_valid_probability_vector():
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu, sigma = np.array([-1.0, 1.0]), np.array([0.5, 0.5])
    pf = RegimeParticleFilter(P, mu, sigma, n_particles=500, seed=0)
    post = pf.step(0.9)
    assert post.shape == (2,)
    assert np.isclose(post.sum(), 1.0)
    assert (post >= 0).all()


def test_run_shape_matches_observation_count():
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu, sigma = np.array([-1.0, 1.0]), np.array([0.5, 0.5])
    pf = RegimeParticleFilter(P, mu, sigma, n_particles=500, seed=0)
    obs = np.random.default_rng(0).normal(0, 1, 40)
    post = pf.run(obs)
    assert post.shape == (40, 2)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-8)


def test_ess_triggers_resampling_and_is_tracked():
    """A very informative emission model (tight sigma, well-separated mu)
    should collapse ESS quickly and trigger at least one resample."""
    P = np.array([[0.95, 0.05], [0.05, 0.95]])
    mu, sigma = np.array([-1.0, 1.0]), np.array([0.05, 0.05])
    pf = RegimeParticleFilter(P, mu, sigma, n_particles=1000, seed=0)
    pf.run(np.full(20, -1.0))
    assert pf.n_resamples > 0
    assert len(pf.ess_history) == 20
    assert all(e <= 1000 for e in pf.ess_history)


def test_degenerate_observation_raises_rather_than_returning_nan():
    """An observation wildly inconsistent with every regime's emission
    density underflows every particle's likelihood to ~0; this must
    raise, not silently produce NaN weights."""
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    mu, sigma = np.array([0.0, 0.0]), np.array([0.001, 0.001])
    pf = RegimeParticleFilter(P, mu, sigma, n_particles=200, seed=0)
    with pytest.raises(FloatingPointError):
        pf.step(1000.0)


def test_particle_filter_converges_to_exact_forward_filtering():
    """The central validation: a 5000-particle filter should closely
    track the EXACT forward-algorithm filtered posterior
    (src.models.hmm.bayesian._forward_pass_numpy) on the same model and
    data, converging as particle count grows. Uses a small toy HMM (not
    a full refit) so the test runs quickly; the full real-data check
    (99.2% hard-label agreement, mean TV 0.0095) is in
    docs/day10_sequential_notes.md."""
    rng = np.random.default_rng(3)
    K, T = 3, 300
    P = np.array([[0.92, 0.04, 0.04], [0.04, 0.92, 0.04], [0.04, 0.04, 0.92]])
    pi0 = np.full(K, 1 / K)
    mu = np.array([-0.02, 0.0, 0.02])
    sigma = np.array([0.01, 0.008, 0.012])

    # Simulate a real path from this HMM so the comparison is on data the
    # model actually generates, not an arbitrary external series.
    state = rng.choice(K, p=pi0)
    obs = np.empty(T)
    for t in range(T):
        state = rng.choice(K, p=P[state])
        obs[t] = rng.normal(mu[state], sigma[state])

    exact_alpha, _ = _forward_pass_numpy(P, pi0, mu, sigma, obs)

    results = {}
    for n_particles in (200, 5000):
        pf = RegimeParticleFilter(P, mu, sigma, n_particles=n_particles, seed=42)
        pf_post = pf.run(obs)
        tv = 0.5 * np.abs(pf_post - exact_alpha).sum(axis=1)
        results[n_particles] = tv.mean()

    assert results[5000] < 0.08           # close to exact with enough particles
    assert results[5000] < results[200]   # and strictly improves with more particles
