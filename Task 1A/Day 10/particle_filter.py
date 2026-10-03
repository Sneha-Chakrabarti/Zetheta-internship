"""Bootstrap particle filter for online regime inference, Section A9.1.

Matches the spec's `RegimeParticleFilter` (propagate through the
transition matrix, reweight by Gaussian emission likelihood, systematic
resample when ESS < N/2), with the particle-propagation step vectorised:
the spec's own code draws each particle's next state with a Python list
comprehension (`[self.rng.choice(self.K, p=self.P[s]) for s in
self.particles]`), which is one Python-level RNG call per particle per
timestep - at N=5000 particles and T~3700 days this is roughly 18.5
million individual `rng.choice` calls, checked to take too long to be
practical here. The vectorised version draws the same distribution (one
categorical draw per particle from its own row of P) via inverse-CDF
sampling against a single `rng.random(N)` draw, not a different
algorithm.
"""
from __future__ import annotations

import numpy as np


class RegimeParticleFilter:
    """Bootstrap particle filter for a K-regime HMM with Gaussian
    emissions. Same constructor signature and `step` semantics as the
    spec; see module docstring for the one implementation difference
    (vectorised propagation, not a different filter)."""

    def __init__(self, P: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                 n_particles: int = 5000, seed: int = 42):
        self.P = P
        self.mu = mu
        self.sigma = sigma
        self.K = P.shape[0]
        self.N = n_particles
        self.rng = np.random.default_rng(seed)
        self.particles = self.rng.integers(0, self.K, size=self.N)
        self.weights = np.full(self.N, 1.0 / self.N)
        self.n_resamples = 0          # diagnostic: how often ESS triggered resampling
        self.ess_history: list[float] = []

    def _propagate_vectorised(self) -> np.ndarray:
        """Draw `self.particles[i]` ~ Categorical(P[self.particles[i]])
        for every i at once: cumulative-sum the transition row each
        particle is currently in, then a single inverse-CDF lookup
        against one `rng.random(N)` draw - the same per-particle
        distribution as the spec's per-particle `rng.choice`, not an
        approximation of it."""
        cum = np.cumsum(self.P[self.particles], axis=1)  # (N, K)
        u = self.rng.random(self.N)[:, None]
        return (u > cum).sum(axis=1)   # number of cumulative bins u exceeds = next state index

    def step(self, obs: float) -> np.ndarray:
        self.particles = self._propagate_vectorised()

        like = np.exp(-0.5 * ((obs - self.mu[self.particles]) / self.sigma[self.particles]) ** 2) \
            / (self.sigma[self.particles] * np.sqrt(2 * np.pi))
        self.weights = self.weights * like
        weight_sum = self.weights.sum()
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            # Every particle has ~0 likelihood (an extreme observation
            # relative to every regime's emission density): fail
            # explicitly rather than silently dividing by ~0, which
            # would otherwise produce NaN weights and a filter that
            # continues to run while reporting garbage.
            raise FloatingPointError(
                f"Particle weights collapsed to zero (obs={obs}); every particle's emission "
                f"likelihood underflowed. Increase n_particles or check `obs` is on the same "
                f"scale as `mu`/`sigma`."
            )
        self.weights = self.weights / weight_sum

        ess = 1.0 / np.sum(self.weights ** 2)
        self.ess_history.append(float(ess))
        if ess < self.N / 2:
            idx = self.rng.choice(self.N, size=self.N, p=self.weights)
            self.particles = self.particles[idx]
            self.weights = np.full(self.N, 1.0 / self.N)
            self.n_resamples += 1

        post = np.bincount(self.particles, weights=self.weights, minlength=self.K)
        return post / post.sum()

    def run(self, observations: np.ndarray) -> np.ndarray:
        """`step` applied to a whole series. Returns (T, K): the online
        regime posterior after each observation."""
        return np.array([self.step(x) for x in observations])
