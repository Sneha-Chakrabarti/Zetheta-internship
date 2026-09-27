"""Bayesian Gaussian-emission HMM, Section A3.4, plus the fix for a real
bug found in that section's own example code.

Section A3.4's code as given:

    states = pm.Categorical('states', p=pi, shape=T)
    obs = pm.Normal('obs', mu=mu[states], sigma=sigma[states], observed=returns)

draws each day's state i.i.d. from a single static vector `pi`. The
transition matrix `P` is declared with a Dirichlet prior but never used
anywhere in the likelihood - there is no `states[t] ~ Categorical(P[states[t-1]])`
dependency. Verified empirically, not just by reading the code: fitting
this exact model and comparing P's posterior mean against its Dirichlet
prior's theoretical mean (diagonal 0.667, off-diagonal 0.083 for
alpha_diag=8, alpha_off=1, K=5) gives posterior diagonal means of
0.661-0.676 - statistically indistinguishable from the prior. P learns
nothing from the data because nothing in the model lets it. This is not
a Bayesian HMM; it is a static 5-component Gaussian mixture with an
unused, decorative transition-matrix parameter. It is also computationally
impractical at this project's scale regardless: PyMC has no choice but to
sample the discrete `states` via Gibbs/Metropolis (visible in its own
sampler summary: "CompoundStep > NUTS: [P, pi, mu, sigma] >
CategoricalGibbsMetropolis: [states]"), and that took 42 seconds for a
60-observation toy series - unworkable for T ~= 3779.

`build_marginalized_hmm` below is the fix: the discrete state path is
marginalised out analytically via the forward algorithm (implemented as
a `pytensor.scan`), leaving only continuous parameters (P, pi, mu, sigma)
for NUTS to sample - the standard, correct way to fit a Bayesian HMM with
gradient-based samplers, and the only one where P actually participates
in the likelihood. Regime probabilities are recovered afterwards via
forward-backward smoothing at each posterior draw (see
`smoothed_state_probs`), not by sampling discrete states at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_LABELS_5 = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]


def build_naive_hmm_from_task_doc(returns: pd.Series, K: int = 5):
    """Section A3.4's example code, verbatim in structure. Do not run this
    at full scale - see module docstring. Kept only so the disconnection
    bug can be demonstrated and regression-tested on a small series."""
    import pymc as pm

    T = len(returns)
    alpha_diag, alpha_off = 8.0, 1.0
    alpha_mat = np.eye(K) * alpha_diag + (1 - np.eye(K)) * alpha_off
    with pm.Model() as model:
        P = pm.Dirichlet("P", a=alpha_mat, shape=(K, K))
        pi = pm.Dirichlet("pi", a=np.ones(K), shape=K)
        mu = pm.Normal("mu", mu=0, sigma=0.02, shape=K)
        sigma = pm.HalfNormal("sigma", sigma=0.03, shape=K)
        states = pm.Categorical("states", p=pi, shape=T)
        pm.Normal("obs", mu=mu[states], sigma=sigma[states], observed=returns.values)
    return model


def _forward_algorithm_logp(P, pi, mu, sigma, y):
    """Marginalised HMM log-likelihood via the (numerically stabilised)
    forward algorithm, implemented as a pytensor.scan so it is
    differentiable and NUTS can sample P, pi, mu, sigma directly through
    it. Returns a scalar.

    Stabilisation: each step's per-state log-emission-density is shifted
    by its own max before exponentiating (the standard log-sum-exp
    trick), with the shift added back into the accumulated log-
    normaliser. Without this, a candidate sigma sampled very close to 0
    during NUTS warmup makes -log(sigma) blow up for an observation that
    happens to land close to the corresponding mu, overflowing exp() -
    seen empirically as a `RuntimeWarning: overflow encountered in dot`
    on this project's data, not a hypothetical concern.
    """
    import pytensor
    import pytensor.tensor as pt

    y_col = y[:, None]  # (T, 1)
    log_emission = (
        -0.5 * np.log(2 * np.pi) - pt.log(sigma[None, :])
        - 0.5 * ((y_col - mu[None, :]) / sigma[None, :]) ** 2
    )  # (T, K)

    log_em_0 = log_emission[0]
    m_0 = pt.max(log_em_0)
    alpha_0_unnorm = pi * pt.exp(log_em_0 - m_0)
    c_0 = pt.sum(alpha_0_unnorm)
    alpha_0 = alpha_0_unnorm / c_0
    log_c_0_total = pt.log(c_0) + m_0

    def step(log_em_t, alpha_prev, P):
        pred = pt.dot(alpha_prev, P)          # (K,) predictive distribution
        m_t = pt.max(log_em_t)
        alpha_t_unnorm = pred * pt.exp(log_em_t - m_t)
        c_t = pt.sum(alpha_t_unnorm)
        return alpha_t_unnorm / c_t, pt.log(c_t) + m_t

    (_, log_cs), _ = pytensor.scan(
        fn=step,
        sequences=[log_emission[1:]],
        outputs_info=[alpha_0, None],
        non_sequences=[P],
        strict=True,
    )
    return log_c_0_total + pt.sum(log_cs)


def build_marginalized_hmm(returns: pd.Series, K: int = 5, alpha_diag: float = 8.0,
                            alpha_off: float = 1.0, mu_sigma_prior: float = 0.02,
                            sigma_prior: float = 0.03):
    """The corrected Bayesian HMM: same priors as Section A3.4 (Dirichlet
    transition rows with alpha_diag/alpha_off, Normal/HalfNormal
    emissions), but the discrete state path is marginalised out via the
    forward algorithm, so P actually participates in the likelihood.

    An ordered transform on mu fixes the label-switching non-identifiability
    that any mixture/HMM has (states 0..K-1 are otherwise an arbitrary
    permutation the sampler is free to relabel between draws or chains,
    which would break R-hat and make posterior summaries meaningless).
    """
    import pymc as pm
    import pytensor.tensor as pt

    y = returns.values
    alpha_mat = np.eye(K) * alpha_diag + (1 - np.eye(K)) * alpha_off
    init_mu = np.linspace(-2 * mu_sigma_prior, 2 * mu_sigma_prior, K)

    with pm.Model() as model:
        P = pm.Dirichlet("P", a=alpha_mat, shape=(K, K))
        pi = pm.Dirichlet("pi", a=np.ones(K), shape=K)
        mu = pm.Normal("mu", mu=0, sigma=mu_sigma_prior, shape=K,
                        transform=pm.distributions.transforms.ordered,
                        initval=init_mu)
        sigma = pm.HalfNormal("sigma", sigma=sigma_prior, shape=K)
        pm.Potential("hmm_loglik", _forward_algorithm_logp(P, pi, mu, sigma, pt.as_tensor_variable(y)))
    return model


def sample_model(model, draws: int = 2000, tune: int = 1000, chains: int = 4,
                  target_accept: float = 0.95, seed: int = 42, cores: int = 1):
    """Wraps pm.sample with this project's required settings (Day 5:
    2000 draws, 1000 tune, 4 chains, target_accept=0.95). cores=1 is
    forced regardless of what's passed: this sandbox reports 1 CPU, and
    PyMC's automatic BLAS-core allocation divides by the requested core
    count, raising ZeroDivisionError if that resolves to 0 - discovered
    empirically, not a hypothetical."""
    import pymc as pm

    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                           target_accept=target_accept, random_seed=seed,
                           cores=1, progressbar=False)
    return trace


def smoothed_state_probs(P: np.ndarray, pi: np.ndarray, mu: np.ndarray,
                          sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Forward-backward smoothed P(state_t = k | all y) for one parameter
    set (e.g. the posterior mean). Pure numpy, not pytensor - this runs
    post-hoc on point estimates, not inside the sampler."""
    T, K = len(y), len(pi)
    log_emission = (
        -0.5 * np.log(2 * np.pi) - np.log(sigma)[None, :]
        - 0.5 * ((y[:, None] - mu[None, :]) / sigma[None, :]) ** 2
    )
    emission = np.exp(log_emission)

    alpha = np.zeros((T, K))
    c = np.zeros(T)
    alpha[0] = pi * emission[0]
    c[0] = alpha[0].sum()
    alpha[0] /= c[0]
    for t in range(1, T):
        pred = alpha[t - 1] @ P
        alpha[t] = pred * emission[t]
        c[t] = alpha[t].sum()
        alpha[t] /= c[t]

    beta = np.zeros((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (P @ (emission[t + 1] * beta[t + 1])) / c[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


def label_regimes_from_means(mu: np.ndarray, sigma: np.ndarray) -> dict[int, str]:
    """Same mean/vol heuristic as the frequentist module's `label_regimes`
    (Section A3.3), applied to posterior-mean mu/sigma instead of an
    hmmlearn model's fitted attributes."""
    K = len(mu)
    if K != 5:
        raise ValueError(f"Only defined for K=5; got K={K}.")
    summary = [(i, mu[i], sigma[i]) for i in range(K)]
    summary.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    return {summary[r][0]: REGIME_LABELS_5[r] for r in range(K)}
