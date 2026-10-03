"""Multivariate Bayesian regime-switching VAR(1), Section A8.3.

Two real gaps in the section's own example code, both verified directly
rather than assumed from reading:

1. `pm.HiddenMarkovChain` does not exist in the installed PyMC (6.3.2) -
   confirmed by `hasattr(pm, 'HiddenMarkovChain')` returning False. The
   section's own note anticipates this and permits marginalising the
   latent chain manually via a forward-algorithm log-likelihood, which is
   what `_forward_algorithm_logp_mvn` below does - the direct multivariate
   extension of the univariate version built for Day 5's HMM
   (`src/models/hmm/bayesian.py`).
2. `pm.LKJCholeskyCov('chol', n=d, ..., shape=(K,))` does not produce K
   independent (d, d) covariance factors - tested directly: it silently
   returns a single (d, d) factor regardless of `shape=(K,)`. But the
   spec's own code then indexes `chol[states]`, which only makes sense if
   `chol` has a leading K dimension. Fixed here by building K independent
   `pm.LKJCholeskyCov` calls in a loop (one per regime, each its own named
   variable) and stacking them - the only way to get regime-conditional
   covariance in this PyMC version.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_LABELS_3 = ["Risk-Off", "Transitional", "Risk-On"]


def unpack_cholesky(packed: np.ndarray, d: int) -> np.ndarray:
    """Unpack PyMC's packed lower-triangular Cholesky representation
    (as stored in the trace for `chol_k`, shape (..., d*(d+1)/2)) into a
    full (..., d, d) lower-triangular matrix. Row-major packing order:
    [L[0,0], L[1,0], L[1,1], L[2,0], L[2,1], L[2,2], ...] - verified
    against `pm.math.expand_packed_triangular` on a known small example,
    not assumed from the docs alone."""
    batch_shape = packed.shape[:-1]
    L = np.zeros(batch_shape + (d, d))
    idx = 0
    for i in range(d):
        for j in range(i + 1):
            L[..., i, j] = packed[..., idx]
            idx += 1
    return L


def relabel_by_nifty_drift(idata):
    """Fix label-switching by permuting regime indices, independently for
    every (chain, draw), so regime 0 is always the highest-nifty_ret-
    intercept regime, regime K-1 the lowest - the same "order by return"
    convention used everywhere else in this project
    (`src.models.hmm.frequentist.label_regimes`,
    `src.models.hmm.bayesian.label_regimes_from_means`).

    Not needed for Day 5's univariate HMM (an `ordered` transform on `mu`
    prevented label-switching at the model level there). No equivalent
    constraint exists here - a d-dimensional VAR intercept has no single
    natural ordering to build into the model - so with 4 independent
    chains, each is free to land on its own arbitrary permutation of the
    K regime labels. Confirmed directly, not assumed: `c[:, :, 0]`
    (the nifty_ret intercept per regime) showed each chain finding the
    same three regimes by drift, in three different index orders, and
    R-hat on `P` came out 1.4-2.4 (should be <1.01) purely from that
    permutation mismatch, not from any real non-convergence within a
    chain.

    Relabels P (both axes), pi, c, A, and the three separately-named
    `chol_k` covariance factors consistently. Returns a plain dict of
    relabelled numpy arrays (chain, draw, ...), not a new InferenceData -
    the caller decides how to package it.
    """
    import numpy as np

    c = idata["posterior"]["c"].values          # (chain, draw, K, d)
    P = idata["posterior"]["P"].values           # (chain, draw, K, K)
    pi = idata["posterior"]["pi"].values         # (chain, draw, K)
    A = idata["posterior"]["A"].values           # (chain, draw, K, d, d)
    n_chains, n_draws, K, d = c.shape
    # chol_k is stored PACKED (chain, draw, d*(d+1)/2), not as a full
    # (d, d) matrix - unpack each before stacking. See `unpack_cholesky`.
    chol_stack = np.stack(
        [unpack_cholesky(idata["posterior"][f"chol_{k}"].values, d) for k in range(K)], axis=2
    )  # (chain, draw, K, d, d)

    c_out = np.empty_like(c)
    P_out = np.empty_like(P)
    pi_out = np.empty_like(pi)
    A_out = np.empty_like(A)
    chol_out = np.empty_like(chol_stack)

    for ch in range(n_chains):
        for dr in range(n_draws):
            order = np.argsort(-c[ch, dr, :, 0])  # descending by nifty_ret intercept
            c_out[ch, dr] = c[ch, dr, order]
            pi_out[ch, dr] = pi[ch, dr, order]
            A_out[ch, dr] = A[ch, dr, order]
            chol_out[ch, dr] = chol_stack[ch, dr, order]
            P_out[ch, dr] = P[ch, dr][order][:, order]

    return {"P": P_out, "pi": pi_out, "c": c_out, "A": A_out, "chol": chol_out}


def build_bayesian_rsvar(Y: np.ndarray, K: int = 3, alpha_diag: float = 8.0,
                          alpha_off: float = 1.0, c_prior: float = 0.05,
                          A_prior: float = 0.3, sd_prior: float = 0.5):
    """Bayesian regime-switching VAR(1). Y: (T, d) standardised feature
    matrix (see `src.models.rsvar.features.build_rsvar_panel`).

    Priors match Section A8.3 (c ~ Normal(0, 0.05), A ~ Normal(0, 0.3),
    persistence-favouring Dirichlet transition rows) except `sd_prior`:
    the spec's HalfNormal(0.03) is calibrated for raw return-scale data
    (~0.01 magnitude); this project's features are standardised to unit
    variance first (see `features.py`), so the emission scale prior is
    HalfNormal(0.5) instead - close to 1 (a regime that exactly matches
    the unconditional variance) with room either side, not clamped near
    zero on a scale these features no longer have.
    """
    import pymc as pm
    import pytensor.tensor as pt

    T, d = Y.shape
    alpha_mat = np.eye(K) * alpha_diag + (1 - np.eye(K)) * alpha_off

    with pm.Model() as model:
        P = pm.Dirichlet("P", a=alpha_mat, shape=(K, K))
        pi = pm.Dirichlet("pi", a=np.ones(K), shape=K)
        c = pm.Normal("c", 0.0, c_prior, shape=(K, d))
        A = pm.Normal("A", 0.0, A_prior, shape=(K, d, d))

        # K independent regime covariances - see module docstring for why
        # this is a loop, not a single shape=(K,) LKJCholeskyCov call.
        chols = []
        for k in range(K):
            chol_k, _, _ = pm.LKJCholeskyCov(
                f"chol_{k}", n=d, eta=2.0,
                sd_dist=pm.HalfNormal.dist(sd_prior, shape=d),
                compute_corr=True,
            )
            chols.append(chol_k)
        chol = pt.stack(chols)  # (K, d, d)

        Y_t = pt.as_tensor_variable(Y)
        loglik = _forward_algorithm_logp_mvn(P, pi, c, A, chol, Y_t)
        pm.Potential("rsvar_loglik", loglik)

    return model


def _mvn_logpdf_batch(x, mu, chol):
    """Log-density of a multivariate normal, batched over a leading
    K dimension: x is (d,), mu is (K, d), chol is (K, d, d) (lower
    Cholesky factors of K covariance matrices). Returns (K,): the
    log-density of the same observation x under each of the K regimes'
    emission distributions - exactly what the forward algorithm needs at
    each timestep."""
    import pytensor.tensor as pt
    from pytensor.tensor.linalg import solve_triangular

    K, d = mu.shape
    diff = x[None, :] - mu  # (K, d)
    # Solve chol_k @ z_k = diff_k for each k (batched triangular solve).
    z = solve_triangular(chol, diff[:, :, None], lower=True).squeeze(-1)  # (K, d)
    quad = pt.sum(z**2, axis=1)  # (K,)
    log_det = 2.0 * pt.sum(pt.log(pt.diagonal(chol, axis1=1, axis2=2)), axis=1)  # (K,)
    return -0.5 * (d * np.log(2 * np.pi) + log_det + quad)


def _forward_algorithm_logp_mvn(P, pi, c, A, chol, Y):
    """Marginalised RS-VAR(1) log-likelihood via the forward algorithm -
    the direct multivariate extension of
    `src.models.hmm.bayesian._forward_algorithm_logp`. Same log-sum-exp
    stabilisation (each step's per-regime log-emission-density is shifted
    by its max before exponentiating), same reason: candidate covariance
    or coefficient draws during warmup can otherwise produce an
    emission density so large or small that exponentiating it overflows.

    y_t | s_t=k, y_{t-1} ~ MVNormal(c_k + A_k @ y_{t-1}, Sigma_k).
    """
    import pytensor
    import pytensor.tensor as pt

    T, d = Y.shape
    K = pi.shape[0]

    def emission_logp_at(t):
        y_prev = Y[t - 1]
        y_t = Y[t]
        mu = c + pt.dot(A, y_prev)  # (K, d): c_k + A_k @ y_prev for every k
        return _mvn_logpdf_batch(y_t, mu, chol)  # (K,)

    # t=0 has no y_{-1}; use the unconditional mean c_k as the regime-0 mean.
    log_em_0 = _mvn_logpdf_batch(Y[0], c, chol)
    m_0 = pt.max(log_em_0)
    alpha_0_unnorm = pi * pt.exp(log_em_0 - m_0)
    c_0 = pt.sum(alpha_0_unnorm)
    alpha_0 = alpha_0_unnorm / c_0
    log_c_0_total = pt.log(c_0) + m_0

    time_indices = pt.arange(1, T)

    def step(t, alpha_prev, P, c, A, chol, Y):
        y_prev = Y[t - 1]
        y_t = Y[t]
        mu = c + pt.dot(A, y_prev)
        log_em_t = _mvn_logpdf_batch(y_t, mu, chol)
        pred = pt.dot(alpha_prev, P)
        m_t = pt.max(log_em_t)
        alpha_t_unnorm = pred * pt.exp(log_em_t - m_t)
        c_t = pt.sum(alpha_t_unnorm)
        return alpha_t_unnorm / c_t, pt.log(c_t) + m_t

    (_, log_cs), _ = pytensor.scan(
        fn=step,
        sequences=[time_indices],
        outputs_info=[alpha_0, None],
        non_sequences=[P, c, A, chol, Y],
        strict=True,
    )
    return log_c_0_total + pt.sum(log_cs)


def sample_model(model, draws: int = 500, tune: int = 500, chains: int = 4,
                  target_accept: float = 0.95, seed: int = 42):
    """Same cores=1 note as src.models.hmm.bayesian.sample_model: this
    sandbox has 1 CPU, and PyMC's automatic BLAS-core allocation raises
    ZeroDivisionError if left to auto-detect."""
    import pymc as pm

    with model:
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                           target_accept=target_accept, random_seed=seed,
                           cores=1, progressbar=False)
    return trace


def _forward_pass_mvn_numpy(P: np.ndarray, pi: np.ndarray, c: np.ndarray,
                             A: np.ndarray, Sigma: list, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Numpy forward pass for the multivariate VAR(1) emission case, one
    parameter set. Mirrors `src.models.hmm.bayesian._forward_pass_numpy`;
    see that function's docstring for what alpha and cnorm mean and why
    filtering (this) and smoothing (below) need to stay separate."""
    from scipy.stats import multivariate_normal

    T, d = Y.shape
    K = len(pi)

    def emission(t):
        y_prev = Y[t - 1] if t > 0 else None
        dens = np.empty(K)
        for k in range(K):
            mu = c[k] if y_prev is None else c[k] + A[k] @ y_prev
            dens[k] = multivariate_normal.pdf(Y[t], mean=mu, cov=Sigma[k])
        return dens

    alpha = np.zeros((T, K))
    cnorm = np.zeros(T)
    alpha[0] = pi * emission(0)
    cnorm[0] = alpha[0].sum()
    alpha[0] /= cnorm[0]
    for t in range(1, T):
        pred = alpha[t - 1] @ P
        alpha[t] = pred * emission(t)
        cnorm[t] = alpha[t].sum()
        alpha[t] /= cnorm[t]
    return alpha, cnorm


def filtered_state_probs_mvn(P: np.ndarray, pi: np.ndarray, c: np.ndarray,
                              A: np.ndarray, Sigma: list, Y: np.ndarray) -> np.ndarray:
    """P(state_t=k | Y_0..Y_t): the causal, no-look-ahead counterpart to
    `smoothed_state_probs_mvn`. Use this, not smoothing, for anything
    claiming to be an out-of-fold or as-of-day-t prediction."""
    alpha, _ = _forward_pass_mvn_numpy(P, pi, c, A, Sigma, Y)
    return alpha


def pointwise_loglik_mvn(P: np.ndarray, pi: np.ndarray, c: np.ndarray,
                          A: np.ndarray, Sigma: list, Y: np.ndarray) -> np.ndarray:
    """log p(Y_t | Y_0..Y_{t-1}) for each t, length T - the per-observation
    shape ArviZ's WAIC/PSIS-LOO require, mirroring
    `src.models.hmm.bayesian.pointwise_loglik`."""
    _, cnorm = _forward_pass_mvn_numpy(P, pi, c, A, Sigma, Y)
    return np.log(cnorm)


def smoothed_state_probs_mvn(P: np.ndarray, pi: np.ndarray, c: np.ndarray,
                              A: np.ndarray, Sigma: list, Y: np.ndarray) -> np.ndarray:
    """Forward-backward smoothed P(state_t=k | all Y) for one parameter
    set (e.g. posterior means). Pure numpy/scipy, mirrors
    `src.models.hmm.bayesian.smoothed_state_probs` for the multivariate
    VAR(1) emission case. Sigma: list of K (d,d) covariance matrices
    (not Cholesky factors, for direct use with scipy)."""
    from scipy.stats import multivariate_normal

    T, d = Y.shape
    K = len(pi)
    alpha, cnorm = _forward_pass_mvn_numpy(P, pi, c, A, Sigma, Y)

    def emission(t):
        y_prev = Y[t - 1] if t > 0 else None
        dens = np.empty(K)
        for k in range(K):
            mu = c[k] if y_prev is None else c[k] + A[k] @ y_prev
            dens[k] = multivariate_normal.pdf(Y[t], mean=mu, cov=Sigma[k])
        return dens

    beta = np.zeros((T, K))
    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (P @ (emission(t + 1) * beta[t + 1])) / cnorm[t + 1]

    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


def impulse_response(A_k: np.ndarray, shock: np.ndarray, horizon: int = 10) -> np.ndarray:
    """VAR(1) impulse response to a one-time shock, for a single regime's
    coefficient matrix A_k: IRF[0] = shock, IRF[h] = A_k @ IRF[h-1].
    Returns (horizon+1, d)."""
    d = A_k.shape[0]
    irf = np.zeros((horizon + 1, d))
    irf[0] = shock
    for h in range(1, horizon + 1):
        irf[h] = A_k @ irf[h - 1]
    return irf
