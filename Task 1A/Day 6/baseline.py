"""Single-feature Markov-switching regression baseline, Section A8.2.

statsmodels' `MarkovRegression` uses a COLUMN-stochastic transition
matrix convention: `res.regime_transition[i, j, t]` = P(state=i at t |
state=j at t-1), so each COLUMN sums to 1. Every other model in this
project (`src/models/hmm/frequentist.py`, `src/models/hmm/bayesian.py`)
uses hmmlearn's ROW-stochastic convention instead (P[i, j] = P(state=j |
state=i), row i sums to 1). `transition_matrix` below transposes on the
way out so it is row-stochastic everywhere in this project, not just
here - getting this backwards silently would make every duration/
stickiness number wrong without erroring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def fit_msm_regression(y: pd.Series, k_regimes: int = 3, switching_variance: bool = True,
                        search_reps: int = 20):
    """Fit Section A8.2's baseline exactly, with `search_reps` random
    restarts (statsmodels' own built-in multi-start) rather than a single
    fit from default starting values - the single-start fit on this
    project's data left most standard errors as NaN and one regime with
    only 16 of 3779 days assigned to it; `search_reps=20` does not fully
    resolve this (see docs/day6_rsvar_notes.md) but is still better than
    a single start and costs only seconds.
    """
    mod = MarkovRegression(y.values, k_regimes=k_regimes, trend="c",
                            switching_variance=switching_variance)
    res = mod.fit(search_reps=search_reps)
    return res


def transition_matrix(res) -> np.ndarray:
    """Row-stochastic transition matrix (see module docstring for why
    this transposes statsmodels' own column-stochastic output)."""
    col_stochastic = res.regime_transition[:, :, 0]
    return col_stochastic.T


def regime_occupancy(res, k_regimes: int) -> np.ndarray:
    """Hard-assignment (argmax of smoothed probabilities) count per
    state, the fastest way to see a collapsed/near-empty regime before
    trusting any parameter estimate attached to it."""
    smoothed = res.smoothed_marginal_probabilities
    hard_states = np.argmax(np.asarray(smoothed), axis=1)
    return np.bincount(hard_states, minlength=k_regimes)


def fit_msm_regression_stable(y: pd.Series, k_regimes: int = 3, switching_variance: bool = True,
                               search_reps: int = 30, seeds=(1, 2, 3, 4, 5)):
    """Fit across several random-search seeds and keep the
    highest-likelihood result.

    Not in Section A8.2's code, which does a single `.fit()` call with no
    seed control. Added because a single default-seed fit on this
    project's data left most standard errors NaN and one regime with only
    16 of 3779 days assigned - and across 5 explicit seeds (search_reps=30
    each), 4 converged to the same qualitative 3-way partition (up to
    label permutation, log-likelihood 12112-12116) while 1 landed in a
    clearly worse local optimum (log-likelihood 12078, a 6-observation
    regime). Picking the best-likelihood seed here is the same principle
    as `fit_regime_hmm_robust` in the frequentist HMM module: the
    underlying MLE point estimates are stable across most seeds, even
    though statsmodels' numerical-Hessian standard errors are frequently
    NaN regardless of which seed wins (see docs/day6_rsvar_notes.md) -
    a known limitation of complex-step differentiation near the
    transition-probability simplex boundary, not evidence the point
    estimates themselves are unreliable.

    Returns (best_result, all_results_by_seed_dict). Seeds whose search
    lands on a transition-probability estimate exactly at the [0, 1]
    boundary are skipped, not allowed to crash the whole comparison:
    statsmodels' logistic un-transform uses a root-finder that can fail
    to converge exactly at that boundary (`ValueError: Could not
    untransform parameters`), seen on this project's data with a small
    window and few search_reps. Raises RuntimeError only if every seed
    fails this way.
    """
    results = {}
    for seed in seeds:
        mod = MarkovRegression(y.values, k_regimes=k_regimes, trend="c",
                                switching_variance=switching_variance)
        try:
            res = mod.fit(search_reps=search_reps, rng=np.random.default_rng(seed))
        except ValueError:
            continue  # boundary transition probability broke the un-transform step
        results[seed] = res
    if not results:
        raise RuntimeError(
            f"All {len(seeds)} seeds failed (boundary transition-probability "
            f"estimates); try more search_reps or different seeds."
        )
    best_seed = max(results, key=lambda s: results[s].llf)
    return results[best_seed], results


def regime_params(res, k_regimes: int) -> pd.DataFrame:
    """Per-regime const (mean) and variance, annualised, read out of
    res.params by name (verified against res.model.param_names, not
    assumed by position)."""
    names = list(res.model.param_names)
    params = res.params
    rows = []
    for k in range(k_regimes):
        const = params[names.index(f"const[{k}]")]
        sigma2 = params[names.index(f"sigma2[{k}]")]
        rows.append({
            "regime": k,
            "mean_annualised": const * 252,
            "vol_annualised": np.sqrt(sigma2 * 252),
        })
    return pd.DataFrame(rows).set_index("regime")
