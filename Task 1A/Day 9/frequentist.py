"""Frequentist Gaussian-emission HMM for regime detection, Section A3.2,
plus post-hoc economic labelling (A3.3) and the regime statistics Day 4
asks for (durations, transition matrix, stickiness, BIC model selection).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn import hmm

REGIME_LABELS_5 = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]


def filtered_state_probs(model: hmm.GaussianHMM, returns: pd.Series) -> np.ndarray:
    """P(state_t = k | y_0..y_t), the causal (no-look-ahead) counterpart
    to hmmlearn's own `.predict_proba`, which is forward-BACKWARD
    (smoothed: uses the whole series, including observations after t).
    hmmlearn does not expose a public filtered-only API, so this
    re-implements the forward pass directly from the fitted model's own
    parameters (`startprob_`, `transmat_`, `means_`, `covars_`) - the
    same stabilised recursion used throughout this project's Bayesian
    modules (`src.models.hmm.bayesian._forward_pass_numpy`), here for a
    diag-covariance Gaussian HMM instead of a fixed univariate one, so it
    generalises to hmmlearn's n_features >= 1 case in one function rather
    than duplicating a 1-D-only version.

    Needed for Day 9: an ensemble member's contribution must reflect only
    what the model could have known as of day t, and smoothed
    probabilities silently leak the future.
    """
    X = returns.values.reshape(-1, 1) if returns.values.ndim == 1 else returns.values
    T, K = len(X), model.n_components
    d = X.shape[1]
    means = model.means_                      # (K, d)
    variances = model.covars_.reshape(K, d)   # diag covariance, one variance per feature

    log_emission = np.empty((T, K))
    for k in range(K):
        diff2 = (X - means[k]) ** 2 / variances[k]
        log_emission[:, k] = -0.5 * (d * np.log(2 * np.pi) + np.log(variances[k]).sum() + diff2.sum(axis=1))
    emission = np.exp(log_emission - log_emission.max(axis=1, keepdims=True))  # per-row shift for stability

    alpha = np.zeros((T, K))
    alpha[0] = model.startprob_ * emission[0]
    alpha[0] /= alpha[0].sum()
    for t in range(1, T):
        pred = alpha[t - 1] @ model.transmat_
        alpha[t] = pred * emission[t]
        alpha[t] /= alpha[t].sum()
    return alpha


def fit_regime_hmm(returns: pd.Series, n_states: int = 5, n_iter: int = 200,
                    seed: int = 42) -> tuple[hmm.GaussianHMM, np.ndarray, np.ndarray]:
    """Fit a Gaussian-emission HMM to a returns series. Matches Section
    A3.2 exactly (diag covariance, tol=1e-5, single fit at a fixed seed).

    Kept for fidelity to the task document and for the comparison in
    `docs/day4_hmm_notes.md`: on this project's data, a single fit at
    seed=42 converges to a DEGENERATE solution (verified empirically, not
    assumed) - two states with nearly identical mean and variance that
    the decoder then flips between almost every single day (self-
    transition probability near 0), and a third state with zero assigned
    observations and a runaway variance. This is what naive single-seed
    EM does when 5 states are asked for but the 1D return series does not
    identify that many distinguishable components well: a well-known
    HMM/GMM pathology, not a code bug. `fit_regime_hmm_robust` below is
    what Day 4's actual deliverable (regime overlays, duration stats,
    transition matrix) is built from.
    """
    X = returns.values.reshape(-1, 1)
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=seed,
        tol=1e-5,
    )
    model.fit(X)
    states = model.predict(X)
    state_probs = model.predict_proba(X)
    return model, states, state_probs


def fit_regime_hmm_robust(returns: pd.Series, n_states: int = 5, n_iter: int = 200,
                           n_restarts: int = 20, base_seed: int = 42,
                           min_state_share: float = 0.01):
    """Multi-restart version of `fit_regime_hmm`: fits `n_restarts`
    independent models from different random seeds, discards any fit
    where a state ends up with fewer than `min_state_share` of
    observations assigned to it (the empty/collapsed-state pathology
    demonstrated in `fit_regime_hmm`'s docstring), and returns the
    surviving fit with the highest log-likelihood.

    Not in the task document's A3.2 code, which does a single fit. Added
    because the single-seed fit provably degenerates on this data (see
    `docs/day4_hmm_notes.md` for the specific numbers), and a degenerate
    model cannot produce the regime overlays, duration statistics, or
    transition matrix Day 4 actually asks for. Standard practice for EM-
    fitted mixture/HMM models generally, not specific to this dataset.

    Returns (model, states, state_probs, seed_used, n_valid_restarts).
    Raises RuntimeError if every restart degenerates.
    """
    X = returns.values.reshape(-1, 1)
    best = None  # (log_likelihood, model, states, probs, seed)
    n_valid = 0
    for i in range(n_restarts):
        seed = base_seed + i
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=seed,
            tol=1e-5,
        )
        try:
            model.fit(X)
            if not np.all(np.isfinite(model.means_)) or not np.all(np.isfinite(model.covars_)):
                continue  # a state's parameters blew up to NaN/inf mid-EM
            states = model.predict(X)
        except (ValueError, np.linalg.LinAlgError):
            continue  # degenerate fit: NaN startprob_, singular covariance, etc.
        counts = np.bincount(states, minlength=n_states)
        if (counts < min_state_share * len(states)).any():
            continue  # a state collapsed; discard this restart
        n_valid += 1
        ll = model.score(X)
        if best is None or ll > best[0]:
            probs = model.predict_proba(X)
            best = (ll, model, states, probs, seed)

    if best is None:
        raise RuntimeError(
            f"All {n_restarts} restarts produced a collapsed state (< "
            f"{min_state_share:.0%} of observations); try more restarts, "
            f"fewer states, or a higher min_state_share tolerance."
        )
    _, model, states, probs, seed = best
    return model, states, probs, seed, n_valid


def label_regimes(model: hmm.GaussianHMM, K: int = 5) -> dict[int, str]:
    """Map numeric states to economic regime names by mean/vol signature,
    exactly Section A3.3's heuristic: sort by (mean, -vol) descending,
    assign labels in that order. Only defined for K=5 - the label list
    has five names and no principled way to extend it to 3 or 7 states,
    so `fit_regime_hmm` calls with other K are compared by BIC only, not
    labelled."""
    if K != 5:
        raise ValueError(
            f"label_regimes only supports K=5 (the five named regimes in the task "
            f"document); got K={K}. Compare other state counts by BIC instead."
        )
    summary = []
    for i in range(K):
        mu = model.means_[i, 0]
        sig = np.sqrt(model.covars_[i, 0, 0])
        summary.append((i, mu, sig))
    summary.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    mapping = {summary[r][0]: REGIME_LABELS_5[r] for r in range(K)}
    return mapping


def regime_duration_stats(states: np.ndarray, label_map: dict[int, str] | None = None) -> pd.DataFrame:
    """Empirical run-length statistics for each state in a Viterbi-decoded
    state sequence: count of runs, mean/median/max run length in trading
    days. This is the OBSERVED complement to the transition matrix's
    THEORETICAL expected duration (see `transition_matrix_stats`) - the
    two can and do differ for a finite sample."""
    runs = []
    current_state = states[0]
    run_length = 1
    for s in states[1:]:
        if s == current_state:
            run_length += 1
        else:
            runs.append((current_state, run_length))
            current_state = s
            run_length = 1
    runs.append((current_state, run_length))

    df = pd.DataFrame(runs, columns=["state", "run_length"])
    stats = df.groupby("state")["run_length"].agg(["count", "mean", "median", "max"])
    if label_map is not None:
        stats.index = [label_map.get(i, i) for i in stats.index]
    return stats.sort_values("mean", ascending=False)


def transition_matrix_stats(model: hmm.GaussianHMM, label_map: dict[int, str] | None = None):
    """Fitted transition matrix as a labelled DataFrame, with the
    self-transition probability (a direct stickiness measure) and the
    theoretical expected sojourn time 1/(1 - p_ii) implied by it, for
    comparison against the empirical run lengths in `regime_duration_stats`.
    Returns (transition_matrix_df, stickiness_and_duration_df)."""
    K = model.transmat_.shape[0]
    index = [label_map.get(i, i) for i in range(K)] if label_map else list(range(K))
    trans_df = pd.DataFrame(model.transmat_, index=index, columns=index)
    stickiness = pd.Series(np.diag(model.transmat_), index=index, name="self_transition_prob")
    expected_duration = (1.0 / (1.0 - stickiness)).rename("expected_duration_days")
    return trans_df, pd.concat([stickiness, expected_duration], axis=1)


def compare_by_bic(returns: pd.Series, state_counts=(3, 5, 7),
                    n_iter: int = 200, n_restarts: int = 20, base_seed: int = 42) -> pd.DataFrame:
    """Fit a Gaussian HMM for each state count in `state_counts` and
    compare by BIC (and AIC, log-likelihood) - Day 4's "compare 3-state,
    5-state, 7-state HMMs by BIC". Lower BIC is preferred; BIC penalises
    the extra parameters of larger state counts more heavily than AIC
    does, which matters here since parameter count grows with K^2
    (transition matrix) not just K.

    Uses `fit_regime_hmm_robust` for every K, not a single seed: a plain
    single-seed fit at K=7 on this data raises (NaN startprob_ from a
    collapsed state) before it ever reaches `.bic()`, and even K=5
    frequently degenerates (see `fit_regime_hmm`'s and
    `fit_regime_hmm_robust`'s docstrings) - an unfair, uninformative
    comparison if larger K fails outright while smaller K happens to
    survive its one seed.
    """
    X = returns.values.reshape(-1, 1)
    rows = []
    for K in state_counts:
        try:
            model, _, _, seed_used, n_valid = fit_regime_hmm_robust(
                returns, n_states=K, n_iter=n_iter, n_restarts=n_restarts, base_seed=base_seed
            )
        except RuntimeError:
            # A genuine finding, not an error to hide: record it as a row
            # rather than letting one K's non-identifiability crash the
            # whole comparison.
            rows.append({
                "n_states": K, "log_likelihood": np.nan, "aic": np.nan,
                "bic": np.nan, "valid_restarts": f"0/{n_restarts}",
            })
            continue
        rows.append({
            "n_states": K,
            "log_likelihood": model.score(X),
            "aic": model.aic(X),
            "bic": model.bic(X),
            "valid_restarts": f"{n_valid}/{n_restarts}",
        })
    return pd.DataFrame(rows).set_index("n_states")
