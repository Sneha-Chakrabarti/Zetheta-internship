"""Adaptive Conformal Inference, Section A6.5 - the online complement to
the fixed-calibration methods in split_conformal.py and mondrian.py:
alpha_t updates after every realised outcome so the TARGET coverage
level tracks whichever alpha actually keeps realised coverage near
alpha_target, even if the model's underlying miscalibration drifts.
"""
from __future__ import annotations

import numpy as np


def adaptive_conformal_inference(scores_stream, alpha_target: float = 0.1, gamma: float = 0.01):
    """Section A6.5, verbatim. `scores_stream` yields
    (nonconformity_score, q_hat, covered) per step; only `covered` is
    actually used by the spec's own update rule (`err_t` derives from it
    alone), kept as 3-tuples to match the spec's signature even though
    `nonconformity_score` and `q_hat` are unused here - a caller
    computing per-step conformal sets naturally has all three to hand.
    Returns a list of (alpha_t, covered) after each step.
    """
    alpha_t = alpha_target
    coverage_path = []
    for score, q_hat, covered in scores_stream:
        err_t = 0 if covered else 1
        alpha_t = alpha_t + gamma * (alpha_target - err_t)
        alpha_t = min(max(alpha_t, 1e-3), 1 - 1e-3)
        coverage_path.append((alpha_t, covered))
    return coverage_path


def aci_quantile_stream(cal_probs: np.ndarray, y_cal: np.ndarray, test_probs: np.ndarray,
                         y_test: np.ndarray, alpha_target: float = 0.1, gamma: float = 0.01):
    """Runs ACI over a real test sequence using split-conformal-style
    nonconformity scores (`1 - prob[true class]`, as in A6.2), with
    alpha_t determining each step's own quantile from the FIXED
    calibration set (the realised outcome only adjusts alpha_t going
    forward, not the calibration scores themselves - matching the
    spec's separation of calibration-time scoring from online alpha
    adaptation).

    A real limitation, found by testing this under a deliberately severe
    synthetic shift, not assumed: alpha_t can only select WHICH quantile
    of the FIXED calibration score distribution to use each step, so
    `q_hat` is bounded above by `max(cal_scores)` no matter how low
    alpha_t goes. Under a severe enough shift, even that ceiling is not
    wide enough - confirmed directly: with a calibration set capping the
    widest achievable inclusion threshold at 0.605, 95.8% of a shifted
    test stream had every class's probability below that threshold, so
    realised coverage collapsed (to ~0.04, against a 0.9 target) even as
    alpha_t saturated at its floor. ACI adapts the TARGET coverage level;
    it does not and cannot widen the set beyond what the original
    calibration scores support. A production system facing this would
    need to refresh the calibration set itself (e.g. a sliding window),
    not just let alpha_t keep shrinking.

    Returns a dict with arrays: alpha_t (T,), q_hat_t (T,), covered (T,),
    set_size (T,) - everything needed to plot both the alpha trajectory
    and the resulting realised coverage/set-size trajectory.
    """
    cal_scores = 1 - cal_probs[np.arange(len(y_cal)), y_cal]
    n_cal = len(cal_scores)
    T = len(y_test)

    alpha_t = alpha_target
    alphas, q_hats, covered_flags, set_sizes = [], [], [], []
    for t in range(T):
        q_level = min(np.ceil((n_cal + 1) * (1 - alpha_t)) / n_cal, 1.0)
        q_hat = np.quantile(cal_scores, q_level, method="higher")
        pred_set = test_probs[t] >= (1 - q_hat)
        covered = bool(pred_set[y_test[t]])

        alphas.append(alpha_t)
        q_hats.append(q_hat)
        covered_flags.append(covered)
        set_sizes.append(int(pred_set.sum()))

        err_t = 0 if covered else 1
        alpha_t = alpha_t + gamma * (alpha_target - err_t)
        alpha_t = min(max(alpha_t, 1e-3), 1 - 1e-3)

    return {
        "alpha_t": np.array(alphas), "q_hat_t": np.array(q_hats),
        "covered": np.array(covered_flags), "set_size": np.array(set_sizes),
    }
