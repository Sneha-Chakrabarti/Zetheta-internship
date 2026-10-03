import warnings

import numpy as np
import pytest

from src.data.loader import DataConfig, load_market_data
from src.models.rsvar.baseline import (
    fit_msm_regression, fit_msm_regression_stable, transition_matrix,
    regime_occupancy, regime_params,
)

warnings.filterwarnings("ignore")

DATA = load_market_data(DataConfig(backend="synthetic", n_years=5.0, seed=7))
RETURNS = DATA["nifty50"]["close"].pct_change().dropna()


def test_fit_msm_regression_runs():
    res = fit_msm_regression(RETURNS, k_regimes=3, search_reps=5)
    assert res.llf is not None


def test_transition_matrix_is_row_stochastic():
    res = fit_msm_regression(RETURNS, k_regimes=3, search_reps=5)
    P = transition_matrix(res)
    assert P.shape == (3, 3)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)


def test_regime_occupancy_sums_to_sample_size():
    res = fit_msm_regression(RETURNS, k_regimes=3, search_reps=5)
    occ = regime_occupancy(res, 3)
    assert occ.sum() == len(RETURNS)
    assert len(occ) == 3


def test_regime_params_reads_named_params_not_positional():
    res = fit_msm_regression(RETURNS, k_regimes=3, search_reps=5)
    df = regime_params(res, 3)
    assert list(df.index) == [0, 1, 2]
    assert {"mean_annualised", "vol_annualised"}.issubset(df.columns)
    assert (df["vol_annualised"] > 0).all()


def test_fit_msm_regression_stable_picks_highest_likelihood():
    best, all_results = fit_msm_regression_stable(RETURNS, k_regimes=3, search_reps=5, seeds=(1, 2, 3))
    assert len(all_results) >= 1  # some seeds may hit a boundary and be skipped
    assert best.llf == max(r.llf for r in all_results.values())


def test_fit_msm_regression_stable_raises_only_if_every_seed_fails(monkeypatch):
    """Regression test for the boundary-transition-probability crash found
    on this project's data: if every seed's .fit() raises ValueError,
    fit_msm_regression_stable should raise RuntimeError, not propagate
    the statsmodels ValueError directly."""
    import src.models.rsvar.baseline as baseline_module

    class _AlwaysFails:
        def fit(self, **kwargs):
            raise ValueError("Could not untransform parameters.")

    monkeypatch.setattr(baseline_module, "MarkovRegression", lambda *a, **k: _AlwaysFails())
    with pytest.raises(RuntimeError):
        fit_msm_regression_stable(RETURNS, k_regimes=3, seeds=(1, 2))
