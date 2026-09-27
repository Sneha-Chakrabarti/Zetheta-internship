import numpy as np
import pandas as pd
import pytest

from src.data.loader import DataConfig, load_market_data
from src.data.regimes import N_REGIMES, REGIME_NAMES, build_transition_matrix, sample_regime_path


def test_transition_matrix_is_row_stochastic():
    P = build_transition_matrix()
    assert P.shape == (N_REGIMES, N_REGIMES)
    np.testing.assert_allclose(P.sum(axis=1), np.ones(N_REGIMES), atol=1e-10)
    assert (P >= 0).all()


def test_regime_path_uses_only_known_labels():
    rng = np.random.default_rng(0)
    path = sample_regime_path(1000, rng)
    assert path.min() >= 0
    assert path.max() < N_REGIMES


def test_synthetic_backend_schema_and_shapes():
    config = DataConfig(backend="synthetic", n_years=2.0, seed=1)
    data = load_market_data(config)

    expected_keys = {
        "nifty50", "nifty_midcap100", "nifty_smallcap100", "india_vix",
        "usdinr", "gilt_10y", "aaa_gilt_spread", "fii_dii_flows",
        "sip_totals", "regime_path",
    }
    assert expected_keys.issubset(data.keys())

    for name in ["nifty50", "nifty_midcap100", "nifty_smallcap100"]:
        df = data[name]
        assert list(df.columns) == ["open", "high", "low", "close"]
        assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
        assert (df > 0).all().all()

    assert data["india_vix"]["value"].between(8.0, 90.0).all()
    assert set(data["regime_path"]["regime"].unique()).issubset(set(REGIME_NAMES))

    # ~2 years of business days, loosely.
    assert 480 <= len(data["nifty50"]) <= 530


def test_synthetic_volatility_increases_down_the_cap_curve():
    """Regression test for a bug where mid/small-cap ended up LESS volatile
    than large-cap: the correlation-mixing weight was also used to scale
    total volatility, shrinking it instead of layering extra vol on top of
    a beta-scaled base. Small-cap > mid-cap > large-cap annualised vol is
    the expected ordering and should hold given the default regime specs."""
    config = DataConfig(backend="synthetic", n_years=10.0, seed=3)
    data = load_market_data(config)

    def annualised_vol(close):
        return np.log(close).diff().std() * np.sqrt(252)

    large_vol = annualised_vol(data["nifty50"]["close"])
    mid_vol = annualised_vol(data["nifty_midcap100"]["close"])
    small_vol = annualised_vol(data["nifty_smallcap100"]["close"])

    assert large_vol < mid_vol < small_vol


def test_csv_backend_raises_clear_error_when_files_absent(tmp_path):
    config = DataConfig(backend="csv", data_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        load_market_data(config)


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        load_market_data(DataConfig(backend="not_a_backend"))
