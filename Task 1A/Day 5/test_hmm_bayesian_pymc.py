import numpy as np
import pandas as pd
import pytest

# --- Tests requiring pymc: skip cleanly if it isn't installed in the
# environment running pytest (it lives in .venv-bayesian, not the main
# environment, because of a real numpy/scikit-learn version conflict
# with giotto-tda - see PROJECT_PLAN.md open decisions). Run these with
# `.venv-bayesian/bin/python3 -m pytest tests/test_hmm_bayesian_pymc.py`. ---
pm = pytest.importorskip("pymc", reason="pymc lives in .venv-bayesian, not the main environment")

from src.models.hmm.bayesian import build_marginalized_hmm, build_naive_hmm_from_task_doc, sample_model
import pandas as pd


@pytest.fixture(scope="module")
def toy_two_block_returns():
    rng = np.random.default_rng(0)
    return pd.Series(np.concatenate([
        rng.normal(0.002, 0.005, 30),
        rng.normal(-0.01, 0.02, 30),
    ]))


def test_naive_hmm_disconnects_P_from_the_likelihood(toy_two_block_returns):
    """Regression test for the bug found in Section A3.4's example code:
    P's posterior should be statistically close to its Dirichlet prior
    mean, because nothing in that model's likelihood depends on P."""
    K = 3
    model = build_naive_hmm_from_task_doc(toy_two_block_returns, K=K)
    with model:
        trace = pm.sample(150, tune=150, chains=1, cores=1, progressbar=False, random_seed=0)
    P_post_diag = np.diag(trace.posterior["P"].mean(dim=["chain", "draw"]).values)
    alpha_diag, alpha_off = 8.0, 1.0
    theoretical_prior_diag = alpha_diag / (alpha_diag + (K - 1) * alpha_off)
    # Posterior should be within a small tolerance of the PRIOR mean,
    # proving it learned essentially nothing from the data.
    assert np.allclose(P_post_diag, theoretical_prior_diag, atol=0.1)


def test_marginalized_hmm_connects_P_to_the_likelihood(toy_two_block_returns):
    """The fix: on a series with an obvious two-block regime switch, the
    marginalized model's P should show clear persistence (diagonal well
    above the flat prior mean), unlike the naive version above."""
    K = 2
    model = build_marginalized_hmm(toy_two_block_returns, K=K, alpha_diag=8.0, alpha_off=1.0)
    trace = sample_model(model, draws=150, tune=150, chains=2, cores=1, seed=0)
    P_post_diag = np.diag(trace.posterior["P"].mean(dim=["chain", "draw"]).values)
    theoretical_prior_diag = 8.0 / (8.0 + 1.0)  # K=2: 0.889
    # Should be noticeably higher than the flat prior mean, reflecting
    # the strong persistence actually present in this toy series.
    assert (P_post_diag > theoretical_prior_diag).all() or (P_post_diag.mean() > 0.9)
