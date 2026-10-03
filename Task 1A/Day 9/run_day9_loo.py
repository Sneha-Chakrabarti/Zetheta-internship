"""Day 9: model comparison (A10.3) for the two genuinely Bayesian
members, Day 5's HMM and Day 6's RS-VAR, via PSIS-LOO.

A10.3 asks for "WAIC / PSIS-LOO ... from ArviZ". The installed ArviZ
(2.x-era, the same DataTree-based release already encountered in Days
5-6) has dropped WAIC entirely - `az.waic` does not exist, and
`az.compare` takes no `ic`/`scale` argument at all, only `method`
(stacking / BB-pseudo-BMA / pseudo-BMA) for combining LOO results
across models. This is a real, checked API change (`hasattr(az, "waic")`
is False), not an oversight: modern ArviZ recommends PSIS-LOO over WAIC
generally, and no longer ships the latter. What follows is PSIS-LOO
only, which is what A10.3's "or" already allows for.

IMPORTANT CAVEAT, repeated in the docs: these two models do not share an
observation space. The HMM's likelihood is built from 1-D Nifty returns;
the RS-VAR's is built from the 6-D standardised feature panel (which
includes those same returns as one of six dimensions). az.compare's
usual use case is several models explaining the SAME observed y (e.g.
two regression specifications for one target series); here it is
comparing two models' fit to their OWN, differently-shaped inputs. The
comparison still means something (which model's implied predictive
density is more consistent with what it was given), but it is not the
apples-to-apples "which model predicts this one series better"
comparison az.compare is usually read as.

  .venv-bayesian/bin/python3 scripts/run_day9_waic.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.abspath("."))
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import arviz as az

from src.data.loader import DataConfig, load_market_data
from src.models.rsvar.features import build_rsvar_panel
from src.models.hmm.bayesian import pointwise_loglik
from src.models.rsvar.bayesian_rsvar import relabel_by_nifty_drift, pointwise_loglik_mvn

OUT = "artifacts_data"

data = load_market_data(DataConfig(backend="synthetic", n_years=15.0, seed=7))
returns_252 = data["nifty50"]["close"].pct_change().dropna().tail(252)
panel6 = build_rsvar_panel(data, tail_days=252)

# --- HMM: pointwise log-lik at every posterior draw. ---
idata5 = az.from_netcdf(f"{OUT}/day5_bayesian_hmm_trace.nc")
P5 = idata5["posterior"]["P"].values      # (chain, draw, 5, 5)
pi5 = idata5["posterior"]["pi"].values
mu5 = idata5["posterior"]["mu"].values
sig5 = idata5["posterior"]["sigma"].values
n_chain, n_draw = P5.shape[:2]

ll5 = np.empty((n_chain, n_draw, len(returns_252)))
for ci in range(n_chain):
    for di in range(n_draw):
        ll5[ci, di] = pointwise_loglik(P5[ci, di], pi5[ci, di], mu5[ci, di], sig5[ci, di], returns_252.values)
idata5_ll = az.from_dict({
    "posterior": {"P": P5, "pi": pi5, "mu": mu5, "sigma": sig5},
    "log_likelihood": {"y": ll5},
})
print("HMM pointwise log-lik array:", ll5.shape, "mean total per draw:", ll5.sum(axis=2).mean(), flush=True)

# --- RS-VAR: pointwise log-lik at every posterior draw (relabelled first). ---
idata6 = az.from_netcdf(f"{OUT}/day6_bayesian_rsvar_trace.nc")
relabeled = relabel_by_nifty_drift(idata6)
P6, pi6, c6, A6, chol6 = relabeled["P"], relabeled["pi"], relabeled["c"], relabeled["A"], relabeled["chol"]
n_chain6, n_draw6 = P6.shape[:2]

ll6 = np.empty((n_chain6, n_draw6, len(panel6.Y)))
for ci in range(n_chain6):
    for di in range(n_draw6):
        Sigma = [chol6[ci, di, k] @ chol6[ci, di, k].T for k in range(3)]
        ll6[ci, di] = pointwise_loglik_mvn(P6[ci, di], pi6[ci, di], c6[ci, di], A6[ci, di], Sigma, panel6.Y)
idata6_ll = az.from_dict({
    "posterior": {"P": P6, "pi": pi6, "c": c6, "A": A6},
    "log_likelihood": {"y": ll6},
})
print("RS-VAR pointwise log-lik array:", ll6.shape, "mean total per draw:", ll6.sum(axis=2).mean(), flush=True)

comparison = az.compare({"bayesian_hmm": idata5_ll, "bayesian_rsvar": idata6_ll})
print(comparison, flush=True)
comparison.to_csv(f"{OUT}/day9_loo_compare.csv")

loo5 = az.loo(idata5_ll, pointwise=True)
loo6 = az.loo(idata6_ll, pointwise=True)
print("\nHMM LOO:\n", loo5, flush=True)
print("\nRS-VAR LOO:\n", loo6, flush=True)
print("\nHMM max Pareto k:", float(np.asarray(loo5.pareto_k).max()), flush=True)
print("RS-VAR max Pareto k:", float(np.asarray(loo6.pareto_k).max()), flush=True)

print("DONE", flush=True)
