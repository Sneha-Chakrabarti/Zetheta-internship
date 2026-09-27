"""Run ONE chain of the marginalized Bayesian HMM and save it to disk.
Must run within a single tool-call's time budget (~280s here) - this
sandbox does not keep background processes alive between separate tool
invocations, discovered empirically (a `nohup ... &` process from one
call was gone by the next call, even with disown). So sampling all 4
chains for Day 5's diagnostics happens as 4 separate invocations of this
script, one per chain index, each saving its own file; a separate step
combines them for R-hat/ESS (both require multiple chains).

Usage: .venv-bayesian/bin/python3 scripts/run_day5_sampling.py <chain_index>
"""
import sys, os, time
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import pymc as pm

from src.data.loader import DataConfig, load_market_data
from src.models.hmm.bayesian import build_marginalized_hmm

TUNE = 500
DRAWS = 500
TARGET_ACCEPT = 0.95
SEED_BASE = 42
WINDOW_YEARS = 1.0  # see docs/day5_bayesian_hmm_notes.md for why this is
                     # much shorter than Day 4's full 15-year frequentist
                     # window - verified empirically, not a guess: at
                     # target_accept=0.95, this environment's single CPU
                     # core does ~1.4s/iteration at T=756 and ~0.49s/
                     # iteration at T=252; the task's literal 2000+1000
                     # draws x 4 chains at the full T=3779 would take
                     # several hours per chain, infeasible here.
OUT_DIR = "artifacts_data"
os.makedirs(OUT_DIR, exist_ok=True)

chain_idx = int(sys.argv[1])

data = load_market_data(DataConfig(backend="synthetic", n_years=15.0, seed=7))
returns_full = data["nifty50"]["close"].pct_change().dropna()
returns = returns_full.tail(int(WINDOW_YEARS * 252))
print(f"T = {len(returns)} (last {WINDOW_YEARS} years of {len(returns_full)} total)", flush=True)

model = build_marginalized_hmm(returns, K=5, alpha_diag=8.0, alpha_off=1.0)

t0 = time.time()
with model:
    trace_c = pm.sample(draws=DRAWS, tune=TUNE, chains=1, cores=1,
                         target_accept=TARGET_ACCEPT, random_seed=SEED_BASE + chain_idx,
                         progressbar=False)
elapsed = time.time() - t0
n_div = int(trace_c.sample_stats["diverging"].sum())
print(f"chain {chain_idx}: {elapsed:.1f}s, {n_div} divergences", flush=True)
trace_c.to_netcdf(f"{OUT_DIR}/day5_chain_{chain_idx}.nc")
print("CHAIN DONE", flush=True)

