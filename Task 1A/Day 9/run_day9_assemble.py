"""Day 9: assemble every member's probability predictions on the common
252-trading-day window (2024-07-11 to 2025-06-27, verified in-session to
be exactly the last 252 days of Day 7/8's 702-day test window), collapsed
to the 3-way regime space (src/models/ensemble/labels.py).

CRITICAL ASYMMETRY, stated here once and repeated in the docs and
notebook: members fall into two genuinely different categories on this
window.

  GENUINELY OUT-OF-FOLD (trained on the first 2806 days, this window is
  held-out test data they never touched in training):
    mc_dropout, variational_bnn, deep_ensemble (Day 7)
    foundation_head (Day 9, TimesFM-style embedding + Bayesian head)

  NOT OUT-OF-FOLD (fit using data that INCLUDES this exact window):
    freq_hmm       - fit on the full 15-year series, which contains it
    bayesian_hmm   - fit on EXACTLY this 252-day window (Day 5)
    bayesian_rsvar - fit on EXACTLY this 252-day window (Day 6)

The second group's apparent quality below is inflated by in-sample
fitting, not a fair test of them as forecasters. Filtering (not
smoothing) removes only the WITHIN-window look-ahead (day t not seeing
day t+5); it does not undo having used this window's data to choose P,
pi, mu, sigma, A, Sigma in the first place. Both facts have to be true at
once for an honest reading: filtering is a real, necessary fix (Day 4's
model would otherwise use the whole window's future at every day), and
it is not sufficient to make these three members out-of-fold.

  .venv-bayesian/bin/python3 scripts/run_day9_assemble.py
"""
import sys, os, warnings
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import arviz as az
import xarray as xr

from src.data.loader import DataConfig, load_market_data
from src.models.rsvar.features import build_rsvar_panel
from src.models.hmm.frequentist import fit_regime_hmm_robust, label_regimes, filtered_state_probs as freq_filtered
from src.models.hmm.bayesian import filtered_state_probs as bhmm_filtered, label_regimes_from_means
from src.models.rsvar.bayesian_rsvar import relabel_by_nifty_drift, filtered_state_probs_mvn
from src.models.ensemble.labels import collapse_probs_5_to_3, collapse_labels_5_to_3, LABELS_3, LABELS_5

OUT = "artifacts_data"
WINDOW_START, WINDOW_END = "2024-07-11", "2025-06-27"

data = load_market_data(DataConfig(backend="synthetic", n_years=15.0, seed=7))
returns_full = data["nifty50"]["close"].pct_change().dropna()
window_mask = (returns_full.index >= WINDOW_START) & (returns_full.index <= WINDOW_END)
window_dates = returns_full.index[window_mask]
print(f"Common window: {window_dates.min().date()} to {window_dates.max().date()}, n={len(window_dates)}", flush=True)

members = {}   # name -> (N, 3) probability array, aligned to window_dates
oof_flag = {}  # name -> True if genuinely out-of-fold on this window

# --- 1. Frequentist HMM (Day 4): refit exactly as Day 4 did, filter, collapse, slice. ---
model_f, states_f, probs_f, seed_f, n_valid_f = fit_regime_hmm_robust(
    returns_full, n_states=5, n_restarts=20, base_seed=42
)
label_map_f = label_regimes(model_f, K=5)
filt_f = freq_filtered(model_f, returns_full)                          # (T_full, 5), hmmlearn's own state order
order_f = [label_map_f[i] for i in range(5)]                            # column i's display label
probs3_f = collapse_probs_5_to_3(filt_f, order_f)
members["freq_hmm"] = probs3_f[window_mask]
oof_flag["freq_hmm"] = False
print(f"freq_hmm: seed={seed_f} valid_restarts={n_valid_f}/20", flush=True)

# --- 2. Bayesian HMM (Day 5): average filtered probs over posterior draws. ---
idata5 = az.from_netcdf(f"{OUT}/day5_bayesian_hmm_trace.nc")
P5 = idata5["posterior"]["P"].values.reshape(-1, 5, 5)
pi5 = idata5["posterior"]["pi"].values.reshape(-1, 5)
mu5 = idata5["posterior"]["mu"].values.reshape(-1, 5)
sig5 = idata5["posterior"]["sigma"].values.reshape(-1, 5)
returns_252 = returns_full.tail(252)
assert (returns_252.index == window_dates).all()
rng = np.random.default_rng(0)
sub5 = rng.choice(P5.shape[0], size=min(60, P5.shape[0]), replace=False)
gammas5 = [bhmm_filtered(P5[i], pi5[i], mu5[i], sig5[i], returns_252.values) for i in sub5]
filt5 = np.mean(gammas5, axis=0)
label_map_5 = label_regimes_from_means(mu5.mean(0), sig5.mean(0))
order_5 = [label_map_5[i] for i in range(5)]
members["bayesian_hmm"] = collapse_probs_5_to_3(filt5, order_5)
oof_flag["bayesian_hmm"] = False
print("bayesian_hmm: averaged over", len(sub5), "posterior draws", flush=True)

# --- 3. Bayesian RS-VAR (Day 6): relabel, average filtered probs over draws. Native 3-way. ---
idata6 = az.from_netcdf(f"{OUT}/day6_bayesian_rsvar_trace.nc")
relabeled = relabel_by_nifty_drift(idata6)
P6 = relabeled["P"].reshape(-1, 3, 3)
pi6 = relabeled["pi"].reshape(-1, 3)
c6 = relabeled["c"].reshape(-1, 3, 6)
A6 = relabeled["A"].reshape(-1, 3, 6, 6)
chol6 = relabeled["chol"].reshape(-1, 3, 6, 6)
panel6 = build_rsvar_panel(data, tail_days=252)
assert (panel6.index == window_dates).all()
sub6 = rng.choice(P6.shape[0], size=min(60, P6.shape[0]), replace=False)
gammas6 = []
for i in sub6:
    Sigma_i = [chol6[i, k] @ chol6[i, k].T for k in range(3)]
    gammas6.append(filtered_state_probs_mvn(P6[i], pi6[i], c6[i], A6[i], Sigma_i, panel6.Y))
members["bayesian_rsvar"] = np.mean(gammas6, axis=0)   # already Risk-On/Transitional/Risk-Off order
oof_flag["bayesian_rsvar"] = False
print("bayesian_rsvar: averaged over", len(sub6), "posterior draws", flush=True)

# --- 4. Day 7 BDL models: slice the saved test-window predictions to the last 252 rows. ---
P7 = np.load(f"{OUT}/day7_final_predictions.npz", allow_pickle=True)
dates7 = pd.to_datetime(P7["dates"])
sl7 = dates7[-252:]
assert (sl7 == window_dates).all()
for key, name in [("mc_probs", "mc_dropout"), ("vi_probs", "variational_bnn"), ("ens_probs", "deep_ensemble")]:
    members[name] = collapse_probs_5_to_3(P7[key][-252:], LABELS_5)
    oof_flag[name] = True

# --- 5. Day 9 foundation head: slice the last 252 rows of its 702-day predictions. ---
P9 = np.load(f"{OUT}/day9_foundation_head_predictions.npz", allow_pickle=True)
dates9 = pd.to_datetime(P9["dates"])
sl9 = dates9[-252:]
assert (sl9 == window_dates).all()
members["foundation_head"] = collapse_probs_5_to_3(P9["probs"][-252:], LABELS_5)
oof_flag["foundation_head"] = True

# --- Labels: the 5-way HMM pseudo-label Days 7-9's supervised members were
# trained on, and the synthetic panel's true regime, both collapsed to 3-way. ---
labels5_hmm = pd.read_csv(f"{OUT}/day7_labels.csv", index_col=0, parse_dates=True)["regime"]
y_hmm_window = labels5_hmm.reindex(window_dates)
assert not y_hmm_window.isna().any()
y_hmm_3way = collapse_labels_5_to_3(y_hmm_window.values)

true_regime = data["regime_path"]["regime"].reindex(window_dates)
y_true_3way = collapse_labels_5_to_3(true_regime.values)

label_to_int = {l: i for i, l in enumerate(LABELS_3)}
y_hmm_int = np.array([label_to_int[l] for l in y_hmm_3way])
y_true_int = np.array([label_to_int[l] for l in y_true_3way])

order = ["freq_hmm", "bayesian_hmm", "bayesian_rsvar", "mc_dropout", "variational_bnn",
         "deep_ensemble", "foundation_head"]
member_array = np.stack([members[n] for n in order])   # (M, N, 3)
for n in order:
    row_sums = members[n].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"{n} rows do not sum to 1"

np.savez_compressed(
    f"{OUT}/day9_members.npz",
    member_array=member_array, member_names=np.array(order),
    is_out_of_fold=np.array([oof_flag[n] for n in order]),
    dates=np.array([str(d.date()) for d in window_dates]),
    y_hmm_int=y_hmm_int, y_true_int=y_true_int, labels_3=np.array(LABELS_3),
)
print("member_array shape:", member_array.shape, flush=True)
print("HMM 3-way label counts:", dict(zip(*np.unique(y_hmm_3way, return_counts=True))), flush=True)
print("TRUE 3-way regime counts:", dict(zip(*np.unique(y_true_3way, return_counts=True))), flush=True)
print("DONE", flush=True)
