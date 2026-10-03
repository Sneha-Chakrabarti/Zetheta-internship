"""Sample-efficiency comparison (Day 8).

For each input representation, train the SAME variational head on random
subsets of the training window of increasing size and score it on the
fixed, later, never-seen test window (the last 702 days, as on Day 7).

Inputs compared:
  engineered        Day 3's 31 engineered features (Day 7's input)
  raw_window        the 252 raw daily returns (same information the
                    foundation models see, no pretraining)
  chronos           Chronos-style pretrained embedding (64-d)
  chronos_rand      the SAME architecture, untrained (control)
  patch             TimesFM-style pretrained embedding (64-d)
  patch_rand        the SAME architecture, untrained (control)
  patch_quantiles   the TimesFM-style model's 10/50/90% forecast features

Results are appended per (method, n, seed) to
artifacts_data/day8/sample_eff_<method>.json, so a call that runs out of
time loses nothing and the next call continues.

  .venv-bayesian/bin/python3 scripts/run_day8_sample_efficiency.py <m1,m2,...> [budget_s]
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.loader import DataConfig, load_market_data
from src.models.bdl.data import load_bdl_data, fit_scaler, apply_scaler
from src.models.bdl.evaluation import classification_metrics
from src.models.foundation.hybrid import fit_head, predict_head

OUT = "artifacts_data/day8"
SIZES = [50, 100, 200, 400, 800, 1600, 2806]
SEEDS = [0, 1, 2]
methods = sys.argv[1].split(",")
budget = float(sys.argv[2]) if len(sys.argv) > 2 else 235.0

X_eng, y, _, _, _ = load_bdl_data(standardize=False)
E = np.load(f"{OUT}/embeddings.npz")
assert len(E["dates"]) == len(y)
inputs = {"engineered": X_eng, "raw_window": E["raw"], "chronos": E["chronos"], "chronos_rand": E["chronos_rand"],
          "patch": E["patch"], "patch_rand": E["patch_rand"], "patch_quantiles": E["patch_quantiles"]}
if os.path.exists(f"{OUT}/embeddings_control.npz"):                      # no-regime-switch pretraining control
    C = np.load(f"{OUT}/embeddings_control.npz")
    assert (C["dates"] == E["dates"]).all()
    inputs["patch_noswitch"] = C["patch_noswitch"]
n_train = int(0.8 * len(y))
te = np.arange(n_train, len(y))
y_te = y[te]
truth = load_market_data(DataConfig(backend="synthetic", n_years=15.0, seed=7))["regime_path"]["regime"]
true_stress = truth.reindex(pd.to_datetime(E["dates"][te])).isin(["risk_off", "post_shock"]).values.astype(int)
hmm_stress = np.isin(y_te, [3, 4]).astype(int)

def scale(name, Xtr, Xte):
    if name == "raw_window":                       # one scalar: 252 per-dim stats from 50 rows would be noise
        s = float(Xtr.std()) + 1e-12
        return (Xtr / s).astype("float32"), (Xte / s).astype("float32")
    m, s = fit_scaler(Xtr)
    return apply_scaler(Xtr, m, s), apply_scaler(Xte, m, s)

t0 = time.time()
for name in methods:
    path = f"{OUT}/sample_eff_{name}.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    for n in SIZES:
        for seed in SEEDS:
            key = f"{n}_{seed}"
            if key in res:
                continue
            if time.time() - t0 > budget:
                json.dump(res, open(path, "w")); print(f"{name}: out of time, saved {len(res)} results", flush=True); sys.exit(0)
            idx = np.random.default_rng(seed).choice(n_train, n, replace=False) if n < n_train else np.arange(n_train)
            Xtr, Xte = scale(name, inputs[name][idx], inputs[name][te])
            model = fit_head(Xtr, y[idx], seed=seed * 1000 + n)
            probs, _ = predict_head(model, Xte)
            m = classification_metrics(y_te, probs)
            counts = np.bincount(y[idx], minlength=5).astype(float)
            prior = (counts + 1) / (counts.sum() + 5)                          # Laplace-smoothed subsample prior
            prior_nll = float(-(np.bincount(y_te, minlength=5) * np.log(prior)).sum() / len(y_te))
            pstress = probs[:, 3] + probs[:, 4]
            res[key] = {"n": n, "seed": seed, "nll": m["nll"], "prior_nll": prior_nll,
                        "nll_minus_prior": m["nll"] - prior_nll, "balanced_accuracy": m["balanced_accuracy"],
                        "accuracy": m["accuracy"], "auroc_hmm_stress": float(roc_auc_score(hmm_stress, pstress)),
                        "auroc_true_stress": float(roc_auc_score(true_stress, pstress)),
                        "n_stress_labels_in_subsample": int(np.isin(y[idx], [3, 4]).sum())}
    json.dump(res, open(path, "w"))
    print(f"{name}: complete ({len(res)} results, {time.time()-t0:.0f}s elapsed)", flush=True)
print("ALL REQUESTED METHODS COMPLETE")
