"""Final chronological 80/20 run of all three Bayesian deep-learning
models (Day 7), saving predictions and uncertainty decompositions for the
notebook and the ensemble members for the SHAP step.

  .venv-bayesian/bin/python3 scripts/run_day7_final.py

Training window: first 80% of days. Test window: last 20% (never seen,
scaler fit on the training window only). The deep ensemble uses the
spec's M=10 with deterministic members and validation-based early
stopping (see docs/day7_bdl_notes.md for why).
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.bdl.data import load_bdl_data, fit_scaler, apply_scaler, REGIME_ORDER
from src.models.bdl.mc_dropout import build_mc_dropout_regime_classifier, mc_predict
from src.models.bdl.variational_bnn import build_variational_regime_classifier, vi_predict
from src.models.bdl.deep_ensemble import train_deep_ensemble, build_deterministic_regime_classifier
from src.models.bdl.evaluation import classification_metrics, majority_baseline, entropy_decomposition

EPOCHS, N_MC, M = 30, 200, 10
OUT = "artifacts_data"
os.makedirs(f"{OUT}/day7_ensemble", exist_ok=True)

X_raw, y_int, y_onehot, feature_names, _ = load_bdl_data(standardize=False)
dates = pd.read_csv("artifacts_data/day7_features.csv", index_col=0, parse_dates=True).index
n_train = int(0.8 * len(X_raw))
mean, std = fit_scaler(X_raw[:n_train])
X_tr, X_te = apply_scaler(X_raw[:n_train], mean, std), apply_scaler(X_raw[n_train:], mean, std)
y_tr, y_te = y_int[:n_train], y_int[n_train:]
Y_tr = y_onehot[:n_train]
d = X_tr.shape[1]
print(f"train {len(X_tr)} ({dates[0].date()}..{dates[n_train-1].date()}), test {len(X_te)} ({dates[n_train].date()}..{dates[-1].date()})", flush=True)
print("train class counts:", np.bincount(y_tr, minlength=5).tolist(), "test:", np.bincount(y_te, minlength=5).tolist(), flush=True)

results, store = {"baseline": majority_baseline(y_tr, y_te)}, {}

t0 = time.time(); tf.random.set_seed(1)
m = build_mc_dropout_regime_classifier(d)
m.fit(X_tr, Y_tr, epochs=EPOCHS, batch_size=64, verbose=0)
mp, sp, preds = mc_predict(m, X_te, n_samples=N_MC)
results["mc_dropout"] = classification_metrics(y_te, mp)
store["mc_probs"] = mp; store["mc_std"] = sp
for k, v in entropy_decomposition(preds).items(): store[f"mc_{k}"] = v
print(f"mc_dropout {time.time()-t0:.1f}s", flush=True)

t0 = time.time(); tf.random.set_seed(2)
m = build_variational_regime_classifier(d, train_size=len(X_tr))
m.fit(X_tr, Y_tr, epochs=EPOCHS, batch_size=64, verbose=0)
mp, sp, preds = vi_predict(m, X_te, n_samples=N_MC)
results["variational_bnn"] = classification_metrics(y_te, mp)
store["vi_probs"] = mp; store["vi_std"] = sp
for k, v in entropy_decomposition(preds).items(): store[f"vi_{k}"] = v
print(f"variational_bnn {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
ens = train_deep_ensemble(lambda: build_deterministic_regime_classifier(d), X_tr, Y_tr,
                           M=M, epochs=50, validation_split=0.2, patience=3)
preds = np.stack([mm.predict(X_te, verbose=0) for mm in ens])
mp = preds.mean(axis=0)
results["deep_ensemble"] = classification_metrics(y_te, mp)
store["ens_probs"] = mp
store["ens_epistemic_std"] = preds.std(axis=0)            # spec A4.4: (batch, classes)
store["ens_aleatoric_p1p"] = (preds * (1 - preds)).mean(axis=0)
for k, v in entropy_decomposition(preds).items(): store[f"ens_{k}"] = v
for i, mm in enumerate(ens):
    mm.save(f"{OUT}/day7_ensemble/member_{i}.h5")
print(f"deep_ensemble {time.time()-t0:.1f}s", flush=True)

store["y_true"] = y_te
store["dates"] = np.array([str(x.date()) for x in dates[n_train:]])
store["X_test_scaled"] = X_te
store["X_train_scaled"] = X_tr
store["feature_names"] = np.array(feature_names)
np.savez_compressed(f"{OUT}/day7_final_predictions.npz", **store)
json.dump(results, open(f"{OUT}/day7_final_metrics.json", "w"), indent=1)
print("FINAL DONE", flush=True)
