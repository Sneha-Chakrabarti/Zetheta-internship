"""One fold of the Section A4.6 time-series cross-validation, comparing
MC Dropout, a class-weighted MC Dropout variant, the variational BNN,
and a deep ensemble. One fold per process invocation (same per-tool-call
time budget reasoning as scripts/run_day5_sampling.py). Usage:
  .venv-bayesian/bin/python3 scripts/run_day7_cv.py <fold_index>

Scaling is fit on each fold's training slice only (no look-ahead).
M and epochs are reduced from the spec's (M=8, epochs=50) to fit the
per-call budget across 5 folds; see docs/day7_bdl_notes.md.
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight

from src.models.bdl.data import load_bdl_data, fit_scaler, apply_scaler, REGIME_ORDER
from src.models.bdl.mc_dropout import build_mc_dropout_regime_classifier, mc_predict
from src.models.bdl.variational_bnn import build_variational_regime_classifier, vi_predict
from src.models.bdl.deep_ensemble import train_deep_ensemble, build_deterministic_regime_classifier
from src.models.bdl.evaluation import classification_metrics, majority_baseline, entropy_decomposition

EPOCHS = 30
M_ENSEMBLE = 5
N_MC = 50
N_SPLITS = 5
N_CLASSES = 5

fold = int(sys.argv[1])
X_raw, y_int, y_onehot, feature_names, _ = load_bdl_data(standardize=False)
train_idx, test_idx = list(TimeSeriesSplit(n_splits=N_SPLITS).split(X_raw))[fold]

mean, std = fit_scaler(X_raw[train_idx])
X_tr, X_te = apply_scaler(X_raw[train_idx], mean, std), apply_scaler(X_raw[test_idx], mean, std)
y_tr, y_te = y_int[train_idx], y_int[test_idx]
Y_tr = y_onehot[train_idx]
d = X_tr.shape[1]

out = {
    "fold": fold, "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
    "train_class_counts": np.bincount(y_tr, minlength=N_CLASSES).tolist(),
    "test_class_counts": np.bincount(y_te, minlength=N_CLASSES).tolist(),
    "baseline": majority_baseline(y_tr, y_te),
    "models": {},
}

def timed(name, fn):
    t0 = time.time()
    res = fn()
    res["seconds"] = round(time.time() - t0, 1)
    out["models"][name] = res
    print(f"fold {fold} {name}: {res['seconds']}s bal_acc={res['balanced_accuracy']:.3f}", flush=True)

def run_mc_dropout(weighted=False):
    tf.random.set_seed(1)
    m = build_mc_dropout_regime_classifier(d, n_regimes=N_CLASSES)
    kwargs = {}
    if weighted:
        present = np.unique(y_tr)
        w = compute_class_weight("balanced", classes=present, y=y_tr)
        cw = {c: 1.0 for c in range(N_CLASSES)}
        cw.update({int(c): float(wi) for c, wi in zip(present, w)})
        kwargs["class_weight"] = cw
    m.fit(X_tr, Y_tr, epochs=EPOCHS, batch_size=64, verbose=0, **kwargs)
    mean_p, std_p, preds = mc_predict(m, X_te, n_samples=N_MC)
    res = classification_metrics(y_te, mean_p, N_CLASSES)
    res["mean_pred_std"] = float(std_p.mean())
    res["mean_entropy_decomp"] = {k: float(v.mean()) for k, v in entropy_decomposition(preds).items()}
    return res

def run_vi():
    tf.random.set_seed(2)
    m = build_variational_regime_classifier(d, n_regimes=N_CLASSES, train_size=len(train_idx))
    m.fit(X_tr, Y_tr, epochs=EPOCHS, batch_size=64, verbose=0)
    mean_p, std_p, preds = vi_predict(m, X_te, n_samples=N_MC)
    res = classification_metrics(y_te, mean_p, N_CLASSES)
    res["mean_pred_std"] = float(std_p.mean())
    res["mean_entropy_decomp"] = {k: float(v.mean()) for k, v in entropy_decomposition(preds).items()}
    return res

def run_ensemble(dropout_members=False, early_stop=False):
    build = (lambda: build_mc_dropout_regime_classifier(d, n_regimes=N_CLASSES)) if dropout_members \
        else (lambda: build_deterministic_regime_classifier(d, n_regimes=N_CLASSES))
    if early_stop:
        ens = train_deep_ensemble(build, X_tr, Y_tr, M=M_ENSEMBLE, epochs=50, validation_split=0.2, patience=3)
    else:
        ens = train_deep_ensemble(build, X_tr, Y_tr, M=M_ENSEMBLE, epochs=EPOCHS)
    preds = np.stack([m.predict(X_te, verbose=0) for m in ens])
    mean_p = preds.mean(axis=0)
    res = classification_metrics(y_te, mean_p, N_CLASSES)
    res["epistemic_std_mean"] = float(preds.std(axis=0).mean())
    res["aleatoric_p1p_mean"] = float((preds * (1 - preds)).mean())
    res["mean_entropy_decomp"] = {k: float(v.mean()) for k, v in entropy_decomposition(preds).items()}
    return res

registry = {
    "mc_dropout": lambda: run_mc_dropout(False),
    "mc_dropout_weighted": lambda: run_mc_dropout(True),
    "variational_bnn": run_vi,
    "deep_ensemble": lambda: run_ensemble(False),
    "deep_ensemble_dropout_members": lambda: run_ensemble(True),
    "deep_ensemble_early_stop": lambda: run_ensemble(False, early_stop=True),
}
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
path = f"artifacts_data/day7_cv_fold{fold}.json"
if only and os.path.exists(path):
    out = json.load(open(path))  # merge into an existing run
for name, fn in registry.items():
    if only is None or name in only:
        timed(name, fn)

with open(f"artifacts_data/day7_cv_fold{fold}.json", "w") as f:
    json.dump(out, f, indent=1)
print("FOLD DONE", flush=True)
