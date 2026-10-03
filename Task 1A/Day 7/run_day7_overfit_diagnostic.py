"""Diagnostic behind the deep-ensemble early-stopping decision (Day 7).

On time-series CV fold 3, trains a deterministic-member ensemble (M=5)
for 3, 10 and 30 epochs and records NLL, accuracy, balanced accuracy and
mean confidence on WRONG days. Saved to JSON so the notebook reads
numbers from an artifact instead of hard-coding them.

This is a diagnostic, not a hyper-parameter search on test data: the
recipe actually adopted (validation-split early stopping on the training
window) never looks at the test fold.

  .venv-bayesian/bin/python3 scripts/run_day7_overfit_diagnostic.py
"""
import sys, os, json
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.models.bdl.data import load_bdl_data, fit_scaler, apply_scaler
from src.models.bdl.deep_ensemble import train_deep_ensemble, build_deterministic_regime_classifier
from src.models.bdl.evaluation import classification_metrics

FOLD = 3
X_raw, y_int, y_onehot, _, _ = load_bdl_data(standardize=False)
tr, te = list(TimeSeriesSplit(n_splits=5).split(X_raw))[FOLD]
mean, std = fit_scaler(X_raw[tr])
Xtr, Xte = apply_scaler(X_raw[tr], mean, std), apply_scaler(X_raw[te], mean, std)

cnt = np.bincount(y_int[tr], minlength=5).astype(float)
prior_nll = float(-(np.bincount(y_int[te], minlength=5) * np.log(cnt / cnt.sum())).sum() / len(te))
out = {"fold": FOLD, "prior_frequency_nll": prior_nll, "by_epochs": {}}

for epochs in (3, 10, 30):
    ens = train_deep_ensemble(lambda: build_deterministic_regime_classifier(Xtr.shape[1]),
                               Xtr, y_onehot[tr], M=5, epochs=epochs)
    probs = np.stack([m.predict(Xte, verbose=0) for m in ens]).mean(0)
    r = classification_metrics(y_int[te], probs)
    wrong = probs.argmax(1) != y_int[te]
    out["by_epochs"][str(epochs)] = {
        "nll": r["nll"], "accuracy": r["accuracy"], "balanced_accuracy": r["balanced_accuracy"],
        "mean_confidence_on_wrong_days": float(probs.max(1)[wrong].mean()),
        "n_wrong": int(wrong.sum()),
    }
    print(epochs, out["by_epochs"][str(epochs)], flush=True)

json.dump(out, open("artifacts_data/day7_overfit_diagnostic.json", "w"), indent=1)
print("DONE")
