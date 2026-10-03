"""Day 9: train the TimesFM-style foundation head at full training size
(n=2806, matching one of Day 8's "2806_*" sample-efficiency arms) and
save PER-DAY predictions on the test window, not just aggregate metrics
- Day 8's sample-efficiency script only saved AUROC/NLL summaries.

Uses the same protocol as scripts/run_day8_sample_efficiency.py (same
fit_head/predict_head, same scaler-per-arm convention) so this is a
genuine re-run of that arm, not a new method.

  .venv-bayesian/bin/python3 scripts/run_day9_foundation_head.py
"""
import sys, os, json
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd

from src.models.bdl.data import load_bdl_data, fit_scaler, apply_scaler
from src.models.foundation.hybrid import fit_head, predict_head

OUT = "artifacts_data/day8"
OUT9 = "artifacts_data"
SEED = 0

X_eng, y, _, _, _ = load_bdl_data(standardize=False)
E = np.load(f"{OUT}/embeddings.npz")
assert len(E["dates"]) == len(y)
n_train = int(0.8 * len(y))
te = np.arange(n_train, len(y))

Xtr_raw, Xte_raw = E["patch"][:n_train], E["patch"][te]
mean, std = fit_scaler(Xtr_raw)
Xtr, Xte = apply_scaler(Xtr_raw, mean, std), apply_scaler(Xte_raw, mean, std)

model = fit_head(Xtr, y[:n_train], seed=SEED * 1000 + n_train, n_classes=5)
probs, preds = predict_head(model, Xte, n_mc=200)

np.savez_compressed(
    f"{OUT9}/day9_foundation_head_predictions.npz",
    probs=probs, preds_std=preds.std(axis=0),
    dates=E["dates"][te], y_true=y[te],
)
print(f"Trained on n={n_train}, predicted on {len(te)} test days.", flush=True)
print("probs shape:", probs.shape, "sum-to-1 check:", np.allclose(probs.sum(1), 1.0), flush=True)
print("DONE", flush=True)
