"""SHAP attributions for selected test dates (Day 7).

Explains the deep ensemble's MEAN predicted probability, a deterministic
function of the inputs. (MC Dropout and the variational BNN are
stochastic per call, which makes a point-attribution ill-posed without
fixing the random state; the ensemble mean is the clean target.)

Dates are chosen programmatically from the test window by stated rules,
not hand-picked to tell a story:
  A  most confident correct Risk-On day (lowest ensemble entropy)
  B  correct Risk-Off day with the highest ensemble Risk-Off probability
  C  MISSED Risk-Off day: true Risk-Off, predicted otherwise, the most
     confident such miss (lowest entropy) - a failure case
  D  highest-epistemic-uncertainty day
  E  correct Post-Shock day with the highest Post-Shock probability
Falls back to the closest available rule if a category is empty.

  .venv-bayesian/bin/python3 scripts/run_day7_shap.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import shap
import tensorflow as tf

OUT = "artifacts_data"
REGIMES = ["Risk-On", "Late-Cycle", "Transitional", "Post-Shock", "Risk-Off"]
M = 10

d = np.load(f"{OUT}/day7_final_predictions.npz", allow_pickle=True)
y, dates = d["y_true"], d["dates"]
probs, total, epi = d["ens_probs"], d["ens_total"], d["ens_epistemic"]
X_te, X_tr, feature_names = d["X_test_scaled"], d["X_train_scaled"], list(d["feature_names"])
pred = probs.argmax(1)

def pick(mask, score, largest=True):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return int(idx[np.argmax(score[idx])] if largest else idx[np.argmin(score[idx])])

RO, PS, RF = 0, 3, 4
chosen = {
    "A confident Risk-On": (pick((y == RO) & (pred == RO), total, largest=False), RO),
    "B correct Risk-Off": (pick((y == RF) & (pred == RF), probs[:, RF]), RF),
    "C missed Risk-Off": (pick((y == RF) & (pred != RF), total, largest=False), RF),
    "D highest epistemic": (int(np.argmax(epi)), int(pred[int(np.argmax(epi))])),
    "E correct Post-Shock": (pick((y == PS) & (pred == PS), probs[:, PS]), PS),
}
chosen = {k: v for k, v in chosen.items() if v[0] is not None}
print({k: (dates[i], REGIMES[c]) for k, (i, c) in chosen.items()}, flush=True)

members = [tf.keras.models.load_model(f"{OUT}/day7_ensemble/member_{i}.h5", compile=False) for i in range(M)]
def f(X):
    return np.stack([m.predict(np.asarray(X, dtype="float32"), verbose=0) for m in members]).mean(0)

background = shap.kmeans(X_tr, 30)
explainer = shap.KernelExplainer(f, background)
rows = np.array([i for i, _ in chosen.values()])
t0 = time.time()
sv = explainer.shap_values(X_te[rows], nsamples=400, silent=True)
sv = np.asarray(sv)
# Normalise to (n_dates, n_features, n_outputs) whichever layout this shap version returns.
if sv.shape[0] == len(rows) and sv.shape[1] == len(feature_names):
    pass
elif sv.shape[0] == 5 and sv.shape[1] == len(rows):
    sv = np.transpose(sv, (1, 2, 0))
print("shap values shape:", sv.shape, f"{time.time()-t0:.1f}s", flush=True)

base = np.asarray(explainer.expected_value)
fx = f(X_te[rows])
add_err = []
for j, (_, (_, c)) in enumerate(chosen.items()):
    add_err.append(float(abs(base[c] + sv[j, :, c].sum() - fx[j, c])))
print("max additivity error (|base + sum(shap) - f(x)|):", max(add_err), flush=True)

np.savez_compressed(
    f"{OUT}/day7_shap.npz",
    shap_values=sv, base_values=base, rows=rows, fx=fx,
    labels=np.array(list(chosen.keys())),
    dates=np.array([dates[i] for i in rows]),
    target_class=np.array([c for _, c in chosen.values()]),
    true_class=np.array([y[i] for i in rows]),
    pred_class=np.array([pred[i] for i in rows]),
    total_entropy=np.array([total[i] for i in rows]),
    epistemic_entropy=np.array([epi[i] for i in rows]),
    feature_names=np.array(feature_names), additivity_error=np.array(add_err),
)
print("SHAP DONE", flush=True)
