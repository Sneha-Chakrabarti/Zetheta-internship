"""Day 9: Bayesian Model Averaging and constrained stacking (A10.1/A10.2),
evaluated with TimeSeriesSplit so the combination weights never see the
data they are scored on. Scored against both label sources: the HMM
pseudo-labels (what the supervised members were trained to predict) and
the synthetic panel's true regime (what actually happened).

  .venv-bayesian/bin/python3 scripts/run_day9_bma_stacking.py
"""
import sys, os, json
sys.path.insert(0, os.path.abspath("."))

import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.models.ensemble.combine import bma_weights, bma_combine, fit_stacking_weights, mean_log_likelihood

OUT = "artifacts_data"
d = np.load(f"{OUT}/day9_members.npz", allow_pickle=True)
member_array, names = d["member_array"], list(d["member_names"])
M, N, K = member_array.shape
label_sets = {"hmm": d["y_hmm_int"], "true": d["y_true_int"]}

N_SPLITS = 4
results = {}
for label_name, y in label_sets.items():
    results[label_name] = {"folds": [], "per_member_mean": {}, "bma_mean": None, "stacking_mean": None,
                            "bma_weights_by_fold": [], "stacking_weights_by_fold": []}
    per_member_ll = {n: [] for n in names}
    bma_ll, stack_ll = [], []
    equal_ll = []

    for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS).split(np.arange(N))):
        y_tr, y_te = y[tr], y[te]
        probs_tr, probs_te = member_array[:, tr, :], member_array[:, te, :]

        log_pred_lik = np.array([mean_log_likelihood(probs_tr[m], y_tr) * len(tr) for m in range(M)])
        w_bma = bma_weights(log_pred_lik)
        w_stack = fit_stacking_weights(probs_tr, y_tr, n_classes=K)

        combined_bma = bma_combine(probs_te, w_bma)
        combined_stack = bma_combine(probs_te, w_stack)      # same tensordot combine, different weights
        combined_equal = bma_combine(probs_te, np.full(M, 1 / M))

        fold_row = {"fold": fold, "n_train": int(len(tr)), "n_test": int(len(te)),
                    "bma_weights": dict(zip(names, w_bma.round(4).tolist())),
                    "stacking_weights": dict(zip(names, w_stack.round(4).tolist())),
                    "ll_bma": mean_log_likelihood(combined_bma, y_te),
                    "ll_stacking": mean_log_likelihood(combined_stack, y_te),
                    "ll_equal_weight": mean_log_likelihood(combined_equal, y_te),
                    "ll_per_member": {n: mean_log_likelihood(probs_te[m], y_te) for m, n in enumerate(names)}}
        results[label_name]["folds"].append(fold_row)
        bma_ll.append(fold_row["ll_bma"]); stack_ll.append(fold_row["ll_stacking"]); equal_ll.append(fold_row["ll_equal_weight"])
        for m, n in enumerate(names):
            per_member_ll[n].append(fold_row["ll_per_member"][n])
        print(f"[{label_name}] fold {fold} (train={len(tr)}, test={len(te)}): "
              f"BMA={fold_row['ll_bma']:+.3f} stacking={fold_row['ll_stacking']:+.3f} equal={fold_row['ll_equal_weight']:+.3f} "
              f"best_member={max(fold_row['ll_per_member'].values()):+.3f}", flush=True)

    results[label_name]["per_member_mean"] = {n: float(np.mean(v)) for n, v in per_member_ll.items()}
    results[label_name]["bma_mean"] = float(np.mean(bma_ll))
    results[label_name]["stacking_mean"] = float(np.mean(stack_ll))
    results[label_name]["equal_weight_mean"] = float(np.mean(equal_ll))
    best_member_mean = max(results[label_name]["per_member_mean"].values())
    results[label_name]["best_individual_member_mean"] = best_member_mean
    results[label_name]["ensemble_beats_every_member"] = {
        "bma": bool(results[label_name]["bma_mean"] > best_member_mean),
        "stacking": bool(results[label_name]["stacking_mean"] > best_member_mean),
    }

json.dump(results, open(f"{OUT}/day9_bma_stacking.json", "w"), indent=1)
for label_name in label_sets:
    r = results[label_name]
    print(f"\n=== {label_name} labels: mean out-of-fold log-likelihood over {N_SPLITS} folds ===")
    for n, v in sorted(r["per_member_mean"].items(), key=lambda kv: -kv[1]):
        print(f"  {n:16s} {v:+.4f}")
    print(f"  {'BMA':16s} {r['bma_mean']:+.4f}")
    print(f"  {'stacking':16s} {r['stacking_mean']:+.4f}")
    print(f"  {'equal-weight':16s} {r['equal_weight_mean']:+.4f}")
    print(f"  best individual member: {r['best_individual_member_mean']:+.4f}")
    print(f"  ensemble beats every member: BMA={r['ensemble_beats_every_member']['bma']} stacking={r['ensemble_beats_every_member']['stacking']}")
print("\nDONE")
