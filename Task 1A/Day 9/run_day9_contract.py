"""Day 9: generate output-contract records (A10.4) for the full common
window, using stacking weights fit on the WHOLE window. This is a
schema demonstration, not a performance claim: these weights are fit on
the data they are then applied to, so the resulting records illustrate
the contract's shape, not held-out quality (that is what
scripts/run_day9_bma_stacking.py's TimeSeriesSplit numbers are for).

  .venv-bayesian/bin/python3 scripts/run_day9_contract.py
"""
import sys, os, json
sys.path.insert(0, os.path.abspath("."))

import numpy as np

from src.models.ensemble.combine import fit_stacking_weights
from src.models.ensemble.contract import build_output_contract_series

OUT = "artifacts_data"
d = np.load(f"{OUT}/day9_members.npz", allow_pickle=True)
member_array, names = d["member_array"], list(d["member_names"])
dates, labels3 = list(d["dates"]), list(d["labels_3"])
y_hmm = d["y_hmm_int"]

w = fit_stacking_weights(member_array, y_hmm, n_classes=3)
print("Final stacking weights (fit on the full window, HMM labels):")
for n, wi in zip(names, w):
    print(f"  {n:16s} {wi:.4f}")

combined = np.tensordot(w, member_array, axes=(0, 0))  # (N, 3)
records = build_output_contract_series(dates, combined, labels3, member_array, names, w, "stacking_v1")

json.dump(records, open(f"{OUT}/day9_output_contract_sample.json", "w"), indent=1)
print(f"\nWrote {len(records)} records. Example (last day):")
print(json.dumps(records[-1], indent=1))
print("DONE")
