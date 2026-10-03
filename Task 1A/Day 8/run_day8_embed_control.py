"""Control for the Day 8 corpus-overlap confound.

Loads the patch model pretrained WITHOUT the Markov regime-switching
family (`patch_noswitch_pretrained.h5`), produces its rolling 252-day
embeddings for the same windows as `run_day8_embed.py`, and compares its
held-out pretext loss against the original model's (pretrained WITH the
family), on three kinds of held-out data: corpus families other than
regime switching, the regime-switching family alone, and the project's
own synthetic Nifty returns.

Saves artifacts_data/day8/embeddings_control.npz and
artifacts_data/day8/pretrain_eval_control.json.

  .venv-bayesian/bin/python3 scripts/run_day8_embed_control.py
"""
import sys, os, json
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd

from src.data.loader import DataConfig, load_market_data
from src.models.foundation.corpus import generate_corpus_batch, FAMILIES
from src.models.foundation import patch_model as pm

OUT = "artifacts_data/day8"
data = load_market_data(DataConfig(backend="synthetic", n_years=15.0, seed=7))
ret = data["nifty50"]["close"].pct_change().dropna()
dates = pd.read_csv("artifacts_data/day7_features.csv", index_col=0, parse_dates=True).index
pos = ret.index.get_indexer(dates)
windows = np.stack([ret.values[p - 251:p + 1] for p in pos]).astype("float32")     # identical to run_day8_embed.py
patches = pm.to_patches(pm.normalize_windows(windows)[0], 12)

orig, orig_enc = pm.build_mini_timesfm(seed=0); orig.load_weights(f"{OUT}/patch_pretrained.h5")
ctrl, ctrl_enc = pm.build_mini_timesfm(seed=0); ctrl.load_weights(f"{OUT}/patch_noswitch_pretrained.h5")
rand, _ = pm.build_mini_timesfm(seed=123)

np.savez_compressed(f"{OUT}/embeddings_control.npz", patch_noswitch=pm.embed_windows(ctrl_enc, patches),
                    dates=np.array([str(d.date()) for d in dates]))

rng = np.random.default_rng(4242)                                                   # never used in pretraining
non_switch = [f for f in FAMILIES if f != "regime_switch"]
sets = {
    "corpus_without_regime_switch": generate_corpus_batch(rng, 400, families=non_switch)[0],
    "regime_switch_family_only": generate_corpus_batch(rng, 400, families=["regime_switch"])[0],
    "project_nifty_returns": np.stack([ret.values[s:s + 252] for s in np.arange(0, len(ret) - 252, 6)]).astype("float32"),
}
pool = pm.normalize_windows(sets["corpus_without_regime_switch"])[0]
static_q = np.quantile(pool.ravel(), pm.QUANTILES)

def static_pinball(p):
    q = np.array(pm.QUANTILES)[None, None, :, None]
    err = p[:, 1:, None, :] - static_q[None, None, :, None]
    return float(np.maximum(q * err, (q - 1) * err).mean())

res = {}
for name, x in sets.items():
    p = pm.to_patches(pm.normalize_windows(x)[0], 12)
    res[name] = {"n": int(len(x)),
                 "pinball_with_switch_family": pm.eval_pinball(orig, p),
                 "pinball_without_switch_family": pm.eval_pinball(ctrl, p),
                 "pinball_untrained": pm.eval_pinball(rand, p),
                 "pinball_static_quantiles": static_pinball(p)}
    r = res[name]
    print(f"{name:30s} with {r['pinball_with_switch_family']:.4f} | without {r['pinball_without_switch_family']:.4f} | "
          f"untrained {r['pinball_untrained']:.4f} | static {r['pinball_static_quantiles']:.4f}", flush=True)
json.dump(res, open(f"{OUT}/pretrain_eval_control.json", "w"), indent=1)
print("CONTROL EMBED DONE")
