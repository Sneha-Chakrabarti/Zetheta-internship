"""Rolling 252-day window embeddings (Section A5.2) for every date in the
Day 7 label set, from trained and untrained copies of both models.

Window for date t = the 252 daily returns ENDING at t inclusive. (The
spec's example excludes day t; inclusive is used so the window matches
the same-day information in Day 7's engineered features and the HMM label
at t, which is decoded from returns up to and including t.)

Saved to artifacts_data/day8/embeddings.npz:
  chronos, chronos_rand     mean-pooled hidden states, (n, 64)
  patch, patch_rand         mean-pooled hidden states, (n, 64)
  patch_quantiles           context-normalised 10/50/90% forecasts of the
                            last 12 days from the preceding 240 (n, 36):
                            the "quantile features" of Section A5.3
  raw                       the return windows themselves, (n, 252)

  .venv-bayesian/bin/python3 scripts/run_day8_embed.py
"""
import sys, os, json
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd

from src.data.loader import DataConfig, load_market_data
from src.models.foundation import chronos_style as cs, patch_model as pm

OUT = "artifacts_data/day8"
data = load_market_data(DataConfig(backend="synthetic", n_years=15.0, seed=7))
ret = data["nifty50"]["close"].pct_change().dropna()
dates = pd.read_csv("artifacts_data/day7_features.csv", index_col=0, parse_dates=True).index

pos = ret.index.get_indexer(dates)
assert (pos >= 251).all(), "some label dates lack a full 252-day window"
windows = np.stack([ret.values[p - 251:p + 1] for p in pos]).astype("float32")
print("windows:", windows.shape, dates[0].date(), "->", dates[-1].date(), flush=True)

tok = cs.MeanScaleTokenizer()
tokens, _ = tok.encode(windows)
patches = pm.to_patches(pm.normalize_windows(windows)[0], 12)

lm, enc = cs.build_mini_chronos(seed=0); lm.load_weights(f"{OUT}/chronos_pretrained.h5")
_, enc0 = cs.build_mini_chronos(seed=123)
fm, fenc = pm.build_mini_timesfm(seed=0); fm.load_weights(f"{OUT}/patch_pretrained.h5")
_, fenc0 = pm.build_mini_timesfm(seed=123)

fc, _ = pm.forecast_next_patch(fm, windows)                  # (n, 3, 12) original units
ctx = windows[:, :-12].astype("float64")
mu, sd = ctx.mean(1)[:, None, None], ctx.std(1)[:, None, None] + 1e-12
qfeat = ((fc - mu) / sd).reshape(len(windows), -1).astype("float32")

np.savez_compressed(
    f"{OUT}/embeddings.npz",
    chronos=cs.embed_windows(enc, tokens), chronos_rand=cs.embed_windows(enc0, tokens),
    patch=pm.embed_windows(fenc, patches), patch_rand=pm.embed_windows(fenc0, patches),
    patch_quantiles=qfeat, raw=windows, dates=np.array([str(d.date()) for d in dates]),
)
z = np.load(f"{OUT}/embeddings.npz")
print({k: z[k].shape for k in z.files}, flush=True)
print("all finite:", all(np.isfinite(z[k]).all() for k in z.files if k != "dates"))
print("EMBED DONE")
