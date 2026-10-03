"""Did pretraining learn anything? (Day 8)

Held-out checks, saved to artifacts_data/day8/pretrain_eval.json:

Chronos-style: next-token cross-entropy of the trained model vs an
untrained copy vs the entropy of the best STATIC token distribution (a
model with no temporal knowledge). Only cross-entropy below the static
baseline shows learned temporal structure.

TimesFM-style: pinball loss vs an untrained copy vs the best static
quantiles (empirical quantiles of the pooled normalised values); plus,
from `forecast_next_patch` (context-only normalisation, no target
leakage), the coverage of the 10-90% interval (target 0.80) and the
median forecast's MAE relative to a "forecast the context mean" baseline.

Each is computed on fresh synthetic series (per family) and on windows
from the project's own synthetic Nifty returns, which are only partly
out-of-distribution (regime switching is one of the nine corpus
families).

  .venv-bayesian/bin/python3 scripts/run_day8_pretrain_eval.py
"""
import sys, os, json
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np

from src.data.loader import DataConfig, load_market_data
from src.models.foundation.corpus import generate_corpus_batch, FAMILIES
from src.models.foundation import chronos_style as cs, patch_model as pm

OUT = "artifacts_data/day8"
N_PER_FAMILY = 60

tok = cs.MeanScaleTokenizer()
lm, enc = cs.build_mini_chronos(seed=0); lm.load_weights(f"{OUT}/chronos_pretrained.h5")
lm0, _ = cs.build_mini_chronos(seed=123)                      # untrained control
fm, fenc = pm.build_mini_timesfm(seed=0); fm.load_weights(f"{OUT}/patch_pretrained.h5")
fm0, _ = pm.build_mini_timesfm(seed=123)

rng = np.random.default_rng(777)                              # never used in pretraining
sets = {}
for j, fam in enumerate(FAMILIES):
    x, _ = generate_corpus_batch(rng, N_PER_FAMILY, families=[fam])
    sets[fam] = x
data = load_market_data(DataConfig(backend="synthetic", n_years=15.0, seed=7))
ret = data["nifty50"]["close"].pct_change().dropna().values
starts = np.arange(0, len(ret) - 252, 6)
sets["project_nifty_returns"] = np.stack([ret[s:s + 252] for s in starts]).astype("float32")
sets["ALL_CORPUS"] = np.concatenate([sets[f] for f in FAMILIES])

# static quantiles for the patch baseline come from the pooled corpus
pool = pm.normalize_windows(sets["ALL_CORPUS"])[0]
static_q = np.quantile(pool.ravel(), pm.QUANTILES)

def static_pinball(patches):
    q = np.array(pm.QUANTILES)[None, None, :, None]
    err = patches[:, 1:, None, :] - static_q[None, None, :, None]
    return float(np.maximum(q * err, (q - 1) * err).mean())

res = {}
for name, x in sets.items():
    tokens, _ = cs.MeanScaleTokenizer().encode(x)
    patches = pm.to_patches(pm.normalize_windows(x)[0], 12)
    fc, tgt = pm.forecast_next_patch(fm, x)
    ctx_mean = x[:, :-12].mean(axis=1, keepdims=True)
    mae_med = np.abs(fc[:, 1, :] - tgt).mean()
    mae_mean = np.abs(ctx_mean - tgt).mean()
    # per-window scale-free comparison so large-scale families do not dominate
    per_win = np.abs(fc[:, 1, :] - tgt).mean(1) / (np.abs(ctx_mean - tgt).mean(1) + 1e-12)
    res[name] = {
        "n": int(len(x)),
        "chronos_ce_trained": cs.lm_cross_entropy(lm, tokens),
        "chronos_ce_untrained": cs.lm_cross_entropy(lm0, tokens),
        "chronos_ce_static_unigram": cs.unigram_entropy(tokens, tok.vocab_size),
        "patch_pinball_trained": pm.eval_pinball(fm, patches),
        "patch_pinball_untrained": pm.eval_pinball(fm0, patches),
        "patch_pinball_static": static_pinball(patches),
        "interval_80_coverage": float(((tgt >= fc[:, 0, :]) & (tgt <= fc[:, 2, :])).mean()),
        "median_mae_over_context_mean_mae_median_of_windows": float(np.median(per_win)),
        "median_mae_over_context_mean_mae_pooled": float(mae_med / mae_mean),
    }
    r = res[name]
    print(f"{name:24s} CE {r['chronos_ce_trained']:.3f} (untrained {r['chronos_ce_untrained']:.3f}, static {r['chronos_ce_static_unigram']:.3f}) | "
          f"pinball {r['patch_pinball_trained']:.3f} (untrained {r['patch_pinball_untrained']:.3f}, static {r['patch_pinball_static']:.3f}) | "
          f"cov80 {r['interval_80_coverage']:.2f} | mae ratio(med of windows) {r['median_mae_over_context_mean_mae_median_of_windows']:.2f}", flush=True)

json.dump(res, open(f"{OUT}/pretrain_eval.json", "w"), indent=1)
print("EVAL DONE")
