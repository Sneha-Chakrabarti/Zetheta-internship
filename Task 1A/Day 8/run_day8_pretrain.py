"""Pretrain one foundation-style model on the synthetic corpus, resumably.

  .venv-bayesian/bin/python3 scripts/run_day8_pretrain.py <chronos|patch> <total_steps> [time_budget_s] [noswitch]

The optional `noswitch` variant excludes the Markov regime-switching
family from the corpus and saves under `<name>_noswitch_*`. It is the
control for the main confound in Day 8: that family is the same kind of
process as the project's synthetic market, so a pretrained model's regime
signal might just reflect having seen regime-switching series.

Runs until `time_budget_s` (default 230, under the per-tool-call limit)
or `total_steps`, then saves weights and the loss history and exits, so
repeated invocations continue where the last stopped. Adam moments are
NOT carried across invocations (weights are), a small inefficiency
accepted to keep resumption simple. The learning rate follows a cosine
decay over `total_steps`, restored from the saved step count.
"""
import sys, os, json, time, math
sys.path.insert(0, os.path.abspath("."))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

from src.models.foundation.corpus import generate_corpus_batch, FAMILIES
from src.models.foundation import chronos_style as cs, patch_model as pm

name, total_steps = sys.argv[1], int(sys.argv[2])
budget = float(sys.argv[3]) if len(sys.argv) > 3 else 230.0
variant = sys.argv[4] if len(sys.argv) > 4 else ""
families = [f for f in FAMILIES if f != "regime_switch"] if variant == "noswitch" else None
tag = f"{name}_{variant}" if variant else name
OUT = "artifacts_data/day8"; os.makedirs(OUT, exist_ok=True)
wpath, spath = f"{OUT}/{tag}_pretrained.h5", f"{OUT}/{tag}_state.json"
BASE_LR, MIN_FRAC = 1e-3, 0.1
BATCH = 32 if name == "chronos" else 64

state = json.load(open(spath)) if os.path.exists(spath) else {"steps": 0, "loss_history": [], "wall_seconds": 0.0}
if name == "chronos":
    tok = cs.MeanScaleTokenizer(); model, _ = cs.build_mini_chronos(seed=0); step, opt = cs.make_train_step(model, BASE_LR)
else:
    model, _ = pm.build_mini_timesfm(seed=0); step, opt = pm.make_train_step(model, BASE_LR)
if os.path.exists(wpath):
    model.load_weights(wpath)
print(f"{tag}: resuming at step {state['steps']} of {total_steps}", flush=True)

rng = np.random.default_rng(10_000 + state["steps"])
t0, run, recent = time.time(), 0, []
while state["steps"] < total_steps and time.time() - t0 < budget:
    x, _ = generate_corpus_batch(rng, BATCH, families=families)
    if name == "chronos":
        batch, _ = tok.encode(x)
    else:
        batch = pm.to_patches(pm.normalize_windows(x)[0], 12)
    frac = state["steps"] / total_steps
    lr = BASE_LR * (MIN_FRAC + (1 - MIN_FRAC) * 0.5 * (1 + math.cos(math.pi * frac)))
    if run % 25 == 0:
        tf.keras.backend.set_value(opt.learning_rate, lr)
    loss = float(step(tf.constant(batch)))
    state["steps"] += 1; run += 1; recent.append(loss)
    if state["steps"] % 100 == 0:
        state["loss_history"].append([state["steps"], float(np.mean(recent))]); recent = []
        print(f"step {state['steps']} loss {state['loss_history'][-1][1]:.4f} lr {lr:.2e}", flush=True)

state["wall_seconds"] += time.time() - t0
model.save_weights(wpath); json.dump(state, open(spath, "w"))
print(f"saved at step {state['steps']} ({run} steps this call, {time.time()-t0:.0f}s)", flush=True)
print("PRETRAIN COMPLETE" if state["steps"] >= total_steps else "PRETRAIN INCOMPLETE, run again", flush=True)
