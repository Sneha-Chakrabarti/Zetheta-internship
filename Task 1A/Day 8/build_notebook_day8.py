"""Builds notebooks/07_foundation_models.ipynb. Run from the repo root
with the bayesian venv's python, then execute with the 'bayesian-env'
kernel. The notebook only LOADS artifacts produced by
scripts/run_day8_*.py (pretraining, embedding and head training ran there,
one stage per tool call)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t))
code = lambda t: cells.append(nbf.v4.new_code_cell(t))

md("""# Day 8: Foundation-model embeddings with a Bayesian head

Regime Lab (Zetheta Algorithms internship, Task 1A), Section D2, Day 8.

**What this is, and is not.** The pretrained Chronos and TimesFM
checkpoints cannot be downloaded from this sandbox (Hugging Face returns
`403 host_not_allowed`, and torch is broken here). So two small
architecture-level re-implementations were built in TensorFlow and
**pretrained on synthetic data**, as instructed:

- a *Chronos-style* model: mean-absolute scaling, uniform binning into a
  token vocabulary, a causal transformer, next-token cross-entropy;
- a *TimesFM-style* model: patches of 12 days, a causal transformer over
  patches, next-patch quantile (pinball) forecasts.

They have about 0.1M parameters each. **Nothing here is Amazon's or
Google's model, and nothing here says whether a real pretrained
foundation model transfers to Indian equities.** It tests the hybrid
pipeline (embedding, then Bayesian head) and whether pretraining on
generic series helps a regime head at all.

**Protocol (continuing Day 7).** Windows are the 252 daily returns ending
on each date. Heads are trained on Day 7's HMM regime labels (Section
A4.6) and scored on the same chronological test window (the last 702
days). Because the panel is synthetic, the generator's *true* regimes are
available for scoring and were never shown to any model. The head is the
same variational (`DenseFlipout`) network for every input, with a fixed
gradient-step budget, so differences come from the input representation.""")

code("""import sys, os, json
sys.path.insert(0, os.path.abspath('..'))
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src.data.loader import DataConfig, load_market_data
from src.utils.plot_style import new_figure, PALETTE

pd.set_option('display.precision', 3); pd.set_option('display.width', 220)
A = '../artifacts_data/day8'
ARMS = ['engineered', 'chronos', 'patch', 'patch_noswitch', 'chronos_rand', 'patch_rand', 'raw_window', 'patch_quantiles']""")

code("""from src.models.foundation import chronos_style as cs, patch_model as pmod
lm, _ = cs.build_mini_chronos(); fm, _ = pmod.build_mini_timesfm()
print(f"Chronos-style parameters: {lm.count_params():,} | TimesFM-style parameters: {fm.count_params():,}")
print(f"Chronos-style: vocab {cs.MeanScaleTokenizer().vocab_size} tokens; TimesFM-style: 252 days = 21 patches of 12")""")

md("## 1. Did pretraining learn anything?")
md("""Pretraining used `src/models/foundation/corpus.py`: series drawn on the
fly from nine generic families with randomised parameters and scales
(AR, random walk with drift, GARCH, **Markov regime switching**, Student-t,
seasonal, trend, jumps, mean-reverting level shifts). The regime-switching
family is the same *kind* of process as the project's synthetic market,
with independently randomised parameters. That overlap could make transfer
look friendlier than it should, so a control was run with that family
**excluded** (Section 4).

A model with no temporal knowledge is the bar to clear, so each loss is
compared with a static baseline: the entropy of the best fixed token
distribution (Chronos-style) and pooled empirical quantiles (TimesFM-style),
both on freshly drawn held-out series.""")

code("""fig, ax = new_figure()
for tag, label, c in [('chronos', 'Chronos-style (cross-entropy)', PALETTE[0]),
                      ('patch', 'TimesFM-style (pinball)', PALETTE[1]),
                      ('patch_noswitch', 'TimesFM-style, no regime-switch family', PALETTE[2])]:
    h = np.array(json.load(open(f'{A}/{tag}_state.json'))['loss_history'])
    ax.plot(h[:, 0], h[:, 1] / h[0, 1], label=f"{label}; first logged value {h[0,1]:.3f}", color=c)
ax.set_xlabel('Pretraining step'); ax.set_ylabel('Loss / first logged loss')
ax.set_title('Pretraining loss (each curve rescaled by its own first value)')
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig('../docs/artifacts/day8_pretraining_loss.png', dpi=110)
for tag in ['chronos', 'patch', 'patch_noswitch']:
    s = json.load(open(f'{A}/{tag}_state.json'))
    print(f"{tag}: {s['steps']} steps, {s['wall_seconds']:.0f}s wall, last logged loss {s['loss_history'][-1][1]:.4f}")""")

code("""ev = json.load(open(f'{A}/pretrain_eval.json'))
rows = []
for k, v in ev.items():
    rows.append({'held-out set': k,
                 'CE trained': v['chronos_ce_trained'], 'CE static': v['chronos_ce_static_unigram'],
                 'CE gain (nats)': v['chronos_ce_static_unigram'] - v['chronos_ce_trained'],
                 'pinball trained': v['patch_pinball_trained'], 'pinball static': v['patch_pinball_static'],
                 'pinball gain %': 100 * (v['patch_pinball_static'] - v['patch_pinball_trained']) / v['patch_pinball_static'],
                 'median-forecast MAE / context-mean MAE': v['median_mae_over_context_mean_mae_median_of_windows'],
                 '10-90% coverage': v['interval_80_coverage']})
pre = pd.DataFrame(rows).set_index('held-out set')
pre""")

md("""**Read this table carefully.** Pretraining learned real structure, but
only in some families. On trend, seasonal and level-shift series the gains
over the static baselines are large (roughly 25 to 34% in pinball loss).
On the return-like families (AR, random walk with drift, GARCH,
regime-switching, Student-t, jumps) the gains are small or absent: within a
few percent in pinball loss, and close to zero or slightly negative in
cross-entropy for several of them. **On the project's own returns the
Chronos-style cross-entropy equals the static baseline, and the median
forecast is no closer to the target than a context mean** (ratio about 1.0).
The 10-90% interval coverage is close to its 0.80 target there.

So these are not useful forecasters of return-like data. Whatever the
embeddings carry about regimes (Section 3) does not come from forecasting
skill on returns.""")

md("""## 2. Embeddings and protocol

`scripts/run_day8_embed.py` produced mean-pooled hidden states (64-d) for
every date in Day 7's label set, from the trained and from an **untrained
copy of the same architecture** (fixed random initialisation). The
untrained copy is the control that separates "pretraining helps" from
"any transformer's random features help". Both models normalise scale away
(Chronos-style by mean absolute value, TimesFM-style by z-scoring), so
their embeddings do not carry the window's volatility *level*.""")

code("""E = np.load(f'{A}/embeddings.npz')
print({k: E[k].shape for k in E.files if k != 'dates'})
n_all = len(E['dates']); n_train = int(0.8 * n_all)
dates = pd.to_datetime(E['dates'])
truth = load_market_data(DataConfig(backend='synthetic', n_years=15.0, seed=7))['regime_path']['regime'].reindex(dates[n_train:])
true_stress = truth.isin(['risk_off', 'post_shock']).values.astype(int)
runs, cur = [], 0
for v in true_stress:
    if v: cur += 1
    elif cur: runs.append(cur); cur = 0
if cur: runs.append(cur)
print(f"windows: {n_all} | training window: first {n_train} | test window: {dates[n_train].date()} to {dates[-1].date()} ({n_all-n_train} days)")
print(f"true Risk-Off/Post-Shock days in the test window: {true_stress.sum()}, in {len(runs)} contiguous episode(s) of {runs} days")""")

md("""Two limits to keep in view. The test window holds only **two** true
stress episodes, so any AUROC below rests on two events, not 146 days. And
each arm uses **3 random training subsets** per size, so the curves are
noisy; error bars are one standard deviation across those 3.""")

md("## 3. Sample efficiency")
code("""rows = []
for m in ARMS:
    for k, v in json.load(open(f'{A}/sample_eff_{m}.json')).items():
        rows.append(dict(method=m, **v))
df = pd.DataFrame(rows)
print(df.groupby('method').size().to_dict())

def curve_plot(metric, ylabel, title, fname, ref=None):
    fig, ax = new_figure(); fig.set_size_inches(9.5, 5)
    g = df.groupby(['method', 'n'])[metric].agg(['mean', 'std']).reset_index()
    style = {'engineered': ('black', '-'), 'chronos': (PALETTE[0], '-'), 'patch': (PALETTE[1], '-'),
             'patch_noswitch': (PALETTE[1], '--'), 'chronos_rand': (PALETTE[0], ':'), 'patch_rand': (PALETTE[1], ':'),
             'raw_window': (PALETTE[5], '-'), 'patch_quantiles': (PALETTE[3], '-')}
    for m in ARMS:
        s = g[g.method == m]; c, ls = style[m]
        ax.errorbar(s['n'], s['mean'], yerr=s['std'], color=c, linestyle=ls, marker='o', markersize=3, linewidth=1.1, capsize=2, label=m)
    if ref is not None:
        for val, lab in ref: ax.axhline(val, color='grey', linewidth=0.8, linestyle='-.'); ax.text(52, val + 0.004, lab, fontsize=7, color='grey')
    ax.set_xscale('log'); ax.set_xlabel('Training examples'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(fname, dpi=110)

curve_plot('auroc_true_stress', 'AUROC of P(stress) vs TRUE stress regime',
           'Sample efficiency: ranking true Risk-Off/Post-Shock days', '../docs/artifacts/day8_sample_efficiency_true_stress.png',
           ref=[(0.5, 'chance')])""")

code("""piv = df.pivot_table(index='n', columns='method', values='auroc_true_stress', aggfunc='mean')[ARMS]
print("AUROC vs TRUE stress, mean over 3 subsets:"); piv.round(3)""")

code("""curve_plot('auroc_hmm_stress', 'AUROC of P(stress) vs HMM stress label',
           'Sample efficiency: ranking the HMM Post-Shock/Risk-Off labels the heads were trained on',
           '../docs/artifacts/day8_sample_efficiency_hmm_stress.png', ref=[(0.5, 'chance')])
big = df[df.n >= 200].groupby('method')[['auroc_true_stress', 'auroc_hmm_stress', 'balanced_accuracy', 'nll_minus_prior', 'accuracy']].mean().loc[ARMS]
print("Mean over training sizes n >= 200 (the curves are close to flat there):"); big.round(3)""")

md("""**What the curves show.**

- **Pretraining matters for these architectures.** The pretrained
  Chronos-style and TimesFM-style embeddings rank true stress days with
  AUROC around 0.87 to 0.88, while the *same architectures untrained* sit at
  chance. The raw 252-return window (no pretraining, no engineering) is also
  at or below chance.
- **They are sample-efficient in the sense of being nearly flat**: AUROC is
  already about 0.85 to 0.88 at 50 training examples. The Chronos-style curve
  stays flat; the two TimesFM-style curves rise by roughly 0.02 to 0.05.
- **But that does not beat the from-scratch route.** The engineered-feature
  arm reaches about the same level at large n, and at the smallest sizes it is
  *higher* (n=50 and n=100 in the table), though noisy. The curves cross; no
  representation dominates. The task's expected result, hybrid ahead of
  scratch on sample efficiency, is **not supported** here.
- **Against the labels the heads were actually trained on (HMM stress), the
  embeddings are much worse** than engineered features (second figure and the
  table above). The embeddings are scale-free, and the HMM states differ in
  volatility *level*, which they cannot see.
- **Calibration and argmax metrics are poor for the embedding arms.** Balanced
  accuracy sits at 0.20 to 0.22 (essentially always predicting Risk-On) against
  about 0.30 for engineered features, and NLL is worse than a class-prior baseline by about
  0.3 nats while engineered features are better than it. The embeddings help
  *rank* stress; the heads on top of them are not usefully calibrated. The
  cause was not investigated (candidates: prior shift between train and test
  windows, head overconfidence, embedding drift); none is tested here.
- The TimesFM-style **quantile-forecast features** (`patch_quantiles`) are
  weaker than the hidden-state embedding but improve broadly with n (about 0.66 at
  n=50 to about 0.83 at n=2806).""")

md("## 4. Control: does the regime-switching family in the corpus explain it?")
code("""ctrl = json.load(open(f'{A}/pretrain_eval_control.json'))
print("Held-out pinball loss (lower is better), TimesFM-style model:")
pd.DataFrame(ctrl).T[['pinball_with_switch_family', 'pinball_without_switch_family', 'pinball_untrained', 'pinball_static_quantiles']]""")

code("""cmp = df[df.n >= 200].groupby('method')['auroc_true_stress'].agg(['mean', 'std']).loc[['patch', 'patch_noswitch', 'patch_rand']]
print("AUROC vs TRUE stress, n >= 200 (mean and std across the 15 runs at those sizes):"); cmp.round(3)""")

md("""Pretraining **without** the regime-switching family gives the same
downstream AUROC as pretraining with it, and the held-out pretext losses are
nearly identical (including on the regime-switching family itself). So the
overlap between the corpus and the target process is **not** what produces the
signal. This control was run for the TimesFM-style model only. The
Chronos-style model was not retrained without the family (about 14 minutes of
compute), so for that model the confound is untested.""")

md("## 5. A trivial baseline the embeddings have to be judged against")
code("""raw = E['raw'][n_train:]
cands = {'std, last 21 days (has scale)': raw[:, -21:].std(1),
         'std, last 63 days (has scale)': raw[:, -63:].std(1),
         'std, full 252 days (has scale)': raw.std(1),
         'std21 / std252 (scale-free)': raw[:, -21:].std(1) / raw.std(1),
         'std63 / std252 (scale-free)': raw[:, -63:].std(1) / raw.std(1),
         'mean|r| last 21 / mean|r| 252 (scale-free)': np.abs(raw[:, -21:]).mean(1) / np.abs(raw).mean(1)}
triv = pd.Series({k: roc_auc_score(true_stress, v) for k, v in cands.items()}, name='AUROC vs true stress')
emb = df[df.n >= 200].groupby('method')['auroc_true_stress'].mean()
print("Pretrained embeddings + Bayesian head (n>=200): chronos %.3f, patch %.3f; engineered %.3f" % (emb['chronos'], emb['patch'], emb['engineered']))
triv.to_frame()""")

md("""**A one-line, scale-free volatility ratio ranks true stress days better
than the pretrained embeddings do**, and so does the last-21-day standard
deviation. The embeddings are informative (well above the untrained
control), but they are not extracting anything a simple statistic does not
already provide. This is the most important context for the results above.
(The heads' inputs are scale-free by construction, so the scale-free rows are
the fair comparison; the rows marked "has scale" show how much the level adds.)""")

md("""## Summary

- Two foundation-style models (Chronos-style tokeniser LM, TimesFM-style
  patch decoder with quantile heads) were implemented in TensorFlow and
  **pretrained on a synthetic corpus**; they are not the real pretrained
  models and cannot say anything about real Chronos or TimesFM.
- Pretraining learned real structure on trend, seasonal and level-shift
  series, and almost none on return-like series. On the project's returns the
  models forecast no better than static baselines.
- Their embeddings nevertheless carry regime information: AUROC about 0.87
  against true stress vs about 0.5 for the untrained architectures. Excluding
  the regime-switching family from the corpus does not change this (tested
  for the TimesFM-style model).
- Hybrid vs from-scratch: **no sample-efficiency advantage**. Engineered
  features match the embeddings at large n and were higher at n=50 and n=100
  (noisy; the order reverses at n=200). Against the HMM labels, the embeddings
  are clearly worse.
- The Bayesian heads on embeddings are poorly calibrated (NLL worse than a
  class prior; balanced accuracy 0.20 to 0.22, i.e. chance).
- A one-line scale-free volatility ratio outperforms the embeddings.
- Limits: 3 training subsets per size; two true stress episodes in the test
  window; single seed per pretraining; no embargo between training and test
  windows (as on Day 7); head uncertainty not analysed; Chronos-style
  control not run.

See `docs/day8_foundation_notes.md`.""")

nb['cells'] = cells
nb['metadata'] = {"kernelspec": {"display_name": "Bayesian (PyMC)", "language": "python", "name": "bayesian-env"},
                   "language_info": {"name": "python", "version": "3.12"}}
with open('notebooks/07_foundation_models.ipynb', 'w') as f:
    nbf.write(nb, f)
print("wrote notebooks/07_foundation_models.ipynb with", len(cells), "cells")
