# Day 8: foundation-model embeddings with a Bayesian head

Scope: `src/models/foundation/` (corpus, Chronos-style model, TimesFM-style
model, hybrid wrapper), scripts `scripts/run_day8_*.py`, notebook
`notebooks/07_foundation_models.ipynb`. Pretraining, embedding and head
training ran in the scripts, one stage per tool call. The notebook loads
their saved outputs.

## What this is, and is not

The pretrained Chronos and TimesFM checkpoints cannot be downloaded here.
Hugging Face returns `403 host_not_allowed` (tested), and torch no longer
imports in the main environment (PROJECT_PLAN, open decision 9). On the
instruction to use synthetic data and implement the models, two small
architecture-level re-implementations were built in TensorFlow and
**pretrained on synthetic series**:

| | Chronos-style | TimesFM-style |
|---|---|---|
| input | mean-abs scaled, 256 uniform bins, 258-token vocabulary | z-scored, 21 patches of 12 days |
| network | causal transformer, 2 layers, d=64 | causal transformer, 3 layers, d=64 |
| objective | next-token cross-entropy | next-patch pinball loss (10/50/90%) |
| parameters | 116,482 | 105,060 |
| pretraining | 3000 steps, batch 32, 863 s | 12000 steps, batch 64, 377 s |
| embedding | mean-pooled hidden states, 64-d | mean-pooled hidden states, 64-d |

**Nothing here is Amazon's or Google's model.** The original models have
millions to hundreds of millions of parameters and are pretrained on vastly
more data. These results say nothing about whether a real pretrained
foundation model transfers to Indian equities. They test the hybrid
pipeline (embedding, then Bayesian head) and whether pretraining on generic
series helps a regime head at all.

## Protocol

Window = the 252 daily returns ending on each date. Heads train on Day 7's
HMM regime labels (Section A4.6). Test window = the last 702 days, as on
Day 7. The head is the same variational network (`DenseFlipout`, 64 and 32
hidden units) for every input, trained for a fixed number of gradient steps
(600), so differences come from the input representation. Training sizes
50, 100, 200, 400, 800, 1600 and 2806, with **3 random subsets** per size.

Because the panel is synthetic, the generator's true regimes are available
and were never shown to any model. Heads are scored by AUROC of
`P(Post-Shock) + P(Risk-Off)` against (a) the true Risk-Off and Post-Shock
regimes and (b) the HMM stress labels they were trained on. Balanced
accuracy and NLL relative to a class-prior baseline are also reported.

Arms: engineered features (Day 7's input), the raw 252-return window,
each pretrained embedding, the **same architecture untrained** (control),
the TimesFM-style quantile forecasts as features and a TimesFM-style
model pretrained without the regime-switching family (control).

## Did pretraining learn anything?

Some structure, in some families. Gains over a static baseline (best fixed
token distribution for cross-entropy; pooled empirical quantiles for
pinball loss) on freshly drawn held-out series:

| family | pinball gain | cross-entropy gain (nats) |
|---|---|---|
| trend | 33.9% | 0.47 |
| level shifts | 27.3% | 1.02 |
| seasonal | 25.1% | 0.18 |
| AR, random walk, GARCH, regime switching, Student-t, jumps | 0.6% to 3.1% | -0.03 to 0.08 |
| **project's own Nifty returns** | **1.1%** | **-0.001** |

On the project's returns the Chronos-style cross-entropy equals the static
baseline (4.427 vs 4.426) and the TimesFM-style median forecast is no closer
to the target than a context mean (ratio 1.00). The 10-90% interval covers
about 79% of targets, close to its nominal 80%. **These models are not
useful forecasters of return-like data.**

## Results

Mean over training sizes n >= 200, where the curves are close to flat:

| input | AUROC vs true stress | AUROC vs HMM stress | balanced acc. | NLL minus class prior |
|---|---|---|---|---|
| engineered features | 0.866 | 0.945 | 0.299 | -0.064 |
| Chronos-style embedding | 0.875 | 0.720 | 0.200 | 0.320 |
| TimesFM-style embedding | 0.876 | 0.676 | 0.221 | 0.292 |
| TimesFM-style, no regime-switch corpus | 0.876 | 0.677 | 0.215 | 0.279 |
| Chronos-style, untrained | 0.474 | 0.335 | 0.199 | 0.714 |
| TimesFM-style, untrained | 0.491 | 0.465 | 0.199 | 0.385 |
| raw 252-return window | 0.454 | 0.522 | 0.199 | 0.453 |
| TimesFM-style quantile forecasts | 0.803 | 0.630 | 0.199 | 0.052 |

- **Pretraining matters for these architectures.** The pretrained
  embeddings rank true stress at about 0.875. The same architectures
  untrained are at chance, and so is the raw window.
- **The corpus overlap does not explain it.** The pretraining corpus
  includes a Markov regime-switching family, the same kind of process as
  the target panel. Retraining the TimesFM-style model without it gives
  the same downstream AUROC (0.876) and nearly identical held-out pretext
  losses, including on the regime-switching family itself. This control
  was not run for the Chronos-style model (about 14 minutes of compute), so
  for that model the confound is untested.
- **There is no sample-efficiency advantage over training from scratch.**
  The embedding curves are nearly flat from n=50 (about 0.85 to 0.88), but
  engineered features are higher at n=50 (0.960 vs 0.883, 0.863 and 0.847)
  and n=100 (0.891 vs 0.853, 0.838 and 0.825), lower at n=200 (0.823 vs 0.876,
  0.861 and 0.841) and level at large n. The curves cross and the
  engineered arm is noisy. The task's expected result, hybrid ahead of
  scratch, is not supported.
- **Against the HMM labels the heads were trained on, the embeddings are
  much worse** (0.68 to 0.72 vs 0.945). The models normalise scale away, and
  the HMM states differ in volatility level.
- **The heads on embeddings are poorly calibrated.** NLL is about 0.3 nats
  worse than a class prior and balanced accuracy is 0.20 to 0.22, i.e. they
  essentially always predict Risk-On. Engineered features are better than
  the prior. The cause was not investigated. Candidates are prior shift
  between training and test windows, head overconfidence and embedding
  drift; none was tested.

## A trivial baseline beats the embeddings

Computed on the same test window with no training:

| statistic | AUROC vs true stress |
|---|---|
| mean absolute return, last 21 days / 252 days (scale-free) | 0.961 |
| std, last 21 days / std, 252 days (scale-free) | 0.928 |
| std, last 63 days / std, 252 days (scale-free) | 0.896 |
| std, last 21 days (has scale) | 0.966 |

A one-line scale-free volatility ratio ranks stress better than the
pretrained embeddings (0.875). The embeddings carry regime information, but
nothing a simple statistic does not already provide. The scale-free rows
are the fair comparison because the embeddings cannot see volatility level.

## Limits

- 3 training subsets per size. Error bars are one standard deviation over 3.
- The test window contains two true stress episodes (62 and 84 days), so
  every AUROC rests on two events, not 146 days.
- One pretraining run per model.
- No embargo between the training and test windows (as on Day 7). Test
  windows overlap the end of the training period in their inputs.
- Heads trained on HMM labels, which carry the Day 4 to 7 limits (88.6% one
  class, decoded with future information).
- Head uncertainty (the Bayesian part) was not analysed here.
- The regime-switching-family control covers the TimesFM-style model only.
- The corpus is nine generic families. Real corpora are far broader.

## Recommendation

Do not treat these embeddings as an improvement on engineered features.
If foundation embeddings are pursued, test a real pretrained model on a
machine with internet access. The code here (`HybridRegimeModel`, the
sample-efficiency script and the controls) is reusable for that. Whatever
model is used, keep three baselines beside it: the untrained architecture,
the raw window and a simple scale-free volatility ratio. Without them, an
AUROC near 0.87 reads as a success that a one-line statistic exceeds.
