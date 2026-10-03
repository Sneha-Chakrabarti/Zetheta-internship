# Day 7: Bayesian deep learning notes

Scope: `src/models/bdl/` (models, evaluation, data helpers), scripts
`scripts/run_day7_*.py`, notebook `notebooks/06_bayesian_deep_learning.ipynb`.
Training ran in the scripts, one stage per tool call. The notebook loads
their saved outputs.

## What was built

- MC Dropout classifier (Section A4.2), variational BNN with
  `DenseFlipout` layers (A4.3) and a deep ensemble with M=10 (A4.4).
- Time-series cross-validation (A4.6) on 5 expanding-window folds.
- Epistemic and aleatoric decomposition, two ways: the spec's
  (cross-member std, mean `p(1-p)`) and an entropy-based one (total,
  aleatoric, epistemic as mutual information).
- SHAP attributions for five dates chosen by stated rules.

Inputs are Day 3's 31 engineered features. Labels are Day 4's HMM
decode, as A4.6 specifies.

## Environment

TensorFlow 2.21, TensorFlow Probability 0.25 and `tf-keras` live in
`.venv-bayesian`, next to PyMC. TFP needs the `tf-keras` shim because
Keras 3 broke the API it expects; `TF_USE_LEGACY_KERAS=1` is set in each
module.

Installing TensorFlow into the main environment failed and left it
damaged (disk full, a partial numpy upgrade, corrupted packages). Torch
no longer imports there. Details and the decision not to repair it are in
`PROJECT_PLAN.md`, open decision 9. Day 3's GCN tests are skipped with an
explicit reason as a result. Nothing in Day 7 uses torch.

## Limits of the labels

These apply to every number below.

- **88.6% of days carry one label** (Risk-On). Late-Cycle has 37 days in
  the whole sample. This is Day 4's weak identifiability inherited by
  construction.
- **Labels use future information.** The HMM is decoded over the whole
  series and was fit on the test periods too. A classifier that only sees
  features up to day *t* cannot reproduce that. This is a property of the
  spec's design and was not changed.
- **The rare labels are clustered in time.** Folds 0 and 1 have no
  Post-Shock or Risk-Off days in their test windows. Per-fold rare-class
  scores rest on 3 to 47 days.
- The scaler is fit on each fold's training slice only. My first
  data-loading helper standardised over the whole series, which leaks
  future statistics into every fold. That option is kept for exploratory
  use and documented as unsuitable for cross-validation.

## Evaluation design

Accuracy alone is misleading here. Always predicting the majority class
scores about 0.89. Every result is reported against two baselines:

- the majority-class predictor (accuracy and balanced accuracy);
- a class-prior predictor that outputs training-set class frequencies
  (negative log-likelihood).

Balanced accuracy, macro-F1 and per-class recall are reported beside
accuracy.

## Cross-validation results

Reduced from the spec's M=8 and 50 epochs to M=5 and 30 epochs so five
folds fit the per-call budget. Means over 5 folds:

| model | accuracy | balanced acc. | NLL | folds with NLL below prior |
|---|---|---|---|---|
| majority / class-prior baseline | 0.893 | 0.253 | 0.502 | n/a |
| MC Dropout | 0.898 | 0.288 | 0.421 | 4 of 5 |
| MC Dropout, class-weighted | 0.813 | 0.366 | 0.543 | 2 of 5 |
| Variational BNN | 0.896 | 0.299 | 0.352 | 5 of 5 |
| Deep ensemble, 30 epochs (spec recipe) | 0.893 | 0.299 | 0.644 | 2 of 5 |
| Deep ensemble, dropout-active members | 0.897 | 0.285 | 0.455 | 4 of 5 |
| Deep ensemble, early stopping | 0.892 | 0.272 | 0.431 | 4 of 5 |

What this supports:

- **No model beats the majority baseline on accuracy.** The three
  standard models sit within about 0.02 of it in every fold.
- **The variational BNN is the most consistent on NLL**, below the class
  prior in every fold. The features therefore carry some information
  about the labels beyond their base rate. That is unsurprising, since
  the labels were decoded from returns and the features include returns
  and volatility.
- **The three architectures are not separable.** Mean differences are
  small next to the fold-to-fold spread.
- **Class weights** lift balanced accuracy above the majority baseline
  in all five folds. They cost 5 to 16 accuracy points in every fold
  and leave mean NLL worse than the prior. Against the unweighted model
  the balanced-accuracy gain held in 3 of 5 folds in the final run and
  5 of 5 in an earlier run, because TensorFlow's CPU kernels are not
  bit-deterministic. That comparison is soft.

## Two methodological problems found and fixed

**1. Ensemble members were not deterministic.** The first ensemble
used the MC Dropout builder, whose dropout is permanently on, so each
member's `predict` was itself random. Cross-member spread then mixed
initialisation disagreement with dropout noise, which is not the
quantity A4.4 describes. Fixed with
`build_deterministic_regime_classifier` (same 128/64/32 layers, no
dropout). The dropout-member variant is kept as a labelled row.

**2. The spec's training recipe overfits here.** The deterministic
ensemble at 30 epochs with no early stopping was worse than the
class-prior baseline on NLL in 3 of 5 folds. Section A4.4 calls deep
ensembles among the most reliable Bayesian methods, so this needed an
explanation. On fold 3, training length was varied:

| epochs | NLL | accuracy | mean confidence on wrong days |
|---|---|---|---|
| 3 | 0.598 | 0.818 | 0.665 |
| 10 | 0.773 | 0.818 | 0.722 |
| 30 | 1.453 | 0.807 | 0.826 |

The class-prior NLL on that fold is 0.815. NLL rises with training
length while accuracy is flat, and the model becomes more confident in
its wrong calls. That is overconfidence under distribution shift from an
unregularised recipe. It is not a property of ensembles. The 30-epoch NLL
varied between 1.33 and 1.45 across three runs I made. The direction did
not.

The fix is early stopping on a chronological validation slice (the last
20% of each training window), which never touches the test fold. The
early-stopped ensemble's mean NLL is 0.431 against 0.644. The epoch
counts in the table above were not chosen by looking at test data.

## Final chronological run

Train on the first 2806 days, test on the last 702 (2022-10-20 to
2025-06-27). The ensemble uses M=10 with early stopping.

| model | accuracy | balanced acc. | NLL | recall Post-Shock | recall Risk-Off |
|---|---|---|---|---|---|
| majority / class-prior baseline | 0.856 | 0.200 | 0.595 | n/a | n/a |
| MC Dropout | 0.876 | 0.284 | 0.574 | 0.24 | 0.18 |
| Variational BNN | 0.876 | 0.305 | 0.461 | 0.46 | 0.08 |
| Deep ensemble (M=10) | 0.873 | 0.268 | 0.494 | 0.32 | 0.03 |

All three get 0.00 recall on Late-Cycle (7 days) and Transitional (14
days). None can be called a working five-regime detector.

## Uncertainty

**Decomposition.** The spec's std and `p(1-p)` measures rank days almost
identically to the entropy-based ones (Spearman 0.998 for epistemic,
1.000 for aleatoric on the ensemble). The spec's formulas are a sound
proxy.

**Epistemic shares are not comparable across methods.** They are 28%
(MC Dropout), 24% (variational BNN) and 8% (ensemble). Dropout noise,
variational weight noise and cross-initialisation disagreement are
different sources of randomness. The ensemble's small share reflects
members trained on the same data agreeing with each other. It does not
show the ensemble is more certain about the world.

**Aleatoric dominates** on every true label. On Post-Shock and Risk-Off
days the ensemble's aleatoric entropy is about 0.9 and its epistemic
entropy about 0.1.

**Is it informative?** AUROC of total entropy for detecting a wrong call
is 0.84 to 0.86, with bootstrap intervals of roughly 0.79 to 0.91 that
overlap across models. Mean entropy is about six times higher on wrong
days than on correct ones. The limit that matters: the models call
Risk-On on 36 to 52 of the 80 Post-Shock and Risk-Off days, and about
one in five of those misses (7 to 11 days) comes with low entropy.
Uncertainty flags many misses. It does not flag all of them.

## Check against the true regimes (synthetic data only)

The generator's regime path was never shown to any model, so it gives a
fairer test than the HMM labels. Two threshold-free signals were compared
as indicators of the true Risk-Off and Post-Shock regimes (146 of 702
test days): predictive entropy and the stress-class probability mass
`P(Post-Shock) + P(Risk-Off)`.

- AUROCs run from 0.77 to 0.93, highest for trailing 21-day means (past
  data only).
- **I expected entropy to be the better indicator. It is not.** The
  probability mass scores as well or slightly better, and the intervals
  overlap heavily. Neither is shown to beat the other.
- **This qualifies the recall results.** The argmax label recovered only
  3 to 46% of stress days. The probabilities behind it rank days by
  stress much better. A rare class seldom wins the argmax against an 89%
  base-rate class, but its probability is systematically higher on stress
  days. For rare regimes the argmax is the wrong operating point.
- **This is thin evidence.** The true stress regime in the test window
  is exactly two contiguous episodes (62 and 84 days). A 21-day block
  bootstrap cannot fully represent how little independent information
  that is, so the intervals are probably optimistic.
- Of the 49 ensemble high-uncertainty days that lie more than 30 days
  from any HMM stress label, 31 were true Post-Shock or Transitional and
  18 were true Risk-On. Most of those days are not false alarms, though
  some are.

## SHAP

KernelExplainer on the ensemble's mean probability (a deterministic
function; MC Dropout and the variational BNN are stochastic per call),
30-point k-means background, 400 samples per date. Additivity holds to
6e-17.

Dates were chosen by rule, in `scripts/run_day7_shap.py`: the most
confident correct Risk-On day, the correct Risk-Off day with the highest
probability, the most confident missed Risk-Off day, the highest
epistemic-uncertainty day and the correct Post-Shock day with the
highest probability.

- INR realised volatility, credit spread and equity volatility dominate
  (`usdinr_vol_21d` and `credit_spread` at about 0.07 mean absolute SHAP,
  the equity volatility features at 0.03 to 0.04 except `vol_63d` at
  0.026). In this synthetic panel those
  variables are regime-dependent by construction, so this shows the
  network found the regime-linked inputs. It is not evidence about which
  drivers matter in Indian markets. SHAP also explains the HMM label, not
  reality.
- **The missed Risk-Off day (2023-02-22)** has P(Risk-Off) of about 1e-4
  and total entropy of 0.011. No feature pushes toward stress and the
  model is confidently wrong. Nothing in the uncertainty output could
  have warned about it.
- Five dates illustrate the method. They are not a global importance
  ranking.

## Limitations

- Cross-validation used M=5 and 30 epochs, not the spec's M=8 and 50.
- One training seed per model. Run-to-run variation in TensorFlow's CPU
  kernels changed some close comparisons.
- Hyperparameters (layer sizes, dropout rate, KL weight) are the spec's
  and were not tuned.
- No calibration curve or expected calibration error was computed. NLL is
  the only calibration-sensitive metric used.
- The labels are in-sample HMM decodes on the full series. Refitting the
  HMM inside each fold would remove the leakage but was out of scope.
- The synthetic ground-truth check rests on two stress episodes.

## Recommendation for later days

Day 9 (ensembling) should consume these models' probabilities, not their
argmax labels. Better labels would help more than a better classifier:
either an HMM refit per fold, or the multivariate regime model from Day 6
given more compute. Rerunning on real data, once available, is the test
that matters.
