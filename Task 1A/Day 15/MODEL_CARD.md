# Model Card: Regime Lab Bayesian Regime Detection Engine

Section D3 Deliverable 5. Documents every member model built across
Days 4-11: inputs, priors where applicable, version/fit provenance, and
consolidated calibration and diagnostic evidence. Numbers below are
restated from each day's own notes, not recomputed here; see the cited
file for full derivation and verification.

## Member models

| Model | Day | Inputs | Priors | States | Fit provenance |
|---|---|---|---|---|---|
| Frequentist HMM | 4 | Nifty 50 daily log returns | None (MLE via EM) | 5 (BIC prefers 3) | `fit_regime_hmm_robust`, 20-restart multi-start, seed selected by best valid log-likelihood |
| Bayesian HMM | 5 | Nifty 50 daily log returns | Dirichlet(1) per transition row; Normal(0, 0.02) per-state mean; HalfNormal(0.02) per-state vol | 5 | PyMC NUTS, 4 chains x 500 draws, 1-year window |
| Bayesian RS-VAR | 6 | 6-D: returns, vol, breadth proxy, INR return, gilt yield change, FII flow proxy | LKJCholeskyCov per regime; Dirichlet(1) transitions | 3 (compute-limited) | PyMC NUTS, 4 chains x 80 draws |
| MC Dropout | 7 | 31 engineered tabular features | N/A (frequentist network, dropout at inference) | 5-class classifier | Trained on first 2806 days, dropout p=0.3, 200 MC passes |
| Variational BNN | 7 | 31 engineered tabular features | Normal(0, 1) weight priors (mean-field VI) | 5-class classifier | Trained on first 2806 days |
| Deep Ensemble | 7 | 31 engineered tabular features | N/A (M=10 independently-initialised networks) | 5-class classifier | Trained on first 2806 days |
| Foundation head (TimesFM-style) | 8 | Synthetic-pretrained patch embeddings (64-D) | N/A | 5-class classifier on frozen embeddings | Pretrained on synthetic corpus (HuggingFace unreachable; real Chronos/TimesFM checkpoints not used - stated explicitly in Day 8's notes, not implied to be the real thing) |
| BMA / stacking ensemble | 9 | All of the above, collapsed to 3-way (Risk-On/Transitional/Risk-Off) | N/A | 3-way | Weights fit per fold, 4-fold TimeSeriesSplit |
| Particle filter | 10 | Nifty 50 daily log returns, frozen HMM parameters | N/A (bootstrap filter) | Matches host HMM's K | 5,000 particles, systematic resampling at ESS < N/2 |
| BOCPD | 10 | Nifty 50 daily log returns, or VIX log-changes (Day 12) | Normal-Gamma, calibrated per-series (see Day 10 for why the spec's literal defaults fail) | N/A (changepoint detector) | Hazard 1/100 |

## Calibration evidence (Day 11, reliability diagrams and ECE)

| Model | ECE vs. HMM training labels | ECE vs. true synthetic regime |
|---|---|---|
| MC Dropout | ~0.03-0.07 (see Day 11 full table) | ~0.30-0.39 |
| Variational BNN | ~0.03-0.07 | ~0.30-0.39 |
| Deep Ensemble | ~0.03-0.07 | ~0.30-0.39 |
| Foundation head | ~0.03-0.07 | ~0.30-0.39 |
| Equal-weight ensemble | ~0.03-0.07 | ~0.30-0.39 |

Every model's ECE against the true synthetic regime is six to ten times
its ECE against the HMM labels it was actually trained on (Day 11's
central calibration finding) - a reminder that a calibration number is
only meaningful relative to the label set it was computed against; see
`docs/day11_conformal_notes.md` for exact per-model figures.

## Rolling conformal coverage (Day 11)

252-day rolling window, equal-weight ensemble, true regime labels:

| Method | min coverage | mean coverage | max coverage |
|---|---|---|---|
| Base (uncalibrated top-1) | 0.369 | 0.779 | 1.000 |
| Split-conformal | 0.952 | 0.998 | 1.000 |
| APS | 0.905 | 0.974 | 1.000 |
| ACI (online) | 0.921 | 0.997 | 1.000 |

## MCMC diagnostics (Days 5, 6)

| Model | R-hat | Divergences | Notes |
|---|---|---|---|
| Bayesian HMM (Day 5) | ~1.00 | 2/2000 | Clean |
| Bayesian RS-VAR (Day 6) | 1.00-1.18 after post-hoc relabelling (1.44-2.43 before - label-switching, not non-convergence) | 0 | PSIS-LOO (Day 9) flags 19/252 held-out points as unreliable (Pareto k > 0.6); p_loo (~205) close to the raw parameter count (~197), indicating an under-sampled posterior, not a diagnostic failure of the LOO computation itself |

## Online/batch reconciliation evidence (Day 10)

Filtered (online) vs. smoothed (batch) posterior TV distance, bucketed
by days since the most recent BOCPD-detected changepoint: did NOT show
the expected monotonic "staleness grows with time" pattern (peaked at
40-80 days, not at the far end) - reported as found, not forced into
the expected story; see `docs/day10_sequential_notes.md` for the full
investigation and the explanation (BOCPD events and the HMM's own
belief-revision frequency are governed by different processes).

## Cross-language validation (Day 14)

Python and R implementations of the HMM (Baum-Welch EM), split-
conformal classifier, and Monte Carlo VaR/CVaR engine were cross-
checked on identical data. Summary: the HMM converges to different
local optima in each language (confirming Day 4's weak-identifiability
finding independently); split-conformal matches exactly after fixing a
quantile-interpolation discrepancy; VaR/CVaR agree within expected
Monte Carlo sampling noise. Full detail: `docs/day14_validation_notes.md`.

## Known limitations, stated once here rather than scattered

- Foundation model heads use synthetic-pretrained re-implementations,
  not real Chronos/TimesFM checkpoints (HuggingFace unreachable from
  this environment, Day 8).
- The GCN sector-graph feature (Day 3) cannot be computed: torch is
  broken in the main environment from a Day 7 TensorFlow installation
  collision (`PROJECT_PLAN.md` open decision 9). 3 tests skip for this
  reason, verified to be exactly 3, not conflated with other skips
  (Day 14's peer review).
- The ensemble does not reliably beat its best individual member (Day
  9): true in 1 of 4 tested cases.
- BOCPD requires per-series threshold recalibration; a threshold
  validated on one series does not transfer to another (Day 10, Day
  12).
