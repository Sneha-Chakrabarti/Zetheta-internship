# Executive Summary: Bayesian Regime Detection Engine

Regime Lab, Zetheta Algorithms internship, Task 1A. Days 1-14 of a
15-day plan. This summary synthesises the project's honest state: what
works, what does not yet, and what the evidence actually supports,
rather than a sales pitch for the engine.

## What was built

A regime-detection stack for Indian equity markets spanning: synthetic
data generation with a known ground truth (Days 1-2), feature
engineering across 31 tabular features plus topological and graph
features (Day 3), a frequentist HMM and a from-scratch-marginalised
Bayesian HMM (Days 4-5), a multivariate regime-switching VAR in PyMC
(Day 6), three Bayesian deep learning architectures (MC Dropout,
variational BNN, deep ensemble, Day 7), synthetic-pretrained foundation
model re-implementations (Day 8), a Bayesian-model-averaging and
stacking ensemble layer with PSIS-LOO model selection (Day 9), a
bootstrap particle filter, Bayesian online changepoint detection, and
streaming conjugate updates (Day 10), a conformal prediction and
calibration layer (split-conformal, APS, Mondrian, ACI, Day 11), four
researched and empirically cross-checked historical case studies (Day
12), a regime-conditioned Monte Carlo engine with an allocation overlay
backtest and Investment Committee artefact generator (Day 13), and
cross-language (Python/R) validation plus end-to-end and online-loop
pipeline checks (Day 14).

Every component is backed by passing tests (156 at last count, 7
skipped for a documented, specific environment limitation - a torch
installation collision from Day 7, unrelated to model correctness) and
an executed notebook with zero errors, every reported number checked
against actual printed output before being written into documentation.

## What the evidence actually supports

**Regime identification from returns alone is weak, and this finding
recurs independently throughout the project, not just once.** Day 4's
BIC prefers 3 states over the briefed 5; the ground-truth match rate
for the best frequentist fit is around 40% against an approximately 20%
random baseline. Day 9's cross-language HMM check (Day 14) finds
Python and R converging to different local optima on identical data,
confirming the same underlying multimodality from a second, independent
angle. Day 13's attempt to fit a point-in-time-valid model for a
backtest found that even 3 states becomes unstable on 8 years of data;
2 states is what that window actually supports. This is not a single
finding dressed up three ways - it is the same property of the data
surfacing in three genuinely different parts of the pipeline.

**The ensemble does not reliably beat its best individual member.**
Checked via real time-series cross-validation (Day 9), this held in 1
of 4 tested cases; against the synthetic panel's true regime, naive
equal-weighting beat both Bayesian model averaging and stacking. The
mechanism is not mysterious: too little data for the number of
combination weights being fit.

**BOCPD, this project's one genuinely online changepoint detector,
correctly found three real historical events in real data on its own
terms (Day 10: August 2015, November 2016, September 2019), but missed
two of the three real crises it was tested against in Day 12's case
study work** (IL&FS 2018, COVID 2020 at its validated threshold) for a
clear, mechanistic reason in each case: IL&FS was a gradual two-month
decline with no single sharp day to detect, and COVID's calibration
window happened to run directly through the crisis onset itself,
contaminating the detector's own notion of "normal."

**Conformal calibration carries a real, demonstrated gap between its
textbook guarantee and this project's own data.** The exchangeability
assumption split-conformal methods rely on is violated by financial
time series in theory (Section A6.5 of the task brief says so
explicitly); Day 11 found a concrete, measured instance of this in this
project's own evaluation window, not just a theoretical caveat: one
model's calibration quality measurably drifted between the calibration
period and the evaluation period for its own majority class.

**Post-Shock, a recovery-rally regime in this project's own synthetic
taxonomy, is easy to mislabel as something to de-risk from if a tilt
rule is built on regime names rather than regime statistics** (Day 13)
- and even a statistically-grounded tilt rule is only as good as the
underlying regime identification, which Day 4 already showed is
unreliable for this state specifically.

## What this means for deployment

The individual components are sound and well-tested; the SYSTEM-LEVEL
claims a naive reading might make (the ensemble is better than any one
model; BOCPD will catch the next crisis; the calibration holds) are
each qualified by specific, demonstrated evidence above. A production
deployment should:

1. Treat single-returns-series regime identification as inherently
   noisy and avoid over-interpreting any one day's hard regime call.
2. Prefer the multivariate RS-VAR's architecture for genuinely
   cross-asset stress (IL&FS, the 2013 Taper Tantrum, COVID), while
   recognising its current posterior sample size is a real, acknowledged
   limitation (Day 6, Day 9), not a solved problem.
3. Recalibrate BOCPD's detection threshold per input series with an
   explicit false-positive budget, rather than reusing a threshold
   validated on one series for another (Day 12's central finding).
4. Build the event-calendar-aware conviction dampener Section C5.3
   describes, which does not yet exist in this project and which Day
   12's election case study shows a reactive detector cannot
   substitute for.
5. Monitor regime-identification quality itself (e.g. via Day 9's own
   per-member log-likelihood tracking), not just the tilt rule applied
   downstream, since Day 13 showed a correct rule fed a wrong estimate
   is no better than a wrong rule.

## What remains

Day 15 (final submission and handover) and the broader Section D3
deliverables checklist (a 40-page formal report, a full R codebase
beyond the three cross-checks built in Day 14, an 18-slide presentation,
and a recorded demonstration video) extend beyond this 14-day working
scope. The day-by-day notebooks and `docs/dayN_*.md` files in this
repository are the working record each of those larger deliverables
would draw from.
