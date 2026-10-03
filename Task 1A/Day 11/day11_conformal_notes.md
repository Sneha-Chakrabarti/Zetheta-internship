# Day 11: conformal prediction and calibration notes

Scope: `src/models/conformal/` (split_conformal, mondrian, aci,
diagnostics), `scripts/run_day11_conformal.py`, notebook
`notebooks/10_conformal_calibration.ipynb`. All computation is plain
numpy/scipy; this ran in the main environment, no PyMC or TensorFlow
needed.

## What was built

- Split-conformal classification and Adaptive Prediction Sets (A6.2,
  A6.3), matching the spec closely, adapted to take probability arrays
  directly rather than a model object.
- Mondrian (class-conditional) conformal as this project's required
  distribution-shift-robust method (A6.5), chosen specifically because
  it targets a problem already established since Day 7: the HMM
  pseudo-labels are about 89% one class.
- Adaptive Conformal Inference (A6.5), matching the spec's update rule
  exactly, plus a wrapper that runs it over a real probability/label
  stream.
- Reliability diagrams and Expected Calibration Error (A6.6), matching
  the spec exactly.

Applied to five models (MC Dropout, variational BNN, deep ensemble from
Day 7, the foundation head from Day 9, and a simple equal-weight
ensemble of the four) on the 702-day held-out test window, all
genuinely trained on strictly prior data. Calibration uses the first 120
days; evaluation uses the remaining 582. Scored against both the HMM
pseudo-labels (what the supervised models were trained toward) and the
synthetic panel's true regime.

## The exchangeability tension, stated once

Conformal prediction's marginal coverage guarantee assumes
exchangeability between calibration and test data. Section A6.5 says
plainly that financial time series violate this. Section 2b below is
not a hypothetical illustration of that warning: it is a measured,
real instance of it inside this project's own evaluation window.

## Split-conformal and APS: verified, not just implemented

Over 50 independent calibration/test splits of a toy model: mean
coverage 0.8996 (target >= 0.9) for split-conformal, 1.0 for APS. The
guarantee is a statement about coverage averaged over the randomness of
the calibration draw; checked that way, not from one realisation (a
single early trial came in at 0.8915, which on its own could have read
as a failure).

**Split-conformal produces genuinely empty prediction sets about 10% of
the time** in that same 50-trial check. This is not a bug: Section
A6.3's own text warns plain split-conformal "can produce trivial
prediction sets (all classes or no classes) when the underlying model is
poorly calibrated", and this confirms it directly rather than only
citing it. An empty set is technically consistent with the marginal
guarantee (coverage holds on average; other points get wider sets to
compensate) but is useless to an actual decision-maker - exactly A6.3's
own stated motivation for APS, which forces the top class into every
set by construction.

## Mondrian conformal: coverage from 0.53 to 1.00, and why

On the real evaluation window, Mondrian's overall coverage ranges from
0.529 (foundation head, HMM labels) to 1.000 (several cases) - not the
clean, uniform fix the method's design promises at first read. Two
separate causes, both checked rather than assumed:

**1. Tiny per-class calibration samples, as expected under severe
imbalance.** With 120 calibration days split five ways and 89% of the
data in one class, some classes get almost no calibration data:
`Late-Cycle` had zero calibration days in the main split (defaults to
always-included, trivial coverage of 1.0, correct behaviour but not
informative), `Transitional` had exactly one (coverage 0.385 for one
model, essentially meaningless with n=1).

**2. A genuine, measured temporal shift in model calibration quality -
the more important finding.** `Risk-On`, the majority class, had 83
calibration days: not a small sample. Yet the foundation head's
Mondrian coverage for it was only 0.471. Traced directly: during
calibration the foundation head was near-perfectly confident about
Risk-On when right (mean nonconformity score 0.0076, 90th percentile
0.017); during evaluation, for the SAME class, it was far less reliable
(mean 0.163, 90th percentile 0.661, max 0.969). This is not sampling
noise from a small calibration window. It is a real difference between
the calibration period and the evaluation period in how trustworthy this
one model's Risk-On probability actually is - precisely the
exchangeability violation Section A6.5 warns about, caught in this
project's own data rather than only described abstractly. Mondrian's
per-class design fixes the cross-class pooling problem; it does nothing
for a calibration period that is not representative of the evaluation
period, because the per-class threshold still assumes calibration and
evaluation scores for that class are exchangeable with each other.

## Adaptive Conformal Inference: a real limitation, found by testing it

**The mechanism**: `q_hat` is a quantile of the FIXED calibration score
distribution at every step; alpha_t only selects WHICH quantile to use.
`q_hat` is therefore bounded above by `max(calibration scores)`
regardless of how low alpha_t goes.

**Tested under a deliberately severe synthetic shift** (calibration from
a confident, well-separated model; test stream from a much weaker
true-class signal): the widest achievable inclusion threshold from
calibration was 0.605, but 94.9% of the shifted test stream had every
class's probability below that threshold. Realised coverage collapsed to
0.027 against a 0.9 target, even as alpha_t saturated near its floor
(0.001) trying to compensate. ACI adapts the coverage TARGET; it cannot
widen what the original calibration scores support, and a production
system facing a shift this severe would need to refresh the calibration
set itself (e.g. a sliding window), not just let alpha_t keep shrinking.

**On the real evaluation window**, the shift is milder: coverage stays
close to or above 0.9 for every model and label set. `alpha_t`'s drift
from its 0.1 starting point varies a lot by model, though - modest for
MC Dropout, the variational BNN and the deep ensemble against the true
regime (0.02-0.06), but substantial for the foundation head and the
equal-weight ensemble (0.32-0.61, moving the opposite direction from the
synthetic stress case: these two became easier to cover over time, not
harder). Coverage holding up even as alpha_t moves this much is a real
result, not an artefact of alpha_t barely changing.

## Reliability diagrams and ECE: the label source changes the number a lot

Every model's ECE against the true synthetic regime (0.30-0.39) is
roughly six to ten times its ECE against the HMM pseudo-labels (0.03-
0.07) it was actually trained on. Unsurprising once stated plainly -
these are confidence scores shaped by training toward one label set, not
toward ground truth the model never saw - but worth stating explicitly:
a calibration number is only meaningful relative to the label set it was
computed against, and both should be reported side by side rather than
one chosen and presented as "the" calibration quality.

## Rolling 252-day coverage: the headline value of conformal wrapping, demonstrated

"Base" = trusting only the top-1 prediction, no calibration. Computed
for the equal-weight ensemble against the true regime, over the 582-day
evaluation period, 252-day rolling window:

| method | min | mean | max |
|---|---|---|---|
| base (top-1 only) | 0.369 | 0.779 | 1.000 |
| split-conformal | 0.952 | 0.998 | 1.000 |
| APS | 0.905 | 0.974 | 1.000 |
| ACI | 0.921 | 0.997 | 1.000 |

Base coverage swings from 0.37 to 1.00 - simply accuracy over time, and
accuracy is not stable on this window (it craters during the stress
period Day 7 already identified in the same months). All three conformal
methods stay within a 0.90-1.00 band across the same window. This is the
concrete, measured version of conformal prediction's core promise: stay
close to the target coverage even when the underlying model's accuracy
is not stable. The methods are over-covering here (comfortably above
0.9 rather than hugging it), which is not a guarantee violation but does
mean the sets likely run larger than strictly necessary - the
efficiency question Section "Mondrian conformal" already showed can be
severe for a poorly-calibrated model.

## Limits

- Calibration uses one fixed 120-day window at the start of the test
  period, not a rolling or re-fit calibration set; Section "Adaptive
  Conformal Inference" shows this matters when calibration and
  evaluation drift apart.
- Mondrian's small-sample classes (`Late-Cycle`, `Transitional`) were
  not given a larger calibration allocation or a fallback pooling
  strategy; their reported coverage is close to uninformative as
  computed.
- CQR (A6.4, for continuous return forecasts rather than discrete
  regime labels) was not implemented; this day's regime models are all
  classifiers.
- The rolling-coverage comparison was run for one model (the
  equal-weight ensemble) against one label set in detail; the full
  calibration/test-split comparison (Section "Mondrian conformal" and
  the main results table) covers all five models and both label sets.

## Recommendation

A production system should not rely on a single fixed calibration
window the way this analysis necessarily did for a clean illustration.
Section "A genuine, measured temporal shift" is the concrete argument
for a rolling or periodically-refreshed calibration set, not just a
theoretical concern: it is the direct, measured cause of Mondrian's
worst coverage failure in this project's own data. ACI with a sliding
calibration window, or periodic recalibration triggered by Day 10's
reconciliation diagnostic, are both more defensible than the fixed
single-split approach used here for a first, verifiable pass.
