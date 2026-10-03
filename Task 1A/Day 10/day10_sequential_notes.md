# Day 10: sequential and online inference notes

Scope: `src/models/sequential/` (particle_filter, bocpd, streaming),
scripts `scripts/run_day10_*.py`, notebook
`notebooks/09_sequential_inference.ipynb`. All computation here is plain
numpy/scipy; nothing needed PyMC or TensorFlow, so this ran in the main
environment on the standard kernel, unlike Days 5-9.

## Particle filter (Section A9.1)

Matches the spec's `RegimeParticleFilter` with one implementation
difference: the propagation step is vectorised. The spec's own code
draws each particle's next state with a Python-level `rng.choice` call
per particle; at N=5000 particles over the ~3700-day series this is
roughly 18.5 million individual calls, checked to be impractically slow
here, not assumed. The vectorised version (cumulative-sum each
particle's own transition row, one batched draw against it) draws from
the identical per-particle distribution, confirmed against the known
transition rows directly before trusting it.

**Validated against the exact forward algorithm**, not just run and
inspected for plausibility: on the full 15-year series with Day 4's
robustly-fit HMM, a 5000-particle filter tracks `filtered_state_probs`
(Day 9) with mean total-variation distance 0.0095, median 0.0062, 99.2%
hard-label agreement. The filter is behaving as a correct Monte Carlo
approximation.

## BOCPD (Section A9.2): two real bugs in the spec's own example, found and fixed

**1. The default priors are scaled for the wrong kind of data.**
`mu0=0, kappa0=1, alpha0=1, beta0=1` imply a prior variance of 1.0 -
about four orders of magnitude above a typical daily return series'
variance (~1e-4). Traced directly on a toy three-block series with an
obvious 5-percentage-point mean shift: the dominant (longest-run-length)
hypothesis's predictive scale stayed near 0.14 throughout (vs. the
series' true 0.01 standard deviation), so a 0.05 shift read as under
half a (wrongly inflated) standard deviation and registered as
unremarkable. `calibrate_normal_gamma_prior` sets the prior's implied
variance from the data's own short-run scale (first 60 points) instead,
with `alpha0=5.0` (not the spec's 1.0) so the predictive scale does not
swing wildly while kappa/alpha are still small.

**2. `R[0, t]`, the obvious reading of "probability a changepoint just
occurred", carries zero information.** Proved, not just observed: for
any nonnegative column `R[:, t]` and any nonnegative predictive
likelihoods, splitting the total mass `S` into `hazard * S` (the
changepoint branch) and `(1 - hazard) * S` (the growth branch) and
normalising recovers exactly `hazard` at the first coordinate, for
*any* `S`. Verified both algebraically and against five random inputs
with different data. This means `R[0, t]` is a constant of the
recursion's structure, not a function of the data at all - using it as
the detection signal (the natural first reading of the output) silently
produces a tool that can never fire, which is exactly what the
uncalibrated-prior case looked like before the real cause (prior scale)
was separated out from this second, independent problem.

The corrected signals: `map_run_length` (the posterior mode run length -
a real changepoint shows up as a sharp drop followed by a steady climb)
and `short_run_length_probability` (aggregated mass on run lengths at or
below a small threshold, which DOES depend on the data since it
aggregates the relative mass among run lengths 1..k, not the fixed row
0). With the calibrated prior, both correctly detect both true breaks in
the toy series (MAP run length collapses from 100 to 1 at each break)
with zero false positives elsewhere, after excluding an expected warm-up
artefact (run length cannot exceed the number of observations seen so
far, so the first few indices of any series trivially look like a
"detection" regardless of data).

## Validation against synthetic ground truth: low recall, explained

Against the synthetic panel's 47 true regime transitions (full 15-year
series, calibrated prior): **recall 4.3% (2/47), precision 75% (3/4)**.
This is not a flaw to explain away - it is consistent with Day 4's own
finding that the 5-regime HMM is weakly identified by returns alone (BIC
preferred 3 states; several state pairs are barely separable). Checked
directly rather than assumed: the two detected transitions are not
simply "the two largest shifts" - one missed transition has an even
larger raw mean shift than either detected case - so the story is real
but not perfectly clean. BOCPD can only catch genuine level/variance
shifts in a single return series; most of the 47 modelled transitions
are between regimes whose short-run return statistics are too similar
for any single-series changepoint test to separate, regardless of
implementation.

## Validation against real data: three of five event clusters independently confirmed

On the real partial Nifty data (2015-2019, from Day 2), BOCPD detects
five clusters of dates. Three are independently verified as genuine,
well-documented Indian market-moving events (the other two are plausible
but unconfirmed, and reported as such rather than claimed):

- **2015-08-24 to 26**: the global equity selloff triggered by the China
  yuan devaluation ("Black Monday") - already independently found by
  Day 2's own large-jump check (-6.1% log return on 2015-08-24).
- **2016-11-15**: the aftermath of the 8 November 2016 demonetisation
  announcement; Nifty/Sensex were near six-month lows the following week
  (Business Standard, 16 Nov 2016).
- **2019-09-20 to 24**: the corporate tax rate cut announcement, 20
  September 2019. Nifty rose 5.3% that day - independently reported by
  multiple sources (Business Standard, LatestLY, 5paisa) as the largest
  single-day Nifty/Sensex gain in a decade.
- **2018-02-02 to 06** and **2019-05-20**: not independently verified.
  Plausibly the February 2018 Union Budget / global "Volmageddon" VIX
  spike, and the May 2019 election result period respectively, but not
  confirmed by search, and not claimed as verified.

## Streaming conjugate updates (Section A9.3)

`StreamingDirichletTransitions` and `StreamingBetaBernoulli` are checked
against the property that makes them worth building at all: a posterior
computed one observation at a time is mathematically identical to one
computed from the full batch of counts at once, and does not depend on
the order updates arrive in. Verified three ways: streaming vs. a single
`update_sequence` batch call, streaming vs. an independent manual count
re-implementation, and streaming vs. the same updates applied in shuffled
order. All identical to floating-point precision.

**A real, explained discrepancy against the batch model's own fitted
transition matrix.** Feeding the streaming tracker the same HMM's hard
Viterbi-decoded state sequence and comparing its posterior mean against
`hmmlearn`'s own fitted `transmat_` gives a maximum difference of 0.196,
on the `Transitional -> Risk-On` entry. This is not a bug: the streaming
tracker counts transitions along the single most likely (hard) path,
while hmmlearn's EM fit estimates `transmat_` from soft, forward-
backward-weighted expected counts across all possible paths - different
estimators that need not agree, especially for a rarely-visited state
(`Transitional` holds only 92 of 3779 hard-decoded days, 2.4%). A
deployed system should not treat this kind of disagreement alone as a
reconciliation failure.

## The reconciliation diagnostic: not the pattern expected going in

"Online" = the HMM's filtered (causal) posterior; "batch" = the same
model's smoothed (full-hindsight) posterior. The natural x-axis A9.3
asks for, days since the last batch refit, is approximated by days
since the most recent BOCPD-detected changepoint (an available, if
imperfect, proxy - this project cannot afford to run a real periodic
20-restart batch refit schedule to get a literal one).

**The expected pattern (TV distance growing monotonically with
staleness) does not hold, checked directly rather than assumed to
hold.** Bucketed by days since the last detected changepoint: 0.042,
0.035, 0.023, 0.036, **0.111** (40-80 days), 0.050 (80+ days). The gap
peaks in the middle, not at the far end. The likely explanation connects
back to the recall finding above: BOCPD only flags sharp return
distribution shifts, while the HMM's own filtered/smoothed disagreement
is driven by its state ambiguity more generally - and most of the 47
true regime transitions are NOT accompanied by a BOCPD-detected event.
"Days since last BOCPD changepoint" therefore does not track "days since
the HMM's belief last needed revising" the way the proxy was intended
to. A production reconciliation system built on this specific proxy
would miss most of the actual sources of online/batch disagreement.

## Limits

- The particle filter was validated against exact filtering on ONE
  fitted model (Day 4's robust HMM); it was not re-validated against
  Day 5's Bayesian HMM or Day 6's RS-VAR.
- BOCPD's prior calibration uses a single fixed 60-day window at the
  start of the series; a rolling recalibration was not implemented or
  tested.
- The reconciliation diagnostic's "batch" reference is a single
  full-series fit, not a sequence of real periodic refits with
  progressively larger training windows - the actual production
  scenario A9.3 describes was approximated, not reproduced.
- Real-event verification used a handful of targeted web searches; the
  two unconfirmed clusters were not pursued further.

## Recommendation

The corrected BOCPD (calibrated prior, MAP-run-length or
short-run-length signal, never `R[0,:]` alone) is ready to use. The
reconciliation finding is the more actionable result for later days:
rather than gate a batch refit decision on "has BOCPD fired recently",
a production reconciliation check should compare filtered against
smoothed directly and trigger on the TV distance itself, since the
changepoint-based proxy tried here does not reliably track it.
