# Day 5: Bayesian HMM notes

Scope: `src/models/hmm/bayesian.py`. Full detail and executed code in
`notebooks/04_bayesian_hmm.ipynb`.

## Section A3.4's example code does not implement an HMM

Its states are drawn i.i.d. from a static `pi` each day; the transition
matrix `P` has a Dirichlet prior but is never referenced in the
likelihood. Proven, not just read off the code: fit it on data with an
obvious two-block regime switch and `P`'s posterior mean diagonal (0.798,
0.797, 0.791) comes out statistically indistinguishable from its
Dirichlet(8,1,1) prior's theoretical mean (0.800). PyMC's own sampler
summary explains why - `P` sits on the continuous NUTS block, `states` on
a separate `CategoricalGibbsMetropolis` step, and nothing connects them.

Separately, this construction is computationally impractical at project
scale regardless of the bug: PyMC has no choice but to Gibbs-sample the
discrete `states`, one scalar categorical variable per trading day.

## The fix: forward-algorithm marginalisation

`build_marginalized_hmm` integrates the discrete state path out
analytically via the forward algorithm (`pytensor.scan`), so `P`,
`pi`, `mu`, and `sigma` are the only sampled quantities and NUTS can
differentiate straight through the likelihood. Verified on the same toy
series: posterior diagonal comes out 0.819, 0.881, 0.855 - genuinely
above the flat 0.800 prior, on data with real persistence to find.

An `ordered` transform on `mu` prevents label-switching (states 0..K-1
are otherwise an arbitrary, sampler-chosen permutation that would wreck
R-hat and make cross-chain summaries meaningless).

A real numerical bug was found and fixed along the way: the initial
implementation exponentiated raw emission log-densities without a
log-sum-exp shift, which overflowed (`RuntimeWarning: overflow
encountered in dot`, inside PyMC's own HMC kinetic-energy computation)
whenever a candidate `sigma` sampled close to 0 during warmup made
`-log(sigma)` blow up for an observation landing close to the
corresponding `mu`. Fixed with the standard per-step max-subtraction.

## The full run required a real, evidenced scope reduction

The task specifies 2000 draws, 1000 tune, 4 chains at the full 15-year
window. This sandbox has exactly one CPU core, and - discovered the hard
way, not assumed - **background processes do not survive between
separate tool calls here**, even under `nohup`. That ruled out the
original plan of kicking off a long background job and polling it.

Real timing measurements taken before choosing a budget:

| T (trading days) | iterations tested | time | s/iteration |
|---|---|---|---|
| 120 (toy) | 600 | 70-90s | ~0.12-0.15 |
| 756 (3 years) | 100 | 140s | ~1.4 |
| 252 (1 year) | 150 | 73s | ~0.49 |

At T=3779 (the full window) and ~1.4-2s/iteration extrapolated, the
task's literal budget (12,000 total iterations across 4 chains) would
take on the order of hours per chain - not feasible in this environment.

**What actually ran**: 1-year window (T=252), 500 draws + 500 tune per
chain, 4 chains, `target_accept=0.95` (kept at spec despite everything
else changing). Each chain ran as a **separate process invocation**
(`scripts/run_day5_sampling.py <chain_index>`, run four times), because
of the no-surviving-background-processes constraint above, then combined
for cross-chain diagnostics. Real per-chain times: 104s, 83s, 90s, 96s.

## Diagnostics on the actual run (not the spec's run)

- R-hat: 1.00 for essentially every `P` and `pi` entry; 1.00-1.02 for
  `sigma` (mildly elevated but not alarming).
- ESS (bulk): 573-2520 across parameters, from 2000 total post-warmup
  draws (4 chains x 500).
- Divergences: 2 out of 2000 (0.1%).
- Trace plots: `docs/artifacts/day5_trace_plots.png`.

This is a genuinely well-behaved posterior at this reduced budget - the
architecture fix (Section 2) is doing real work, not just avoiding a
crash.

## Credible intervals

90% credible intervals for self-transition probability and implied
expected duration (`1/(1-p_ii)`) are in the notebook, Section 5, per
state. Median durations run 2.7-3.7 days across the five states, with
90% credible intervals reaching up to 6-12 days - short and wide, an
honest reflection of only 252 observations informing 5 states' worth of
transition rows, not a claim of precision the sample size cannot
support.

## Frequentist comparison: two different questions, two different answers

**Does the informative prior help on a short window?** Decisively yes:
the frequentist approach cannot produce a valid 5-state fit on this same
252-day window at all - every one of 30 restarts collapses a state. The
Bayesian model, same window, samples cleanly (see diagnostics above).

**Does better-behaved sampling mean better regime recovery?** No, and
the notebook's own numbers say so plainly rather than implying
otherwise. Checked numerically before writing this up, because the
first-pass number looked more favourable than it turned out to be:

- Bayesian exact-label match rate against the synthetic panel's ground
  truth: 21.0%, only **0.4 standard errors** above the 20% random-guess
  baseline for 5 labels (SE = 2.5% at n=252) - statistically
  indistinguishable from chance.
- Frequentist (Day 4's model, fit on the full 15 years, sliced to this
  same year): 29.4%, a genuine **3.7 SE** above baseline.

The Bayesian model's informative prior fixes an estimation pathology
(degenerate, redundant, or empty states). It does not and cannot fix the
underlying information bottleneck: both models see nothing but Nifty
daily returns, and no fitting procedure manufactures information that
was never in the input. This is Day 4's conclusion again, from a
different angle - BIC preferring fewer states there, a statistically
chance-level match rate here.

## Recommendation

The corrected model (`build_marginalized_hmm`) is architecturally right
and worth keeping; the reduced-budget run demonstrates it works well
when it gets enough compute. Day 6's RS-VAR, which adds volatility,
breadth-proxy, flow, INR, and gilt-yield dimensions rather than asking
one Bayesian prior to compensate for a single scalar feature, is the
more promising direction for actually improving regime recovery - not
because Bayesian methods failed here, but because they succeeded at the
problem they can solve (estimation) and, honestly, could not solve the
one they can't (information).
