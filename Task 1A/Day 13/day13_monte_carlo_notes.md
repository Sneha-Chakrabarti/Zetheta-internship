# Day 13: Monte Carlo, allocation overlay, and backtest notes

Scope: `src/models/monte_carlo/` (regime_mc, ic_artefact),
`src/backtest/` (overlay, engine), `scripts/run_day13_backtest.py`,
notebook `notebooks/11_monte_carlo_backtest.ipynb`. Plain numpy/scipy/
pandas; no PyMC or TensorFlow needed.

## The central methodological decision: point-in-time parameters, not just point-in-time probabilities

A genuinely valid 2019-2024 backtest needs a model fit using only data
available before the window starts. Day 4's published 5-state HMM was
fit on the full 15-year series, including 2025 data, which is "future"
relative to this window. Reusing it here, even with filtered
(causal) probabilities within the backtest window, would still carry
parameter-level look-ahead bias: the transition matrix and per-regime
drift/vol were estimated partly from data the backtest window should
not have access to. This is the same category of problem Day 9 flagged
prominently for several of its ensemble members, applied here rather
than set aside for convenience.

Refitting on pre-2019 data alone (2011-01-04 to 2018-12-31, 2085 days)
was attempted at K=5 (Day 4's original choice) and K=3 (Day 4's own
BIC-preferred count):

- **K=5 fails outright**: 0 of 30 restarts produce a valid fit (worse
  than the full series' already-fragile 2 of 20).
- **K=3 technically fits (15 of 30 valid) but degenerates**: two of the
  three states have zero self-transition probability and occupy only 35
  of 2085 days each - single-day outlier-catching states, not real
  multi-day regimes.
- **K=2 is robust**: 30 of 30 restarts valid, two genuinely persistent
  states (self-transition probabilities 0.999 and 0.977), with a
  sensible Calm (annualised drift +0.118, vol 0.153) / Crisis (drift
  -1.121, vol 0.358) structure.

K=2 is used for the headline backtest. This is a further, independent
confirmation of Day 4's "simpler models are more robust with limited
data" finding, extended one step further: on a shorter window, even
three states becomes too many. It is not a convenience fallback chosen
to avoid work; K=5 and K=3 were both attempted and failed first.

## The Monte Carlo engine and overlay: one real numerical fix

`RegimeConditionedMC` (A13.1) and `regime_var` (A13.2) match the spec
closely. One fix: the spec's inverse-CDF state sampling
(`(u[:, None] < cum).argmax(axis=1)`) relies on each transition row's
cumulative sum reaching exactly 1.0 at its last column. For a matrix
built from floating-point accumulation (an EM-fitted `transmat_`), that
last entry can land a few ULPs below 1.0; if a sampled `u` ever lands in
that gap, `argmax` on an all-False row silently returns index 0 instead
of the last state. Not observed to cause a problem in this project's own
fitted matrices, but checked directly rather than assumed negligible: a
random-Dirichlet test case came within 1e-16 of the boundary, and
`n_sims * horizon` draws routinely runs into the millions, so rare is
not the same as impossible. `RegimeConditionedMC.__init__` renormalises
`P`'s rows to be exactly stochastic, eliminating the gap outright.

The allocation overlay (`src/backtest/overlay.py`) is built around a
real subtlety in this project's own regime taxonomy: Post-Shock has the
SECOND-HIGHEST annualised drift of any of the five regimes (0.25,
exceeded only informally in rank by nothing - Risk-On is 0.15), because
it represents a recovery rally, not continued stress. A tilt rule based
on regime NAMES (Transitional, Post-Shock and Risk-Off all sound
"risk-off", de-risk accordingly) gets this backwards. Checked directly:
a certain Post-Shock call gets weight 1.0 from both proper,
risk-adjusted tilt rules (`tilt_probability_weighted`,
`tilt_conviction_scaled`) and weight 0.3 (the floor, identical to
certain Risk-Off) from the naive name-based rule kept specifically to
make this contrast visible.

## Backtest results, 2019-2024: read with the data source in mind

All figures below use the SYNTHETIC price path for this project's own
2019-2024 calendar slice, not real Nifty data. This is restated
deliberately rather than once: the real Nifty market roughly doubled
over the real 2019-2024 period, and nothing here should be read as a
claim about what actually happened.

| strategy | total return | annualised vol | max drawdown | IR vs. buy-hold | tracking error |
|---|---|---|---|---|---|
| buy-hold | -15.1% | 16.4% | -41.1% | - | - |
| prob-weighted overlay | -10.3% | 14.2% | -36.5% | 0.157 | 3.45% |
| conviction-scaled overlay | -10.8% | 15.6% | -39.3% | 0.232 | 2.81% |
| naive name-based overlay | -8.5% | 12.4% | -32.5% | 0.134 | 4.68% |

All three overlay variants reduce drawdown and achieve a positive
information ratio against the same synthetic buy-hold benchmark, by
de-risking during the same detected Crisis-probability days. The naive
rule happens to perform reasonably here specifically BECAUSE this is a
2-state Calm/Crisis taxonomy with no Post-Shock-like state to mislabel -
see the next section for why this does not contradict the overlay
module's own finding.

Regime-conditioned drawdowns (one row per CONTIGUOUS run of a dominant
label, not pooled across non-adjacent occurrences) show the worst
individual episodes are long Calm runs with a grinding decline inside
them, a real limitation of a 2-state taxonomy: "Calm" describes the
regime's typical annualised drift, not a guarantee against drawdown
within any one run of it.

## The Post-Shock finding has two layers, not one

The overlay module's own tests demonstrate the naming-trap problem using
the TRUE synthetic generator's known regime parameters - a clean,
controlled check that Post-Shock's actual drift (0.25) should earn it a
high weight. Making this concrete in an actual backtest, using Day 4's
full 5-state model (reused here only for this illustration, and
carrying exactly the look-ahead bias the K=2 analysis above exists to
avoid) surfaced a second, more interesting problem:

**The fitted model's own ESTIMATED Post-Shock drift came out NEGATIVE
(-0.809 annualised), not the true generative +0.25.** This is a
different failure from the naming trap: it is not that the tilt rule
ignores Post-Shock's real character, it is that the model's own
estimate of that character is wrong for this state. This directly
extends Day 4's own established finding (this 5-state HMM is weakly
identified; BIC preferred 3 states; ground-truth match rate around 40%)
to a concrete downstream consequence: a drift/vol-aware tilt rule is
only as good as the regime identification feeding it, and fixing the
RULE does not help if the ESTIMATE is wrong.

This is reported as illustrative, not as a robust result: only 2 days
in the entire 2019-2024 window have P(Post-Shock) > 0.5, far too few to
draw a confident conclusion from. The direction (naive weight 0.345,
"proper" weight 0.475 on those 2 days) is at least consistent with the
proper rule still doing somewhat better even with a corrupted drift
estimate, likely because the blended probability-weighted calculation
draws on other, better-identified states too - but this is a two-data-
point observation, stated as such.

## Monte Carlo projection and the Investment Committee artefact

Projected forward from the end of the backtest window (31 December
2024, P(Calm)=34.8%, P(Crisis)=65.2%), 10,000 simulated paths, 252-day
horizon, using the point-in-time-fit K=2 model's own parameters:

- 90% one-year prediction interval: -49.0% to +36.4%
- Probability of negative one-year return: 54.9%
- VaR(5%): -49.0%; CVaR(5%): -59.2%
- Conditional on at least half of the first 90 days being spent in
  Crisis (probability of this: 26.4%), the worst-case (5th percentile)
  one-year return is -63.3%.

These numbers are wide and the outlook is pessimistic because the
regime distribution at the end of this particular synthetic realisation
happens to be Crisis-dominated (65.2%) - a direct, mechanical
consequence of this synthetic window's own regime path, not a
forecast about real markets.

`ICArtefact` (A13.3) carries explicit lineage with every statement: the
model name, exact fit data window, fit seed, number of valid restarts,
simulation seed, path count, and horizon - so any number in the
rendered statements can be traced back to exactly what produced it and
re-run, rather than taken as an assertion.

## Limits

- The backtest's 1-day lag between a filtered probability and the
  resulting portfolio weight is a simplifying assumption; real
  implementation lag (data availability, trading costs, slippage) is
  not modelled.
- Cash return is held at exactly 0.0 throughout; a more realistic
  backtest would use a real short-rate series for the non-equity sleeve.
- The K=2 Calm/Crisis taxonomy is a genuine simplification relative to
  the project's own 5-state regime definitions; it was chosen because it
  is what the available pre-2019 data robustly supports, not because
  two states are believed sufficient in general.
- The Post-Shock mis-estimation finding (Section above) rests on 2 data
  points and should not be generalised beyond "this is a real risk
  worth checking for," which is exactly how it is reported.
- Transaction costs, position limits, and any real-world implementation
  constraint are entirely absent from the overlay backtest.

## Recommendation

The K=2 point-in-time-valid approach should be the template for any
production backtest claim, even though it is a coarser regime space
than the project's headline 5-state ensemble: a backtest's credibility
depends on genuine point-in-time validity more than on regime-space
richness. The Post-Shock two-layer finding argues for monitoring
regime-identification QUALITY (e.g. via Day 9's own per-member
log-likelihood tracking) alongside the tilt rule itself, since a correct
rule fed a wrong estimate is no better than a wrong rule.
