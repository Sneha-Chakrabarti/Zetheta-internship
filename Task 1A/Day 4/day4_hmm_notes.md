# Day 4: frequentist HMM notes

Scope: `src/models/hmm/frequentist.py`. Full detail and executed code in
`notebooks/03_frequentist_hmm.ipynb`.

## The task document's exact recipe degenerates on this data

Section A3.2's code (one `GaussianHMM` fit, `seed=42`, `n_iter=200`)
converges to a genuinely broken model, verified by inspecting the fitted
parameters, not assumed from a convergence flag:

- States 0 and 1: means 0.079 and 0.069 annualised, vols 0.151 and 0.149
  annualised - statistically indistinguishable. Self-transition
  probabilities near 0: the Viterbi decode flips between them roughly
  96-99% of the time, on 93% of all trading days. Two redundant states,
  not two regimes.
- State 2: zero observations assigned, variance blown up to 502
  annualised (a numerical artefact of EM starving a component of data).

This is a known EM/HMM pathology (too many states for what a single 1-D
series identifies), not a bug in the fitting code, and not specific to
this dataset's synthetic origin - the same thing happens to real return
series fit this way.

## Fix: multi-restart selection, not a different algorithm

`fit_regime_hmm_robust` fits N different seeds, discards any restart
where a state collapses below 1% of observations or produces non-finite
parameters (both observed failure modes above), and keeps the
highest-likelihood survivor. At 20 restarts, only 2/20 survived; at 50,
5/50. This alone is informative: even the "valid" fits are a minority.

The best surviving 5-state fit is usable but not clean: one dominant,
reasonably persistent state (89% of days, self-transition 0.825, expected
duration 5.7 days) plus four smaller states, two of which still show
near-zero self-transition (alternating rather than persisting).

## BIC says 5 states isn't well supported here

Day 4 asks for a 3/5/7-state BIC comparison. Using the same multi-restart
fitting for a fair comparison (a plain single-seed fit raises outright at
K=7 before reaching `.bic()`):

| states | log-likelihood | BIC | valid restarts |
|---|---|---|---|
| 3 | 12066.7 | -24018.0 | 26/50 |
| 5 | 12058.3 | -23836.5 | 5/50 |
| 7 | - | - | 0/50 (no valid fit found) |

Lower BIC is better: **3 states is preferred over 5**, and **7 states
could not be reliably fit at all** in 50 restarts - not "fit worse", but
genuinely unidentifiable at this sample size with one feature. This
directly follows Section A3.7's own evaluation guidance (parsimony via
BIC) and is reported as the honest answer even though the rest of Day
4's deliverable (regime overlay, duration stats, transition matrix)
specifically calls for a 5-state model. Both are done; they are not in
tension - the comparison's job is to say whether 5 is supported, and
here it is not.

## Cross-checked against the synthetic panel's ground truth

Only possible with synthetic data: `regime_path` in the data dict carries
the actual regime used to generate each day, never given to any model.
Comparing the HMM's post-hoc labels (assigned purely by the mean/vol
heuristic in Section A3.3, not fitted to match) against it:

- Overall exact-label match rate: 42.2% (vs. 20% for random guessing
  among 5 labels).
- True `Risk-On` days: 1415/1418 correctly labelled `Risk-On` - the one
  state the HMM identifies well.
- True `Late-Cycle` and `Transitional` days: mostly absorbed into the
  HMM's `Risk-On` label (1358/1414 and 438/484 respectively) - these true
  regimes have return distributions too similar to `Risk-On` for a
  returns-only model to separate, consistent with the near-redundant
  states found in the first two sections above.
- True `Risk-Off` days: split between `Post-Shock` and `Risk-Off` labels
  (75 and 73 of 170) - reasonably concentrated in the two
  negative-return states, not scattered randomly.

This is the same finding as the BIC table, from an independent angle: the
model behaves like it found roughly 2-3 real clusters (calm/positive,
moderate-stress, acute-stress), not 5, which is exactly what forcing 5
states onto information that only supports 2-3 produces.

## What this means for later days

Day 5's Bayesian HMM adds an informative Dirichlet prior on the
transition matrix (Section A3.4: self-transition pseudo-count 8 vs. 1
off-diagonal) - a direct, principled fix for the alternation pathology
seen here, since the prior penalises exactly the near-zero
self-transition rows this frequentist fit produces. Day 6's RS-VAR adds
volatility, breadth, flows, INR, and gilt yield as additional observed
dimensions, which gives the model more than one scalar number per day to
separate regimes with. Both are motivated by a specific, demonstrated
limitation here, not by "more sophisticated is generally better."

## Recommendation

Use the multi-restart 5-state fit (not the single-seed one) as the
frequentist baseline going forward, with the caveat above attached. Do
not present the frequentist regime labels as more reliable than the
42.2% ground-truth match rate and 3-state BIC preference actually
support.
