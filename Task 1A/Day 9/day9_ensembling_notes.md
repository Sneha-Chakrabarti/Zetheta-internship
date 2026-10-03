# Day 9: ensembling notes

Scope: `src/models/ensemble/` (labels, combine, contract), new filtered/
pointwise functions added to `src/models/hmm/frequentist.py`,
`src/models/hmm/bayesian.py` and `src/models/rsvar/bayesian_rsvar.py`,
scripts `scripts/run_day9_*.py`, notebook `notebooks/08_ensembling.ipynb`.

## Two scope decisions, both forced, not chosen for convenience

**The regime space is 3-way.** Day 6's multivariate RS-VAR only supports
K=3 at feasible compute (documented there). Rather than fabricate a
5-way split for it, every member is collapsed to Risk-On / Transitional
/ Risk-Off using the mapping Day 6 already established
(`src/models/ensemble/labels.py`): Risk-On absorbs Post-Shock,
Transitional absorbs Late-Cycle.

**The evaluation window is 252 days** (2024-07-11 to 2025-06-27),
verified in-session to be exactly the last 252 days of Day 7-9's 702-day
held-out test period, chosen because it is the only window every member
has native coverage of (Day 5/6's models were fit on exactly this span).

## Members are not equally out-of-fold, and that matters for every number below

| member | trained on | out-of-fold on this window |
|---|---|---|
| `mc_dropout`, `variational_bnn`, `deep_ensemble` (Day 7) | first 2806 days | yes |
| `foundation_head` (Day 9, re-run of a Day 8 arm) | first 2806 days | yes |
| `freq_hmm` (Day 4) | full 15-year series | no, fit on data including this window |
| `bayesian_hmm` (Day 5) | exactly this 252-day window | no, this window IS its training data |
| `bayesian_rsvar` (Day 6) | exactly this 252-day window | no, this window IS its training data |

Filtering (causal, no-look-ahead probabilities) fixes the WITHIN-window
leak, day *t* no longer sees day *t+5*, but does not undo having used
this window's data to choose the model's parameters in the first place.
Both facts matter together: filtering is necessary (Day 4's smoothed
probabilities would otherwise use the whole window's future at every
day) and not sufficient to make these three members genuinely
out-of-fold. Their strong individual showing below should be read with
that in mind.

## Infrastructure built: none of this existed before today

Every HMM/RS-VAR module had a *smoothed* (forward-backward) probability
function, built for earlier days' regime overlays. None had a *filtered*
(forward-only) one, needed here because a member's contribution to an
out-of-fold ensemble must reflect only what it could have known as of
day *t*. Added `filtered_state_probs` (frequentist HMM, via a numpy port
of hmmlearn's own parameters since hmmlearn exposes no public filtered
API), `filtered_state_probs` (Bayesian HMM) and
`filtered_state_probs_mvn` (Bayesian RS-VAR), each verified against its
smoothed counterpart: they must agree exactly at the final timestep
(nothing left to look ahead to) and disagree earlier on data with real
ambiguity. All three passed. Also added `pointwise_loglik` and
`pointwise_loglik_mvn` (per-observation log-likelihood, needed for
PSIS-LOO), each verified to sum to the model's total log-likelihood.

`src/models/ensemble/combine.py` implements Section A10.1 (BMA) and
A10.2 (stacking) close to verbatim, with one fix: the spec infers the
number of classes from `base_probs.shape[2]`, which silently breaks if
a rare class is absent from `y_true` in a given fold. Made the class
count an explicit, checked argument.

## Bayesian Model Averaging and stacking: evaluated out-of-fold, not assumed

`TimeSeriesSplit`, 4 folds, weights fit on each fold's training slice
only, scored on its held-out test slice. Both label sources tested: the
HMM pseudo-labels (what the supervised members were trained toward) and
the synthetic panel's true regime.

Mean out-of-fold log-likelihood, 4 folds:

| | vs. HMM labels | vs. true regime |
|---|---|---|
| best individual member | -0.3765 (`freq_hmm`) | -0.9022 (`bayesian_hmm`) |
| equal-weight average | -0.4827 | -0.9918 |
| BMA | -0.4703 | -1.4994 |
| stacking | -0.3713 | -1.3712 |

**A10.4 asks the engine to show the ensemble beats every individual
member on out-of-sample calibrated log-likelihood. Checked honestly,
this holds in exactly one of four cases**: stacking against the HMM
labels, and only narrowly. It fails for BMA against both label sets and
for stacking against the true regime, where naive equal-weighting beats
both learned methods. With 50-202 observations per fold and up to 7
weights to fit, BMA and stacking are prone to fitting noise that does
not generalise; this is the concrete mechanism, not just a label for the
result. Reported as found, not as the "ensemble strictly dominates"
result the task document's framing anticipates.

## Model selection: PSIS-LOO, not WAIC, for a real and checked reason

Section A10.3 asks for "WAIC / PSIS-LOO ... from ArviZ". The installed
ArviZ (the same 2.x DataTree-based release already met in Days 5-6) has
dropped WAIC entirely: `hasattr(az, "waic")` is `False`, and
`az.compare` takes no `ic` or `scale` argument at all, only `method` for
combining LOO results across models. Checked directly, not assumed.
What follows is PSIS-LOO only, which A10.3's own "or" already permits.

A second caveat applies to the comparison itself: Day 5's HMM and Day
6's RS-VAR do not share an observation space (1-D returns vs. the 6-D
feature panel). `az.compare` is usually read as "which model predicts
this one series better"; here it compares two models' fit to their own,
differently-shaped inputs. The ranking below still means something, but
not that.

```
                rank  elpd_diff   dse    p    elpd    se  weight
bayesian_hmm       0        0.0   0.0  5.9   790.0  14.0     1.0
bayesian_rsvar     1    -2190.0  39.0 205.0 -1390.0  47.0     0.0
```

## A finding independent of the comparability caveat: RS-VAR's own LOO is flagged unreliable

PSIS-LOO reports a Pareto-k diagnostic per held-out point; k > 0.7 means
the importance-sampling approximation for that point cannot be trusted.

- **HMM**: p_loo (effective parameters) = 5.89. All 252 points "good"
  (max k = 0.653).
- **RS-VAR**: p_loo = 204.95, against roughly 197 raw parameters (K=3:
  `A` alone is 3x6x6=108, `c`=18, three 6x6 covariance factors at 21
  packed elements each=63, `P`=6, `pi`=2). 233/252 points "good", 13
  "bad" (0.6-1), 6 "very bad" (>1); max k = 2.870.

An effective parameter count this close to the raw count means the
priors are barely shrinking the fit relative to 252 observations, close
to interpolating the training window rather than generalising. This is
a second, independent symptom (LOO importance-sampling breakdown on 19
of 252 points) of the same problem Day 6 already found from a different
angle (wide credible intervals, a chance-level ground-truth match): 320
posterior draws on a 3-regime, 6-dimensional VAR is not enough to
support a trustworthy comparison number here, not just a trustworthy
point estimate.

## The combined output contract (Section A10.4)

Implemented as a typed dataclass (`RegimeOutputRecord`) with every field
either computed or explicitly `None`:

- `probability`, `regime_id`/`regime_label`: the combined vector's
  argmax and its probability.
- `epistemic_uncertainty` / `aleatoric_uncertainty`: cross-member std of
  the dominant class' probability, and mean per-member entropy, the same
  two decompositions used in Days 7-8, here applied across different
  MODELS rather than stochastic passes of one network. Noted as a
  different notion of "epistemic" (model-form disagreement, not
  within-model weight uncertainty), not presented as identical.
- `dominant_model` / `dominant_model_weight`: the highest-weighted
  member, which is not necessarily the most confident one (tested: a
  low-confidence, high-weight member is correctly reported over a
  high-confidence, low-weight one).
- `conformal_lower` / `conformal_upper`: a **placeholder** fixed-width
  band, flagged `conformal_is_placeholder=True`. Real conformal
  calibration is Day 11's deliverable.
- `changepoint_flag`, `reconciliation_gap`, `ood_score`: always `None`.
  They depend on Day 10's online/changepoint work, not built yet.
  Explicit `None` rather than omission, so a caller cannot mistake a
  missing field for a computed zero.

Stacking weights fit on the full window (illustrative, not a
performance claim, Section 2 above is the honest number) put all weight
on two members (`freq_hmm` 0.30, `mc_dropout` 0.70) and zero on the
other five: a sparse solution from constrained cross-entropy
minimisation with several correlated members on 252 days of data.

## Limits

- Only one embedding-based head was re-trained for Day 9
  (`foundation_head`, the TimesFM-style embedding, matching Day 8's
  better-validated arm); the Chronos-style embedding was not added as an
  eighth member.
- BMA/stacking evaluation uses 4 folds of 50-202 days; small-sample
  noise in the reported comparison is real and should not be over-read
  fold to fold.
- The output contract's epistemic/aleatoric split conflates model-level
  and within-model uncertainty, noted above but not resolved.
- No calibration curve (reliability diagram) was computed for the
  combined output; only mean log-likelihood.
- `bayesian_hmm` and `bayesian_rsvar` are compared by PSIS-LOO but that
  comparison is not on equal footing (different observation spaces);
  stated, not corrected, since correcting it would mean re-deriving one
  model's likelihood on the other's inputs.

## Recommendation

Treat the "ensemble beats every member" framing as unproven here, not
disproven for good: the mechanism identified (too little data for the
number of weights being fit) suggests more history, not a different
combination method, is the fix. Day 10-11's changepoint and conformal
work should slot into the `RegimeOutputRecord` schema as-is rather than
prompting a schema change. The RS-VAR's LOO reliability issue is a
direct argument for more posterior draws before trusting any
information-criterion-based comparison involving it.
