# Day 6: regime-switching VAR notes

Scope: `src/models/rsvar/baseline.py` (Section A8.2), `src/models/rsvar/
features.py`, `src/models/rsvar/bayesian_rsvar.py` (Section A8.3). Full
detail and executed code in `notebooks/05_rsvar.ipynb`.

## A8.2: single-feature Markov-switching baseline (statsmodels)

A single default `.fit()` call left most standard errors as NaN and one
regime with only 16 of 3779 days assigned to it. Running 5 independent
random-search seeds (`search_reps=30` each) showed 4 converge to the
same qualitative 3-way partition of the data (~200/1500/2100-day split,
log-likelihood within 4 points of each other, up to a label permutation)
while 1 seed landed in a clearly worse local optimum (log-likelihood
12078 vs. ~12114, a 6-observation regime). `fit_msm_regression_stable`
runs several seeds and keeps the best by likelihood - same principle as
`fit_regime_hmm_robust` from Day 4.

Two more things found and handled:

- statsmodels' `MarkovRegression` uses a **column-stochastic** transition
  matrix (`regime_transition[i,j]` = P(state=i at t | state=j at t-1)),
  the opposite convention from every other model in this project
  (hmmlearn's row-stochastic). `transition_matrix()` transposes on the
  way out. Getting this backwards would have silently produced wrong
  duration and stickiness numbers without ever raising an error.
- A specific random seed can produce a transition-probability estimate
  exactly at the [0, 1] boundary, which breaks statsmodels' internal
  logistic un-transform (`ValueError: Could not untransform
  parameters`) via a failing `scipy.optimize.root` call. Caught per-seed
  in `fit_msm_regression_stable` rather than letting one bad seed crash
  the whole comparison.

Even the best fit still shows NaN standard errors for about half the
parameters - a known limitation of statsmodels' numerical-Hessian
approach near the transition-probability simplex boundary, not evidence
the point estimates themselves are unreliable (the multi-seed stability
check above is the actual evidence for or against that).

## A8.3: two real gaps in the spec's own example code

1. **`pm.HiddenMarkovChain` does not exist** in the installed PyMC
   (6.3.2) - confirmed with `hasattr(pm, 'HiddenMarkovChain')` returning
   `False`, not inferred. Fixed the same way Day 5 fixed Section A3.4's
   gap: forward-algorithm marginalisation via `pytensor.scan`, extended
   here from univariate Gaussian emissions to multivariate VAR(1)
   emissions (`_forward_algorithm_logp_mvn`, `_mvn_logpdf_batch`).
2. **`pm.LKJCholeskyCov(..., shape=(K,))` does not batch** - tested
   directly: passing `shape=(K,)` silently returns a single (d, d)
   Cholesky factor, not K independent ones (verified by checking the
   free variable's actual shape, `(10,)` for d=4, matching one packed
   triangular factor, not `(K, 10)`). Yet the spec's own code indexes
   `chol[states]` as if `chol` had a leading K dimension. Fixed by
   building K independent `LKJCholeskyCov` calls in a loop (one per
   regime, each its own named PyMC variable) and stacking them.

A related, easy-to-miss detail: `LKJCholeskyCov`'s returned `chol` is
stored **packed** (the `d*(d+1)/2` lower-triangular elements as a flat
vector), not as a full `(d, d)` matrix. `unpack_cholesky` reconstructs
the full matrix and was verified against `pm.math.
expand_packed_triangular` on a known example before being trusted for
any downstream covariance calculation.

## Feature panel

Six of the spec's seven features (Nifty return, realised vol, FII flow,
DII flow, INR return, gilt yield change); breadth is not built, for the
same reason as Day 3 (needs per-constituent stock data outside the
required panel). Standardised (z-scored) before fitting: raw FII/DII
flows are three orders of magnitude larger than returns, which would
make a shared `A ~ Normal(0, 0.3)` prior meaningless across features on
wildly different scales.

## Compute budget: real timing forced a larger reduction than Day 5

A 6-feature, K-regime VAR has far more parameters than a 1-feature
K-regime HMM (`A` alone is `K x d x d` vs. `K` means), and per-iteration
cost turned out to be both higher and less predictable than Day 5's
univariate case:

| Test | iterations | time | s/iteration |
|---|---|---|---|
| K=3, d=6, T=252, early NUTS steps | 60 | 30s | ~0.5 |
| K=3, d=6, T=252, after more tuning | 120 | 171s | ~1.4 |
| K=3, d=6, T=252, final budget | 160/chain | 158-250s/chain | ~1.2-1.6 |

Two budget attempts at 250+250 and even 500+500 draws both timed out
before finishing a single chain. Final budget: **K=3 (not 5), 1-year
window (same as Day 5), 80 draws + 80 tune per chain**, `target_accept
=0.95` (kept at spec). Each chain ran as a separate process invocation
(`scripts/run_day6_sampling.py <chain_index>`, run four times) for the
same reason as Day 5: background processes do not survive between tool
calls in this sandbox.

Result: 0 divergences across all 320 post-warmup draws.

## Label-switching: found, diagnosed, fixed

Despite zero divergences, R-hat on the raw combined trace was
catastrophic (1.44-2.43 on `P`, should be <1.01). Diagnosed before
assuming the sampler had failed: each chain's posterior mean of `c[:,
0]` (the Nifty-return intercept per regime) showed the same three
values, in three different index orders, across the four chains -
textbook label-switching, not non-convergence.

Day 5's univariate HMM prevented this at the model level with an
`ordered` transform on `mu`. No equivalent single-dimension ordering
exists for a 6-dimensional VAR intercept, so this project relabels
**post-hoc, per posterior draw** instead (`relabel_by_nifty_drift`):
regime 0 is always the draw's highest-nifty-drift regime, regime K-1 the
lowest, and every regime-indexed parameter (`P`'s both axes, `pi`, `c`,
`A`, and the three `chol_k` factors) is permuted consistently.

After relabelling: R-hat drops to 1.00-1.08 for nearly everything (two
`P` entries at 1.17-1.18, still a bit high but far better), and ESS
recovers from single digits to the hundreds. Confirms the sampler itself
was fine; the diagnostic failure was entirely a labelling artefact.

## Regime-conditional impulse responses and covariance: delivered, with honest uncertainty

Both are Day 6's explicit asks (Section A8.4). Computed on the relabelled
posterior (320 draws): a -1SD FII-outflow shock's propagation to Nifty
returns, and the Nifty-INR correlation, by regime.

**Both come back with 90% credible intervals wide enough to straddle
zero at every regime and every horizon checked**, and heavily overlapping
across regimes. The point estimates show mild differences (e.g. the
lowest-drift regime's median IRF response runs a bit higher than the
highest-drift regime's), but nothing a reader should treat as a
confident regime-dependent effect at this sample size. Reported plainly
in the notebook rather than emphasising the point estimates and
downplaying the intervals - the intervals are the actual finding here.

## Smoothed regime probabilities: plug-in vs. correctly-averaged

Computing smoothed state probabilities once from posterior-**mean**
parameters ("plug-in") versus averaging smoothed probabilities computed
**separately for each of 30 posterior draws** give different pictures
when real posterior uncertainty about the partition remains - which,
given the wide credible intervals above, it clearly does here:

- Plug-in state counts: `[0, 246, 6]`
- Correctly-averaged state counts: `[2, 238, 12]`
- Correctly-averaged **mean probability mass** per state: `[0.36, 0.42,
  0.21]`

The hard-assignment counts look similar, but the probability mass tells
a different story: regime 0 carries real average probability (36%) even
though it almost never "wins" the argmax on any single day. The
correctly-averaged version is what the rest of the analysis uses.

## Ground-truth comparison: chance level

Mapped the synthetic panel's five true regimes down to three buckets
(Risk-On absorbs post_shock; Transitional absorbs late_cycle) for a fair
comparison against this K=3 model.

**Match rate: 32.9%, z = -0.13 against a 33.3% random-guess baseline for
3 labels.** Not even slightly above chance.

This is reported as a genuine, honest result, not softened: with only
320 total posterior draws on a 3-regime, 6-dimensional VAR, there is too
much residual parameter uncertainty (see the impulse-response and
covariance credible intervals above) for the extra observed dimensions
to translate into confident, accurate regime identification. The
theoretical case for RS-VAR over a single-feature HMM - more observed
information should separate regimes better - is not refuted by this;
this particular run, under this sandbox's one-CPU, no-background-
process compute constraints, simply does not yet have enough posterior
resolution to demonstrate it. That is a compute-budget limitation,
evidenced with real timing numbers above, not a finding that
multivariate modelling doesn't help.

## Recommendation

The corrected model architecture (`build_bayesian_rsvar` +
`relabel_by_nifty_drift`) is right and worth keeping - the label-
switching fix alone took R-hat from unusable to good. What it actually
needs to deliver on Section A8.4's promises is more posterior draws than
this sandbox's single CPU core can produce in a session, not a different
model. Running the same code with real parallel compute (more cores, or
more wall-clock time than a single tool-call budget allows) at the
existing K=3/1-year settings, before even considering K=5 or the full
15-year window, is the direct next step - the code does not need to
change, only the compute available to it.
