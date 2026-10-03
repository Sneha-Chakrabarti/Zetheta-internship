# R cross-validation codebase

Scope and honest limits, checked directly rather than assumed:

**CRAN is unreachable from this sandbox.** `install.packages()` fails
(`cannot open URL 'https://cloud.r-project.org/src/contrib/PACKAGES'`),
checked directly before writing any R code. This rules out the specific
packages the project brief names (depmixS4, MSwM, bcp, changepoint,
conformal, mlr3): none are pre-installed, and none can be installed
here. What IS pre-installed (checked via `installed.packages()`):
`rstan`, `rstanarm`, `brms`, `loo`, `bayesplot`, `strucchange`, `forecast`,
`tseries`, and a large general-purpose set, but none of the specific
regime-switching or conformal packages the brief names.

**Given that, this R codebase implements the underlying algorithms
directly in base R**, rather than attempting a different, substitute
package (e.g. using `strucchange` to stand in for Bayesian online
changepoint detection would be a different algorithm with different
properties, not a cross-check of the same one). This is arguably a
MORE rigorous cross-language check than a packaged alternative would
have been: it tests whether two independent implementations of the
SAME mathematics (Baum-Welch EM, split-conformal, regime-conditioned
Monte Carlo) agree, not whether two different software packages happen
to produce similar output.

## Files

- `hmm_lib.R` - Gaussian HMM via Baum-Welch EM (forward-backward,
  multi-restart robustness matching `fit_regime_hmm_robust`'s own
  criterion: reject any restart with a state occupying under 1% of
  observations).
- `hmm_baum_welch.R` - command-line entry point: `Rscript
  hmm_baum_welch.R <K> <n_restarts> <seed>`, reads
  `nifty_returns_full.csv`, writes filtered/smoothed probabilities and
  fitted parameters.
- `conformal_and_var.R` - split-conformal classification (A6.2) and
  regime-conditioned Monte Carlo VaR/CVaR (A13.1/A13.2), both in base R.
- `run_cross_validation.R` - runs all three cross-checks against
  Python's saved outputs and writes `cross_validation_results.json`.
- `nifty_returns_full.csv` - the project's own synthetic Nifty return
  series, exported from Python for R to read (R cannot call the Python
  synthetic generator directly).

## Results summary (see `docs/day14_validation_notes.md` for full detail)

1. **HMM (Baum-Welch EM, K=2, full series)**: the two independent
   implementations converge to DIFFERENT local optima (R's best
   log-likelihood 12077.19 vs Python's best-of-30 12073.49) - a genuine,
   expected finding given Day 4's own established weak-identifiability
   result, not a bug. The resulting filtered regime probabilities still
   agree reasonably in practice: 85.5% hard-label agreement, mean total
   variation distance 0.154.
2. **Split-conformal (A6.2)**: found a real, if modest, discrepancy
   (q_hat 0.9843 in R vs 0.9934 in Python on identical data) traced to
   R's generic `quantile()` type parameter using a different
   interpolation convention than numpy's `method="higher"`. Fixed by
   computing the exact order statistic numpy's formula uses directly;
   after the fix, R and Python match to the precision printed
   (q_hat, coverage, and mean set size all identical).
3. **Monte Carlo VaR/CVaR (A13.1/A13.2)**: given the identical model
   specification (same P, mu, sigma, initial regime distribution),
   VaR and CVaR from independent 10,000-path simulations (different RNG
   streams, R's Mersenne Twister vs numpy's PCG64) agree within 0.4-0.6
   percentage points - consistent with expected Monte Carlo sampling
   noise at this path count, not a discrepancy in the underlying
   computation.
