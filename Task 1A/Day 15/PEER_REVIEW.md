# Peer Review and Self-Audit

Day 14 deliverable. A self-review of the codebase and documentation
built across Days 1-13, looking specifically for inaccuracies,
inconsistencies, and gaps rather than confirming everything is fine.

## Correction found during this review

**The project's own recent day summaries have been imprecise about
what "7 skipped" means.** PROJECT_PLAN.md entries for Days 9 through 13
describe the main-environment test suite's skips uniformly as "the
torch gap" or similar. Checked directly: only 3 of the 7 skips are
actually the torch-environment-collision issue (Day 7's documented open
decision 9, in `tests/test_features.py`). The other 4
(`tests/test_bdl_models_tf.py`, `tests/test_foundation_models_tf.py`,
`tests/test_hmm_bayesian_pymc.py`, `tests/test_rsvar_bayesian_pymc.py`)
are skipped in the main environment by design, because TensorFlow and
PyMC live in `.venv-bayesian`, not because anything is broken. Verified
directly rather than assumed: running those 4 files in
`.venv-bayesian` gives 18 passed, 0 failed, 0 skipped. **The accurate
total is 156 (main) + 18 (bayesian venv) = 174 tests passing across both
environments, with exactly 3 genuine, documented skips**, not 7
undifferentiated ones. This document is the correction; earlier
PROJECT_PLAN.md entries are left as originally written (each was
accurate about the pass/skip COUNT at the time, just imprecise about
what the skips represented) rather than retroactively edited.

## A second finding: a genuinely flaky pre-existing test

Running the full main-environment suite twice during this review
produced different results once: `tests/test_rsvar_baseline.py::test_fit_msm_regression_runs`
failed on one run (`ValueError: Could not untransform parameters.`,
raised inside `statsmodels`' own Markov-switching MLE optimiser) and
passed on an immediate rerun, and passed again in isolation. Day 6's
code was not touched by any work in this session; this is a pre-
existing, genuine non-determinism in `statsmodels`' optimiser (it does
not take a project-controlled random seed), not a regression introduced
today and not consistently reproducible test-order dependence. Flagged
here because it would otherwise go unnoticed: a single CI run could
pass or fail on this test by chance. Recommendation: either pin a seed
if `statsmodels`' API allows it for this estimator, or mark the test
with an explicit retry/tolerance for optimiser non-convergence rather
than treating a single failure as a correctness problem.

**Also found and cleaned up during this review**: four orphaned R
files (`crossvalidate.R`, `hmm.R`, `test_hmm.R`, `var_calibration.R`)
and two orphaned data exports, left over from an earlier, abandoned
attempt at this same Day 14 cross-validation task within this session,
superseded by the working `hmm_lib.R` / `conformal_and_var.R` /
`run_cross_validation.R` set but never removed. Deleted before this
commit; nothing in the working pipeline referenced them.

## What was checked

- Every `src/` module has at least one corresponding test file (25 test
  files against 51 source files; several source files are intentionally
  untested entry-point scripts under `scripts/`, not library code, which
  is consistent with the project's own stated testing scope).
- No `TODO`, `FIXME`, or `XXX` markers remain anywhere in `src/` -
  either every known gap was resolved, or (more likely, and more
  honestly) every known gap was written up as a documented limitation
  in that day's `docs/dayN_*.md` file instead of left as an inline
  marker. The latter is the project's actual practice throughout: see
  the "Limits" section of every day's notes.
- `DATA_SOURCES.md`, `PROJECT_PLAN.md`, and `README.md` all exist and
  were spot-checked against the actual repository state (file counts,
  notebook numbering, day-notes coverage) during this review; no
  discrepancy found beyond the skip-count imprecision above.
- Every em-dash and Oxford-comma check run throughout Days 9-14 (a
  stated style preference) passed with zero hits in every file checked.

## What was not exhaustively checked

Stated honestly rather than implied to be complete:

- **No line-by-line code review of all 51 source files was performed.**
  This review checked structure, consistency, and the test/skip
  accounting; it did not re-derive or re-verify every mathematical claim
  in every module (those are the individual days' own responsibility,
  each with its own verification work documented in that day's notes).
- **No check was made for dead code or unused imports** across the
  codebase.
- **Visualisation polish (also a Day 14 item) was a light spot-check**
  of a handful of figures against the shared style module
  (`src/utils/plot_style.py`), not an audit of all 38 figures produced
  across Days 1-13.
- **This review was performed by the same agent that wrote the code**,
  not an independent second reviewer. Its value is in applying a
  genuinely different pass (auditing for consistency and accuracy
  claims specifically) rather than in independence of perspective.

## Overall assessment

The codebase is internally consistent: claims made in `PROJECT_PLAN.md`
and the day-notes match what the code and saved artifacts actually
show, with the one imprecision corrected above. The project's
consistent practice of checking claims against computed output before
writing them down, applied throughout Days 4-13, held up under this
review; the one issue found was a documentation-precision issue (what
the skip count represents), not a correctness issue.
