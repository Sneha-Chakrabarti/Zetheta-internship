# Exit Documentation and Knowledge Transfer

Day 15 deliverable. Written for whoever picks this project up next.

## What this document is honest about upfront

Three of Day 15's six literal bullet points require actions outside
what this working session can actually perform:

- **"Transfer GitHub repository ownership to @ZethetaIntern"**: there
  is no real external GitHub repository connected to this working
  directory to transfer. What exists is a local git repository
  (`/home/claude/regime-lab`, 18 commits) with full history. Transfer
  requires a human with GitHub account access to push this history to
  a real remote and change ownership there.
- **"Upload all deliverables to shared drive"**: no shared drive is
  connected to this session. The repository's files (code, docs,
  notebooks, this documentation) are the deliverables; a human needs to
  perform the actual upload.
- **"Record a 10-minute demonstration video"**: this session cannot
  record video or audio. `docs/VIDEO_WALKTHROUGH_SCRIPT.md` is a
  timestamped script for that video, built from the project's own real
  commands and outputs, ready for a human to record from directly.

Flagging this is the honest version of this deliverable, not a
workaround for it: pretending to have done any of the three above would
be a false claim. What follows is everything that COULD be done to
leave the project in the best possible state for handover.

## Repository state at handover

- **18 commits**, Days 1-14 of the 15-day plan, each a working,
  tested, documented unit.
- **Main test suite**: 156 passed, 7 skipped, 0 failed (one test is a
  known, confirmed non-deterministic flake in Day 6's `statsmodels`-
  based MSM fit - see `docs/PEER_REVIEW.md` for the investigation).
  Bayesian-venv suite (`.venv-bayesian`): 18 passed for the 4 files that
  need PyMC/TensorFlow.
- **11 executed notebooks** (`notebooks/01_*.ipynb` through
  `11_*.ipynb`), each with 0 execution errors and every narrative claim
  checked against its own printed output before being written.
- **14 day-notes files** (`docs/dayN_*.md`), one per working day,
  each with its own "Limits" section stating what was NOT done or NOT
  verified, not just what was.

## How to pick this up

1. **Read `docs/EXECUTIVE_SUMMARY.md` first**, not `PROJECT_PLAN.md`.
   The summary states what the evidence actually supports; the plan is
   a day-by-day task log.
2. **For any specific technical claim, the source is the day-notes
   file plus its executed notebook**, not this document or the
   executive summary - both of those are summaries of summaries, and
   the notebooks are where every number was actually checked.
3. **Two environments exist and matter**: the main environment
   (numpy/scipy/hmmlearn/sklearn) and `.venv-bayesian`
   (PyMC/TensorFlow/ArviZ). `PROJECT_PLAN.md`'s "Environment
   Architecture" section and each affected day's notes say which.
4. **`r/` is a separate, working R cross-validation codebase**
   (Day 14), not a full port of the Python engine - see `r/README.md`
   for exactly what it covers and why (CRAN is unreachable from this
   sandbox, so the scope was adapted honestly rather than silently
   reduced).

## Immediate next steps for a human continuing this work

In priority order, based on what the project's own findings point to
as most valuable (see `docs/EXECUTIVE_SUMMARY.md`'s "What this means
for deployment" section for the full reasoning):

1. **Increase the Bayesian RS-VAR's posterior sample size** (currently
   320 draws, flagged unreliable by PSIS-LOO in Day 9). This is the
   component most directly relevant to three of Day 12's four case
   studies (IL&FS, Taper Tantrum, COVID all have a cross-asset
   signature the RS-VAR is built to capture but its current posterior
   cannot yet support confident claims about).
2. **Build the event-calendar-aware conviction dampener** Section C5.3
   of the task brief describes. Does not exist anywhere in this
   project; Day 12's election case study shows concretely why a
   reactive detector (BOCPD) cannot substitute for it.
3. **Recalibrate BOCPD's detection threshold per input series with an
   explicit false-positive budget**, rather than reusing one series'
   validated threshold on another - Day 12's central, most actionable
   finding.
4. **Real data acquisition**: this project trains and validates almost
   entirely on synthetic data; only two small real datasets exist
   (`data/raw/real_partial/`, Nifty 2015-2019 and VIX 2020-2026). A
   production system needs a real, continuous Nifty/breadth/flow data
   pipeline - the loader (`src/data/loader.py`) already has a
   `backend` switch for this, but no real backend has been
   implemented.
5. **The remaining Section D3 deliverables not built in this 15-day
   window**: a full 40-page consolidated report (the day-notes are the
   substance; `docs/REPORT_INDEX.md` maps them to the required
   structure but does not merge them into one document), and the R
   codebase's remaining scope beyond the three cross-checks built in
   Day 14 (a full depmixS4/MSwM-equivalent, Stan-based Bayesian regime
   model, and R-side changepoint detection, blocked here by CRAN being
   unreachable - a human with a different network environment could
   complete these directly).

## Contact and provenance

All work in this repository was produced in a single extended working
session (Claude, Anthropic), Days 1 through 14 of the stated 15-day
plan, under the git identity `Sneha Chakrabarti <sneha@regime-lab.local>`
for commit continuity. Every commit message documents what was built,
what was found, and what was fixed for that day; `git log` is itself
part of the project's documentation, not just version history.
