# Video Walkthrough Script: Raw Data to Investment Committee Artefact

Day 15 deliverable: Section D3 Deliverable 6 asks for a 10-minute
recorded demonstration video. This session cannot record video or
audio (stated plainly in `docs/EXIT_DOCUMENTATION.md`). What follows is
a timestamped script, built from the project's own real commands and
real output (`scripts/run_day14_e2e_validation.py`, run live during
Day 14), ready for a human to record directly: every number below is
what the pipeline actually produced, not a mockup.

**Target runtime: 10 minutes. Screen: terminal + the two figures named
below.**

---

### 0:00-0:45 - Open on the question, not the code

> "If you ask this engine 'what regime is the Indian market in right
> now, and what should a fund do about it', here is the honest answer
> it gives, end to end, from raw data to a number an Investment
> Committee can act on. I'm going to run the actual pipeline live, not
> a mockup."

Show: `PROJECT_PLAN.md` title and the Day 1-14 checklist, scrolled
past quickly to establish scale (14 working days, each with its own
tests and notebook).

### 0:45-2:00 - Raw data

Run:
```
python3 -c "
from src.data.loader import DataConfig, load_market_data
data = load_market_data(DataConfig(backend='synthetic', n_years=15.0, seed=7))
returns = data['nifty50']['close'].pct_change().dropna()
print(f'{len(returns)} days, {returns.index.min().date()} to {returns.index.max().date()}')
print(f'Return on 2024-06-14: {returns.loc[\"2024-06-14\"]:+.4%}')
"
```

> "This is synthetic data with a known ground truth, not real Nifty
> prices - stated once here, and repeated everywhere in this project's
> documentation that it matters. 3,779 trading days, 2011 to 2025.
> Today's target date, 14 June 2024, had a return of minus 1.03%."

### 2:00-3:30 - Feature engineering and the regime model

Run (or show the already-executed cells in `notebooks/03_frequentist_hmm.ipynb`):
```
python3 scripts/run_day14_e2e_validation.py
```

Let it run through stage [1] and [2] on screen, then pause on stage [3]:

> "This fits a 5-state Hidden Markov Model with 20 random restarts and
> keeps the best one - here, only 2 of 20 converged to something
> usable. That number matters: it's a direct, quantified measure of how
> fragile this specific fit is, and it goes into every downstream
> number as its lineage. On 14 June 2024, filtered through everything
> the model has seen up to and including that day: 79% Risk-On, 18%
> Transitional, 3% Late-Cycle, under 1% each for Post-Shock and
> Risk-Off."

### 3:30-4:15 - The honest gap: ensembling

Let stage [4] print on screen.

> "This project built a full Bayesian model averaging and stacking
> ensemble in Day 9 - and found, honestly, that it doesn't reliably
> beat its best single member. For today's specific date, which
> predates that ensemble's fitted window, this pipeline check uses the
> single HMM's own probability, and says so on screen rather than
> quietly substituting something else."

### 4:15-5:30 - Tilt recommendation

Let stage [5] print.

> "The tilt rule is built around something specific to this project's
> own regime definitions: Post-Shock, despite the name, is actually the
> HIGHEST-drift regime in the synthetic data - a recovery rally, not
> continued stress. A naive rule that de-risks on the name alone gets
> this backwards. Today's call: 79% Risk-On, conviction-scaled equity
> weight of 0.92 against a base of 0.70."

Show figure: `docs/artifacts/day13_backtest_cumulative.png` (the
2019-2024 overlay backtest) as supporting evidence for why this tilt
rule is used, captioned on screen: "synthetic price path, not real
Nifty - shown for methodology, not as a market forecast."

### 5:30-7:30 - Monte Carlo and the Investment Committee artefact

Let stage [6] print, all four IC statements on screen as they appear.

> "5,000-path Monte Carlo projection, conditioned on today's regime
> call. 90% one-year interval: minus 26% to plus 35%. 45% probability
> of a negative year. And this fourth line is the one worth pausing on: 'if
> the regime spends at least 45 of the next 90 days in Risk-Off,
> probability of that today is 0% - none of the 5,000 simulated paths
> meet it - there are too few simulated paths to give a reliable
> worst-case number.' The system is designed
> to say 'I don't have enough evidence' instead of making up a number
> when the evidence genuinely isn't there."

Show figure: `docs/artifacts/day13_mc_fan_chart.png`.

> "And every one of these numbers carries lineage: which model, which
> exact date range it was fit on, which random seed, how many restarts
> converged, how many simulated paths. Open `docs/IC_BRIEFING_TEMPLATE.md`
> to show the full structured version of this."

### 7:30-9:00 - What this project found, not just what it built

> "The most valuable part of 14 days of work isn't the models - it's
> what they revealed when checked honestly. Three examples, fast:"

1. "BOCPD, the changepoint detector, independently found three REAL
   historical market events in real data - but missed two of three
   real crises it was tested against in a later case study, for
   specific, traceable reasons each time." (Show
   `docs/artifacts/day12_bocpd_vix_signal.png`.)
2. "Cross-validating this project's HMM in Python against an
   independent implementation in R found them converging to different
   answers on the same data - not a bug, but direct evidence of how
   weakly identified this problem genuinely is." (Show the comparison
   table in `docs/day14_validation_notes.md`.)
3. "And a peer review of this project's OWN work, done on the last
   day, caught a real documentation imprecision and a genuinely flaky
   pre-existing test - the project checks itself, not just the
   market."

### 9:00-9:45 - The gamification framing

> "This repository is called Regime Lab for a reason: the project
> brief's own Part B describes a gamified training platform by that
> exact name, with a 9-level campaign that maps almost exactly onto
> what got built, level by level, across these 14 days - HMM, Bayesian
> HMM, multivariate RS-VAR, Bayesian deep learning, foundation models,
> ensembling, and the audit-defensibility level this video is
> demonstrating right now." (Show the mapping slide from
> `REGIME_LAB_PRESENTATION.pptx`.)

### 9:45-10:00 - Close

> "Everything shown here is reproducible from this repository: 156
> passing tests, 11 executed notebooks, 18 commits, and every number
> checked against real output before it went into documentation. Thank
> you."

---

*Script source: `docs/VIDEO_WALKTHROUGH_SCRIPT.md`. Built from a real
execution of `scripts/run_day14_e2e_validation.py` on 2024-06-14;
re-run that script for a current date before recording to get fresh,
live numbers rather than reading last session's output on screen.*
