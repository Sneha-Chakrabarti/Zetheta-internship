---
title: "Indian Market Case Study Research and Replication"
subtitle: "Regime Lab, Task 1A, Day 12 Deliverable"
author: "Zetheta Algorithms Internship"
date: "October 2026"
---

# Indian Market Case Study Research and Replication

## Executive summary

This report researches, documents and quantifies four Indian market
episodes from Section C of the project brief, cross-checks the brief's
own figures against independently sourced data, and assesses honestly
how the regime-detection engine built across Days 1 to 11 of this
project would have performed in each episode.

Four of the five available case studies were chosen: the 2020 COVID
regime transition, the 2018 IL&FS credit shock, the 2013 Taper Tantrum,
and the 2024 election verdict event. The 2017-2018 mid-cap rally and
correction was set aside because its core lesson (large-cap calm masking
mid- and small-cap stress) substantially duplicates the IL&FS case; the
other four cover four genuinely distinct failure modes of a naive,
single-index regime engine.

Two real datasets, already part of this project since Day 2, ground
much of the analysis directly rather than relying only on cited
figures: real Nifty 50 large-cap OHLC for 2015-2019 (covering IL&FS in
full), and real India VIX for 2020-2026 (covering COVID and the 2024
election in full). Where neither applies (the 2013 episode, and the
mid- and small-cap side of IL&FS), figures are drawn from published,
cited sources and checked against the brief's own numbers rather than
copied uncritically.

The central, most load-bearing finding is not flattering to this
project's own engine, and is reported as such. Day 10's Bayesian
online changepoint detector, run on real Nifty 50 returns, **does not
fire anywhere near the actual IL&FS default date** (14 September 2018):
the large-cap index's daily returns that week were unremarkable in
magnitude, because the shock was a gradual two-month grind, not a sharp
break. The same detector, tested fresh on real VIX data for this
report, also **misses the entire COVID volatility spike at its
originally validated 0.5 detection threshold**, and only catches it, and
the 2024 election, at a substantially lower threshold (0.10) that was
not calibrated or validated in Day 10's work. This is a genuine,
specific, previously undocumented gap in this project's sequential
inference layer, found by testing it against new real episodes rather
than assumed to generalise from Day 10's synthetic validation.

## Methodology and data sources

| Case study | Primary source | Verification |
|---|---|---|
| 2020 COVID | Real VIX (2020-2026), this project | Nifty price levels cross-checked against five independent published sources |
| 2018 IL&FS | Real Nifty 50 large-cap OHLC (2015-2019), this project | Large-cap drawdown computed directly; mid/small-cap figures cited |
| 2013 Taper Tantrum | Published sources (no real data coverage) | INR figure cross-checked against a BIS central-bank speech; equity figures against contemporary news reports |
| 2024 Election | Real VIX (2020-2026), this project | Nifty price-move figures cross-checked against six independent published sources |

All P&L tables in this report are **illustrative**, in the same spirit
as Section C1.3 of the project brief: simple, transparent arithmetic on
a small number of named hypothetical allocation rules, not a backtest of
any model built in this project. Where this project's own engine
(frequentist/Bayesian HMM, RS-VAR, BDL models, ensembling, BOCPD,
conformal calibration) is assessed, the assessment draws on the actual,
previously documented results from Days 4 to 11, including their
established limitations, not a re-run with hindsight-improved settings.

---

## Case Study 1: The 2020 COVID Regime Transition

### Background, verified

Nifty 50 peaked near 12,430 on 20 January 2020 (sources place the exact
peak date and level in a narrow band: 12,168-12,431, depending on
whether a closing or intraday reference is used) and fell to a trough of
7,511-7,610 on 23 March 2020, a peak-to-trough decline most
independently reported in the 29-40% range depending on the exact
reference points used. The project brief's own figures (12,201 to
7,610, -37.6%) sit squarely inside this verified range.

India VIX is directly verifiable from this project's own real data: it
closed at **13.62 on 14 February 2020**, climbing to a closing peak of
**83.61 on 24 March 2020**. This is close to, but not identical to, the
brief's claimed peak of 86.6 - a modest discrepancy worth noting rather
than silently adopting the higher figure, and plausibly explained by
intraday-versus-closing measurement. VIX was **23.24 on 28 February**
in the real data, below the brief's claimed "above 25 by 28 February",
though the broader characterisation (a sharp rise from the low-teens to
the mid-20s within two weeks) holds.

Figure 1 shows the real VIX path.

![COVID regime transition: real India VIX, Jan-Jun 2020](artifacts/day12_covid_vix.png)

### How this project's engine would have performed: assessed honestly

**BOCPD, re-tested on real VIX data for this report, misses the event
at its Day-10-validated threshold.** Day 10 validated a detection rule
(`short_run_length_probability > 0.5`) that cleanly separated two true
breaks from routine noise on a synthetic series. Applied fresh to real
VIX log-changes around the actual COVID spike, that same threshold
**never fires** in the window Section A9.2 of the brief specifically
calls out (28 February to late March 2020) - the signal is visibly
elevated (peaking near 0.30, against a roughly 0.02-0.05 baseline) but
stays well under 0.5 throughout. A lower threshold (0.10), not
previously validated anywhere in this project, would have caught it
with four distinct crossings inside the window - at the cost of a higher
baseline false-positive rate across the full real VIX series (5.3% of
all days flagged, against an essentially negligible rate at the 0.5
threshold on Day 10's synthetic test). Figure 2 shows this directly.

![BOCPD short-run-length signal on real VIX, Day 10's threshold vs. a lower one, both the COVID and election windows](artifacts/day12_bocpd_vix_signal.png)

This is a genuine gap, not a restatement of a known limitation: Day 10
never tested BOCPD against real VIX data, only real Nifty returns
(where it found three *different*, independently verified real matches
- August 2015, November 2016, September 2019 - none of which are this
episode). The threshold that worked on a clean synthetic series does
not transfer to this real series without recalibration, and the
project's Day 9/10 documentation did not previously establish what
threshold real deployment would need.

**The frequentist and Bayesian HMMs (Days 4-5) offer little help here
either, for a reason already established, not a new one.** Day 4 found
that a single-feature (returns-only) 5-state HMM is weakly identified on
this project's own synthetic data (BIC preferred 3 states; ground-truth
match rate around 40%). A VIX-driven shock of this speed and magnitude
is exactly the kind of sharp break a returns-only model is least
equipped to contextualise in real time, since by the time enough
post-break observations accumulate to shift a smoothed regime
probability, the acute phase is often already over.

**The RS-VAR (Day 6) is the component of this project's stack most
directly aimed at this failure mode**, since VIX is one of its six input
dimensions. Day 6 is explicit that its own posterior was severely
under-sampled (320 draws, PSIS-LOO reliability warnings) and that its
regime-conditional outputs carried credible intervals too wide to
support confident claims. The honest assessment is that the
*architecture* targets exactly this problem; the *posterior quality*
delivered in this project to date does not yet support claiming it would
have worked.

### Illustrative P&L

| Allocation strategy | Drawdown (Mar 2020) | Notes |
|---|---|---|
| 100% equity (passive) | -38% (verified range: -29% to -40%) | Buy-and-hold benchmark |
| Static 80/20 equity/cash | approx. -30% | Linear scaling of the passive drawdown |
| Regime overlay, BOCPD at 0.5 threshold | approx. -38% (no de-risking signal fires) | This project's validated threshold does not trigger in this window |
| Regime overlay, BOCPD at 0.10 threshold | approx. -28% to -30%, assuming a 25% equity trim on the first 28 Feb crossing | Illustrative only; this threshold was never validated against a false-positive budget |

The last row is deliberately hedged: it is not a claim that this
project's engine, as actually built and validated, would have produced
this outcome. It shows what a *correctly threshold-calibrated* version
of Day 10's own method could plausibly have achieved, which is a
different and more defensible statement.

---

## Case Study 2: The 2018 IL&FS Credit Shock

### Background, verified directly from real data

IL&FS Financial Services defaulted on commercial paper on 14 September
2018. This project's own real Nifty 50 large-cap data gives a precise,
directly computed answer for the large-cap side: **peak of 11,738.5 on
28 August 2018, trough of 10,030.0 on 26 October 2018, a drawdown of
-14.55%** - closely matching the brief's own "~14%" claim. Published
figures for the mid- and small-cap segments, not covered by this
project's real data, report roughly -25% and -35% respectively over the
same window; these are cited, not independently re-derived.

Figure 3 shows the real large-cap path.

![IL&FS credit shock: real large-cap Nifty 50, Jul-Nov 2018](artifacts/day12_ilfs_nifty.png)

**The large-cap decline was gradual, not sharp, and this is the whole
story for why detection failed.** Daily log returns in the ten trading
days around the actual default date (10-20 September 2018) ranged from
-1.33% to +1.27%, all within about 1.5 standard deviations of the
index's own full-sample daily volatility (0.86%). The index's peak
(28 August) predates the default itself by more than two weeks, and the
decline unfolds over roughly two months, not two days.

### How this project's engine would have performed: the central finding of this report

**Day 10's BOCPD, already run on this exact real data, does not fire
anywhere near 14 September 2018.** The nearest detected changepoint
clusters are 2 to 6 February 2018 (more than seven months earlier,
plausibly related to the Union Budget and the global "Volmageddon" VIX
spike, neither confirmed) and 20 to 24 September 2019 (a full year
later, independently confirmed as the corporate tax cut rally). Nothing
fires in the entire autumn of 2018. This was already computed in Day
10's own validation run; this report's contribution is recognising its
direct relevance to Section C2 and stating the connection explicitly,
rather than treating Day 10's "low recall, explained by subtle
transitions" finding as a separate, abstract result.

This is precisely the failure mode Section C2 of the brief describes in
the abstract, now demonstrated concretely with this project's own tools
on real data: **a regime engine relying on large-cap returns alone,
including this project's own sequential-inference layer, would have
produced no detectable signal during the IL&FS episode**, because the
large-cap index itself carried no sharp signal to detect. The stress was
real, severe, and entirely absent from the one series (large-cap daily
returns) that most of this project's models (the frequentist and
Bayesian HMMs, Day 4-5) are built on.

**This is the single strongest empirical argument in this project's own
results for Day 6's RS-VAR**, which was built specifically to use
breadth, credit spreads and cap-segmented features rather than headline
returns alone - the project brief makes this argument in the abstract
(Section C2.2); this report is able to make it concretely, using data
and a model this project actually has.

### Illustrative P&L

| Allocation strategy | Large-cap drawdown | Mid/small-cap drawdown (cited) | Notes |
|---|---|---|---|
| 100% large-cap equity | -14.6% (real data) | n/a | Would have looked like a routine correction |
| 100% mid/small-cap equity | n/a | -25% to -35% (cited) | Acute stress, invisible to a large-cap-only engine |
| Blended 60/40 large/mid-small | approx. -21% | | Illustrative weighted average |
| Regime overlay (large-cap signal only) | -14.6% (no distinct signal fires) | -25% to -35% (unmitigated) | The large-cap-only engine offers no protection to the mid/small-cap sleeve |

---

## Case Study 3: The 2013 Taper Tantrum

### Background, verified against an independent source

Following Fed Chair Bernanke's May 2013 tapering remarks, the Indian
rupee depreciated sharply. A BIS (Bank for International Settlements)
central-bank source independently confirms the brief's own figures
closely: "the rupee went into a tailspin and depreciated sharply by over
19 per cent, touching a historic low of 68.85 on August 28, 2013"
during the May-August 2013 episode - consistent with the brief's "INR
fell from approximately 55 to 68" (a 19-20% depreciation). Contemporary
market reports from the period (Business Standard, Moneylife) describe
Nifty moves in the range of single-digit percentage points on individual
sessions (for example, a 2.9% single-day fall on 20 June 2013, the
largest one-day move in nearly two years at the time) rather than a
sustained crash, consistent with the brief's characterisation of a
"moderate" ~10% equity correction against acute, EM-specific currency
and flow stress.

This project holds no real market data for 2013; both the frequentist
HMM (Day 4) and the RS-VAR (Day 6) were fit on synthetic data standing
in for this period, not on actual 2013 observations.

### How this project's engine would have performed

This is the one case study in this report where no part of this
project's engine can be tested against real contemporaneous data at
all, and that absence is itself the finding worth stating plainly: a
regime engine trained and validated entirely on a single market's
equity-return history, as the HMM and BDL components of this project
substantially are, has no mechanism to have learned anything
specific to this kind of EM-currency-driven stress, because the stress
signature here lives almost entirely in INR, FII flows and the credit
curve, series this project's HMM and BDL components (Days 4, 5, 7, 8)
do not use as inputs at all.

The RS-VAR (Day 6) is again the component built to use the relevant
series (INR return, gilt yield change, FII flow are three of its six
dimensions) and so is, in principle, the right tool. Day 6's own
reported limitation applies with full force here: the posterior
delivered in this project is not yet precise enough to support a
specific claim about how it would have classified this particular
episode, synthetic or real.

### Illustrative P&L

| Exposure | Approximate move (May-Aug 2013) | Notes |
|---|---|---|
| INR (unhedged foreign investor, equity flat) | -19% to -20% (BIS-confirmed) | Currency alone, equity position unchanged |
| Nifty (INR terms) | approx. -10% (cited) | The brief's "moderate Late-Cycle correction" framing |
| Nifty (USD terms, unhedged) | approx. -28% to -30% (combined) | Illustrative: equity decline compounded with INR depreciation |
| Regime overlay using equity signal only | approx. -10% (no distinct signal from equity alone) | The EM-specific stress channel is invisible to an equity-only engine by construction |

---

## Case Study 4: The 2024 Election Verdict Event

### Background, verified directly from real data and independent sources

On 4 June 2024, with the ruling alliance's seat count falling short of
exit-poll projections, Nifty fell 5.93% (1,379 points) to close at
21,884.50, with an intraday low around 8.5% below the prior close -
independently confirmed across six published sources, closely matching
the brief's own "~5.9%" figure.

This project's own real VIX data adds a nuance the brief's simplified
"spiked from 14 to 27" framing does not capture: **VIX had already
climbed from a April 2024 baseline of 10-14 to 21-24 in the final week
of May**, on exit-poll-driven anticipation, before the acute spike to a
closing value of 26.75 on the result day itself (consistent with an
intraday peak near 27-28, as independently reported). Most of the
headline "14 to 27" move had, in other words, already happened before
the result was known; the sharpest, most acute leg was the final one.
VIX then fell rapidly: 18.9 by 5 June, 16.8 by 6 June, 12.8 by 14 June -
back near its pre-anticipation baseline within roughly two weeks,
consistent with Section C5's own framing of this as an event-driven
shock, not a regime transition.

Figures 2 (right panel) and 4 show the real VIX path and this project's
BOCPD signal on it.

![2024 election verdict event: real India VIX, Apr-Jun 2024](artifacts/day12_election_vix.png)

### How this project's engine would have performed

**BOCPD, re-tested fresh on real VIX data for this report, DOES fire
close to the event, but only at the same lower, previously unvalidated
threshold identified in the COVID case.** The `short_run_length_probability`
signal peaks just after 4 June (visible in Figure 2's right panel),
crossing 0.10 within the window but never approaching Day 10's original
0.5 threshold. The same honest caveat applies as in Case Study 1: this
is a real, positive result for a threshold this project has not actually
validated for a false-positive budget, not a validated capability.

**This episode is also the clearest test of Section A6.5's event-
calendar-aware conviction dampening (Section C5.3 of the brief), which
this project has not implemented.** Day 11's conformal prediction work
(ACI, Mondrian) addresses *statistical* miscalibration and distribution
shift; it has no mechanism for *known, scheduled* discontinuities like a
declared election count date. A system that knew 4 June 2024 was a
scheduled event could have halved conviction in the preceding days,
exactly as Section C5.3 proposes, and widened Day 11's conformal
intervals pre-emptively rather than reactively. This project's Day 11
work is reactive by construction (it responds to realised nonconformity
scores); the event-calendar layer the brief describes is a different,
complementary mechanism this project has not yet built.

**Day 9's ensembling work is relevant in a specific, limited way.** The
combined output contract (Section A10.4, built Day 9) already has a
`changepoint_flag` field reserved and explicitly set to `None`, pending
this exact kind of work. This episode is a concrete example of what
would populate it.

### Illustrative P&L

| Allocation strategy | Single-day move (4 Jun 2024) | 2-week recovery | Notes |
|---|---|---|---|
| 100% equity (passive, no adjustment) | -5.9% | Most of the loss recovered (cited, not independently re-verified to an exact figure) | Buy-hold benchmark |
| Static 80/20 equity/cash | approx. -4.7% | Partial | Standard tactical strategy |
| Event-calendar-aware (conviction halved, 5 days pre-event) | approx. -3.0% to -4.4%, assuming a 25-50% pre-trim | Full, plus avoided re-entry cost | Illustrative application of Section C5.3's own proposed rule |
| BOCPD-reactive overlay (0.10 threshold, fires day after) | -5.9% realised, de-risks AFTER the move | Full | A reactive detector cannot protect against a single-day gap; it can only avoid being caught offside for a continuation that, here, did not come |

The fourth row makes a point worth stating plainly: for a genuine
one-day event shock that resolves quickly, a *reactive* changepoint
detector (BOCPD, as built) is structurally the wrong tool. It can only
help once the move has already happened. Section C5.3's calendar-aware
pre-emptive dampening is the correct complementary mechanism for exactly
this reason, and this project does not yet have it.

---

## Cross-cutting synthesis: how would the trained ensemble engine have performed

Read across all four episodes, honestly rather than charitably:

1. **The single-feature, returns-only components of this project's
   stack (the frequentist and Bayesian HMMs, Days 4-5) would likely have
   underperformed in three of the four episodes** (IL&FS, the Taper
   Tantrum, and the anticipatory phase of the 2024 election), because the
   relevant stress signal in each case lived substantially or entirely
   outside large-cap equity returns. COVID is the partial exception: a
   sufficiently sharp equity move would eventually register, though with
   the lag this project's own Day 4 findings (weak state separability)
   would predict.

2. **The multivariate RS-VAR (Day 6) is the architecturally correct tool
   for three of the four episodes** (IL&FS, Taper Tantrum, COVID, all of
   which have a cross-asset or cap-segment signature its six input
   dimensions are built to capture), but this project's own Day 6 results
   do not support a confident claim that its *current, under-sampled*
   posterior would have called any of them correctly. This is a capacity
   gap in compute, documented honestly in Day 6, not a design flaw.

3. **BOCPD (Day 10), this project's one sequential, online-capable
   detector, is the component most directly tested against real episodes
   in this report, and the results are mixed in a specific, informative
   way.** It independently found three real historical events on its own
   terms in Day 10 (none of which are in this report's four). Tested
   fresh against two more real episodes for this report, it required a
   substantially lower, previously unvalidated threshold to fire at all,
   and missed one entirely (IL&FS) regardless of threshold, because the
   underlying large-cap return series simply did not carry the signal.

4. **Day 9's ensemble and Day 11's conformal layer would not have
   changed any of this materially.** Both operate on the probability
   outputs of the member models; if no member model's inputs contain the
   relevant signal (IL&FS, Taper Tantrum), no combination or calibration
   step downstream can recover it. Day 11's own finding, that
   calibration quality itself can drift over time in ways that look like
   exactly this kind of regime-dependent failure, is a second, compounding
   concern for episodes like these.

5. **The event-calendar-aware mechanism Section C5.3 describes is
   entirely absent from this project**, and the 2024 election case study
   is the clearest illustration of why that gap matters: a reactive
   detector is the wrong primary tool for a known, scheduled
   discontinuity.

## Proposed regime-specific adaptations

Each proposal below is tied to a specific finding in this report or in
the cited prior day's work, not a generic best practice.

**1. Add real breadth and cap-segmented features (Day 3, Day 6).** The
IL&FS case study is this project's clearest evidence that large-cap
returns alone miss acute stress. Day 3 already documents this exact gap
(breadth features not built, real constituent data unavailable); this
report adds a concrete, quantified real-data demonstration of its cost.

**2. Recalibrate BOCPD's detection threshold per series, with an
explicit false-positive budget, rather than reusing Day 10's synthetic-
validated 0.5 threshold on new data.** This report's central finding:
the same threshold that cleanly separated signal from noise on a clean
synthetic series required lowering by 5x to detect two of three real
episodes tested here, at a materially higher baseline alarm rate (5.3%
of days, against near-zero on the synthetic case). A production system
needs this trade-off made explicitly and validated per input series, not
inherited from a different series' validation.

**3. Build the event-calendar-aware conviction dampener (Section
C5.3), currently entirely unbuilt.** The 2024 election case study shows
concretely why a reactive detector cannot substitute for this: BOCPD can
only respond after a scheduled discontinuity has already moved the
market. This slots directly into the `changepoint_flag` field Day 9's
output contract already reserves for it.

**4. Prioritise RS-VAR posterior sample size over adding further model
variety.** Three of this report's four case studies (IL&FS, Taper
Tantrum, COVID) have their signal concentrated in the cross-asset
dimensions the RS-VAR already models. Day 6 and Day 9 both independently
found its posterior under-sampled; this report adds three concrete
historical episodes where a better-sampled RS-VAR posterior, not a new
model, would plausibly matter most.

**5. Treat gradual, grinding declines (IL&FS) and sharp, acute shocks
(COVID, the 2024 election) as genuinely different detection problems,**
not variations on one. BOCPD-style changepoint detection is well-suited
to the latter and poorly suited to the former by construction (a
changepoint detector looks for a discontinuity; IL&FS had none in
large-cap returns). A gradual-decline detector would need a different
statistic, such as a rolling drawdown-versus-volatility ratio, not a
better-tuned changepoint test.

## Limitations of this report

- P&L tables throughout are illustrative, simple-arithmetic
  constructions in the style of the brief's own Section C1.3, not
  backtests of any model actually built in this project.
- Real data coverage is partial: large-cap Nifty 50 only for 2015-2019,
  VIX only for 2020-2026. The 2013 episode and the mid/small-cap side of
  IL&FS rely entirely on cited published figures, cross-checked against
  at least one independent source each but not independently
  recomputed.
- The lower BOCPD threshold (0.10) identified in Cases 1 and 4 is
  reported honestly as *unvalidated*: it was not tested against a
  broader set of real non-event periods beyond the baseline alarm-rate
  check in this report, and should not be read as a recommended
  production setting without that further work.
- This report does not re-run or retrain any Day 1-11 model; it applies
  already-built tools (BOCPD) to new real data, and reasons about the
  others from their documented prior results.

## Sources

- BIS (Bank for International Settlements): Gurumoorthy Mahalingam,
  "Some thoughts on forex markets in India", India Treasury Summit,
  Mumbai, 25 February 2015 - 2013 INR depreciation figure.
- Business Standard, Moneylife, 5paisa, Outlook Business: contemporary
  and retrospective reporting on the 4 June 2024 election market
  reaction and the 2013 Taper Tantrum equity moves.
- Upstox, Vestai, BackTestIndia, Grip Invest, 1StopInvestment, Moneyvesta,
  Inves21: independent retrospective figures for the 2020 COVID
  drawdown, cross-checked for a consensus range rather than a single
  adopted figure.
- This project's own real data: `data/raw/real_partial/nifty50_2015_2019.csv`,
  `data/raw/real_partial/india_vix_2020_2026.csv` (see `DATA_SOURCES.md`).
- This project's own prior results: `docs/day4_hmm_notes.md`,
  `docs/day6_rsvar_notes.md`, `docs/day9_ensembling_notes.md`,
  `docs/day10_sequential_notes.md`, `docs/day11_conformal_notes.md`.
