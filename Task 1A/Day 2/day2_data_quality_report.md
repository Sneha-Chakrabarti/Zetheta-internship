# Day 2: data quality report

Scope: the synthetic regime-switching panel (`src/data/synthetic.py`),
cross-checked against the two real partial series recovered from GitHub
(`data/raw/real_partial/`, see `DATA_SOURCES.md`). Full detail and code in
`notebooks/01_data_exploration.ipynb`.

## Missing values

None, in any of the nine synthetic series. Expected: the generator never
introduces gaps.

## Business-day / calendar gaps

Synthetic panel: zero gaps longer than 4 calendar days, since it is built
directly on a business-day calendar with no holiday calendar layered on.
Real Nifty 50 (2015-2019): four gaps of 5 calendar days, at Holi/Good
Friday-adjacent weeks in April 2015, March 2016, April 2016 and March
2018. These are genuine NSE trading holidays, not missing data, and are
exactly the class of gap this check exists to surface once real data
replaces the synthetic backend: a business-day calendar alone will not
know about them.

## Large-jump flags (proxy for shocks and corporate actions)

Synthetic Nifty, 8% threshold: zero days flagged over 15 years, consistent
with the regime specs (no single-day discontinuity is built in beyond the
changepoint-day VIX spike).

Real Nifty 2015-2019, 6% threshold: one day flagged, 24 August 2015, log
return -6.1%. This is the real "Black Monday" global selloff triggered by
the Chinese yuan devaluation and China growth concerns; correctly
recovered by the check, and a useful sign the method works before it is
relied on for real data later in the project.

## Volatility ordering bug, found and fixed

The first version of the synthetic generator scaled mid/small-cap total
volatility by the correlation-mixing weight (0.75, 0.65) instead of by
beta, which made mid/small-cap *less* volatile than large-cap: backwards.
Caught by inspecting the actual output stats before writing the notebook
summary (annualised std was Nifty 0.163, Midcap 0.142, Smallcap 0.150),
not by assuming the code matched its comment. Fixed in
`src/data/synthetic.py`; corrected ordering (large 0.163 < mid 0.181 <
small 0.207 annualised) is now covered by a regression test in
`tests/test_data_loader.py`.

## Known synthetic-data limitation

Nifty-return / VIX-change rolling correlation oscillates around zero
(mean 0.016) rather than showing the persistent negative correlation
(leverage effect) real markets exhibit. The synthetic VIX is generated as
its own regime-dependent process with no same-day coupling to the Nifty
return. Not a defect to fix for pipeline-development purposes, but a gap
to remember: any leverage-effect feature built later will show nothing
on synthetic data and needs real VIX data to test properly.

## Recommendation

Proceed to Day 3 (feature engineering) on the synthetic panel. Switch
`DataConfig(backend="csv")` once real data covering the nine required
series is available; no other code change needed.
