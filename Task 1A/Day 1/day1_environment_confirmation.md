# Day 1: environment confirmation

Core tier (`requirements/core.txt`) installed clean, versions in
`docs/artifacts/day1_installed_versions.txt`; all meet the minimums in
Section E1 of the task document.

Data layer smoke test (`tests/test_data_loader.py`, 5 tests) passes:
transition matrix is row-stochastic, regime path uses only the five known
labels, synthetic backend produces the full schema with sane OHLC
relationships, and the CSV backend fails with a clear message when files
are absent rather than silently returning nothing.

15-year synthetic path generated for a visual check
(`docs/artifacts/day1_synthetic_nifty_check.png`): visible extended
drawdowns and recoveries consistent with regime switching, not pure random
walk. This is a development fixture, not market data; see
`src/data/synthetic.py` docstring and `PROJECT_PLAN.md` open decision 1.

Remaining Day 1 item: confirm real data source (CSV pack vs. Yahoo Finance
vs. synthetic-only for now) before Day 2 ingestion work.
