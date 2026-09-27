# Day 3: feature engineering notes

Scope: `src/features/engineer.py` (tabular features), `src/features/
topology.py` (TDA), `src/features/sector_gnn.py` (GCN). Full detail and
executed code in `notebooks/02_feature_engineering.ipynb`.

## What was implemented

31 tabular features across seven groups (return, trend, volatility, VIX,
macro, flow, cap-segment), matching Section A4.5 plus the Day 3 additions
in Section D2. TDA persistence-landscape features (Section A7.2) and a
segment-graph GCN embedding (Section A7.3), both over a nine-series daily
panel (three cap-segment indices, VIX change, INR return, gilt yield
change, credit spread change, FII flow, DII flow).

## What was not implemented, and why

Three items the task document asks for need data outside the required
panel (Section A1):

- **Breadth features** (advance/decline ratio, % above 50DMA, new highs
  minus new lows). Need per-constituent OHLC for all Nifty 50 members;
  the panel has only index-level series.
- **Real interest rates.** Need a CPI series; not in the panel.
- **INR "DXY-orthogonalised".** Needs a US Dollar Index series; not in
  the panel.

Two more are implemented, but as labelled proxies rather than the real
thing, because the same data gap applies:

- `rbi_stance_proxy` uses the 63-day change in the 10Y gilt yield in
  place of actual policy-rate/MPC data.
- `midcap_large_ratio_z` / `smallcap_large_ratio_z` use a rolling
  z-score of the cap-segment price ratio in place of a P/E or P/B-based
  valuation z-score (no fundamentals data in the panel).

All five are called out inline in `engineer.py`'s docstring as well, so
the caveat travels with the code, not just this document.

## The graph and the point cloud are nine series, not a stock/sector universe

Section A7's TDA and GCN examples assume "the universe of stocks/
sectors". The panel has three cap-segment indices and six single-series
macro/flow inputs: nine series total, used as both the TDA point cloud
and the GCN's graph nodes. This is a real, working demonstration of both
methods, not a placeholder, but it is a different and much smaller graph
than a genuine sectoral one (Nifty IT, Nifty Bank, Nifty Auto, and
similar NSE sectoral indices). Adding those series to the data schema
later would let this code run on an actual sector graph without changing
`topology.py` or `sector_gnn.py` themselves.

## The GCN is untrained by design at this stage

`SectorGCN`'s weights are a fixed seeded random initialisation, not
trained. There is no supervised regime label yet for it to train against
(that arrives with the HMM pseudo-labels in Section A4.6, used starting
Day 9's ensembling). This mirrors how Section A5 uses foundation-model
embeddings: frozen feature extraction now, fine-tuning once there is
something to fine-tune against.

## Claims checked numerically before being written, not left as visual impressions

Two things that looked like findings from the plots turned out to need a
closer look before being stated as fact:

- The GCN embedding norm plot visually appears to show elevated stretches
  near all three case-study markers (2013, 2020, 2024). Checked in a
  ±45-day window around each: only 2020 (window mean 1.41 vs. an overall
  mean of 0.77) is actually elevated. 2013 (0.57) and 2024 (0.43) are at
  or below the overall mean; the visually elevated stretches near them in
  the full plot sit close to, but not aligned with, those specific dates.
  Since the synthetic regime path is one random Markov-chain realisation
  with no connection to real calendar dates, the 2020 alignment is very
  likely coincidence at this sample size, not a detection result.
- The same check on the TDA landscape L2 norm found none of the three
  windows elevated relative to its own overall mean (8.04, 8.10, 8.24 vs.
  8.06 overall) - so no claim about the TDA norm reacting to these
  breakpoints is made, unlike the (partial, caveated) one for the GCN
  norm.
- The first several columns of every TDA feature row are numerically
  identical (to float precision) regardless of the window. This is a
  mathematical property of dimension-0 persistence landscapes (every
  connected-component bar is born at filtration value 0, so the early
  landscape is a fixed ramp), not a bug in `topology_features`.

## Recommendation

Proceed to Day 4 (frequentist HMM) using `engineer_regime_features` as
the input feature matrix. TDA and GCN embeddings are separate feature
frames on their own (sparser, every-5th-day) calendar; joining them to
the daily feature matrix is a Day 9 ensembling-stage decision, not
needed before the single-series HMM work in Day 4-5.
