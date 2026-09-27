# Data sources: search results and recommendations

This environment has no network access to NSE, RBI DBIE, AMFI, MOSPI, SEBI,
or Yahoo Finance (the sources named in Section E4 of the task document).
Its outbound network is restricted to code-hosting and package-registry
domains (GitHub, PyPI, npm and similar). So real market data can only be
pulled here from files already published on GitHub, not from a live API
or a bank/exchange website. This file records what that search turned up.

## Found and included (`data/raw/real_partial/`)

| File | Series | Coverage | Source |
|---|---|---|---|
| `nifty50_2015_2019.csv` | Nifty 50 OHLC | 2015-01-01 to 2019-12-31 (1232 rows) | github.com/abulbasar/data, `nifty50-index.csv` |
| `india_vix_2020_2026.csv` | India VIX OHLC | 2020-01-02 to 2026-08-24 (1627 rows) | github.com/tanya459/india-vix-market-data-analysis |

Both are third-party re-publications of NSE data under their own repos; no
license file was found in either repo, so treat as unverified provenance
and re-derive from an official source before anything goes in a
regulator-facing artifact. Good enough for pipeline development and
sanity-checking model code against a real, if short and partial, window.

## Searched but not found as a downloadable bulk file on GitHub

- **Nifty 50, full 15-year+ daily history in one file.** Several repos
  cover pieces of the range (2000-2020 per-stock data, 2015-2024
  minute-level data that would need resampling) but none as a single
  clean daily-OHLC file spanning 2011-2026.
- **Nifty Midcap 100 / Nifty Smallcap 100 daily OHLC.** No GitHub-hosted
  file found at all. NSE Indices (niftyindices.com) publishes this but
  requires a live site fetch.
- **USD/INR for 2011-2026 specifically.** A full daily series since 1973
  exists at `eco3min.fr/dataset/usd-inr-exchange-rate.csv` (FRED series
  DEXINUS, Federal Reserve H.10), but it is hosted on eco3min.fr, not
  GitHub, and the fetch tool available here truncates the response before
  reaching 2011 (the file is served oldest-first with no way to request a
  later slice). The 1973-2001 portion was retrieved but is not useful for
  this project's window.
- **10Y Gilt yield, AAA-Gilt spread.** No free bulk historical file found;
  most sources (Bloomberg, Cbonds, Investing.com, TradingEconomics) gate
  the download behind a paid plan or a login.
- **FII/DII daily flows, monthly SIP totals.** No bulk historical CSV
  found; live dashboards exist (e.g. the FII/DII Data Dashboard on
  GitHub) but they show recent snapshots, not a downloadable multi-year
  series.

## Recommended paths to complete the data

Any of these need to run somewhere with normal internet access (the
user's own machine, Colab, or any non-sandboxed environment), not from
this chat:

1. **`jugaad-data` or `nseindiapy` (Python, PyPI).** Both wrap NSE
   Indices' own historical-data API: `index_raw("NIFTY MIDCAP 100", ...)`
   style calls return OHLC for any NSE index, auto-paginated. This is the
   most direct way to get Nifty 50, Midcap 100, Smallcap 100, and India
   VIX in one consistent format for the full 15 years.
2. **`yfinance`** for Nifty 50 (`^NSEI`), India VIX (`^INDIAVIX`), and
   USD/INR (`INR=X`) — already in `requirements/core.txt`.
3. **RBI DBIE** (dbie.rbi.org.in) for the 10Y gilt yield and AAA-corporate
   spread; manual monthly/weekly download, no API.
4. **AMFI** (amfiindia.com) monthly Excel files for SIP totals; NSDL/CDSL
   for FII/DII flow data.
5. If Zetheta's "provided CSV pack" (mentioned in Section D2, Day 1)
   actually exists, that is the fastest path and should be checked first.

Once any of these produce the nine files named in `src/data/loader.py`'s
`_CSV_SCHEMA`, drop them under `data/raw/` and set
`DataConfig(backend="csv")` — no other code changes needed.
