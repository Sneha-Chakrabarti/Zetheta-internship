"""Single entry point for market data, regardless of source.

Every downstream module should call `load_market_data(config)` and never
touch a CSV path or the synthetic generator directly. That keeps the
choice of data source a one-line config change (see PROJECT_PLAN.md,
"Open decisions", item 1) instead of a branch scattered through the
codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .synthetic import generate_synthetic_market

# Filenames expected under data_dir for the CSV backend. Each must have a
# 'date' column plus the listed value column(s); index frames additionally
# need open/high/low/close.
_CSV_SCHEMA = {
    "nifty50": ["open", "high", "low", "close"],
    "nifty_midcap100": ["open", "high", "low", "close"],
    "nifty_smallcap100": ["open", "high", "low", "close"],
    "india_vix": ["value"],
    "usdinr": ["value"],
    "gilt_10y": ["value"],
    "aaa_gilt_spread": ["value"],
    "fii_dii_flows": ["fii_cr", "dii_cr"],
    "sip_totals": ["sip_cr"],
}


@dataclass
class DataConfig:
    backend: str = "synthetic"  # "synthetic" or "csv"
    data_dir: str = "data/raw"
    seed: int = 7
    n_years: float = 15.0
    date_column: str = "date"


def _load_csv_backend(config: DataConfig) -> dict[str, pd.DataFrame]:
    data_dir = Path(config.data_dir)
    out: dict[str, pd.DataFrame] = {}
    missing = []
    for name, columns in _CSV_SCHEMA.items():
        path = data_dir / f"{name}.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path, parse_dates=[config.date_column])
        df = df.set_index(config.date_column).sort_index()
        absent_cols = [c for c in columns if c not in df.columns]
        if absent_cols:
            raise ValueError(
                f"{path} is missing expected column(s) {absent_cols}; "
                f"schema requires {columns}."
            )
        out[name] = df[columns]
    if missing:
        raise FileNotFoundError(
            "CSV backend selected but the following files were not found: "
            + ", ".join(missing)
            + f". Place them under {data_dir} or switch DataConfig.backend to 'synthetic'."
        )
    return out


def load_market_data(config: DataConfig | None = None) -> dict[str, pd.DataFrame]:
    """Load the full multi-asset panel using the backend named in `config`.

    Returned dict keys and shapes are identical across backends: nifty50,
    nifty_midcap100, nifty_smallcap100 (OHLC), india_vix, usdinr, gilt_10y,
    aaa_gilt_spread (single 'value' column), fii_dii_flows ('fii_cr',
    'dii_cr'), sip_totals (monthly 'sip_cr'). The synthetic backend also
    returns a 'regime_path' key with the ground-truth label used to
    generate the data; the CSV backend does not, since real regimes are
    unobserved. Code that consumes this dict must not assume 'regime_path'
    is always present.
    """
    config = config or DataConfig()
    if config.backend == "synthetic":
        return generate_synthetic_market(n_years=config.n_years, seed=config.seed)
    elif config.backend == "csv":
        return _load_csv_backend(config)
    else:
        raise ValueError(f"Unknown backend '{config.backend}'; expected 'synthetic' or 'csv'.")
