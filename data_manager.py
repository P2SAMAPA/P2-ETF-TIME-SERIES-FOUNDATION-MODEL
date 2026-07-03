"""
data_manager.py — P2-ETF-TIME-SERIES-FOUNDATION-MODEL (TSFM)

Loads the shared master dataset from HuggingFace and prepares per-ticker
close-price series for zero-shot forecasting with Chronos.

Follows the established suite-wide data-loading pattern:
  - hf_hub_download + pd.read_parquet (NOT HfFileSystem.open for reads)
  - dropna restricted to REQUIRED_COLS (price + MACRO_COLS_CORE only)
  - MACRO_COLS_EXTENDED (IG_SPREAD, HY_SPREAD, DGS*) NEVER included in dropna subset
  - Prices are bare ticker columns; log returns = log(price_t / price_{t-1})
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

from config import (
    MASTER_DATASET_REPO,
    MASTER_DATASET_FILENAME,
    FI_COMMODITIES,
    EQUITY_SECTORS,
    MACRO_COLS_CORE,
    REQUIRED_COLS,
)


def load_master_data() -> pd.DataFrame:
    """Download and load the shared master parquet dataset."""
    local_path = hf_hub_download(
        repo_id=MASTER_DATASET_REPO,
        filename=MASTER_DATASET_FILENAME,
        repo_type="dataset",
    )
    df = pd.read_parquet(local_path)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # Critical fix: dropna restricted to price + core macro columns only.
    # Options/ADV columns and MACRO_COLS_EXTENDED are NaN for most of history
    # and must never be part of this subset, or dropna wipes all history.
    dropna_subset = [c for c in REQUIRED_COLS if c in df.columns]
    df = df.dropna(subset=dropna_subset)

    return df


def get_universe_tickers(universe: str) -> list:
    """Return ticker list for a given universe. Never combine FI and EQUITY."""
    if universe == "FI_COMMODITIES":
        return FI_COMMODITIES
    elif universe == "EQUITY_SECTORS":
        return EQUITY_SECTORS
    else:
        raise ValueError(f"Unknown universe: {universe}")


def get_price_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    """Return a clean close-price series (bare ticker column) for one asset."""
    if ticker not in df.columns:
        raise KeyError(f"Ticker '{ticker}' not found in master dataset columns")

    series = df.set_index("date")[ticker].dropna()
    series = series.astype(float)
    return series


def get_log_returns(price_series: pd.Series) -> pd.Series:
    """Compute log returns from a raw price series."""
    return np.log(price_series / price_series.shift(1)).dropna()


def get_macro_context(df: pd.DataFrame) -> pd.DataFrame:
    """Return the core macro columns (VIX, DXY, T10Y2Y) aligned on date."""
    cols = ["date"] + [c for c in MACRO_COLS_CORE if c in df.columns]
    return df[cols].dropna(subset=[c for c in MACRO_COLS_CORE if c in df.columns])


def build_universe_price_panel(df: pd.DataFrame, universe: str) -> pd.DataFrame:
    """Build a date-indexed price panel (columns = tickers) for a universe."""
    tickers = get_universe_tickers(universe)
    available = [t for t in tickers if t in df.columns]
    missing = sorted(set(tickers) - set(available))
    if missing:
        print(f"[data_manager] Warning: missing tickers for {universe}: {missing}")

    panel = df.set_index("date")[available].dropna(how="all")
    return panel
