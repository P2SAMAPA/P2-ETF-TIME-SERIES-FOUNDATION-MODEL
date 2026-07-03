import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config


def load_master_data() -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=config.DATA_REPO,
        filename="master_data.parquet",
        repo_type="dataset",
        token=config.HF_TOKEN,
    )
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.sort_index(inplace=True)
    return df


def prepare_prices(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    prices = pd.DataFrame(index=df.index)
    for ticker in tickers:
        if ticker in df.columns:
            col = df[ticker]
            if not col.isna().all():
                prices[ticker] = col.ffill()
    return prices.dropna(how="all")


def prepare_macro(df: pd.DataFrame) -> pd.DataFrame:
    avail_core = [c for c in config.MACRO_COLS_CORE     if c in df.columns]
    avail_ext  = [c for c in config.MACRO_COLS_EXTENDED if c in df.columns]
    avail_all  = avail_core + avail_ext
    if not avail_all:
        return pd.DataFrame(index=df.index)
    macro = df[avail_all].copy()
    if avail_core:
        macro = macro.dropna(subset=avail_core)
    if avail_ext:
        macro[avail_ext] = macro[avail_ext].ffill().fillna(0.0)
    return macro


# ── TSFM-specific helpers (thin wrappers over the above) ──────────────────

def get_universe_tickers(universe: str) -> list:
    """Return ticker list for a given universe. Never combine FI and EQUITY."""
    if universe not in config.UNIVERSES:
        raise ValueError(f"Unknown universe: {universe}")
    return config.UNIVERSES[universe]


def build_universe_price_panel(df: pd.DataFrame, universe: str) -> pd.DataFrame:
    """Build a date-indexed price panel (columns = tickers) for a universe."""
    tickers = get_universe_tickers(universe)
    panel = prepare_prices(df, tickers)

    missing = sorted(set(tickers) - set(panel.columns))
    if missing:
        print(f"[data_manager] Warning: missing/all-NaN tickers for {universe}: {missing}")

    return panel


def get_log_returns(price_series: pd.Series) -> pd.Series:
    """Compute log returns from a raw price series."""
    return np.log(price_series / price_series.shift(1)).dropna()
