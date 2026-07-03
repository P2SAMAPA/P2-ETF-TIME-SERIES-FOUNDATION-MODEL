# P2-ETF-TIME-SERIES-FOUNDATION-MODEL (TSFM)

Zero-shot ETF price forecasting using Amazon Chronos, a pretrained time
series foundation model. Unlike the other engines in the P2Quant suite,
TSFM requires no training — Chronos is applied directly (zero-shot) to
each ticker's price history to produce probabilistic forecasts.

## Model

- **Variant:** `chronos-small` (`amazon/chronos-t5-small`, ~46M parameters)
- Chosen for CPU feasibility inside GitHub Actions runners (no GPU required)
- Runs zero-shot: no fine-tuning step, no `trainer.py` optimization loop —
  `trainer.py` here orchestrates inference, not gradient training

## Universes

Forecasts are generated separately for each universe. They are never combined:

- `FI_COMMODITIES`
- `EQUITY_SECTORS`

## Pipeline

1. `data_manager.py` — loads the shared master dataset from
   `P2SAMAPA/fi-etf-macro-signal-master-data` via `hf_hub_download` +
   `pd.read_parquet`, restricting `dropna` to `REQUIRED_COLS`
   (price + `MACRO_COLS_CORE`) so options/ADV and `MACRO_COLS_EXTENDED`
   columns never wipe historical rows.
2. `tsfm_engine.py` — wraps `ChronosPipeline`, generates sampled forecast
   trajectories per ticker, and derives point forecasts, 10/90 quantile
   bands, a directional signal (`LONG` / `SHORT` / `NEUTRAL`), and a
   confidence score from forecast dispersion.
3. `trainer.py` — orchestrates the daily run across both universes and
   assembles the results payload.
4. `push_results.py` — uploads the results JSON to
   `P2SAMAPA/p2-etf-tsfm-results` using `HfApi.upload_file` (not
   `HfFileSystem.open`, which silently fails on write).
5. `streamlit_app.py` — dashboard reading the latest results via
   `requests`. All markdown strings are plain strings / concatenation,
   never f-strings, to avoid the `st.markdown()` NameError bug seen
   elsewhere in the suite.
6. `us_calendar.py` — NYSE trading-day check used to skip non-trading days.

## Workflow

Single GitHub Actions job (no matrix — Chronos inference across both
universes runs sequentially within one job, unlike engines that parallelize
per-universe). Scheduled weekdays at 21:30 UTC, after US market close.

## Output

Results are pushed to `P2SAMAPA/p2-etf-tsfm-results` as:
- `results/<YYYY-MM-DD>.json` — dated snapshot
- `results/latest.json` — always overwritten with the most recent run

Each ticker result includes: last price, median/low/high forecast paths
over the configured horizon, horizon median return, signal, and confidence.

## Requirements

Standard suite dependencies plus `torch` and `chronos-forecasting`.
