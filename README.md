# P2-ETF-TIME-SERIES-FOUNDATION-MODEL (TSFM)

Zero-shot ETF price forecasting using Amazon Chronos, a pretrained time
series foundation model. Unlike the other engines in the P2Quant suite,
TSFM requires no training — Chronos is applied directly (zero-shot) to
each ticker's price history to produce probabilistic forecasts.

## Model

- **Variant:** `chronos-small` (`amazon/chronos-t5-small`, ~46M parameters)
- Chosen for CPU feasibility inside GitHub Actions runners (no GPU required)
- Runs zero-shot: no fine-tuning step — `trainer.py` orchestrates inference
  and ranking, not gradient training
- Context length: longest configured window (`WINDOWS`, currently 504 days)
- Forecast horizon: `PRED_HORIZON` (21 trading days)

## Universes

Forecasts are generated separately for each universe defined in
`config.UNIVERSES`. They are never combined:

- `FI_COMMODITIES`
- `EQUITY_SECTORS`

(`COMBINED` exists in config for other engines' use but is intentionally
not forecasted separately here — it's the union of the two above.)

## Composite scoring and top picks

Each ticker's Chronos sample paths yield a composite score:

```
composite_score = WEIGHT_MEAN * mean_return
                 + WEIGHT_PROB * prob_positive
                 + WEIGHT_SHARPE * sharpe
```

where `mean_return`, `prob_positive`, and `sharpe` are computed across the
`NUM_SAMPLES` sampled terminal-horizon outcomes. Tickers are ranked by
`composite_score` within each universe, and the top `TOP_N` are flagged
`top_pick: true` in the results payload.

## Pipeline

1. `data_manager.py` — loads the shared master dataset from
   `DATA_REPO` (`P2SAMAPA/fi-etf-macro-signal-master-data`) via
   `hf_hub_download` + `pd.read_parquet` (parquet filename auto-detected),
   restricting `dropna` to `UNIVERSES["COMBINED"]` price columns +
   `MACRO_COLS_CORE` so options/ADV and `MACRO_COLS_EXTENDED` columns never
   wipe historical rows.
2. `tsfm_engine.py` — wraps `ChronosPipeline`, generates sampled forecast
   trajectories per ticker, and derives point forecasts, 10/90 quantile
   bands, a directional signal (`LONG` / `SHORT` / `NEUTRAL`), and the
   composite score above.
3. `trainer.py` — orchestrates the daily run across both universes, ranks
   and flags top picks, and assembles the results payload.
4. `push_results.py` — uploads the results JSON to
   `OUTPUT_REPO` (`P2SAMAPA/p2-etf-tsfm-results`) using
   `HfApi.upload_file` (not `HfFileSystem.open`, which silently fails on
   write).
5. `streamlit_app.py` — dashboard reading the latest results via
   `requests`, surfacing top picks and composite scores. All markdown
   strings are plain strings / concatenation, never f-strings, to avoid
   the `st.markdown()` NameError bug seen elsewhere in the suite.
6. `us_calendar.py` — NYSE trading-day check used to skip non-trading days.

## Workflow

Single GitHub Actions job (no matrix — Chronos inference across both
universes runs sequentially within one job). Scheduled weekdays at 21:30
UTC, after US market close.

## Output

Results are pushed to `P2SAMAPA/p2-etf-tsfm-results` as:
- `results/<YYYY-MM-DD>.json` — dated snapshot
- `results/latest.json` — always overwritten with the most recent run

Each ticker result includes: last price, median/low/high forecast paths
over the horizon, horizon median return, mean return, probability positive,
Sharpe, composite score, rank, top-pick flag, signal, and confidence.

## Requirements

Standard suite dependencies plus `torch`, `chronos-forecasting`, and
`pandas_market_calendars`.
