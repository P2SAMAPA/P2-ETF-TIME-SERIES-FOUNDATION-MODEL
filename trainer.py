"""
trainer.py — P2-ETF-TIME-SERIES-FOUNDATION-MODEL (TSFM)

Chronos is used zero-shot (no fine-tuning), so "training" here means
orchestrating the daily forecast run: load data, run inference for each
universe separately (FI_COMMODITIES and EQUITY_SECTORS are never combined),
rank tickers by composite_score, and assemble the results payload for
push_results.py.
"""

import datetime as dt

from data_manager import load_master_data, build_universe_price_panel
from tsfm_engine import TSFMEngine
from config import MODEL_VARIANT, TOP_N


def _rank_and_flag_top_picks(results: list) -> list:
    """Sort ok results by composite_score desc; flag the top TOP_N as picks."""
    ok = [r for r in results if r.get("status") == "ok"]
    not_ok = [r for r in results if r.get("status") != "ok"]

    ok.sort(key=lambda r: r["composite_score"], reverse=True)
    for i, r in enumerate(ok):
        r["rank"] = i + 1
        r["top_pick"] = i < TOP_N

    for r in not_ok:
        r["rank"] = None
        r["top_pick"] = False

    return ok + not_ok


def run_daily_forecast() -> dict:
    """Run the full daily TSFM forecast cycle across both universes."""
    print("[trainer] Loading master dataset...")
    df = load_master_data()

    print(f"[trainer] Initializing Chronos engine ({MODEL_VARIANT})...")
    engine = TSFMEngine()

    all_results = {}
    for universe in ["FI_COMMODITIES", "EQUITY_SECTORS"]:
        print(f"[trainer] Building price panel for {universe}...")
        panel = build_universe_price_panel(df, universe)

        print(f"[trainer] Running zero-shot forecasts for {universe} "
              f"({len(panel.columns)} tickers)...")
        results = engine.forecast_universe(panel)
        results = _rank_and_flag_top_picks(results)
        all_results[universe] = results

    payload = {
        "engine": "P2-ETF-TIME-SERIES-FOUNDATION-MODEL",
        "model_variant": MODEL_VARIANT,
        "top_n": TOP_N,
        "run_date": dt.datetime.utcnow().strftime("%Y-%m-%d"),
        "run_timestamp_utc": dt.datetime.utcnow().isoformat(),
        "results": all_results,
    }

    n_ok = sum(
        1 for u in all_results.values() for r in u if r.get("status") == "ok"
    )
    n_total = sum(len(u) for u in all_results.values())
    print(f"[trainer] Completed: {n_ok}/{n_total} tickers forecasted successfully.")

    for universe, results in all_results.items():
        picks = [r["ticker"] for r in results if r.get("top_pick")]
        print(f"[trainer] Top {TOP_N} picks for {universe}: {picks}")

    return payload


if __name__ == "__main__":
    run_daily_forecast()
