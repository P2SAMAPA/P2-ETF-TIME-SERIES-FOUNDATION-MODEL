"""
tsfm_engine.py — P2-ETF-TIME-SERIES-FOUNDATION-MODEL (TSFM)

Zero-shot probabilistic forecasting using Amazon Chronos (chronos-small,
~46M params, CPU-feasible). Chronos treats each price series as a token
sequence and produces a set of sampled forecast trajectories, from which
we derive a point forecast, a directional signal, and a composite score
(WEIGHT_MEAN * mean_return + WEIGHT_PROB * prob_positive + WEIGHT_SHARPE *
sharpe) used to rank tickers for the top-N picks in trainer.py.
"""

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

from config import (
    MODEL_VARIANT,
    MODEL_MAP,
    PRED_HORIZON,
    NUM_SAMPLES,
    WEIGHT_MEAN,
    WEIGHT_PROB,
    WEIGHT_SHARPE,
    WINDOWS,
)

# Longest configured lookback window is used as the Chronos context length.
LOOKBACK_WINDOW = max(WINDOWS)


class TSFMEngine:
    """Loads a Chronos pipeline once and reuses it across tickers."""

    def __init__(self, model_variant: str = None):
        self.model_variant = model_variant or MODEL_VARIANT
        model_id = MODEL_MAP.get(self.model_variant, MODEL_MAP["chronos-small"])

        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    def forecast_ticker(self, price_series: pd.Series, ticker: str) -> dict:
        """
        Generate a zero-shot forecast for a single ticker.

        Returns point forecast, quantile bands, a directional signal, and a
        composite score for cross-sectional ranking.
        """
        series = price_series.tail(LOOKBACK_WINDOW).astype(float)
        if len(series) < 30:
            return {
                "ticker": ticker,
                "status": "insufficient_history",
                "n_obs": int(len(series)),
            }

        context = torch.tensor(series.values, dtype=torch.float32)

        # Passed positionally (not context=) — some chronos-forecasting
        # versions renamed/repositioned this first parameter, and calling
        # positionally works across versions.
        forecast = self.pipeline.predict(
            context,
            prediction_length=PRED_HORIZON,
            num_samples=NUM_SAMPLES,
        )
        # forecast shape: [num_series, num_samples, prediction_length]
        samples = forecast[0].numpy()  # [num_samples, prediction_length]

        low, median, high = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)

        last_price = float(series.iloc[-1])

        # Per-sample terminal returns, used for the composite score.
        terminal_returns = (samples[:, -1] - last_price) / last_price
        mean_return = float(np.mean(terminal_returns))
        prob_positive = float(np.mean(terminal_returns > 0))
        std_return = float(np.std(terminal_returns))
        sharpe = float(mean_return / std_return) if std_return > 1e-9 else 0.0

        composite_score = (
            WEIGHT_MEAN * mean_return
            + WEIGHT_PROB * prob_positive
            + WEIGHT_SHARPE * sharpe
        )

        horizon_median_return = float((median[-1] - last_price) / last_price)

        if horizon_median_return > 0.005:
            signal = "LONG"
        elif horizon_median_return < -0.005:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"

        confidence = float(
            1.0 - (high[-1] - low[-1]) / max(abs(median[-1]), 1e-6) / 2.0
        )
        confidence = max(0.0, min(1.0, confidence))

        return {
            "ticker": ticker,
            "status": "ok",
            "last_price": last_price,
            "forecast_horizon": PRED_HORIZON,
            "median_forecast": [float(x) for x in median],
            "low_forecast": [float(x) for x in low],
            "high_forecast": [float(x) for x in high],
            "horizon_median_return": horizon_median_return,
            "mean_return": mean_return,
            "prob_positive": prob_positive,
            "sharpe": sharpe,
            "composite_score": composite_score,
            "signal": signal,
            "confidence": confidence,
        }

    def forecast_universe(self, price_panel: pd.DataFrame) -> list:
        """Run forecast_ticker across every column in a price panel."""
        import traceback

        results = []
        for ticker in price_panel.columns:
            series = price_panel[ticker].dropna()
            try:
                result = self.forecast_ticker(series, ticker)
            except Exception as e:
                print(f"[tsfm_engine] ERROR forecasting {ticker}: {e}")
                traceback.print_exc()
                result = {"ticker": ticker, "status": "error", "error": str(e)}
            results.append(result)
        return results
