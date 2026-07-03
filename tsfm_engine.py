"""
tsfm_engine.py — P2-ETF-TIME-SERIES-FOUNDATION-MODEL (TSFM)

Zero-shot probabilistic forecasting using Amazon Chronos (chronos-small,
~46M params, CPU-feasible). Chronos treats each price series as a token
sequence and produces a set of sampled forecast trajectories, from which
we derive point forecasts, quantiles, and a directional trading signal.
"""

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

from config import MODEL_VARIANT, LOOKBACK_WINDOW, FORECAST_HORIZON, NUM_SAMPLES

_MODEL_MAP = {
    "chronos-small": "amazon/chronos-t5-small",
    "chronos-tiny": "amazon/chronos-t5-tiny",
    "chronos-mini": "amazon/chronos-t5-mini",
    "chronos-base": "amazon/chronos-t5-base",
}


class TSFMEngine:
    """Loads a Chronos pipeline once and reuses it across tickers."""

    def __init__(self, model_variant: str = None):
        self.model_variant = model_variant or MODEL_VARIANT
        model_id = _MODEL_MAP.get(self.model_variant, _MODEL_MAP["chronos-small"])

        self.pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    def forecast_ticker(self, price_series: pd.Series, ticker: str) -> dict:
        """
        Generate a zero-shot forecast for a single ticker.

        Returns a dict with point forecast, quantiles, and a directional
        signal derived from the median forecast vs. the last observed price.
        """
        series = price_series.tail(LOOKBACK_WINDOW).astype(float)
        if len(series) < 30:
            return {
                "ticker": ticker,
                "status": "insufficient_history",
                "n_obs": int(len(series)),
            }

        context = torch.tensor(series.values, dtype=torch.float32)

        forecast = self.pipeline.predict(
            context=context,
            prediction_length=FORECAST_HORIZON,
            num_samples=NUM_SAMPLES,
        )
        # forecast shape: [num_series, num_samples, prediction_length]
        samples = forecast[0].numpy()

        low, median, high = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)

        last_price = float(series.iloc[-1])
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
            "forecast_horizon": FORECAST_HORIZON,
            "median_forecast": [float(x) for x in median],
            "low_forecast": [float(x) for x in low],
            "high_forecast": [float(x) for x in high],
            "horizon_median_return": horizon_median_return,
            "signal": signal,
            "confidence": confidence,
        }

    def forecast_universe(self, price_panel: pd.DataFrame) -> list:
        """Run forecast_ticker across every column in a price panel."""
        results = []
        for ticker in price_panel.columns:
            series = price_panel[ticker].dropna()
            try:
                result = self.forecast_ticker(series, ticker)
            except Exception as e:
                result = {"ticker": ticker, "status": "error", "error": str(e)}
            results.append(result)
        return results
