"""
streamlit_app.py — P2-ETF-TIME-SERIES-FOUNDATION-MODEL (TSFM)

Dashboard for the Chronos zero-shot forecasting engine.

IMPORTANT: All st.markdown() calls use plain strings built via string
concatenation or .format(), NEVER f-strings. F-strings containing '**',
'_varname_', or subscripts inside st.markdown() have caused NameErrors
across the suite; this pattern avoids that class of bug entirely.
"""

import streamlit as st
import pandas as pd
import requests

from config import OUTPUT_REPO, MODEL_VARIANT

st.set_page_config(page_title="TSFM — Chronos Zero-Shot Forecaster", layout="wide")

RESULTS_URL = "https://huggingface.co/datasets/" + OUTPUT_REPO + "/resolve/main/results/latest.json"


@st.cache_data(ttl=3600)
def load_results():
    resp = requests.get(RESULTS_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


def signal_color(signal):
    if signal == "LONG":
        return "🟢"
    elif signal == "SHORT":
        return "🔴"
    else:
        return "⚪"


st.title("TSFM — Time Series Foundation Model (Chronos Zero-Shot)")
st.markdown("Zero-shot ETF forecasting powered by Amazon Chronos (" + MODEL_VARIANT + ").")

try:
    data = load_results()
except Exception as e:
    st.error("Could not load results: " + str(e))
    st.stop()

st.markdown("**Run date:** " + data.get("run_date", "unknown"))

universe_tabs = st.tabs(["FI_COMMODITIES", "EQUITY_SECTORS"])

for tab, universe in zip(universe_tabs, ["FI_COMMODITIES", "EQUITY_SECTORS"]):
    with tab:
        results = data.get("results", {}).get(universe, [])
        rows = []
        for r in results:
            if r.get("status") != "ok":
                rows.append({
                    "Ticker": r.get("ticker", "?"),
                    "Status": r.get("status", "unknown"),
                    "Signal": "-",
                    "Last Price": "-",
                    "Median Return": "-",
                    "Confidence": "-",
                })
                continue

            rows.append({
                "Ticker": r["ticker"],
                "Status": "ok",
                "Signal": signal_color(r["signal"]) + " " + r["signal"],
                "Last Price": round(r["last_price"], 2),
                "Median Return": str(round(r["horizon_median_return"] * 100, 2)) + "%",
                "Confidence": round(r["confidence"], 2),
            })

        df_display = pd.DataFrame(rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("**Ticker detail**")
        tickers_ok = [r["ticker"] for r in results if r.get("status") == "ok"]
        if tickers_ok:
            selected = st.selectbox("Select ticker", tickers_ok, key="select_" + universe)
            detail = next(r for r in results if r["ticker"] == selected)

            chart_df = pd.DataFrame({
                "step": list(range(1, len(detail["median_forecast"]) + 1)),
                "median": detail["median_forecast"],
                "low": detail["low_forecast"],
                "high": detail["high_forecast"],
            }).set_index("step")

            st.line_chart(chart_df)

            summary_line = (
                selected + " — last price " + str(round(detail["last_price"], 2))
                + ", signal " + detail["signal"]
                + ", confidence " + str(round(detail["confidence"], 2))
            )
            st.markdown(summary_line)
        else:
            st.markdown("No successful forecasts available for this universe.")

st.markdown("---")
st.markdown("Engine: P2-ETF-TIME-SERIES-FOUNDATION-MODEL | Model: " + MODEL_VARIANT)
