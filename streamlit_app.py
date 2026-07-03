"""
streamlit_app.py — P2-ETF-TIME-SERIES-FOUNDATION-MODEL (TSFM)

Dashboard for the Chronos zero-shot forecasting engine.

IMPORTANT: All st.markdown() calls use plain strings built via string
concatenation, NEVER f-strings — avoids the st.markdown() NameError bug
seen elsewhere in the suite.

IMPORTANT: st.dataframe() columns must never mix strings and numbers
(e.g. "-" alongside floats) — PyArrow serialization fails on mixed dtypes
and raises errors in the UI. Failed/insufficient-history tickers are kept
in a separate table, not blended into the numeric results table.
"""

import streamlit as st
import pandas as pd
import requests

import config

st.set_page_config(page_title="TSFM — Chronos Zero-Shot Forecaster", layout="wide")

RESULTS_URL = "https://huggingface.co/datasets/" + config.OUTPUT_REPO + "/resolve/main/results/latest.json"


@st.cache_data(ttl=3600)
def load_results():
    headers = {}
    if config.HF_TOKEN:
        headers["Authorization"] = "Bearer " + config.HF_TOKEN
    resp = requests.get(RESULTS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def signal_badge(signal):
    if signal == "LONG":
        return "🟢 LONG"
    elif signal == "SHORT":
        return "🔴 SHORT"
    else:
        return "⚪ NEUTRAL"


st.title("TSFM — Time Series Foundation Model (Chronos Zero-Shot)")
st.caption("Zero-shot ETF forecasting powered by Amazon Chronos (" + config.MODEL_VARIANT + ")")

try:
    data = load_results()
except requests.exceptions.HTTPError as e:
    st.error(
        "Could not load results (HTTP error). If "
        + config.OUTPUT_REPO
        + " is a private dataset, confirm HF_TOKEN is set in this app's secrets. Details: "
        + str(e)
    )
    st.stop()
except Exception as e:
    st.error("Could not load results: " + str(e))
    st.stop()

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown("**Run date:** " + data.get("run_date", "unknown"))
with header_col2:
    st.markdown("**Top N:** " + str(data.get("top_n", config.TOP_N)))

st.divider()

universe_tabs = st.tabs(["FI_COMMODITIES", "EQUITY_SECTORS"])

for tab, universe in zip(universe_tabs, ["FI_COMMODITIES", "EQUITY_SECTORS"]):
    with tab:
        results = data.get("results", {}).get(universe, [])

        ok_results = [r for r in results if r.get("status") == "ok"]
        other_results = [r for r in results if r.get("status") != "ok"]

        picks = [r["ticker"] for r in ok_results if r.get("top_pick")]

        metric_cols = st.columns(4)
        metric_cols[0].metric("Tickers forecasted", str(len(ok_results)) + " / " + str(len(results)))
        metric_cols[1].metric("Top picks", ", ".join(picks) if picks else "—")
        n_long = sum(1 for r in ok_results if r["signal"] == "LONG")
        n_short = sum(1 for r in ok_results if r["signal"] == "SHORT")
        metric_cols[2].metric("Long signals", str(n_long))
        metric_cols[3].metric("Short signals", str(n_short))

        st.markdown("")

        if ok_results:
            rows = []
            for r in ok_results:
                rows.append({
                    "Rank": int(r.get("rank")) if r.get("rank") is not None else 999,
                    "Pick": "⭐" if r.get("top_pick") else "",
                    "Ticker": r["ticker"],
                    "Signal": signal_badge(r["signal"]),
                    "Last Price": float(round(r["last_price"], 2)),
                    "Median Return %": float(round(r["horizon_median_return"] * 100, 2)),
                    "Composite Score": float(round(r["composite_score"], 4)),
                    "Confidence": float(round(r["confidence"], 2)),
                })
            df_display = pd.DataFrame(rows).sort_values("Rank").reset_index(drop=True)

            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn(format="%d"),
                    "Median Return %": st.column_config.NumberColumn(format="%.2f%%"),
                    "Confidence": st.column_config.ProgressColumn(
                        min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                },
            )
        else:
            st.info("No successful forecasts available for this universe.")

        if other_results:
            with st.expander(str(len(other_results)) + " ticker(s) skipped or errored"):
                skip_rows = []
                for r in other_results:
                    skip_rows.append({
                        "Ticker": r.get("ticker", "?"),
                        "Status": r.get("status", "unknown"),
                        "Detail": r.get("error", r.get("n_obs", "")),
                    })
                st.dataframe(pd.DataFrame(skip_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**Ticker detail**")
        tickers_ok = [r["ticker"] for r in ok_results]
        if tickers_ok:
            selected = st.selectbox("Select ticker", tickers_ok, key="select_" + universe)
            detail = next(r for r in ok_results if r["ticker"] == selected)

            chart_df = pd.DataFrame({
                "step": list(range(1, len(detail["median_forecast"]) + 1)),
                "median": detail["median_forecast"],
                "low": detail["low_forecast"],
                "high": detail["high_forecast"],
            }).set_index("step")

            st.line_chart(chart_df)

            detail_cols = st.columns(4)
            detail_cols[0].metric("Last price", str(round(detail["last_price"], 2)))
            detail_cols[1].metric("Signal", detail["signal"])
            detail_cols[2].metric("Composite score", str(round(detail["composite_score"], 4)))
            detail_cols[3].metric("Confidence", str(round(detail["confidence"], 2)))
        else:
            st.info("No successful forecasts available for this universe.")

st.divider()
st.caption("Engine: P2-ETF-TIME-SERIES-FOUNDATION-MODEL | Model: " + config.MODEL_VARIANT)
