import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-tsfm-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "XLB",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

MACRO_COLS_CORE     = ["VIX", "DXY", "T10Y2Y"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]

WINDOWS = [63, 126, 252, 504]

# ── Foundation model ──────────────────────────────────────────────────────────
# "chronos-tiny"  : amazon/chronos-t5-tiny    8M  — fastest
# "chronos-mini"  : amazon/chronos-t5-mini   20M
# "chronos-small" : amazon/chronos-t5-small  46M  — default
MODEL_VARIANT = "chronos-small"

MODEL_MAP = {
    "chronos-tiny":  "amazon/chronos-t5-tiny",
    "chronos-mini":  "amazon/chronos-t5-mini",
    "chronos-small": "amazon/chronos-t5-small",
}

PRED_HORIZON = 21
NUM_SAMPLES  = 20

WEIGHT_MEAN   = 0.50
WEIGHT_PROB   = 0.30
WEIGHT_SHARPE = 0.20

TOP_N = 3
