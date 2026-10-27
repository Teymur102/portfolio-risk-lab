"""Data loading with an on-disk cache so every run uses identical inputs.

Prices: daily dividend/split-adjusted closes from Yahoo Finance (yfinance).
Risk-free: FRED DTB3 (3-month T-bill, secondary market, annualised % discount basis).
If the CSVs in data/ exist they are used as-is; delete them to re-download.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_CSV = DATA_DIR / "prices.csv"
RF_CSV = DATA_DIR / "dtb3.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"

# Downloaded set. DBC is fetched for reference but excluded from the
# optimisation universe because its late start (Feb 2006) would push the first
# out-of-sample month past the 2008 crisis (see README).
ALL_TICKERS = ["SPY", "EFA", "EEM", "AGG", "TLT", "LQD", "GLD", "VNQ", "DBC"]
UNIVERSE = ["SPY", "EFA", "EEM", "AGG", "TLT", "LQD", "GLD", "VNQ"]
END_DATE = "2026-09-30"  # fixed end so the sample is reproducible


def load_prices() -> pd.DataFrame:
    if not PRICES_CSV.exists():
        import yfinance as yf

        DATA_DIR.mkdir(exist_ok=True)
        raw = yf.download(
            ALL_TICKERS,
            start="1993-01-01",
            end=pd.Timestamp(END_DATE) + pd.Timedelta(days=1),
            auto_adjust=True,
            progress=False,
        )["Close"]
        raw = raw[ALL_TICKERS]
        raw.index.name = "Date"
        raw.to_csv(PRICES_CSV, float_format="%.6f")
    px = pd.read_csv(PRICES_CSV, index_col="Date", parse_dates=True)
    return px.loc[:END_DATE]


def load_rf() -> pd.Series:
    """Daily risk-free rate as a simple daily return (DTB3/100/252)."""
    if not RF_CSV.exists():
        DATA_DIR.mkdir(exist_ok=True)
        with urllib.request.urlopen(FRED_URL, timeout=60) as r:
            txt = r.read().decode()
        RF_CSV.write_text(txt)
    df = pd.read_csv(RF_CSV)
    df.columns = ["Date", "DTB3"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["DTB3"] = pd.to_numeric(df["DTB3"], errors="coerce")
    s = df.set_index("Date")["DTB3"].loc[:END_DATE]
    return s / 100.0 / 252.0


def rf_on(index: pd.DatetimeIndex, rf: pd.Series) -> pd.Series:
    """Align daily rf to a trading calendar (last available value carried forward)."""
    return rf.reindex(rf.index.union(index)).ffill().reindex(index).fillna(0.0)
