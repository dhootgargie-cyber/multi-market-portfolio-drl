"""
data_pipeline.py
-----------------
Phase 1: Data Collection

Downloads daily OHLCV data for the 10-asset universe (US + Indian ADRs)
using yfinance, and saves one CSV per ticker to data/raw/.

Run this on your own machine (needs internet access to Yahoo Finance).
"""

import os
import time
import pandas as pd
import yfinance as yf

# ---- Asset universe -------------------------------------------------------
US_TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "JNJ", "AMZN"]
INDIA_TICKERS = ["INFY", "WIT", "HDB", "IBN"]  # NYSE-listed ADRs, fetchable directly via yfinance
ALL_TICKERS = US_TICKERS + INDIA_TICKERS

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

RAW_DATA_DIR = "data/raw"


def fetch_ticker(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Download OHLCV data for a single ticker with basic retry logic."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            df = df.reset_index()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df["Ticker"] = ticker
            return df
        except Exception as e:
            print(f"  [{ticker}] attempt {attempt}/{retries} failed: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to fetch {ticker} after {retries} attempts")


def fetch_all(tickers=ALL_TICKERS, start=START_DATE, end=END_DATE, out_dir=RAW_DATA_DIR) -> dict:
    """Download all tickers and save each to its own CSV. Returns dict of DataFrames."""
    os.makedirs(out_dir, exist_ok=True)
    data = {}
    for ticker in tickers:
        print(f"Fetching {ticker} ...")
        df = fetch_ticker(ticker, start, end)
        out_path = os.path.join(out_dir, f"{ticker}.csv")
        df.to_csv(out_path, index=False)
        print(f"  saved {len(df)} rows -> {out_path}")
        data[ticker] = df
    return data


if __name__ == "__main__":
    print(f"Downloading {len(ALL_TICKERS)} tickers from {START_DATE} to {END_DATE}")
    fetch_all()
    print("Done. Raw CSVs are in data/raw/")
