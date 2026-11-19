"""
generate_synthetic_data.py
---------------------------
Fallback data source used automatically by other scripts ONLY when
data/raw/ (real data from data_pipeline.py) is not present. Generates
realistic-looking daily OHLCV data for the same 10-ticker universe, using
geometric Brownian motion with per-asset drift/volatility so different
assets behave differently.

This is NOT part of the final project deliverable -- it exists purely so
the rest of the pipeline can be sanity-checked without internet access.
"""

import numpy as np
import pandas as pd

TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "JNJ", "AMZN", "INFY", "WIT", "HDB", "IBN"]

ASSET_PARAMS = {
    "AAPL":  (0.15, 0.28), "MSFT": (0.14, 0.26), "GOOGL": (0.12, 0.30),
    "JPM":   (0.08, 0.24), "JNJ":  (0.06, 0.16), "AMZN":  (0.13, 0.34),
    "INFY":  (0.10, 0.27), "WIT":  (0.07, 0.25), "HDB":   (0.09, 0.29),
    "IBN":   (0.08, 0.31),
}


def generate_ticker(ticker: str, n_days: int, seed: int) -> pd.DataFrame:
    mu, sigma = ASSET_PARAMS[ticker]
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    daily_drift = mu * dt
    daily_vol = sigma * np.sqrt(dt)

    shocks = rng.normal(daily_drift, daily_vol, n_days)
    log_prices = np.cumsum(shocks)
    close = 100 * np.exp(log_prices)

    high = close * (1 + np.abs(rng.normal(0, 0.006, n_days)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n_days)))
    open_ = close * (1 + rng.normal(0, 0.003, n_days))
    volume = rng.integers(1_000_000, 8_000_000, n_days)

    dates = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame({
        "Date": dates, "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume, "Ticker": ticker,
    })


def generate_all(n_days: int = 2500) -> dict:
    return {t: generate_ticker(t, n_days, seed=i) for i, t in enumerate(TICKERS)}


if __name__ == "__main__":
    data = generate_all()
    for ticker, df in data.items():
        print(f"{ticker}: {len(df)} rows, close range [{df['Close'].min():.1f}, {df['Close'].max():.1f}]")
