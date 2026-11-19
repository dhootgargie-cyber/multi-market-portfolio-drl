"""
feature_engineering.py
-----------------------
Phase 1: Feature Engineering

Computes the grouped feature set described on the "Feature Selection
Rationale" slide, for a single ticker's OHLCV DataFrame:

  Group 1 - Price / Returns   : daily_return, log_return
  Group 2 - Momentum          : rsi_14, macd, macd_signal
  Group 3 - Volatility        : bb_pct, rolling_std_20
  Group 4 - Trend context     : sma_10, sma_30, price_vs_sma30

No external TA library is required -- every indicator is implemented
directly with pandas so this only depends on pandas/numpy.
"""

import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    df = df.copy()
    df["daily_return"] = df[price_col].pct_change()
    df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def compute_rsi(df: pd.DataFrame, price_col: str = "Close", period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50)
    return df


def compute_macd(df: pd.DataFrame, price_col: str = "Close",
                  fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    return df


def compute_bollinger(df: pd.DataFrame, price_col: str = "Close", period: int = 20, num_std: int = 2) -> pd.DataFrame:
    df = df.copy()
    sma = df[price_col].rolling(window=period, min_periods=period).mean()
    std = df[price_col].rolling(window=period, min_periods=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = (upper - lower).replace(0, np.nan)
    df["bb_pct"] = ((df[price_col] - lower) / band_width).clip(0, 1)
    df["rolling_std_20"] = std / df[price_col]
    return df


def compute_trend(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    df = df.copy()
    df["sma_10"] = df[price_col].rolling(window=10, min_periods=10).mean()
    df["sma_30"] = df[price_col].rolling(window=30, min_periods=30).mean()
    df["price_vs_sma30"] = (df[price_col] - df["sma_30"]) / df["sma_30"]
    return df


FEATURE_COLUMNS = [
    "daily_return", "log_return",
    "rsi_14", "macd", "macd_signal",
    "bb_pct", "rolling_std_20",
    "sma_10", "sma_30", "price_vs_sma30",
]

# Named groups, matching the "Feature Selection Rationale" slide -- used by
# feature_ablation.py to test each group's contribution incrementally.
FEATURE_GROUPS = {
    "price":      ["daily_return", "log_return"],
    "momentum":   ["rsi_14", "macd", "macd_signal"],
    "volatility": ["bb_pct", "rolling_std_20"],
    "trend":      ["sma_10", "sma_30", "price_vs_sma30"],
}


def build_features(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Apply the full feature pipeline to one ticker's OHLCV DataFrame."""
    df = compute_returns(df, price_col)
    df = compute_rsi(df, price_col)
    df = compute_macd(df, price_col)
    df = compute_bollinger(df, price_col)
    df = compute_trend(df, price_col)
    df = df.dropna().reset_index(drop=True)
    return df


def temporal_train_test_split(df: pd.DataFrame, date_col: str, split_date: str):
    """Strict temporal split -- everything before split_date is train, rest is test."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    train = df[df[date_col] < split_date].reset_index(drop=True)
    test = df[df[date_col] >= split_date].reset_index(drop=True)
    return train, test


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 100
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    demo_df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n, freq="B"),
        "Close": price,
    })
    out = build_features(demo_df)
    print(f"Input rows: {n} -> Output rows after warm-up drop: {len(out)}")
    print(out[["Date"] + FEATURE_COLUMNS].head())
