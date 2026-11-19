"""
build_dataset.py
-----------------
Takes per-ticker OHLCV DataFrames -- either loaded from real CSVs saved by
data_pipeline.py, or from generate_synthetic_data.py -- runs
feature_engineering.py on each, aligns them on a common date index, and
stacks them into the (T, N, F) feature matrix + (T, N) price matrix that
trading_env.py expects.
"""

import os
import numpy as np
import pandas as pd

from feature_engineering import build_features, FEATURE_COLUMNS


def load_raw_csvs(data_dir: str = "data/raw", tickers: list = None) -> dict:
    """Loads the real CSVs saved by data_pipeline.py into {ticker: DataFrame}."""
    if tickers is None:
        tickers = [f[:-4] for f in os.listdir(data_dir) if f.endswith(".csv")]

    raw_data = {}
    for ticker in tickers:
        path = os.path.join(data_dir, f"{ticker}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Could not find {path}. Run data_pipeline.py first to download real data."
            )
        df = pd.read_csv(path, parse_dates=["Date"])
        raw_data[ticker] = df
    return raw_data


def build_multi_asset_matrices(raw_data: dict):
    """
    raw_data: dict {ticker: raw OHLCV DataFrame with a 'Date' and 'Close' column}
    Returns: feature_matrix (T,N,F), price_matrix (T,N), dates (T,), tickers (N,)
    """
    tickers = sorted(raw_data.keys())
    featured = {t: build_features(raw_data[t]) for t in tickers}

    common_dates = None
    for t in tickers:
        dates = pd.to_datetime(featured[t]["Date"])
        common_dates = dates if common_dates is None else common_dates[common_dates.isin(dates)]
    common_dates = sorted(common_dates)

    T, N, F = len(common_dates), len(tickers), len(FEATURE_COLUMNS)
    feature_matrix = np.zeros((T, N, F), dtype=np.float32)
    price_matrix = np.zeros((T, N), dtype=np.float32)

    for n, t in enumerate(tickers):
        df = featured[t].copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").loc[common_dates]
        feature_matrix[:, n, :] = df[FEATURE_COLUMNS].values
        price_matrix[:, n] = df["Close"].values

    return feature_matrix, price_matrix, np.array(common_dates), tickers

def temporal_split(feature_matrix, price_matrix, dates, split_frac=0.8):
    """Strict temporal split; features standardized using TRAIN statistics only."""
    split_idx = int(len(dates) * split_frac)
    tr_f = feature_matrix[:split_idx]
    te_f = feature_matrix[split_idx:]

    mean = tr_f.mean(axis=0, keepdims=True)
    std = tr_f.std(axis=0, keepdims=True) + 1e-8
    tr_f = np.clip((tr_f - mean) / std, -10, 10)
    te_f = np.clip((te_f - mean) / std, -10, 10)

    train = (tr_f, price_matrix[:split_idx], dates[:split_idx])
    test = (te_f, price_matrix[split_idx:], dates[split_idx:])
    return train, test


def temporal_train_val_test_split(
    feature_matrix, price_matrix, dates, train_frac=0.65, val_frac=0.15
):
    """Strict 3-way temporal split (Train / Validation / Test).
    By default, train (65%) + val (15%) = 80%, keeping the final 20% test window
    identical to temporal_split(split_frac=0.8).
    Features are standardized using TRAIN statistics only (no lookahead).
    """
    n = len(dates)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    tr_f = feature_matrix[:train_end]
    va_f = feature_matrix[train_end:val_end]
    te_f = feature_matrix[val_end:]

    mean = tr_f.mean(axis=0, keepdims=True)
    std = tr_f.std(axis=0, keepdims=True) + 1e-8
    tr_f = np.clip((tr_f - mean) / std, -10, 10)
    va_f = np.clip((va_f - mean) / std, -10, 10)
    te_f = np.clip((te_f - mean) / std, -10, 10)

    train = (tr_f, price_matrix[:train_end], dates[:train_end])
    val = (va_f, price_matrix[train_end:val_end], dates[train_end:val_end])
    test = (te_f, price_matrix[val_end:], dates[val_end:])
    return train, val, test


if __name__ == "__main__":
    if os.path.isdir("data/raw") and os.listdir("data/raw"):
        print("Found data/raw/ -- loading REAL data from data_pipeline.py output.")
        raw = load_raw_csvs("data/raw")
    else:
        print("No data/raw/ found -- falling back to synthetic data.")
        print("Run data_pipeline.py first, then re-run this script, to use real data.")
        from generate_synthetic_data import generate_all
        raw = generate_all(n_days=1000)

    feat_mat, price_mat, dates, tickers = build_multi_asset_matrices(raw)
    print(f"Feature matrix shape: {feat_mat.shape}  (T, N, F)")
    print(f"Price matrix shape:   {price_mat.shape}  (T, N)")
    print(f"Tickers (N={len(tickers)}): {tickers}")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    (tr_f, tr_p, tr_d), (te_f, te_p, te_d) = temporal_split(feat_mat, price_mat, dates)
    print(f"Train: {tr_f.shape[0]} days | Test: {te_f.shape[0]} days")
