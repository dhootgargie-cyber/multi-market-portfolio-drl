"""
xgboost_baseline.py
--------------------
Supervised-ML baseline for the comparative study.

Predicts next-day price DIRECTION (up/down) from the same feature set used
by the RL agents, and a simple rule converts that prediction into a
buy/sell/hold decision -- the "predict-then-decide" approach discussed in
the Introduction slide, included so its results can be contrasted against
the RL agents that learn the decision directly.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from feature_engineering import build_features, FEATURE_COLUMNS, temporal_train_test_split


def make_labels(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """Label = 1 if next day's close is higher than today's, else 0."""
    df = df.copy()
    df["label"] = (df[price_col].shift(-1) > df[price_col]).astype(int)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    return df


def train_xgb_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label"]

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, digits=3))
    return model, preds


def predictions_to_actions(preds: np.ndarray) -> np.ndarray:
    """Predicted up -> buy signal (1), predicted down -> sell signal (-1)."""
    return np.where(preds == 1, 1, -1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    price = 100 + np.cumsum(rng.normal(0, 1, n))
    demo_df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "Close": price,
    })
    feat_df = build_features(demo_df)
    labeled_df = make_labels(feat_df)

    train_df, test_df = temporal_train_test_split(labeled_df, "Date", split_date="2021-06-01")
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    model, preds = train_xgb_baseline(train_df, test_df)
    actions = predictions_to_actions(preds)
    print("Sample actions (first 10):", actions[:10])
