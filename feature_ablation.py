"""
feature_ablation.py
---------------------
Phase 3: Feature-ablation study.

Tests whether each feature group actually helps, rather than assuming it
does, by training a DQN agent on cumulative feature subsets:

  Stage 1: price only
  Stage 2: price + momentum
  Stage 3: price + momentum + volatility
  Stage 4: price + momentum + volatility + trend  (= full feature set,
           same as every other script in this project)

Each stage uses the SAME environment design, SAME training length, and
SAME train/test split -- the only thing that changes is which columns of
the feature matrix are visible to the agent. If Sharpe ratio improves as
groups are added, that's empirical evidence the group helps; if it doesn't,
that's an equally valid (and honest) finding to report.

TOTAL_TIMESTEPS is deliberately smaller than train_dqn.py's default, since
this script trains 4 separate models in one run -- increase it once you've
confirmed the mechanics work, matching TOTAL_TIMESTEPS in train_dqn.py for
a fully rigorous final version.

Run directly: python feature_ablation.py
"""

import os
import numpy as np
from stable_baselines3 import DQN

from feature_engineering import FEATURE_COLUMNS, FEATURE_GROUPS
from build_dataset import build_multi_asset_matrices, temporal_train_val_test_split, load_raw_csvs
from trading_env import PortfolioTradingEnv
from train_dqn import (
    sharpe_ratio,
    max_drawdown,
    buy_and_hold_baseline,
    evaluate_agent,
    BestValidationModelCallback,
)

TOTAL_TIMESTEPS = 300_000
EVAL_FREQ = 10_000

# Cumulative stages, in the order described in the Feature Selection Rationale.
STAGES = [
    ("Price only", ["price"]),
    ("+ Momentum", ["price", "momentum"]),
    ("+ Volatility", ["price", "momentum", "volatility"]),
    ("+ Trend (full feature set)", ["price", "momentum", "volatility", "trend"]),
]


def column_indices_for_groups(group_names: list) -> list:
    """Returns the indices into FEATURE_COLUMNS for the given group names,
    preserving FEATURE_COLUMNS' original order."""
    wanted_cols = set()
    for g in group_names:
        wanted_cols.update(FEATURE_GROUPS[g])
    return [i for i, col in enumerate(FEATURE_COLUMNS) if col in wanted_cols]


def load_data():
    if os.path.isdir("data/raw") and os.listdir("data/raw"):
        print("Using REAL data from data/raw/\n")
        raw = load_raw_csvs("data/raw")
    else:
        print("No data/raw/ found -- using synthetic data (pipeline check only).\n")
        from generate_synthetic_data import generate_all
        raw = generate_all(n_days=2000)
    return raw


def run_stage(stage_name: str, group_names: list, full_feat_mat: np.ndarray,
              price_mat: np.ndarray, dates) -> dict:
    col_idx = column_indices_for_groups(group_names)
    feat_subset = full_feat_mat[:, :, col_idx]  # keep only the selected feature columns

    (tr_f, tr_p, _), (va_f, va_p, _), (te_f, te_p, _) = temporal_train_val_test_split(
        feat_subset, price_mat, dates
    )

    train_env = PortfolioTradingEnv(tr_f, tr_p, lookback=30)
    val_env = PortfolioTradingEnv(va_f, va_p, lookback=30)
    model = DQN(
        "MlpPolicy", train_env,
        learning_rate=1e-3, buffer_size=50_000, learning_starts=500,
        batch_size=64, train_freq=4, target_update_interval=250,
        exploration_fraction=0.3, seed=42, verbose=0,
    )
    safe_prefix = f"ablation_{len(col_idx)}feat"
    val_cb = BestValidationModelCallback(
        val_env, save_path="checkpoints", name_prefix=safe_prefix, eval_freq=EVAL_FREQ, verbose=1
    )
    print(f"  Training on [{stage_name}] -- {len(col_idx)} features/asset -- "
          f"{TOTAL_TIMESTEPS:,} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=val_cb, progress_bar=True)
    if os.path.exists(val_cb.best_model_path + ".zip"):
        print(f"  -> Loaded best validation checkpoint from step {val_cb.best_step:,}")
        model = DQN.load(val_cb.best_model_path, env=train_env)

    test_env = PortfolioTradingEnv(te_f, te_p, lookback=30)
    values = evaluate_agent(model, test_env)
    returns = np.diff(values) / values[:-1]

    return {
        "stage": stage_name,
        "n_features": len(col_idx),
        "final_value": values[-1],
        "total_return_pct": (values[-1] / values[0] - 1) * 100,
        "sharpe": sharpe_ratio(returns),
        "max_drawdown_pct": max_drawdown(values) * 100,
    }


def main():
    print("=== Feature-Ablation Study ===\n")
    raw = load_data()
    full_feat_mat, price_mat, dates, tickers = build_multi_asset_matrices(raw)
    print(f"Full feature set: {len(FEATURE_COLUMNS)} features/asset across {len(tickers)} assets\n")

    results = []
    for stage_name, group_names in STAGES:
        results.append(run_stage(stage_name, group_names, full_feat_mat, price_mat, dates))

    bh_values = buy_and_hold_baseline(price_mat[int(len(dates) * 0.8) + 30:])
    bh_sharpe = sharpe_ratio(np.diff(bh_values) / bh_values[:-1])

    print("\n" + "=" * 78)
    print(f"{'Stage':<28}{'# Features':>12}{'Total Return':>15}{'Sharpe':>10}{'Max DD':>10}")
    print("=" * 78)
    for r in results:
        print(f"{r['stage']:<28}{r['n_features']:>12}{r['total_return_pct']:>14.2f}%"
              f"{r['sharpe']:>10.3f}{r['max_drawdown_pct']:>9.2f}%")
    print("-" * 78)
    print(f"{'(Reference: Buy & Hold)':<28}{'--':>12}{'--':>15}{bh_sharpe:>10.3f}{'--':>10}")
    print("=" * 78)

    print("\nHow to read this table:")
    print("- If Sharpe increases as each group is added, that group appears to help.")
    print("- If Sharpe does NOT increase (or drops), that's a legitimate finding too --")
    print("  it means that feature group isn't earning its place in the state space,")
    print("  at least at this training length. Report it honestly either way.")
    print(f"\nNote: {TOTAL_TIMESTEPS:,} steps per stage is modest, chosen so all 4 stages")
    print("finish in one run. Increase it (matching train_dqn.py's TOTAL_TIMESTEPS)")
    print("for a more rigorous final version of this study.")


if __name__ == "__main__":
    main()
