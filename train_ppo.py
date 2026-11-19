"""
train_ppo.py
------------
Trains a PPO agent on the SAME PortfolioTradingEnv used for DQN, so the
comparison between the two is on identical footing -- same state, same
actions, same reward, same data split.

TOTAL_TIMESTEPS below controls training length -- see train_dqn.py for
guidance on how long different values take.

Run directly: python train_ppo.py
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

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
CHECKPOINT_FREQ = 25_000


def main():
    print("=== data -> features -> env -> PPO training ===\n")

    if os.path.isdir("data/raw") and os.listdir("data/raw"):
        print("1) Loading REAL data from data/raw/ (data_pipeline.py output)...")
        raw = load_raw_csvs("data/raw")
        using_real_data = True
    else:
        print("1) No data/raw/ found -- generating synthetic data instead.")
        from generate_synthetic_data import generate_all
        raw = generate_all(n_days=2000)
        using_real_data = False

    print("2) Building feature matrix for all 10 assets...")
    feat_mat, price_mat, dates, tickers = build_multi_asset_matrices(raw)
    print(f"   Shape: {feat_mat.shape} (T, N, F) across {len(tickers)} tickers")

    print("3) Temporal train/val/test split (65/15/20, no shuffling, no lookahead)...")
    (tr_f, tr_p, tr_d), (va_f, va_p, va_d), (te_f, te_p, te_d) = temporal_train_val_test_split(
        feat_mat, price_mat, dates
    )
    print(f"   Train: {tr_f.shape[0]} days | Val: {va_f.shape[0]} days | Test: {te_f.shape[0]} days")

    print(f"\n4) Training PPO agent for {TOTAL_TIMESTEPS:,} timesteps...")
    train_env = PortfolioTradingEnv(tr_f, tr_p, lookback=30)
    val_env = PortfolioTradingEnv(va_f, va_p, lookback=30)
    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=1024, batch_size=64,
        n_epochs=10, gamma=0.99, ent_coef=0.01, seed=42, verbose=0,
    )
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ, save_path="checkpoints", name_prefix="ppo"
    )
    val_cb = BestValidationModelCallback(
        val_env, save_path="checkpoints", name_prefix="ppo", eval_freq=CHECKPOINT_FREQ, verbose=1
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([checkpoint_cb, val_cb]),
        progress_bar=True,
    )
    print(f"   Training complete ({TOTAL_TIMESTEPS:,} timesteps).")
    if os.path.exists(val_cb.best_model_path + ".zip"):
        print(f"   Loading best validation checkpoint (step {val_cb.best_step:,}, Val Sharpe {val_cb.best_sharpe:.3f})...")
        model = PPO.load(val_cb.best_model_path, env=train_env)

    print("\n5) Evaluating on held-out test data...")
    test_env = PortfolioTradingEnv(te_f, te_p, lookback=30)
    ppo_values = evaluate_agent(model, test_env)
    ppo_returns = np.diff(ppo_values) / ppo_values[:-1]

    bh_values = buy_and_hold_baseline(te_p[30:])
    bh_returns = np.diff(bh_values) / bh_values[:-1]

    print("\n=== Results on held-out test period ===")
    print(f"{'Metric':<22}{'PPO Agent':>15}{'Buy & Hold':>15}")
    print(f"{'Final value':<22}{ppo_values[-1]:>15,.0f}{bh_values[-1]:>15,.0f}")
    print(f"{'Total return':<22}{(ppo_values[-1]/ppo_values[0]-1)*100:>14.2f}%{(bh_values[-1]/bh_values[0]-1)*100:>14.2f}%")
    print(f"{'Sharpe ratio':<22}{sharpe_ratio(ppo_returns):>15.3f}{sharpe_ratio(bh_returns):>15.3f}")
    print(f"{'Max drawdown':<22}{max_drawdown(ppo_values)*100:>14.2f}%{max_drawdown(bh_values)*100:>14.2f}%")

    model.save("ppo_portfolio_demo")
    print("\nModel saved to ppo_portfolio_demo.zip")

    if using_real_data:
        print("\nThis run used REAL market data. 20,000 timesteps is still a")
        print("short run -- increase total_timesteps for a trustworthy result.")
    else:
        print("\nNOTE: synthetic data used -- this only validates the pipeline,")
        print("not real trading performance. Run data_pipeline.py for real data.")


if __name__ == "__main__":
    main()
