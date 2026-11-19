"""
compare_all_models.py
-----------------------
Runs the full comparative study in one script -- DQN, PPO, Buy & Hold,
Equal-Weighted, and Markowitz -- all on the exact same train/test split,
so the comparison is fair. This is the script whose output table becomes
your Results section.

TOTAL_TIMESTEPS controls how long DQN and PPO train for -- see
train_dqn.py for guidance on run length vs. time.

Run directly: python compare_all_models.py
"""

import os
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from build_dataset import build_multi_asset_matrices, temporal_train_val_test_split, load_raw_csvs
from trading_env import PortfolioTradingEnv
from train_dqn import sharpe_ratio, max_drawdown, evaluate_agent, BestValidationModelCallback
from classical_baselines import buy_and_hold, equal_weighted_rebalanced, markowitz_backtest

TOTAL_TIMESTEPS = 300_000
CHECKPOINT_FREQ = 25_000
EVAL_FREQ = 10_000


def load_data():
    if os.path.isdir("data/raw") and os.listdir("data/raw"):
        print("Using REAL data from data/raw/")
        raw = load_raw_csvs("data/raw")
    else:
        print("No data/raw/ found -- using synthetic data (pipeline check only).")
        from generate_synthetic_data import generate_all
        raw = generate_all(n_days=2000)
    return raw


def report_row(name, values):
    returns = np.diff(values) / values[:-1]
    total_return = (values[-1] / values[0] - 1) * 100
    return {
        "name": name,
        "final_value": values[-1],
        "total_return_pct": total_return,
        "sharpe": sharpe_ratio(returns),
        "max_drawdown_pct": max_drawdown(values) * 100,
    }


def main():
    raw = load_data()
    feat_mat, price_mat, dates, tickers = build_multi_asset_matrices(raw)
    (tr_f, tr_p, tr_d), (va_f, va_p, va_d), (te_f, te_p, te_d) = temporal_train_val_test_split(
        feat_mat, price_mat, dates
    )
    print(
        f"Train: {tr_f.shape[0]} days | Val: {va_f.shape[0]} days | "
        f"Test: {te_f.shape[0]} days | Assets: {tickers}\n"
    )

    results = []
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Training DQN ({TOTAL_TIMESTEPS:,} steps, selecting best validation checkpoint)...")
    train_env = PortfolioTradingEnv(tr_f, tr_p, lookback=30)
    val_env = PortfolioTradingEnv(va_f, va_p, lookback=30)
    dqn = DQN(
        "MlpPolicy", train_env,
        learning_rate=1e-3, buffer_size=50_000, learning_starts=500,
        batch_size=64, train_freq=4, target_update_interval=250,
        exploration_fraction=0.3, seed=42, verbose=0,
    )
    dqn_ckpt = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path="checkpoints", name_prefix="dqn")
    dqn_val_cb = BestValidationModelCallback(
        val_env, save_path="checkpoints", name_prefix="dqn", eval_freq=EVAL_FREQ, verbose=1
    )
    dqn.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([dqn_ckpt, dqn_val_cb]),
        progress_bar=True,
    )
    if os.path.exists(dqn_val_cb.best_model_path + ".zip"):
        print(f"  -> Loaded best DQN validation checkpoint from step {dqn_val_cb.best_step:,}\n")
        dqn = DQN.load(dqn_val_cb.best_model_path, env=train_env)
    test_env = PortfolioTradingEnv(te_f, te_p, lookback=30)
    results.append(report_row("DQN", evaluate_agent(dqn, test_env)))

    print(f"Training PPO ({TOTAL_TIMESTEPS:,} steps, selecting best validation checkpoint)...")
    train_env = PortfolioTradingEnv(tr_f, tr_p, lookback=30)
    val_env = PortfolioTradingEnv(va_f, va_p, lookback=30)
    ppo = PPO(
        "MlpPolicy", train_env,
        learning_rate=3e-4, n_steps=1024, batch_size=64,
        n_epochs=10, gamma=0.99, ent_coef=0.01, seed=42, verbose=0,
    )
    ppo_ckpt = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path="checkpoints", name_prefix="ppo")
    ppo_val_cb = BestValidationModelCallback(
        val_env, save_path="checkpoints", name_prefix="ppo", eval_freq=EVAL_FREQ, verbose=1
    )
    ppo.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([ppo_ckpt, ppo_val_cb]),
        progress_bar=True,
    )
    if os.path.exists(ppo_val_cb.best_model_path + ".zip"):
        print(f"  -> Loaded best PPO validation checkpoint from step {ppo_val_cb.best_step:,}\n")
        ppo = PPO.load(ppo_val_cb.best_model_path, env=train_env)
    test_env = PortfolioTradingEnv(te_f, te_p, lookback=30)
    results.append(report_row("PPO", evaluate_agent(ppo, test_env)))

    print("Computing classical baselines...")
    train_val_prices = np.concatenate([tr_p, va_p], axis=0)
    results.append(report_row("Buy & Hold", buy_and_hold(te_p[30:])))
    results.append(report_row("Equal-Weighted", equal_weighted_rebalanced(te_p[30:])))
    results.append(report_row("Markowitz", markowitz_backtest(train_val_prices, te_p[30:])))

    print("\n" + "=" * 70)
    print(f"{'Model':<18}{'Final Value':>14}{'Total Return':>15}{'Sharpe':>10}{'Max DD':>10}")
    print("=" * 70)
    for r in results:
        print(f"{r['name']:<18}{r['final_value']:>14,.0f}{r['total_return_pct']:>14.2f}%"
              f"{r['sharpe']:>10.3f}{r['max_drawdown_pct']:>9.2f}%")
    print("=" * 70)
    print("\nReminder: check whether TOTAL_TIMESTEPS at the top of this file")
    print("is high enough to trust before reporting these numbers in the paper.")


if __name__ == "__main__":
    main()
