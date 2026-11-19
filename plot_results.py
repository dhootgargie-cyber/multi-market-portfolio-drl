"""
plot_results.py
---------------
Generates publication-ready charts for the paper and presentation slides
using the saved best-validation checkpoints (`checkpoints/ppo_best_val.zip`
and `checkpoints/dqn_best_val.zip`) -- runs in a few seconds without retraining.

Outputs saved to `figures/`:
  1. figures/equity_curves.png       : Portfolio value over the held-out test period
  2. figures/drawdown_curves.png     : Underwater drawdown curves across models
  3. figures/feature_ablation.png    : Bar chart of Return, Sharpe, and Max DD by stage

Run directly: python plot_results.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO

from build_dataset import build_multi_asset_matrices, temporal_train_val_test_split, load_raw_csvs
from trading_env import PortfolioTradingEnv
from train_dqn import evaluate_agent
from classical_baselines import buy_and_hold, equal_weighted_rebalanced, markowitz_backtest


def compute_drawdown_series(values: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(values)
    return -((running_max - values) / running_max) * 100.0


def main():
    os.makedirs("figures", exist_ok=True)

    raw = load_raw_csvs("data/raw")
    feat_mat, price_mat, dates, tickers = build_multi_asset_matrices(raw)
    (tr_f, tr_p, tr_d), (va_f, va_p, va_d), (te_f, te_p, te_d) = temporal_train_val_test_split(
        feat_mat, price_mat, dates
    )

    test_dates = pd.to_datetime(te_d[30:])
    curves = {}

    if os.path.exists("checkpoints/ppo_best_val.zip"):
        ppo = PPO.load("checkpoints/ppo_best_val.zip")
        curves["PPO (Best Val)"] = evaluate_agent(ppo, PortfolioTradingEnv(te_f, te_p, lookback=30))

    if os.path.exists("checkpoints/dqn_best_val.zip"):
        dqn = DQN.load("checkpoints/dqn_best_val.zip")
        curves["DQN (Best Val)"] = evaluate_agent(dqn, PortfolioTradingEnv(te_f, te_p, lookback=30))

    train_val_prices = np.concatenate([tr_p, va_p], axis=0)
    curves["Buy & Hold"] = buy_and_hold(te_p[30:])
    curves["Equal-Weighted"] = equal_weighted_rebalanced(te_p[30:])
    curves["Markowitz (MPT)"] = markowitz_backtest(train_val_prices, te_p[30:])

    # 1. Equity Curves Plot
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)

    styles = {
        "PPO (Best Val)": ("#1f77b4", 2.6, "-"),
        "DQN (Best Val)": ("#ff7f0e", 2.2, "-"),
        "Buy & Hold": ("#2ca02c", 1.8, "--"),
        "Equal-Weighted": ("#9467bd", 1.8, "-."),
        "Markowitz (MPT)": ("#7f7f7f", 1.6, ":"),
    }

    for name, values in curves.items():
        color, lw, ls = styles.get(name, ("black", 1.5, "-"))
        n = min(len(test_dates), len(values))
        ax.plot(test_dates[:n], values[:n], label=f"{name} (${values[-1]:,.0f})", color=color, linewidth=lw, linestyle=ls)

    ax.set_title("Out-of-Sample Portfolio Value Comparison (Test Period)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.legend(frameon=True, fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig("figures/equity_curves.png")
    plt.close(fig)
    print("Saved: figures/equity_curves.png")

    # 2. Drawdown Plot
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=200)
    for name, values in curves.items():
        color, lw, ls = styles.get(name, ("black", 1.5, "-"))
        dd = compute_drawdown_series(values)
        n = min(len(test_dates), len(dd))
        ax.plot(test_dates[:n], dd[:n], label=f"{name} (Max DD: {abs(dd.min()):.2f}%)", color=color, linewidth=lw, linestyle=ls)

    ax.set_title("Portfolio Drawdown Over Time (Test Period)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.legend(frameon=True, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig("figures/drawdown_curves.png")
    plt.close(fig)
    print("Saved: figures/drawdown_curves.png")

    # 3. Feature Ablation Summary Chart
    stages = ["Stage 1:\nPrice Only\n(2 feat)", "Stage 2:\n+ Momentum\n(5 feat)", "Stage 3:\n+ Volatility\n(7 feat)", "Stage 4:\n+ Trend (Full)\n(10 feat)"]
    returns_pct = [40.35, 16.85, 26.87, 57.39]
    sharpes = [1.932, 1.230, 1.424, 1.899]
    max_dds = [7.30, 5.29, 6.42, 10.09]

    x = np.arange(len(stages))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)

    bars1 = ax1.bar(x - width / 2, returns_pct, width, label="Total Return (%)", color="#1f77b4")
    bars2 = ax1.bar(x + width / 2, max_dds, width, label="Max Drawdown (%)", color="#d62728")
    ax1.set_title("Feature Ablation: Return vs. Max Drawdown", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=9)
    ax1.set_ylabel("Percentage (%)")
    ax1.legend(frameon=True)
    for b in list(bars1) + list(bars2):
        ax1.annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")

    bars3 = ax2.bar(x, sharpes, width=0.45, color="#2ca02c", label="DQN Sharpe Ratio")
    ax2.axhline(1.875, color="#ff7f0e", linestyle="--", linewidth=1.8, label="Buy & Hold Sharpe (1.875)")
    ax2.set_title("Feature Ablation: Out-of-Sample Sharpe Ratio", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages, fontsize=9)
    ax2.set_ylabel("Annualized Sharpe Ratio")
    ax2.set_ylim(0, 2.3)
    ax2.legend(frameon=True, loc="lower right")
    for b in bars3:
        ax2.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")

    fig.tight_layout()
    fig.savefig("figures/feature_ablation.png")
    plt.close(fig)
    print("Saved: figures/feature_ablation.png")


if __name__ == "__main__":
    main()
