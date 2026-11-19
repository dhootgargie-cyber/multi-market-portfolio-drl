"""
train_dqn.py
------------
Trains a DQN agent on PortfolioTradingEnv and evaluates it against a
Buy & Hold baseline. Automatically uses REAL data from data/raw/ (saved by
data_pipeline.py) if present, otherwise falls back to synthetic data for a
pipeline-only smoke test.

TOTAL_TIMESTEPS below controls training length:
  - 20,000   : quick pipeline smoke test (~1-2 min), NOT a real result
  - 300,000  : a reasonable first real training run (~15-40 min on a laptop CPU)
  - 500,000+ : more trustworthy, but proportionally longer

Training saves a checkpoint every CHECKPOINT_FREQ steps to checkpoints/, so
a crash or interruption doesn't lose all progress -- you can resume from
the latest checkpoint file instead of starting over.

Run directly: python train_dqn.py
"""

import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

from build_dataset import (
    build_multi_asset_matrices,
    temporal_split,
    temporal_train_val_test_split,
    load_raw_csvs,
)
from trading_env import PortfolioTradingEnv

# ---- Training length: change this one number to control run length ----
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 25_000


def sharpe_ratio(returns: np.ndarray, eps: float = 1e-9) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / (returns.std() + eps) * np.sqrt(252))


def max_drawdown(values: np.ndarray) -> float:
    running_max = np.maximum.accumulate(values)
    drawdown = (running_max - values) / running_max
    return float(drawdown.max())


def evaluate_agent(model, env: PortfolioTradingEnv):
    """Runs the trained agent through the environment once (deterministic)."""
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    return np.array(env._portfolio_value_history)


class BestValidationModelCallback(BaseCallback):
    """Evaluates the agent on a held-out validation environment every `eval_freq`
    steps and saves the checkpoint with the best validation Sharpe ratio.
    """

    def __init__(
        self,
        val_env: PortfolioTradingEnv,
        save_path: str = "checkpoints",
        name_prefix: str = "model",
        eval_freq: int = 10_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.val_env = val_env
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.eval_freq = eval_freq
        self.best_sharpe = -np.inf
        self.best_step = 0
        self.best_model_path = os.path.join(save_path, f"{name_prefix}_best_val")
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            values = evaluate_agent(self.model, self.val_env)
            returns = np.diff(values) / values[:-1]
            val_sharpe = sharpe_ratio(returns)
            if val_sharpe > self.best_sharpe:
                self.best_sharpe = val_sharpe
                self.best_step = self.num_timesteps
                self.model.save(self.best_model_path)
                if self.verbose > 0:
                    print(
                        f"  [Val] New best {self.name_prefix} at {self.best_step:,} steps "
                        f"| Val Sharpe: {val_sharpe:.3f} | Val Final: {values[-1]:,.0f}"
                    )
        return True


def buy_and_hold_baseline(price_matrix: np.ndarray, initial_cash: float = 100_000.0) -> np.ndarray:
    """Equal-dollar buy & hold across all N assets, held for the whole period."""
    n_assets = price_matrix.shape[1]
    cash_per_asset = initial_cash / n_assets
    shares = cash_per_asset / price_matrix[0]
    values = (shares * price_matrix).sum(axis=1)
    return values


def main():
    print("=== data -> features -> env -> DQN training ===\n")

    if os.path.isdir("data/raw") and os.listdir("data/raw"):
        print("1) Loading REAL data from data/raw/ (data_pipeline.py output)...")
        raw = load_raw_csvs("data/raw")
        using_real_data = True
    else:
        print("1) No data/raw/ found -- generating synthetic data instead.")
        print("   (Run data_pipeline.py first to use real market data.)")
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

    print(f"\n4) Training DQN agent for {TOTAL_TIMESTEPS:,} timesteps...")
    train_env = PortfolioTradingEnv(tr_f, tr_p, lookback=30)
    val_env = PortfolioTradingEnv(va_f, va_p, lookback=30)
    model = DQN(
        "MlpPolicy", train_env,
        learning_rate=1e-3, buffer_size=50_000, learning_starts=500,
        batch_size=64, train_freq=4, target_update_interval=250,
        exploration_fraction=0.3, seed=42, verbose=0,
    )
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ, save_path="checkpoints", name_prefix="dqn"
    )
    val_cb = BestValidationModelCallback(
        val_env, save_path="checkpoints", name_prefix="dqn", eval_freq=CHECKPOINT_FREQ, verbose=1
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=CallbackList([checkpoint_cb, val_cb]),
        progress_bar=True,
    )
    print(f"   Training complete ({TOTAL_TIMESTEPS:,} timesteps).")
    if os.path.exists(val_cb.best_model_path + ".zip"):
        print(f"   Loading best validation checkpoint (step {val_cb.best_step:,}, Val Sharpe {val_cb.best_sharpe:.3f})...")
        model = DQN.load(val_cb.best_model_path, env=train_env)

    print("\n5) Evaluating on held-out test data...")
    test_env = PortfolioTradingEnv(te_f, te_p, lookback=30)
    dqn_values = evaluate_agent(model, test_env)
    dqn_returns = np.diff(dqn_values) / dqn_values[:-1]

    bh_values = buy_and_hold_baseline(te_p[30:])
    bh_returns = np.diff(bh_values) / bh_values[:-1]

    print("\n=== Results on held-out test period ===")
    print(f"{'Metric':<22}{'DQN Agent':>15}{'Buy & Hold':>15}")
    print(f"{'Final value':<22}{dqn_values[-1]:>15,.0f}{bh_values[-1]:>15,.0f}")
    print(f"{'Total return':<22}{(dqn_values[-1]/dqn_values[0]-1)*100:>14.2f}%{(bh_values[-1]/bh_values[0]-1)*100:>14.2f}%")
    print(f"{'Sharpe ratio':<22}{sharpe_ratio(dqn_returns):>15.3f}{sharpe_ratio(bh_returns):>15.3f}")
    print(f"{'Max drawdown':<22}{max_drawdown(dqn_values)*100:>14.2f}%{max_drawdown(bh_values)*100:>14.2f}%")

    model.save("dqn_portfolio_demo")
    print("\nModel saved to dqn_portfolio_demo.zip")

    if using_real_data:
        print("\nThis run used REAL market data. 20,000 timesteps is still a")
        print("short run -- increase total_timesteps for a trustworthy result.")
    else:
        print("\nNOTE: synthetic data used -- this only validates the pipeline,")
        print("not real trading performance. Run data_pipeline.py for real data.")


if __name__ == "__main__":
    main()
