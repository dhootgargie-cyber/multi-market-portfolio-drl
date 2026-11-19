import os
import numpy as np
from stable_baselines3 import PPO
from build_dataset import build_multi_asset_matrices, temporal_train_val_test_split, load_raw_csvs
from trading_env import PortfolioTradingEnv

raw = load_raw_csvs("data/raw")
feat, price, dates, tickers = build_multi_asset_matrices(raw)
(tr_f, tr_p, _), (va_f, va_p, _), (te_f, te_p, _) = temporal_train_val_test_split(feat, price, dates)

ckpt_path = (
    "checkpoints/ppo_best_val.zip"
    if os.path.exists("checkpoints/ppo_best_val.zip")
    else "checkpoints/ppo_300000_steps.zip"
)
print(f"Evaluating: {ckpt_path}")
model = PPO.load(ckpt_path)
env = PortfolioTradingEnv(te_f, te_p, lookback=30)
obs, _ = env.reset()
done, actions = False, []
while not done:
    a, _ = model.predict(obs, deterministic=True)
    actions.append(int(a))
    obs, r, term, trunc, info = env.step(int(a))
    done = term or trunc
print("Action counts (index 0 = hold):", np.bincount(actions, minlength=21))
print(f"Final value: {info['portfolio_value']:,.0f}")
