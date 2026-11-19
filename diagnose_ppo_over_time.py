import glob, os, re
import numpy as np
from stable_baselines3 import PPO
from build_dataset import build_multi_asset_matrices, temporal_train_val_test_split, load_raw_csvs
from trading_env import PortfolioTradingEnv
from train_dqn import sharpe_ratio

raw = load_raw_csvs("data/raw")
feat, price, dates, tickers = build_multi_asset_matrices(raw)
(tr_f, tr_p, _), (va_f, va_p, _), (te_f, te_p, _) = temporal_train_val_test_split(feat, price, dates)

def steps(path):
    return int(re.search(r"ppo_(\d+)_steps", path).group(1))

def eval_on(model, f_mat, p_mat):
    env = PortfolioTradingEnv(f_mat, p_mat, lookback=30)
    obs, _ = env.reset()
    done, actions = False, []
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        actions.append(int(a))
        obs, r, term, trunc, info = env.step(int(a))
        done = term or trunc
    values = np.array(env._portfolio_value_history)
    rets = np.diff(values) / values[:-1]
    hold_pct = 100 * np.mean(np.array(actions) == 0)
    return hold_pct, sharpe_ratio(rets), info["portfolio_value"]

# Evaluate the 25k..300k checkpoints from the current 300,000-step run
files = [p for p in sorted(glob.glob("checkpoints/ppo_*_steps.zip"), key=steps) if steps(p) <= 300_000]
for path in files:
    model = PPO.load(path)
    hold_pct, val_sharpe, _ = eval_on(model, va_f, va_p)
    _, test_sharpe, test_final = eval_on(model, te_f, te_p)
    print(
        f"{steps(path):>9,} steps | hold {hold_pct:5.1f}% | "
        f"val Sharpe {val_sharpe:6.3f} | test Sharpe {test_sharpe:6.3f} | test final {test_final:,.0f}"
    )

if os.path.exists("checkpoints/ppo_best_val.zip"):
    model = PPO.load("checkpoints/ppo_best_val.zip")
    hold_pct, val_sharpe, _ = eval_on(model, va_f, va_p)
    _, test_sharpe, test_final = eval_on(model, te_f, te_p)
    print("-" * 86)
    print(
        f"{'BEST VAL':>15} | hold {hold_pct:5.1f}% | "
        f"val Sharpe {val_sharpe:6.3f} | test Sharpe {test_sharpe:6.3f} | test final {test_final:,.0f}"
    )
