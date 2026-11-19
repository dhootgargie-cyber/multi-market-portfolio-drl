# Multi-Market Portfolio Optimization with Deep Reinforcement Learning

A controlled comparison of **DQN** and **PPO** trading agents against three classical strategies
(**Buy & Hold**, **Equal-Weighted**, **Markowitz**) on a 10-asset portfolio of US equities and
India-focused NYSE ADRs.

> **Research / education project. Not financial advice.**
> Results below are preliminary: single random seed, one test period.

## Overview

Both agents are trained in the same custom Gymnasium environment (same state, actions, reward and
constraints), and all five strategies are evaluated on the same held-out test period.

- **Assets (10):** AAPL, MSFT, GOOGL, JPM, JNJ, AMZN (US) and INFY, WIT, HDB, IBN (India-focused NYSE ADRs)
- **Data:** daily OHLCV from Yahoo Finance via `yfinance`, 2015-01-01 to 2024-12-31
  (2015-02-13 to 2024-12-30 after indicator warm-up and date alignment)
- **Features per asset (10):** daily/log return; RSI, MACD, MACD signal; Bollinger-band %, rolling volatility;
  SMA10, SMA30, price vs SMA30. The state also includes each asset's current portfolio weight and the cash ratio.
- **Environment:** 30-day lookback; `Discrete(2N+1)` actions (hold / buy asset *i* / sell asset *i*);
  0.1% transaction cost; 30% maximum position per asset; 10% trade fraction
- **Reward:** rolling Sharpe-like term, minus a drawdown penalty, minus a penalty for blocked (invalid) trades
- **Split:** strict temporal split, about 65% train / 15% validation / 20% test
  (1615 / 373 / 498 trading days). Features are standardized using **training-set statistics only**.
- **Model selection:** the best checkpoint is chosen by validation Sharpe ratio, not by test results.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python data_pipeline.py          # download data into data/raw/ (needs internet)
python build_dataset.py          # sanity check: should print "Found data/raw/"
python compare_all_models.py     # train DQN + PPO, evaluate all five strategies
python feature_ablation.py       # feature-group ablation (DQN)
```

Training length is set by `TOTAL_TIMESTEPS` at the top of the scripts. On the author's laptop,
300,000 steps took about 13 minutes for DQN and 21 minutes for PPO.
If `data/raw/` is missing, scripts fall back to synthetic data. That is only a pipeline check, not a result.

## Preliminary results (single seed, test period, $100,000 starting capital)

| Strategy | Total return | Sharpe | Max drawdown | Final value |
|---|---|---|---|---|
| PPO | 89.84% | 2.229 | 9.70% | $189,843 |
| DQN | 57.39% | 1.899 | 10.09% | $157,392 |
| Buy & Hold | 57.81% | 1.875 | 9.83% | $157,811 |
| Equal-Weighted | 56.49% | 1.905 | 9.72% | $156,487 |
| Markowitz | 49.22% | 1.608 | 9.46% | $149,224 |

Notes:
- DQN and PPO are evaluated at their validation-selected checkpoints (DQN: step 10,000; PPO: step 230,000).
  PPO's validation Sharpe at its selected checkpoint was only 0.026.
- The test window was a strong market (Buy & Hold returned about 58%), and the validation window was a harder one.
- These numbers come from one run with one seed. They are **not** evidence that PPO beats the baselines.
  Multi-seed runs (mean ± std) are planned.

## PPO stability note

An early version of the pipeline used unscaled features, and PPO drifted to holding cash on nearly every
day in later checkpoints (one checkpoint stayed at exactly $100,000). After standardizing features with
training-set statistics, later PPO checkpoints traded steadily. The change is consistent with feature scale
being the cause, but the split also changed at the same time, so this is not a controlled experiment.
`diagnose_ppo_over_time.py` reproduces the checkpoint-by-checkpoint check.

## Limitations

- Single random seed; RL results can vary a lot between seeds.
- One test period (a strong rally), so results may not generalize to other market regimes.
- Classical baselines are computed without transaction costs, while DQN and PPO pay 0.1% per trade.
- No slippage, taxes or liquidity constraints.
- Feature ablation is preliminary (single seed, DQN only).

## Project structure

| File | Purpose |
|---|---|
| `data_pipeline.py` | Downloads OHLCV data for the 10 tickers |
| `feature_engineering.py` | Computes the four feature groups |
| `build_dataset.py` | Aligns assets, builds (T, N, F) tensors, train/val/test split + standardization |
| `trading_env.py` | Custom Gymnasium trading environment |
| `train_dqn.py` | DQN training, evaluation, validation-checkpoint callback |
| `train_ppo.py` | PPO training and evaluation |
| `classical_baselines.py` | Buy & Hold, Equal-Weighted, Markowitz |
| `compare_all_models.py` | Trains both agents and prints the five-strategy comparison |
| `feature_ablation.py` | Price-only → +momentum → +volatility → +trend ablation |
| `xgboost_baseline.py` | Supervised direction-prediction baseline (not part of the main comparison) |
| `generate_synthetic_data.py` | Synthetic fallback data for pipeline checks |
| `diagnose_ppo.py`, `diagnose_ppo_over_time.py` | Inspect what a trained PPO agent does on the test period |

## Acknowledgements and AI assistance

Parts of the code and documentation were developed with AI assistance (Claude, Anthropic).
All reported numbers were produced by running the code in this repository.
<!-- Add: supervisor name, institution, and any prior work this builds on. -->

## License

MIT. See `LICENSE`.
