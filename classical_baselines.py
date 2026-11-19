"""
classical_baselines.py
------------------------
The three classical (non-ML) strategies from the Scope slide, computed
directly from price data -- no training involved.

  - Buy & Hold        : equal dollar amount in each asset, held forever
  - Equal-Weighted     : rebalanced periodically back to equal weights
  - Markowitz (MPT)    : mean-variance optimal weights computed from the
                          TRAINING period's historical returns, then held
                          fixed through the test period (deliberately
                          static/backward-looking, per the Literature
                          Review slide).
"""

import numpy as np


def buy_and_hold(price_matrix: np.ndarray, initial_cash: float = 100_000.0) -> np.ndarray:
    """Equal dollar amount in each asset on day 0, never rebalanced."""
    n_assets = price_matrix.shape[1]
    cash_per_asset = initial_cash / n_assets
    shares = cash_per_asset / price_matrix[0]
    return (shares * price_matrix).sum(axis=1)


def equal_weighted_rebalanced(price_matrix: np.ndarray, initial_cash: float = 100_000.0,
                               rebalance_every: int = 21) -> np.ndarray:
    """Equal-weighted, rebalanced back to equal weights every ~month (21 trading days)."""
    T, N = price_matrix.shape
    values = np.zeros(T)
    cash_per_asset = initial_cash / N
    shares = cash_per_asset / price_matrix[0]
    values[0] = initial_cash

    for t in range(1, T):
        portfolio_value = (shares * price_matrix[t]).sum()
        values[t] = portfolio_value
        if t % rebalance_every == 0:
            shares = (portfolio_value / N) / price_matrix[t]

    return values


def markowitz_weights(train_returns: np.ndarray, risk_free_rate: float = 0.0) -> np.ndarray:
    """Maximum-Sharpe (tangency) portfolio weights from historical training returns."""
    mean_returns = train_returns.mean(axis=0)
    cov_matrix = np.cov(train_returns.T)

    inv_cov = np.linalg.pinv(cov_matrix)
    excess_returns = mean_returns - risk_free_rate

    raw_weights = inv_cov @ excess_returns
    raw_weights = np.clip(raw_weights, 0, None)  # no short-selling
    if raw_weights.sum() == 0:
        return np.ones(len(mean_returns)) / len(mean_returns)
    return raw_weights / raw_weights.sum()


def markowitz_backtest(train_price_matrix: np.ndarray, test_price_matrix: np.ndarray,
                        initial_cash: float = 100_000.0) -> np.ndarray:
    """Computes Markowitz weights ONCE from training data, holds fixed through test period."""
    train_returns = np.diff(train_price_matrix, axis=0) / train_price_matrix[:-1]
    weights = markowitz_weights(train_returns)

    dollar_alloc = weights * initial_cash
    shares = dollar_alloc / test_price_matrix[0]
    return (shares * test_price_matrix).sum(axis=1)


if __name__ == "__main__":
    from generate_synthetic_data import generate_all
    from build_dataset import build_multi_asset_matrices, temporal_split

    raw = generate_all(n_days=1000)
    feat_mat, price_mat, dates, tickers = build_multi_asset_matrices(raw)
    (tr_f, tr_p, tr_d), (te_f, te_p, te_d) = temporal_split(feat_mat, price_mat, dates)

    bh = buy_and_hold(te_p)
    ew = equal_weighted_rebalanced(te_p)
    mw = markowitz_backtest(tr_p, te_p)

    print(f"{'Strategy':<20}{'Final Value':>15}{'Total Return':>15}")
    for name, values in [("Buy & Hold", bh), ("Equal-Weighted", ew), ("Markowitz", mw)]:
        total_return = (values[-1] / values[0] - 1) * 100
        print(f"{name:<20}{values[-1]:>15,.0f}{total_return:>14.2f}%")
