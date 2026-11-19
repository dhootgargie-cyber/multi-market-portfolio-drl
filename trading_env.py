"""
trading_env.py
--------------
Custom Gymnasium trading environment.

  State (per step)  : for each of the N assets -> [FEATURE_COLUMNS] (10 features)
                       + portfolio-state features per asset -> current_weight (1)
                       + a single global cash_ratio feature
                       => N * 11 + 1 values, flattened into one Box observation.

  Action space       : Discrete(2*N + 1)
                        0            -> hold everything
                        1..N         -> buy asset (i-1) using a fixed trade size
                        N+1..2N      -> sell asset (i-N-1)

  Reward             : change in a rolling Sharpe-like ratio, minus a
                        drawdown penalty.

  Constraints         : transaction cost (default 0.1%) applied on every buy
                        /sell trade; max position size cap (default 30% of
                        portfolio value) enforced by blocking oversized buys.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PortfolioTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_matrix: np.ndarray,
        price_matrix: np.ndarray,
        initial_cash: float = 100_000.0,
        transaction_cost: float = 0.001,
        max_position_pct: float = 0.30,
        trade_fraction: float = 0.10,
        lookback: int = 30,
    ):
        super().__init__()
        assert feature_matrix.shape[:2] == price_matrix.shape, "feature/price shape mismatch"

        self.features = feature_matrix.astype(np.float32)
        self.prices = price_matrix.astype(np.float32)
        self.T, self.N, self.F = self.features.shape

        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_position_pct = max_position_pct
        self.trade_fraction = trade_fraction
        self.lookback = lookback

        self.action_space = spaces.Discrete(2 * self.N + 1)

        obs_dim = self.N * (self.F + 1) + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._returns_history = []
        self._prev_sharpe = 0.0
        self._prev_drawdown = 0.0
        self._reset_state()

    def _reset_state(self):
        self.t = self.lookback
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.N, dtype=np.float64)
        self._returns_history = []
        self._portfolio_value_history = [self.initial_cash]
        self._prev_sharpe = 0.0
        self._prev_drawdown = 0.0

    def _portfolio_value(self, t: int) -> float:
        return float(self.cash + np.sum(self.holdings * self.prices[t]))

    def _get_obs(self) -> np.ndarray:
        pv = self._portfolio_value(self.t)
        asset_values = self.holdings * self.prices[self.t]
        weights = asset_values / pv if pv > 0 else np.zeros(self.N)
        cash_ratio = self.cash / pv if pv > 0 else 1.0

        feat_t = self.features[self.t]
        obs = np.concatenate([feat_t, weights.reshape(-1, 1)], axis=1).flatten()
        obs = np.append(obs, cash_ratio).astype(np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action: int):
        prev_value = self._portfolio_value(self.t)

        trade_blocked = False
        if action != 0:
            if action <= self.N:
                asset_idx = action - 1
                trade_blocked = not self._try_buy(asset_idx, prev_value)
            else:
                asset_idx = action - self.N - 1
                trade_blocked = not self._try_sell(asset_idx)

        self.t += 1
        terminated = self.t >= self.T - 1
        truncated = False

        new_value = self._portfolio_value(self.t)
        step_return = (new_value / prev_value) - 1.0 if prev_value > 0 else 0.0
        self._returns_history.append(step_return)
        self._portfolio_value_history.append(new_value)

        reward = self._compute_reward()
        if trade_blocked:
            reward -= 0.05  # small penalty so the agent learns not to repeat a now-invalid action

        obs = self._get_obs()
        info = {"portfolio_value": new_value, "step_return": step_return}
        return obs, reward, terminated, truncated, info

    def _try_buy(self, asset_idx: int, portfolio_value: float) -> bool:
        trade_value = self.trade_fraction * portfolio_value
        cost = trade_value * self.transaction_cost
        total_needed = trade_value + cost

        current_asset_value = self.holdings[asset_idx] * self.prices[self.t, asset_idx]
        projected_pct = (current_asset_value + trade_value) / portfolio_value if portfolio_value > 0 else 1.0

        if total_needed > self.cash or projected_pct > self.max_position_pct:
            return False  # blocked: insufficient cash or breaches position-size cap

        shares_bought = trade_value / self.prices[self.t, asset_idx]
        self.holdings[asset_idx] += shares_bought
        self.cash -= total_needed
        return True

    def _try_sell(self, asset_idx: int) -> bool:
        if self.holdings[asset_idx] <= 0:
            return False  # nothing to sell
        shares_to_sell = min(
            self.holdings[asset_idx],
            (self.trade_fraction * self._portfolio_value(self.t)) / self.prices[self.t, asset_idx],
        )
        proceeds = shares_to_sell * self.prices[self.t, asset_idx]
        cost = proceeds * self.transaction_cost
        self.holdings[asset_idx] -= shares_to_sell
        self.cash += proceeds - cost
        return True

    def _compute_reward(self, window: int = 20) -> float:
        """Change in rolling Sharpe-like ratio + step return, minus drawdown penalty."""
        returns = np.array(self._returns_history[-window:])
        step_ret = self._returns_history[-1] if self._returns_history else 0.0

        if len(returns) < 2 or returns.std() < 1e-8:
            sharpe_term = 0.0
        else:
            sharpe_term = float(np.clip(returns.mean() / (returns.std() + 1e-6), -5.0, 5.0))

        sharpe_delta = sharpe_term - self._prev_sharpe
        self._prev_sharpe = sharpe_term

        values = np.array(self._portfolio_value_history[-window:])
        running_max = np.maximum.accumulate(values)
        drawdown = (running_max - values) / running_max
        current_dd = float(drawdown[-1]) if len(drawdown) else 0.0
        dd_increase = max(0.0, current_dd - self._prev_drawdown)
        self._prev_drawdown = current_dd

        return float((step_ret * 100.0) + sharpe_delta - 0.5 * current_dd - 10.0 * dd_increase)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, N, F = 200, 3, 10
    prices = 100 + np.cumsum(rng.normal(0, 1, size=(T, N)), axis=0)
    features = rng.normal(0, 1, size=(T, N, F)).astype(np.float32)

    env = PortfolioTradingEnv(features, prices, lookback=30)
    obs, _ = env.reset()
    print("Observation shape:", obs.shape, "| Action space:", env.action_space)

    total_reward = 0.0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Ran 50 random steps OK. Final portfolio value: {info['portfolio_value']:.2f}, "
          f"cumulative reward: {total_reward:.4f}")
