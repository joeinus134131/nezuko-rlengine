from __future__ import annotations

import gymnasium as gym
import numpy as np
from numpy import random as rd


class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        initial_account=1e6,
        gamma=0.99,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=1e2,
        initial_capital=1e6,
        buy_cost_pct=1.6e-3,
        sell_cost_pct=3.5e-3,
        reward_scaling=2**-11,
        initial_stocks=None,
        lot_size=100,
        stop_loss_pct=0.0,
    ):
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        turbulence_ary = config["turbulence_array"]
        if_train = config["if_train"]
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        self.turbulence_ary = turbulence_ary

        self.tech_ary = self.tech_ary * 2**-7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        self.turbulence_ary = (
            self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5
        ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        # Support configuration overrides for costs, lot sizes, and risk controls
        self.buy_cost_pct = float(config.get("buy_cost_pct", buy_cost_pct))
        self.sell_cost_pct = float(config.get("sell_cost_pct", sell_cost_pct))
        self.lot_size = int(config.get("lot_size", lot_size))
        self.stop_loss_pct = float(config.get("stop_loss_pct", stop_loss_pct))
        self.reward_scaling = reward_scaling
        self.initial_capital = float(config.get("initial_capital", initial_capital))
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.cost_basis = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockEnv"
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        self.day = 0
        price = self.price_ary[self.day]

        if self.if_train:
            self.stocks = (
                self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            if self.lot_size > 1:
                self.stocks = (self.stocks // self.lot_size) * self.lot_size
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = (
                self.initial_capital * rd.uniform(0.95, 1.05)
                - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.amount = self.initial_capital

        self.cost_basis = np.zeros_like(self.stocks)
        for i in range(len(self.stocks)):
            if self.stocks[i] > 0 and price[i] > 0:
                self.cost_basis[i] = price[i]

        self.total_asset = self.amount + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price), {}  # state

    def step(self, actions):
        actions = (actions * self.max_stock).astype(int)

        self.day += 1
        price = self.price_ary[self.day]
        self.stocks_cool_down += 1

        # 1. Hard position-level stop-loss check (if enabled)
        if self.stop_loss_pct > 0:
            for index in range(len(self.stocks)):
                if self.stocks[index] > 0 and self.cost_basis[index] > 0 and price[index] > 0:
                    if price[index] <= self.cost_basis[index] * (1.0 - self.stop_loss_pct):
                        sell_num_shares = self.stocks[index]
                        self.stocks[index] = 0
                        self.amount += (
                            price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                        )
                        self.cost_basis[index] = 0.0
                        self.stocks_cool_down[index] = 0

        # 2. Main execution logic
        if self.turbulence_bool[self.day] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd

            # Sells first: free up liquidity for new purchases
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0 and self.stocks[index] > 0:
                    raw_sell = min(self.stocks[index], -actions[index])
                    if self.lot_size > 1:
                        sell_num_shares = (raw_sell // self.lot_size) * self.lot_size
                    else:
                        sell_num_shares = raw_sell
                    if sell_num_shares > 0:
                        self.stocks[index] -= sell_num_shares
                        self.amount += (
                            price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                        )
                        if self.stocks[index] == 0:
                            self.cost_basis[index] = 0.0
                        self.stocks_cool_down[index] = 0

            # Buys: proportional fair capital allocation without alphabetical ordering bias
            buy_indices = np.where(actions > min_action)[0]
            valid_buy_indices = [idx for idx in buy_indices if price[idx] > 0]
            if len(valid_buy_indices) > 0 and self.amount > 0:
                total_available_cash = self.amount
                total_buy_desire = sum(actions[idx] for idx in valid_buy_indices)
                for index in valid_buy_indices:
                    weight = (
                        actions[index] / total_buy_desire
                        if total_buy_desire > 0
                        else (1.0 / len(valid_buy_indices))
                    )
                    allocated_cash = min(self.amount, total_available_cash * weight)
                    effective_cost = price[index] * (1 + self.buy_cost_pct)
                    max_affordable = allocated_cash // effective_cost
                    raw_buy = min(max_affordable, actions[index])
                    if self.lot_size > 1:
                        buy_num_shares = (raw_buy // self.lot_size) * self.lot_size
                    else:
                        buy_num_shares = raw_buy

                    if buy_num_shares > 0:
                        total_cost = price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                        if total_cost <= self.amount:
                            new_total_shares = self.stocks[index] + buy_num_shares
                            old_cost = self.cost_basis[index] * self.stocks[index]
                            new_cost = price[index] * buy_num_shares
                            self.cost_basis[index] = (old_cost + new_cost) / new_total_shares
                            self.stocks[index] = new_total_shares
                            self.amount -= total_cost
                            self.stocks_cool_down[index] = 0

        else:  # sell all when turbulence
            self.amount += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.cost_basis[:] = 0
            self.stocks_cool_down[:] = 0

        state = self.get_state(price)
        total_asset = self.amount + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, False, dict()

    def get_state(self, price):
        amount = np.array(self.amount * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        return np.hstack(
            (
                amount,
                self.turbulence_ary[self.day],
                self.turbulence_bool[self.day],
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_ary[self.day],
            )
        )  # state.astype(np.float32)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh

