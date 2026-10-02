from __future__ import annotations

import numpy as np
import pytest

from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv


def _dummy_config(num_stocks=2, num_days=5):
    # Day 0: 1000, 1000
    # Day 1: 1000, 1000
    # Day 2: 900, 1000 (stock 0 drops 10%)
    price_array = np.array([
        [1000.0, 1000.0],
        [1000.0, 1000.0],
        [900.0, 1000.0],
        [900.0, 1000.0],
        [900.0, 1000.0],
    ], dtype=np.float32)
    tech_array = np.zeros((num_days, 2), dtype=np.float32)
    turbulence_array = np.zeros(num_days, dtype=np.float32)
    return {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }


def test_lot_size_constraint():
    config = _dummy_config()
    env = StockTradingEnv(
        config=config,
        initial_capital=10_000_000,
        lot_size=100,
        buy_cost_pct=0.0016,
        sell_cost_pct=0.0035,
    )
    env.reset()
    # Action asks to buy 150 shares for stock 0, and 75 shares for stock 1
    # Normalized actions [-1, 1] scaled by max_stock=100
    # Let's directly step with action [1.0, 0.5] -> 100 shares, 50 shares
    state, reward, done, truncated, info = env.step(np.array([1.0, 0.5]))
    
    # With lot_size=100:
    # Stock 0: 100 shares requested -> 100 shares bought (1 lot)
    # Stock 1: 50 shares requested -> 0 shares bought (cannot form 1 lot of 100)
    assert env.stocks[0] % 100 == 0
    assert env.stocks[1] % 100 == 0
    assert env.stocks[0] == 100
    assert env.stocks[1] == 0


def test_fair_proportional_allocation():
    config = _dummy_config()
    # Limited capital: only enough for 200 shares total
    # Price is 1000. 200 shares = 200,000 IDR
    env = StockTradingEnv(
        config=config,
        initial_capital=210_000,
        lot_size=100,
        max_stock=500,
        buy_cost_pct=0.0,
    )
    env.reset()
    # Both stock 0 and stock 1 have identical buy desire: [1.0, 1.0] (500 shares each)
    # Total available cash is 210,000 -> max 200 shares total.
    # Fair allocation gives 50% budget to each -> 100 shares each.
    # Previously, sequential greedy allocation would give 200 shares to stock 0 and 0 to stock 1.
    env.step(np.array([1.0, 1.0]))
    assert env.stocks[0] == 100
    assert env.stocks[1] == 100


def test_hard_stop_loss():
    config = _dummy_config()
    env = StockTradingEnv(
        config=config,
        initial_capital=1_000_000,
        lot_size=100,
        max_stock=100,
        stop_loss_pct=0.05,  # 5% stop loss
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
    )
    env.reset()
    # Day 0 -> Day 1: Buy stock 0 at price 1000
    env.step(np.array([1.0, 0.0]))
    assert env.stocks[0] == 100
    assert env.cost_basis[0] == 1000.0

    # Day 1 -> Day 2: Price of stock 0 drops to 900 (-10%, which exceeds 5% stop loss)
    # Position should be automatically liquidated by stop loss
    env.step(np.array([0.0, 0.0]))
    assert env.stocks[0] == 0
    assert env.cost_basis[0] == 0.0
