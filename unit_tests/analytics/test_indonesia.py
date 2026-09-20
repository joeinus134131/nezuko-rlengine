from __future__ import annotations

import numpy as np
import pandas as pd

from finrl.analytics.indonesia import build_idx_analysis


def test_build_idx_analysis_produces_ranked_market_views():
    dates = pd.bdate_range("2023-01-02", periods=280)
    rows = []
    for index, ticker in enumerate(("BBCA.JK", "TLKM.JK", "ASII.JK")):
        price = 1_000 * np.cumprod(np.full(len(dates), 1.0005 + index * 0.0002))
        for date, close in zip(dates, price):
            rows.append({"timestamp": date, "tic": ticker, "close": close, "volume": 1_000_000 + index * 100_000})

    result = build_idx_analysis(pd.DataFrame(rows))

    assert set(result.screener.index) == {"BBCA.JK", "TLKM.JK", "ASII.JK"}
    assert result.screener["composite_score"].is_monotonic_decreasing
    assert result.correlation.shape == (3, 3)
    assert {"advancers_pct", "above_sma20_pct", "above_sma50_pct"} <= set(result.breadth.columns)
    assert result.normalized_prices.iloc[0].round(8).eq(100).all()
