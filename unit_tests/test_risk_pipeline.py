from __future__ import annotations

import numpy as np
import pandas as pd

from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor


def test_short_dataset_can_use_turbulence_when_vix_is_disabled() -> None:
    dates = pd.bdate_range("2024-01-02", periods=20)
    rows = []
    for ticker, offset in (("BBCA.JK", 0), ("BBRI.JK", 100)):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "timestamp": date,
                    "tic": ticker,
                    "close": 1_000 + offset + index,
                    "feature": float(index),
                }
            )

    processor = YahooFinanceProcessor()
    data = processor.add_turbulence(pd.DataFrame(rows))
    price, technical, risk = processor.df_to_array(data, ["feature"], if_vix=False)

    assert "turbulence" in data.columns
    assert price.shape == (20, 2)
    assert technical.shape == (20, 2)
    assert risk.shape == (20,)
    assert np.isfinite(risk).all()
