from __future__ import annotations

import pandas as pd

from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor


class _Response:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "chart": {
                "error": None,
                "result": [
                    {
                        "timestamp": [1704067200, 1704153600],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100.0, 102.0],
                                    "high": [103.0, 104.0],
                                    "low": [99.0, 101.0],
                                    "close": [102.0, 103.0],
                                    "volume": [1000, 1200],
                                }
                            ],
                            "adjclose": [{"adjclose": [51.0, 51.5]}],
                        },
                    }
                ],
            }
        }


def test_chart_api_fallback_returns_adjusted_ohlcv(monkeypatch) -> None:
    monkeypatch.setattr(
        "finrl.meta.data_processors.processor_yahoofinance.requests.get",
        lambda *args, **kwargs: _Response(),
    )
    frame = YahooFinanceProcessor._download_chart_api(
        "BBCA.JK", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03"), "1d"
    )
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert frame.index.name == "Date"
    assert frame.iloc[0]["Close"] == 51.0
    assert frame.iloc[0]["Open"] == 50.0
