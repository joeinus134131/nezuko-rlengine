from __future__ import annotations

import pandas as pd

from finrl.monitoring.telegram_monitor import _signal_label


def test_signal_labels_are_research_labels_not_orders() -> None:
    positive = pd.Series(
        {
            "composite_score": 80,
            "above_sma20": True,
            "above_sma50": True,
            "above_sma200": False,
        }
    )
    caution = pd.Series(
        {
            "composite_score": 20,
            "above_sma20": False,
            "above_sma50": False,
            "above_sma200": False,
        }
    )
    assert _signal_label(positive) == "WATCH POSITIVE"
    assert _signal_label(caution) == "CAUTION"
