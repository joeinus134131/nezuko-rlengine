from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from finrl.decision.backtest_guard import BacktestGuardSettings
from finrl.decision.backtest_guard import evaluate_equity
from finrl.decision.explainer import build_decision_card
from finrl.decision.explainer import export_decision_markdown
from finrl.decision.explainer import export_portfolio_decision_pack
from finrl.decision.ledger import evaluate_ledger_accuracy
from finrl.decision.ledger import load_ledger
from finrl.decision.ledger import record_decision
from finrl.decision.ledger import update_realized_outcomes
from finrl.decision.scoring import PlannerSettings
from finrl.decision.scoring import build_trade_plan
from finrl.decision.scoring import score_universe
from finrl.decision.scoring import select_diversified
from finrl.decision.scoring import suggest_weights
from finrl.decision.signal_study import run_signal_study
from finrl.fundamentals.idx import score_fundamentals
from finrl.integrations.aggregator import build_evidence_pack
from finrl.integrations.aggregator import grounding_report
from finrl.macro.regime import MacroSnapshot
from finrl.macro.regime import assess_regime
from finrl.macro.regime import market_internals_from_prices


@pytest.fixture
def mock_market_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=280)
    data = {}
    for i, ticker in enumerate(["BBCA.JK", "BBRI.JK", "BMRI.JK", "ADRO.JK", "TLKM.JK", "ASII.JK"]):
        trend = 1.0006 + 0.0003 * (i % 3)
        prices = 5000.0 * np.cumprod(np.full(len(dates), trend))
        data[ticker] = prices
    return pd.DataFrame(data, index=dates)


def test_fundamentals_scoring():
    tickers = ["BBCA.JK", "ADRO.JK", "TLKM.JK"]
    scores = score_fundamentals(tickers=tickers)
    assert len(scores) == 3
    assert "fundamental_score" in scores.columns
    assert "data_quality" in scores.columns
    assert (scores["fundamental_score"] >= 0).all() and (scores["fundamental_score"] <= 100).all()


def test_macro_regime_classification(mock_market_prices):
    internals = market_internals_from_prices(mock_market_prices)
    assert internals.index_above_sma50 is not None
    assert internals.breadth_above_sma50 is not None

    macro = MacroSnapshot(
        bi_rate=6.0,
        inflation_yoy=2.5,
        idr_change_1m=1.0,
        yield_10y_change_1m_bp=10.0,
        coal_change_1m=5.0,
    )
    regime = assess_regime(macro, internals)
    assert regime.regime in ("RISK_ON", "NEUTRAL", "RISK_OFF")
    assert 0.0 <= regime.score <= 100.0
    assert len(regime.reasons) > 0


def test_scoring_universe_and_trade_planning(mock_market_prices):
    settings = PlannerSettings.for_profile("Seimbang", {"capital": 100_000_000.0})
    internals = market_internals_from_prices(mock_market_prices)
    macro = MacroSnapshot(bi_rate=6.0, inflation_yoy=2.5)
    study = run_signal_study(mock_market_prices, horizon=20, step=5)

    table = score_universe(
        prices=mock_market_prices,
        fundamentals=score_fundamentals(tickers=list(mock_market_prices.columns)),
        macro=macro,
        internals=internals,
        signal_study=study,
        settings=settings,
    )

    assert "score" in table.columns
    assert "label" in table.columns
    assert "confidence" in table.columns
    assert (table["score"] >= 0).all() and (table["score"] <= 100).all()

    # Trade plan
    top_ticker = table.index[0]
    plan = build_trade_plan(table.loc[top_ticker], settings)
    assert plan["valid"] is True
    assert plan["stop"] < plan["entry_ref"]
    assert plan["target1"] > plan["entry_ref"]
    assert plan["lots"] > 0
    assert plan["shares"] == plan["lots"] * 100
    assert plan["reward_risk_t1"] >= 1.5

    # Diversification & Weights
    chosen = select_diversified(table, settings)
    assert len(chosen) <= settings.max_positions
    weights = suggest_weights(table, chosen, settings)
    assert "KAS" in weights.index
    assert np.isclose(weights.sum(), 1.0, atol=1e-3)


def test_explainer_and_markdown_exports(mock_market_prices):
    settings = PlannerSettings()
    table = score_universe(mock_market_prices, settings=settings)
    ticker = table.index[0]
    row = table.loc[ticker]
    plan = build_trade_plan(row, settings)

    card = build_decision_card(row, plan, {"regime": "NEUTRAL"})
    assert card["ticker"] == ticker
    assert len(card["bull_cases"]) > 0
    assert len(card["bear_cases"]) > 0
    assert len(card["invalidation"]) > 0
    assert len(card["verification_checklist"]) > 0

    single_md = export_decision_markdown(card)
    assert f"Kartu Keputusan Investasi: {ticker}" in single_md
    assert "Rencana Transaksi" in single_md

    chosen = select_diversified(table, settings)
    weights = suggest_weights(table, chosen, settings)
    plans = {t: build_trade_plan(table.loc[t], settings) for t in chosen}
    pack_md = export_portfolio_decision_pack(table, chosen, weights, plans)
    assert "Paket Perencanaan Investasi & Keputusan Portofolio" in pack_md
    assert "Alokasi Portofolio yang Disarankan" in pack_md


def test_decision_ledger(tmp_path):
    ledger_file = tmp_path / "test_ledger.json"
    rec = record_decision(
        as_of="2023-01-02",
        ticker="BBCA.JK",
        score=75.0,
        label="STRONG WATCH POSITIVE",
        confidence=80.0,
        entry_ref=8000.0,
        stop=7600.0,
        target1=8800.0,
        ledger_path=ledger_file,
    )
    assert rec["ticker"] == "BBCA.JK"
    assert rec["outcome_status"] == "PENDING"

    # Future prices hitting target
    dates = pd.bdate_range("2023-01-02", periods=10)
    prices = pd.DataFrame({"BBCA.JK": [8000, 8100, 8300, 8500, 8900, 9000, 9100, 9200, 9300, 9400]}, index=dates)

    updated, pending = update_realized_outcomes(prices, ledger_file)
    assert updated == 1
    assert pending == 0

    acc = evaluate_ledger_accuracy(ledger_file)
    assert acc["total_decisions"] == 1
    assert acc["resolved_decisions"] == 1
    assert acc["hit_rate"] == 1.0


def test_backtest_guard():
    # Healthy upward equity curve
    dates = pd.bdate_range("2023-01-02", periods=250)
    equity = 100_000_000.0 * np.cumprod(np.random.normal(1.001, 0.008, len(dates)))
    res = evaluate_equity(equity, BacktestGuardSettings())
    assert res.status in ("LULUS", "WASPADA", "DITOLAK")
    assert "sharpe" in res.metrics
    assert "psr_vs_rf" in res.metrics
    assert "max_drawdown" in res.metrics


def test_aggregator_grounding():
    row = pd.Series({
        "ticker": "BBCA.JK",
        "price": 9000.0,
        "score": 80.0,
        "label": "STRONG WATCH POSITIVE",
        "confidence": 75.0,
        "technical_score": 82.0,
        "risk_score": 78.0,
        "fundamental_score": 75.0,
        "regime_score": 70.0,
        "rsi14": 55.0,
        "sharpe": 1.4,
        "max_drawdown": -0.15,
        "avg_value_idr_20d": 500e9,
        "sector": "Financial Services",
    })
    pack = build_evidence_pack(row)
    assert pack["price"] == 9000.0
    assert pack["rsi14"] == 55.0

    regime = SimpleNamespace(
        regime="RISK_ON", score=72.0, reasons=["breadth positif"], caps=[]
    )
    regime_pack = build_evidence_pack(
        ticker="BBCA.JK", row=row, plan={"lots": 2}, regime=regime
    )
    assert regime_pack["ticker"] == "BBCA.JK"
    assert regime_pack["rezim"]["label"] == "RISK_ON"
    assert regime_pack["rezim"]["skor"] == 72.0

    # Grounded judgement
    grounded_judgement = {
        "bullish_case": ["Harga berada pada 9000.0 dan RSI di 55.0"],
        "bearish_case": ["Drawdown pernah mencapai -0.15"],
    }
    rep = grounding_report(grounded_judgement, pack)
    assert rep["ratio"] >= 0.90

    # Hallucinated judgement with numbers not in pack
    hallucinated_judgement = {
        "bullish_case": ["Target harga adalah 15850.25 dan laba tumbuh 99.4%"],
    }
    rep_bad = grounding_report(hallucinated_judgement, pack)
    assert rep_bad["ratio"] < 0.50
    assert len(rep_bad["ungrounded"]) > 0
