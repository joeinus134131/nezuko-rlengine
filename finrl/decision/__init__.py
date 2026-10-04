"""Decision engine: evidence-based scoring, validation guards, explanations, ledger.

Modules
-------
- ``scoring``        : composite moment score, gates, labels, trade plans, diversification.
- ``signal_study``   : point-in-time historical evidence (IC, hit rate, Wilson CI, Brier).
- ``backtest_guard`` : overfitting guards for RL/backtests (PSR, DSR, bootstrap, walk-forward).
- ``explainer``      : human-readable decision cards and Markdown packages.
- ``ledger``         : forward-test decision ledger and calibration evaluation.

All outputs are research classifications, not buy/sell instructions.
"""
from finrl.decision.backtest_guard import BacktestGuardSettings
from finrl.decision.backtest_guard import GuardResult
from finrl.decision.backtest_guard import evaluate_equity_curve
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

__all__ = [
    "PlannerSettings",
    "score_universe",
    "build_trade_plan",
    "select_diversified",
    "suggest_weights",
    "run_signal_study",
    "BacktestGuardSettings",
    "GuardResult",
    "evaluate_equity_curve",
    "build_decision_card",
    "export_decision_markdown",
    "export_portfolio_decision_pack",
    "record_decision",
    "load_ledger",
    "update_realized_outcomes",
    "evaluate_ledger_accuracy",
]
