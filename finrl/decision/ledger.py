"""Forward-test decision ledger and calibration evaluation.

This module provides mathematical accountability: every decision produced by
the engine can be recorded at time T and audited against realized market prices
at T+5, T+20, and T+60 sessions.

Key features:
1. Append-only or persistent JSON storage for forward-test recommendations.
2. Realized outcome calculation (target hit, stop hit, return at horizon).
3. Empirical performance auditing: Win Rate, Profit Factor, Brier Score calibration,
   and performance broken down by macro regime.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_LEDGER_PATH = Path("data/decision_ledger.json")


@dataclass
class DecisionRecord:
    id: str
    timestamp: str
    as_of: str
    ticker: str
    score: float
    label: str
    confidence: float
    entry_ref: float
    stop: float
    target1: float
    target2: float
    regime: str
    profile: str
    lots: int = 0
    outcome_status: str = "PENDING"  # PENDING, TARGET_HIT, STOP_HIT, EXPIRED
    realized_return_5d: float | None = None
    realized_return_20d: float | None = None
    realized_return_60d: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ledger(path: Path | str | None = None) -> list[dict[str, Any]]:
    target = Path(path) if path else DEFAULT_LEDGER_PATH
    if not target.exists():
        return []
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_ledger(records: list[dict[str, Any]], path: Path | str | None = None) -> None:
    target = Path(path) if path else DEFAULT_LEDGER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def record_decision(
    as_of: str,
    ticker: str,
    score: float,
    label: str,
    confidence: float,
    entry_ref: float,
    stop: float,
    target1: float,
    target2: float = 0.0,
    regime: str = "NEUTRAL",
    profile: str = "Seimbang",
    lots: int = 0,
    notes: str = "",
    ledger_path: Path | str | None = None,
) -> dict[str, Any]:
    """Append a new decision record to the persistent ledger."""
    records = load_ledger(ledger_path)
    rec_id = f"{as_of}_{ticker}_{int(datetime.now().timestamp())}"
    rec = DecisionRecord(
        id=rec_id,
        timestamp=datetime.now().isoformat(),
        as_of=str(as_of),
        ticker=str(ticker),
        score=float(score),
        label=str(label),
        confidence=float(confidence),
        entry_ref=float(entry_ref),
        stop=float(stop),
        target1=float(target1),
        target2=float(target2),
        regime=str(regime),
        profile=str(profile),
        lots=int(lots),
        notes=str(notes),
    )
    records.append(rec.to_dict())
    save_ledger(records, ledger_path)
    return rec.to_dict()


def update_realized_outcomes(
    prices: pd.DataFrame,
    ledger_path: Path | str | None = None,
) -> tuple[int, int]:
    """Check future prices to resolve pending decision records.

    Parameters
    ----------
    prices : pd.DataFrame
        Datetime-indexed DataFrame with tickers as columns containing close prices.
    ledger_path : Path | str | None
        Path to the decision ledger file.

    Returns
    -------
    tuple[int, int]
        (total_updated, total_pending)
    """
    records = load_ledger(ledger_path)
    if not records or prices.empty:
        return 0, len(records)

    prices = prices.sort_index()
    updated_count = 0
    pending_count = 0

    for rec in records:
        ticker = rec.get("ticker")
        if ticker not in prices.columns:
            continue

        as_of_dt = pd.Timestamp(rec.get("as_of"))
        series = prices[ticker].dropna()
        future_prices = series[series.index >= as_of_dt]

        if len(future_prices) <= 1:
            pending_count += 1
            continue

        entry = rec.get("entry_ref", future_prices.iloc[0])
        stop = rec.get("stop", 0.0)
        target1 = rec.get("target1", 0.0)

        # Check subsequent sessions (excluding day 0 entry)
        eval_window = future_prices.iloc[1:]
        min_p = eval_window.min()
        max_p = eval_window.max()

        if rec["outcome_status"] == "PENDING":
            if max_p >= target1 > 0:
                rec["outcome_status"] = "TARGET_HIT"
                updated_count += 1
            elif min_p <= stop and stop > 0:
                rec["outcome_status"] = "STOP_HIT"
                updated_count += 1
            elif len(eval_window) >= 60:
                rec["outcome_status"] = "EXPIRED"
                updated_count += 1
            else:
                pending_count += 1

        # Calculate horizon returns if available
        if len(eval_window) >= 5 and rec.get("realized_return_5d") is None:
            rec["realized_return_5d"] = round(float(eval_window.iloc[4] / entry - 1), 4)
        if len(eval_window) >= 20 and rec.get("realized_return_20d") is None:
            rec["realized_return_20d"] = round(float(eval_window.iloc[19] / entry - 1), 4)
        if len(eval_window) >= 60 and rec.get("realized_return_60d") is None:
            rec["realized_return_60d"] = round(float(eval_window.iloc[59] / entry - 1), 4)

    save_ledger(records, ledger_path)
    return updated_count, pending_count


def evaluate_ledger_accuracy(ledger_path: Path | str | None = None) -> dict[str, Any]:
    """Calculate rigorous empirical accuracy, win rate, and Brier calibration."""
    records = load_ledger(ledger_path)
    if not records:
        return {
            "total_decisions": 0,
            "resolved_decisions": 0,
            "pending_decisions": 0,
            "hit_rate": None,
            "brier_score": None,
            "positive_label_hit_rate": None,
            "profit_factor": None,
            "regime_breakdown": {},
        }

    df = pd.DataFrame(records)
    total = len(df)
    resolved = df[df["outcome_status"].isin(["TARGET_HIT", "STOP_HIT", "EXPIRED"])]
    pending = df[df["outcome_status"] == "PENDING"]

    if resolved.empty:
        return {
            "total_decisions": total,
            "resolved_decisions": 0,
            "pending_decisions": len(pending),
            "hit_rate": None,
            "brier_score": None,
            "positive_label_hit_rate": None,
            "profit_factor": None,
            "regime_breakdown": {},
        }

    # Hit Rate: Target Hit vs Stop Hit (for resolved trades)
    target_hits = (resolved["outcome_status"] == "TARGET_HIT").sum()
    stop_hits = (resolved["outcome_status"] == "STOP_HIT").sum()
    trade_trials = target_hits + stop_hits
    hit_rate = float(target_hits / trade_trials) if trade_trials > 0 else None

    # Positive labels accuracy (STRONG WATCH POSITIVE / WATCH POSITIVE)
    pos_mask = resolved["label"].isin(["STRONG WATCH POSITIVE", "WATCH POSITIVE"])
    pos_subset = resolved[pos_mask]
    pos_target_hits = (pos_subset["outcome_status"] == "TARGET_HIT").sum()
    pos_trials = (pos_subset["outcome_status"].isin(["TARGET_HIT", "STOP_HIT"])).sum()
    pos_hit_rate = float(pos_target_hits / pos_trials) if pos_trials > 0 else None

    # Realized returns & Profit factor
    rets_20d = resolved["realized_return_20d"].dropna()
    gains = rets_20d[rets_20d > 0].sum()
    losses = abs(rets_20d[rets_20d < 0].sum())
    profit_factor = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else None)

    # Brier score: confidence (0-1) vs actual binary outcome (1 = target hit, 0 = stop hit)
    brier = None
    if trade_trials > 0:
        trades_df = resolved[resolved["outcome_status"].isin(["TARGET_HIT", "STOP_HIT"])]
        prob = trades_df["confidence"].fillna(50.0) / 100.0
        actual = (trades_df["outcome_status"] == "TARGET_HIT").astype(float)
        brier = float(np.mean((prob - actual) ** 2))

    # Breakdown by regime
    regime_breakdown: dict[str, dict[str, Any]] = {}
    for reg, group in resolved.groupby("regime"):
        reg_hits = (group["outcome_status"] == "TARGET_HIT").sum()
        reg_trials = (group["outcome_status"].isin(["TARGET_HIT", "STOP_HIT"])).sum()
        reg_rate = float(reg_hits / reg_trials) if reg_trials > 0 else None
        regime_breakdown[str(reg)] = {
            "total": len(group),
            "hit_rate": reg_rate,
            "avg_return_20d": float(group["realized_return_20d"].dropna().mean()) if not group["realized_return_20d"].dropna().empty else None,
        }

    return {
        "total_decisions": total,
        "resolved_decisions": len(resolved),
        "pending_decisions": len(pending),
        "hit_rate": hit_rate,
        "positive_label_hit_rate": pos_hit_rate,
        "profit_factor": profit_factor,
        "brier_score": brier,
        "mean_return_20d": float(rets_20d.mean()) if not rets_20d.empty else None,
        "regime_breakdown": regime_breakdown,
    }
