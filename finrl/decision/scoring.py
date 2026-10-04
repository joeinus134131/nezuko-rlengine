"""Composite "moment" scoring, gating, timing, and risk-based trade plans.

Pipeline per ticker::

    technical (relative + absolute) ─┐
    risk-adjusted                     ├─ weighted by profile ─ gates ─ label
    fundamental (shrunk if missing)   │                         │
    regime (+ sector/commodity tilt) ─┘                         └─ confidence

Labels are relative research classifications, never buy/sell instructions.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
import pandas as pd

from finrl.analytics.indonesia import build_idx_analysis
from finrl.analytics.scales import blended_score
from finrl.analytics.scales import weighted_mean
from finrl.fundamentals.idx import sector_for

PROFILES: dict[str, dict[str, Any]] = {
    "Konservatif": {
        "weights": {"technical": 0.20, "risk": 0.35, "fundamental": 0.30, "regime": 0.15},
        "max_drawdown_tolerance": -0.25,
        "min_liquidity_idr": 5e9,
        "risk_per_trade": 0.005,
        "max_position_pct": 0.15,
        "stop_atr_mult": 1.5,
        "max_positions": 5,
        "risk_off_cap": 50.0,
        "drawdown_penalty": 15.0,
        "exposure": {"RISK_ON": 0.80, "NEUTRAL": 0.55, "RISK_OFF": 0.25},
    },
    "Seimbang": {
        "weights": {"technical": 0.30, "risk": 0.25, "fundamental": 0.25, "regime": 0.20},
        "max_drawdown_tolerance": -0.35,
        "min_liquidity_idr": 2e9,
        "risk_per_trade": 0.01,
        "max_position_pct": 0.20,
        "stop_atr_mult": 2.0,
        "max_positions": 6,
        "risk_off_cap": 55.0,
        "drawdown_penalty": 12.0,
        "exposure": {"RISK_ON": 1.00, "NEUTRAL": 0.70, "RISK_OFF": 0.40},
    },
    "Agresif": {
        "weights": {"technical": 0.40, "risk": 0.15, "fundamental": 0.20, "regime": 0.25},
        "max_drawdown_tolerance": -0.50,
        "min_liquidity_idr": 1e9,
        "risk_per_trade": 0.015,
        "max_position_pct": 0.25,
        "stop_atr_mult": 2.5,
        "max_positions": 8,
        "risk_off_cap": 58.0,
        "drawdown_penalty": 10.0,
        "exposure": {"RISK_ON": 1.00, "NEUTRAL": 0.85, "RISK_OFF": 0.50},
    },
}

LABEL_THRESHOLDS = (
    (75.0, "STRONG WATCH POSITIVE"),
    (62.0, "WATCH POSITIVE"),
    (45.0, "NEUTRAL"),
    (0.0, "CAUTION"),
)
POSITIVE_LABELS = ("STRONG WATCH POSITIVE", "WATCH POSITIVE")
ILLIQUID_LABEL = "TIDAK LIKUID"
BUY_FEE = 0.0015
SELL_FEE = 0.0025  # broker fee + 0.1% final sales tax, typical retail IDX


@dataclass
class PlannerSettings:
    profile: str = "Seimbang"
    capital: float = 100_000_000.0
    risk_per_trade: float = 0.01
    min_liquidity_idr: float = 2e9
    max_drawdown_tolerance: float = -0.35
    max_position_pct: float = 0.20
    stop_atr_mult: float = 2.0
    max_positions: int = 6
    max_correlation: float = 0.70
    max_per_sector: int = 2
    risk_off_cap: float = 55.0
    drawdown_penalty: float = 12.0
    risk_free_rate: float = 0.06
    lot_size: int = 100
    weights: dict[str, float] = field(default_factory=lambda: dict(PROFILES["Seimbang"]["weights"]))
    exposure: dict[str, float] = field(default_factory=lambda: dict(PROFILES["Seimbang"]["exposure"]))

    @classmethod
    def for_profile(cls, profile: str, overrides: dict[str, Any] | None = None) -> "PlannerSettings":
        base = PROFILES.get(profile, PROFILES["Seimbang"])
        values: dict[str, Any] = {
            "profile": profile if profile in PROFILES else "Seimbang",
            "risk_per_trade": base["risk_per_trade"],
            "min_liquidity_idr": base["min_liquidity_idr"],
            "max_drawdown_tolerance": base["max_drawdown_tolerance"],
            "max_position_pct": base["max_position_pct"],
            "stop_atr_mult": base["stop_atr_mult"],
            "max_positions": base["max_positions"],
            "risk_off_cap": base["risk_off_cap"],
            "drawdown_penalty": base["drawdown_penalty"],
            "weights": dict(base["weights"]),
            "exposure": dict(base["exposure"]),
        }
        for key, value in (overrides or {}).items():
            if key in cls.__dataclass_fields__ and value is not None:
                values[key] = value
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- helpers
def idx_tick_size(price: float) -> int:
    """IDX price fraction (fraksi harga) for equities."""
    if price < 200:
        return 1
    if price < 500:
        return 2
    if price < 2000:
        return 5
    if price < 5000:
        return 10
    return 25


def round_to_tick(price: float, direction: str = "down") -> float:
    if not np.isfinite(price) or price <= 0:
        return float("nan")
    tick = idx_tick_size(price)
    rounder = math.floor if direction == "down" else math.ceil
    return float(max(tick, rounder(price / tick) * tick))


def label_for(score: float) -> str:
    for threshold, label in LABEL_THRESHOLDS:
        if score >= threshold:
            return label
    return "CAUTION"


def _normalize_input_data(data: pd.DataFrame) -> pd.DataFrame:
    if "tic" in data.columns or "ticker" in data.columns:
        return data
    frame = data.reset_index()
    first_col = frame.columns[0]
    melted = frame.melt(id_vars=[first_col], var_name="tic", value_name="close")
    melted.rename(columns={first_col: "timestamp"}, inplace=True)
    melted["volume"] = 10_000_000.0
    return melted


def _matrices(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = _normalize_input_data(data)
    date_col = "timestamp" if "timestamp" in data.columns else "date"
    frame = data.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col, "tic", "close"])
    out: dict[str, pd.DataFrame] = {}
    for column in ("close", "high", "low", "volume"):
        if column in frame.columns:
            out[column] = (
                frame.pivot_table(index=date_col, columns="tic", values=column, aggfunc="last")
                .sort_index()
                .ffill(limit=5)
            )
    return out


def price_matrix(data: pd.DataFrame) -> pd.DataFrame:
    return _matrices(data)["close"]


def compute_technical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Latest technical state per ticker computed directly from OHLCV."""
    matrices = _matrices(data)
    close = matrices["close"]
    if close.empty:
        raise ValueError("Data harga kosong.")
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.where(loss != 0, 100.0)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    if "high" in matrices and "low" in matrices:
        high, low = matrices["high"], matrices["low"]
        previous = close.shift(1)
        true_range = pd.concat(
            [(high - low), (high - previous).abs(), (low - previous).abs()], keys=range(3)
        ).groupby(level=1).max()
        true_range = true_range.reindex(close.index)
    else:
        true_range = close.diff().abs()
    atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=5).mean()
    sma20 = close.rolling(20, min_periods=10).mean()
    sma50 = close.rolling(50, min_periods=20).mean()
    sma200 = close.rolling(200, min_periods=60).mean()
    below50 = close.lt(sma50)
    streak = below50.astype(int).iloc[::-1].cumprod().sum()

    def _momentum_12_1(series: pd.Series) -> float:
        valid = series.dropna()
        if len(valid) > 253:
            return float(valid.iloc[-22] / valid.iloc[-253] - 1)
        if len(valid) > 127:
            return float(valid.iloc[-22] / valid.iloc[-127] - 1)
        return float("nan")

    latest = close.iloc[-1]
    features = pd.DataFrame(
        {
            "price": latest,
            "sma20": sma20.iloc[-1],
            "sma50": sma50.iloc[-1],
            "sma200": sma200.iloc[-1],
            "rsi14": rsi.iloc[-1],
            "macd_hist": hist.iloc[-1],
            "macd_hist_prev": hist.iloc[-6] if len(hist) > 6 else np.nan,
            "atr14": atr.iloc[-1],
            "momentum_12_1": close.apply(_momentum_12_1),
            "high_52w": close.tail(252).max(),
            "below_sma50_streak": streak,
            "observations": close.notna().sum(),
            "last_date": close.apply(lambda s: s.last_valid_index()),
        }
    )
    features["atr_pct"] = features["atr14"] / features["price"]
    features["dist_52w_high"] = features["price"] / features["high_52w"] - 1
    features["sma50_above_sma200"] = features["sma50"] > features["sma200"]
    features.index.name = "ticker"
    return features


def _rsi_zone_score(rsi: pd.Series) -> pd.Series:
    bins = [-np.inf, 30, 45, 65, 75, 80, np.inf]
    values = [45.0, 55.0, 85.0, 70.0, 45.0, 25.0]
    return pd.cut(rsi, bins=bins, labels=values, ordered=False).astype(float)


def _macd_score(hist: pd.Series, previous: pd.Series) -> pd.Series:
    rising = hist > previous
    return pd.Series(
        np.select(
            [(hist > 0) & rising, hist > 0, (hist <= 0) & rising],
            [100.0, 65.0, 45.0],
            default=20.0,
        ),
        index=hist.index,
    ).where(hist.notna())


def _timing(row: pd.Series) -> str:
    price, sma20, sma50, sma200, rsi = (row.get(k) for k in ("price", "sma20", "sma50", "sma200", "rsi14"))
    if not np.isfinite(price):
        return "Data tidak cukup"
    if np.isfinite(sma200) and price < sma200 and row.get("momentum_12_1", 0) < 0:
        return "Tren turun primer: hanya pantau, jangan tangkap pisau jatuh"
    if np.isfinite(rsi) and rsi >= 75:
        return f"Overextended (RSI {rsi:.0f}): tunggu pullback ke area SMA20 ~{sma20:,.0f}"
    if np.isfinite(sma50) and price < sma50 and (not np.isfinite(sma200) or price > sma200):
        return f"Pullback dalam tren naik: tunggu konfirmasi close > SMA20 ~{sma20:,.0f}"
    if np.isfinite(sma20) and price > sma20 and np.isfinite(rsi) and 45 <= rsi < 70:
        return "Momentum sehat: entry bertahap (2-3 tahap) layak dipertimbangkan"
    return "Konsolidasi: tunggu arah jelas (breakout volume atau tembus SMA20)"


# --------------------------------------------------------------------------- scoring
def score_universe(
    data: pd.DataFrame | None = None,
    settings: PlannerSettings | None = None,
    fundamentals: pd.DataFrame | None = None,
    regime: Any = None,
    signal_study: Any = None,
    as_of: pd.Timestamp | None = None,
    prices: pd.DataFrame | None = None,
    macro: Any = None,
    internals: Any = None,
) -> pd.DataFrame:
    """Score every ticker in ``data`` and return a ranked decision table."""
    if data is None:
        data = prices
    if data is None:
        raise ValueError("Data pasar wajib diberikan.")
    data = _normalize_input_data(data)
    if settings is None:
        settings = PlannerSettings()
    if regime is None and macro is not None:
        from finrl.macro.regime import assess_regime
        regime = assess_regime(internals, macro)
    analysis = build_idx_analysis(data, risk_free_rate=settings.risk_free_rate)
    screener = analysis.screener
    features = compute_technical_features(data)
    table = features.join(
        screener[[
            "return_1m", "return_3m", "return_6m", "return_12m", "annual_volatility",
            "sharpe", "sortino", "max_drawdown", "daily_var_95", "daily_cvar_95",
            "avg_value_idr_20d", "liquidity_change",
        ]],
        how="left",
    )
    tickers = table.index

    # Technical pillar: momentum, trend, RSI zone, MACD, 52-week-high proximity.
    trend = (
        table["price"].gt(table["sma20"]).astype(float)
        + table["price"].gt(table["sma50"]).astype(float)
        + table["price"].gt(table["sma200"]).astype(float)
        + table["sma50_above_sma200"].astype(float)
    ) / 4 * 100
    technical_parts = pd.DataFrame(
        {
            "momentum": blended_score(table["momentum_12_1"], -0.30, 0.40),
            "trend": trend,
            "rsi": _rsi_zone_score(table["rsi14"]),
            "macd": _macd_score(table["macd_hist"], table["macd_hist_prev"]),
            "proximity": blended_score(table["dist_52w_high"], -0.40, 0.0),
        },
        index=tickers,
    )
    table["technical_score"] = weighted_mean(
        technical_parts, {"momentum": 0.35, "trend": 0.25, "rsi": 0.15, "macd": 0.10, "proximity": 0.15}
    )

    risk_parts = pd.DataFrame(
        {
            "sharpe": blended_score(table["sharpe"], -0.5, 1.5),
            "sortino": blended_score(table["sortino"], -0.5, 2.0),
            "volatility": blended_score(table["annual_volatility"], 0.60, 0.15),
            "drawdown": blended_score(table["max_drawdown"], -0.60, -0.10),
            "cvar": blended_score(table["daily_cvar_95"], -0.06, -0.015),
        },
        index=tickers,
    )
    table["risk_score"] = risk_parts.mean(axis=1, skipna=True)

    if fundamentals is not None and not fundamentals.empty:
        fundamental = fundamentals.reindex(tickers)
        table["fundamental_score"] = fundamental["fundamental_score"].fillna(50.0)
        table["fund_completeness"] = fundamental["completeness"].fillna(0.0)
        table["fund_quality"] = fundamental["data_quality"].fillna("kosong")
        provided_sector = fundamental.get("sector")
    else:
        table["fundamental_score"] = 50.0
        table["fund_completeness"] = 0.0
        table["fund_quality"] = "kosong"
        provided_sector = None
    table["sector"] = [
        sector_for(t, provided_sector.get(t) if provided_sector is not None else None) for t in tickers
    ]

    regime_label = getattr(regime, "label", "NEUTRAL")
    if regime is not None:
        table["regime_score"] = [regime.score_for(t, s) for t, s in zip(tickers, table["sector"])]
        table["regime_tilt_reasons"] = [regime.tilt_for(t, s)[1] for t, s in zip(tickers, table["sector"])]
    else:
        table["regime_score"] = 50.0
        table["regime_tilt_reasons"] = [[] for _ in tickers]

    pillars = table[["technical_score", "risk_score", "fundamental_score", "regime_score"]].rename(
        columns=lambda c: c.replace("_score", "")
    )
    table["composite_raw"] = weighted_mean(pillars, settings.weights)

    # ---------------------------------------------------------------- gates
    scores, gates_all, eligible_all = [], [], []
    for ticker, row in table.iterrows():
        score = float(row["composite_raw"]) if np.isfinite(row["composite_raw"]) else 0.0
        gates: list[str] = []
        eligible = True
        liquidity = row.get("avg_value_idr_20d")
        if not np.isfinite(liquidity) or liquidity < settings.min_liquidity_idr:
            eligible = False
            score = min(score, 40.0)
            shown = liquidity if np.isfinite(liquidity) else 0.0
            gates.append(
                f"Likuiditas Rp{shown / 1e9:,.1f} M/hari < minimum Rp{settings.min_liquidity_idr / 1e9:,.1f} M"
            )
        if regime_label == "RISK_OFF" and score > settings.risk_off_cap:
            score = settings.risk_off_cap
            gates.append(f"Rezim RISK_OFF: skor dibatasi {settings.risk_off_cap:.0f}")
        if np.isfinite(row["max_drawdown"]) and row["max_drawdown"] < settings.max_drawdown_tolerance:
            score -= settings.drawdown_penalty
            gates.append(
                f"Max drawdown {row['max_drawdown']:.0%} > toleransi {settings.max_drawdown_tolerance:.0%}: -{settings.drawdown_penalty:.0f}"
            )
        if (
            np.isfinite(row["sma200"]) and row["price"] < row["sma200"]
            and np.isfinite(row["momentum_12_1"]) and row["momentum_12_1"] < 0 and score > 60
        ):
            score = 60.0
            gates.append("Di bawah SMA200 + momentum 12-1 negatif: skor maks 60 (anti pisau jatuh)")
        if np.isfinite(row["rsi14"]) and row["rsi14"] > 80:
            score -= 5
            gates.append(f"RSI {row['rsi14']:.0f} > 80: -5 (rawan profit taking)")
        if row["observations"] < 200:
            score -= 5
            gates.append(f"Histori {int(row['observations'])} sesi < 200: -5")
        if row["fund_completeness"] < 0.4:
            gates.append("Fundamental belum lengkap: skor fundamental netral 50, confidence diturunkan")
        scores.append(float(np.clip(score, 0, 100)))
        gates_all.append(gates)
        eligible_all.append(eligible)
    table["score"] = np.round(scores, 1)
    table["gates"] = gates_all
    table["eligible"] = eligible_all
    table["label"] = [label_for(s) if e else ILLIQUID_LABEL for s, e in zip(table["score"], table["eligible"])]

    # ---------------------------------------------------------- confidence
    pillar_std = pillars.std(axis=1, ddof=0).fillna(25.0)
    agreement = (1 - pillar_std / 50).clip(0.0, 1.0)
    data_quality = 0.6 + 0.4 * table["fund_completeness"].clip(0, 1)
    data_quality = data_quality * np.where(table["observations"] < 200, 0.9, 1.0)
    strength = signal_study.evidence_strength() if signal_study is not None else "tidak diketahui"
    evidence_factor = {"kuat": 1.0, "sedang": 0.9, "lemah": 0.8, "tidak ada / terbalik": 0.65}.get(strength, 0.8)
    table["confidence"] = (100 * agreement * data_quality * evidence_factor).round(0)
    table["evidence_strength"] = strength

    if signal_study is not None and len(getattr(signal_study, "latest_scores", [])):
        latest_signal = signal_study.latest_scores.reindex(tickers)
        probabilities = [signal_study.probability_for(v) for v in latest_signal]
        table["signal_score_pit"] = latest_signal
        table["hist_prob_beat_median"] = [p for p, _ in probabilities]
        table["hist_prob_n"] = [n for _, n in probabilities]
    else:
        table["signal_score_pit"] = np.nan
        table["hist_prob_beat_median"] = np.nan
        table["hist_prob_n"] = 0

    table["timing"] = table.apply(_timing, axis=1)
    reference = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now().normalize()
    table["stale_days"] = [
        (reference - pd.Timestamp(d)).days if pd.notna(d) else np.nan for d in table["last_date"]
    ]
    for pillar_name, part in (("tech", technical_parts), ("risk", risk_parts)):
        for column in part.columns:
            table[f"{pillar_name}_{column}"] = part[column].round(1)
    table["ticker"] = table.index
    table = table.sort_values(["eligible", "score"], ascending=[False, False])
    table.attrs["regime_label"] = regime_label
    table.attrs["settings"] = settings.to_dict()
    table.attrs["correlation"] = analysis.correlation.round(4).fillna(0.0).to_dict()
    return table


# ------------------------------------------------------------------ trade planning
def build_trade_plan(row: pd.Series, settings: PlannerSettings) -> dict[str, Any]:
    """Risk-first scenario: stop from ATR, size from risk budget, IDX ticks & lots."""
    price = float(row["price"])
    atr = float(row.get("atr14", np.nan))
    if not np.isfinite(price) or price <= 0:
        return {"valid": False, "reason": "Harga tidak valid"}
    if not np.isfinite(atr) or atr <= 0:
        atr = price * 0.02
    raw_stop = max(price - settings.stop_atr_mult * atr, price * 0.85)
    stop = round_to_tick(raw_stop, "down")
    if not np.isfinite(stop) or stop >= price:
        stop = round_to_tick(price - idx_tick_size(price) * 2, "down")
    risk_per_share = price - stop + price * (BUY_FEE + SELL_FEE)
    target1 = round_to_tick(price + 2 * (price - stop), "up")
    target2 = round_to_tick(price + 3 * (price - stop), "up")
    lot = max(1, int(settings.lot_size))
    risk_budget = settings.capital * settings.risk_per_trade
    lots_by_risk = math.floor(risk_budget / (risk_per_share * lot)) if risk_per_share > 0 else 0
    lots_by_cap = math.floor(settings.capital * settings.max_position_pct / (price * lot * (1 + BUY_FEE)))
    lots = max(0, min(lots_by_risk, lots_by_cap))
    value = lots * lot * price
    return {
        "valid": lots > 0,
        "entry_ref": round_to_tick(price, "up"),
        "stop": stop,
        "stop_pct": stop / price - 1,
        "target1": target1,
        "target2": target2,
        "reward_risk_t1": (target1 - price) / (price - stop) if price > stop else float("nan"),
        "lots": int(lots),
        "shares": int(lots * lot),
        "position_value": float(value),
        "position_pct": float(value / settings.capital) if settings.capital else 0.0,
        "risk_amount": float(lots * lot * risk_per_share),
        "risk_pct_capital": float(lots * lot * risk_per_share / settings.capital) if settings.capital else 0.0,
        "limited_by": "batas % modal" if lots_by_cap < lots_by_risk else "budget risiko",
        "reason": "" if lots > 0 else "Modal/risk budget tidak cukup untuk 1 lot dengan stop ini",
    }


def select_diversified(
    table: pd.DataFrame, settings: PlannerSettings, correlation: pd.DataFrame | None = None
) -> list[str]:
    """Greedy pick by score with correlation and sector caps. Never pads with weak names."""
    if correlation is None and table.attrs.get("correlation"):
        correlation = pd.DataFrame(table.attrs["correlation"])
    candidates = table[table["eligible"] & table["label"].isin(POSITIVE_LABELS)]
    chosen: list[str] = []
    per_sector: dict[str, int] = {}
    for ticker, row in candidates.iterrows():
        if len(chosen) >= settings.max_positions:
            break
        if per_sector.get(row["sector"], 0) >= settings.max_per_sector:
            continue
        if correlation is not None and chosen and ticker in correlation.index:
            pairs = correlation.loc[ticker, [c for c in chosen if c in correlation.columns]]
            if pairs.notna().any() and pairs.max() > settings.max_correlation:
                continue
        chosen.append(ticker)
        per_sector[row["sector"]] = per_sector.get(row["sector"], 0) + 1
    return chosen


def suggest_weights(table: pd.DataFrame, chosen: list[str], settings: PlannerSettings) -> pd.Series:
    """Inverse-volatility weights scaled by regime exposure; remainder is cash."""
    if not chosen:
        return pd.Series({"KAS": 1.0})
    regime_label = table.attrs.get("regime_label", "NEUTRAL")
    exposure = float(settings.exposure.get(regime_label, 0.7))
    vol = table.loc[chosen, "annual_volatility"].astype(float).replace(0, np.nan)
    inverse = (1 / vol).fillna((1 / vol).mean() if vol.notna().any() else 1.0)
    weights = inverse / inverse.sum() * exposure
    cap = settings.max_position_pct
    for _ in range(10):  # iterative capping, redistribute excess within exposure
        excess = (weights - cap).clip(lower=0).sum()
        if excess <= 1e-9:
            break
        weights = weights.clip(upper=cap)
        free = weights < cap
        if not free.any():
            break
        weights[free] += excess * weights[free] / weights[free].sum()
    weights = weights.clip(upper=cap)
    weights["KAS"] = max(0.0, 1.0 - weights.sum())
    return weights.round(4)
