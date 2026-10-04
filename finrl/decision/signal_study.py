"""Point-in-time historical study of the technical ranking signal.

This is the honest answer to "how accurate is the engine?". At every
rebalance date the score is computed **only from data available on that
date**, then compared with what happened over the next ``horizon`` sessions.

Outputs:

* Information Coefficient (Spearman rank correlation score vs forward return),
  its t-stat adjusted for overlapping horizons, and ICIR
* hit rate of the top quintile beating the universe median, with a Wilson
  95% confidence interval
* top-minus-bottom quintile spread
* a reliability table: empirical P(beat median) per score bucket, used to
  attach a *measured* historical probability to today's scores

Fundamentals are excluded because free sources do not provide point-in-time
history; including today's fundamentals in the past would be look-ahead bias.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
import pandas as pd


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return float(centre - margin), float(centre + margin)


def brier_score(probabilities: pd.Series | np.ndarray, outcomes: pd.Series | np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    mask = np.isfinite(p) & np.isfinite(o)
    if not mask.any():
        return float("nan")
    return float(np.mean((p[mask] - o[mask]) ** 2))


def point_in_time_scores(prices: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional technical score for every date using only past data.

    Components (all backward-looking rolling windows):
    12-1 momentum, trend (price vs SMA20/50/200), proximity to 52-week high,
    and low 60-day volatility.
    """
    prices = prices.sort_index()
    returns = prices.pct_change(fill_method=None)
    momentum = prices.shift(21) / prices.shift(252) - 1
    trend = (
        prices.gt(prices.rolling(20).mean()).astype(float)
        + prices.gt(prices.rolling(50).mean()).astype(float)
        + prices.gt(prices.rolling(200).mean()).astype(float)
    ) / 3
    proximity = prices / prices.rolling(252).max() - 1
    low_vol = -returns.rolling(60).std()
    components = [momentum, trend, proximity, low_vol]
    weights = [0.40, 0.25, 0.20, 0.15]
    ranked = [component.rank(axis=1, pct=True) for component in components]
    score = sum(rank * weight for rank, weight in zip(ranked, weights))
    valid = momentum.notna() & proximity.notna() & low_vol.notna()
    return (score * 100).where(valid)


@dataclass
class SignalStudy:
    horizon: int
    step: int
    dates: int
    observations: int
    mean_ic: float
    ic_std: float
    icir: float
    ic_tstat: float
    top_hit_rate: float
    top_hit_ci: tuple[float, float]
    top_mean_excess: float
    spread_q5_q1: float
    reliability: pd.DataFrame
    ic_series: pd.Series
    latest_scores: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    notes: list[str] = field(default_factory=list)

    def probability_for(self, score: float) -> tuple[float, int]:
        """Empirical P(beat universe median over horizon) for a score value."""
        if self.reliability.empty or not np.isfinite(score):
            return float("nan"), 0
        for _, row in self.reliability.iterrows():
            if row["score_low"] <= score <= row["score_high"]:
                return float(row["p_beat_median"]), int(row["n"])
        nearest = (self.reliability["score_mid"] - score).abs().idxmin()
        row = self.reliability.loc[nearest]
        return float(row["p_beat_median"]), int(row["n"])

    def evidence_strength(self) -> str:
        if not np.isfinite(self.ic_tstat):
            return "tidak diketahui"
        if self.ic_tstat >= 3 and self.mean_ic > 0.03:
            return "kuat"
        if self.ic_tstat >= 2 and self.mean_ic > 0:
            return "sedang"
        if self.mean_ic > 0:
            return "lemah"
        return "tidak ada / terbalik"

    def summary(self) -> dict[str, Any]:
        return {
            "horizon_sesi": self.horizon,
            "rebalance_tiap_sesi": self.step,
            "tanggal_uji": self.dates,
            "observasi": self.observations,
            "mean_ic": round(self.mean_ic, 4),
            "icir": round(self.icir, 3),
            "ic_tstat": round(self.ic_tstat, 2),
            "hit_rate_top_quintile": round(self.top_hit_rate, 4),
            "hit_rate_ci95": [round(self.top_hit_ci[0], 4), round(self.top_hit_ci[1], 4)],
            "excess_top_quintile_rata2": round(self.top_mean_excess, 5),
            "spread_q5_q1": round(self.spread_q5_q1, 5),
            "kekuatan_bukti": self.evidence_strength(),
            "catatan": self.notes,
        }


def run_signal_study(
    prices: pd.DataFrame,
    horizon: int = 20,
    step: int = 5,
    min_tickers: int = 5,
    buckets: int = 5,
) -> SignalStudy | None:
    """Run the walk-forward signal study. Returns ``None`` if data is insufficient."""
    prices = prices.sort_index().ffill(limit=5)
    if prices.shape[1] < min_tickers or len(prices) < 252 + horizon + step * 10:
        return None
    scores = point_in_time_scores(prices)
    forward = prices.shift(-horizon) / prices - 1
    excess = forward.sub(forward.median(axis=1), axis=0)
    dates = scores.index[252 : len(scores) - horizon : step]

    ics: dict[pd.Timestamp, float] = {}
    records: list[pd.DataFrame] = []
    spreads: list[float] = []
    for date in dates:
        s = scores.loc[date]
        e = excess.loc[date]
        f = forward.loc[date]
        mask = s.notna() & f.notna()
        if mask.sum() < min_tickers:
            continue
        ics[date] = float(s[mask].rank().corr(f[mask].rank()))
        quintile = pd.qcut(s[mask].rank(method="first"), q=min(5, int(mask.sum())), labels=False)
        top = quintile == quintile.max()
        bottom = quintile == quintile.min()
        spreads.append(float(f[mask][top].mean() - f[mask][bottom].mean()))
        records.append(pd.DataFrame({"score": s[mask], "excess": e[mask], "top": top}))
    if len(ics) < 10:
        return None
    ic_series = pd.Series(ics).sort_index()
    pooled = pd.concat(records)
    overlap = max(1.0, horizon / step)
    n_eff = len(ic_series) / overlap
    ic_std = float(ic_series.std(ddof=1))
    mean_ic = float(ic_series.mean())
    tstat = mean_ic / ic_std * np.sqrt(n_eff) if ic_std > 0 else float("nan")
    top_rows = pooled[pooled["top"]]
    hits = int((top_rows["excess"] > 0).sum())
    trials = int(len(top_rows))

    pooled["bucket"] = pd.cut(pooled["score"], bins=np.linspace(0, 100, buckets + 1), include_lowest=True)
    reliability = (
        pooled.groupby("bucket", observed=True)
        .agg(n=("excess", "size"), p_beat_median=("excess", lambda x: float((x > 0).mean())), mean_excess=("excess", "mean"))
        .reset_index()
    )
    intervals = list(reliability["bucket"])
    reliability["score_low"] = [max(0.0, float(b.left)) for b in intervals]
    reliability["score_high"] = [float(b.right) for b in intervals]
    reliability["score_mid"] = (reliability["score_low"] + reliability["score_high"]) / 2
    reliability["bucket"] = reliability["bucket"].astype(str)

    notes = [
        "Skor teknikal point-in-time; fundamental tidak diikutkan (tidak ada histori point-in-time).",
        "Universe = ticker yang dimuat hari ini (risiko survivorship bias, hasil cenderung optimis).",
        f"t-stat dikoreksi overlap horizon {horizon} vs langkah {step} sesi.",
    ]
    return SignalStudy(
        horizon=horizon,
        step=step,
        dates=int(len(ic_series)),
        observations=int(len(pooled)),
        mean_ic=mean_ic,
        ic_std=ic_std,
        icir=mean_ic / ic_std if ic_std > 0 else float("nan"),
        ic_tstat=float(tstat),
        top_hit_rate=hits / trials if trials else float("nan"),
        top_hit_ci=wilson_interval(hits, trials),
        top_mean_excess=float(top_rows["excess"].mean()) if trials else float("nan"),
        spread_q5_q1=float(np.nanmean(spreads)) if spreads else float("nan"),
        reliability=reliability,
        ic_series=ic_series,
        latest_scores=scores.iloc[-1].dropna(),
        notes=notes,
    )
