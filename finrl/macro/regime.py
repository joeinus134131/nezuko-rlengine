"""Transparent, rule-based market regime for IDX equities.

Every point added or removed is recorded as a reason so the user (and the AI
aggregator) can audit exactly why the regime is RISK_ON / NEUTRAL / RISK_OFF.

Units used by :class:`MacroSnapshot` (chosen for easy manual entry):

* rates, inflation, PMI: plain percent / index values (``6.0`` == 6%)
* ``*_change_1m``: percent change over ~1 month (``2.5`` == +2.5%)
* ``idr_change_1m``: change of USD/IDR. Positive == rupiah *weakening*
* ``yield_10y_change_1m_bp``: basis points (``30`` == +0.30pp)
* ``foreign_flow_20d_trn``: foreign net buy over 20 sessions in IDR trillion
"""
from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import Any

import numpy as np
import pandas as pd

from finrl.fundamentals.idx import base_ticker
from finrl.fundamentals.idx import sector_for

# Commodity sensitivity of common IDX names (positive = revenue rises with price).
COMMODITY_EXPOSURE: dict[str, tuple[str, ...]] = {
    "coal": ("ADRO", "AADI", "ITMG", "PTBA", "HRUM", "INDY", "UNTR"),
    "nickel": ("INCO", "ANTM", "MBMA", "NCKL", "MDKA"),
    "gold": ("ANTM", "MDKA", "BRMS"),
    "oil": ("MEDC", "AKRA", "PGAS", "ELSA"),
    "cpo": ("AALI", "LSIP", "SIMP", "DSNG"),
    "tin": ("TINS",),
    "pulp": ("INKP", "TKIM"),
}
USD_EXPORTERS = {ticker for group in ("coal", "nickel", "gold", "cpo", "tin", "pulp") for ticker in COMMODITY_EXPOSURE[group]}
BI_INFLATION_TARGET = (1.5, 3.5)  # 2.5% +/- 1%
DIVIDEND_SEASON_MONTHS = (3, 4, 5)  # AGM season for large IDX banks/BUMN


@dataclass
class MacroSnapshot:
    as_of: str = ""
    source: str = "manual"
    bi_rate: float | None = None
    bi_rate_change_3m: float | None = None
    inflation_yoy: float | None = None
    idr_change_1m: float | None = None
    yield_10y: float | None = None
    yield_10y_change_1m_bp: float | None = None
    pmi: float | None = None
    foreign_flow_20d_trn: float | None = None
    dxy_change_1m: float | None = None
    us10y_change_1m_bp: float | None = None
    coal_change_1m: float | None = None
    nickel_change_1m: float | None = None
    gold_change_1m: float | None = None
    oil_change_1m: float | None = None
    cpo_change_1m: float | None = None
    tin_change_1m: float | None = None
    pulp_change_1m: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MacroSnapshot":
        payload = payload or {}
        allowed = {item.name for item in fields(cls)}
        clean: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in allowed:
                continue
            if key in ("as_of", "source"):
                clean[key] = str(value or "")
            elif value is None or value == "":
                clean[key] = None
            else:
                try:
                    number = float(value)
                    clean[key] = number if np.isfinite(number) else None
                except (TypeError, ValueError):
                    clean[key] = None
        return cls(**clean)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def filled_fields(self) -> int:
        return sum(
            1 for key, value in self.to_dict().items() if key not in ("as_of", "source") and value is not None
        )


@dataclass
class MarketInternals:
    as_of: str
    source: str
    index_above_sma50: bool | None = None
    index_above_sma200: bool | None = None
    index_return_1m: float | None = None
    index_drawdown_52w: float | None = None
    breadth_above_sma20: float | None = None
    breadth_above_sma50: float | None = None
    vol_percentile: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegimeAssessment:
    label: str
    score: float
    reasons: list[dict[str, Any]] = field(default_factory=list)
    sector_tilts: dict[str, float] = field(default_factory=dict)
    commodity_tilts: dict[str, float] = field(default_factory=dict)
    usd_exporter_tilt: float = 0.0
    internals: dict[str, Any] = field(default_factory=dict)
    macro: dict[str, Any] = field(default_factory=dict)
    caps: list[str] = field(default_factory=list)

    def tilt_for(self, ticker: str, sector: str | None = None) -> tuple[float, list[str]]:
        """Points to add to the regime pillar for one ticker plus the reasons."""
        reasons: list[str] = []
        resolved_sector = sector_for(ticker, sector)
        points = float(self.sector_tilts.get(resolved_sector, 0.0))
        if points:
            reasons.append(f"Sektor {resolved_sector}: {points:+.0f}")
        code = base_ticker(ticker)
        for commodity, delta in self.commodity_tilts.items():
            if delta and code in COMMODITY_EXPOSURE.get(commodity, ()):
                points += delta
                reasons.append(f"Eksposur {commodity}: {delta:+.0f}")
        if self.usd_exporter_tilt and code in USD_EXPORTERS:
            points += self.usd_exporter_tilt
            reasons.append(f"Eksportir USD (efek kurs): {self.usd_exporter_tilt:+.0f}")
        return points, reasons

    def score_for(self, ticker: str, sector: str | None = None) -> float:
        points, _ = self.tilt_for(ticker, sector)
        return float(np.clip(self.score + points, 0.0, 100.0))

    @property
    def regime(self) -> str:
        return self.label

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _price_matrix(data: pd.DataFrame) -> pd.DataFrame:
    if {"tic", "close"}.issubset(data.columns):
        date_col = "timestamp" if "timestamp" in data.columns else "date"
        frame = data[[date_col, "tic", "close"]].copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        return frame.pivot_table(index=date_col, columns="tic", values="close", aggfunc="last").sort_index().ffill(limit=5)
    return data.sort_index()


def market_internals_from_prices(data: pd.DataFrame, benchmark: pd.Series | None = None) -> MarketInternals:
    """Derive trend, breadth, and volatility state from the loaded universe.

    If ``benchmark`` (e.g. ``^JKSE`` close) is given it is used as the index,
    otherwise an equal-weight index of the universe is built (documented as such).
    """
    prices = _price_matrix(data)
    if prices.empty:
        raise ValueError("Data harga kosong untuk menghitung rezim.")
    returns = prices.pct_change(fill_method=None)
    if benchmark is not None and len(benchmark.dropna()) > 60:
        index = benchmark.dropna().astype(float).sort_index()
        source = "benchmark"
    else:
        index = (1 + returns.mean(axis=1).fillna(0)).cumprod() * 100
        source = "equal-weight universe"
    index_returns = index.pct_change().dropna()

    def _last_above(window: int, minimum: int) -> bool | None:
        if len(index) < minimum:
            return None
        sma = index.rolling(window, min_periods=minimum).mean().iloc[-1]
        return bool(index.iloc[-1] > sma) if np.isfinite(sma) else None

    vol_percentile = None
    rolling_vol = index_returns.rolling(20).std().dropna()
    if len(rolling_vol) >= 60:
        window = rolling_vol.tail(252)
        vol_percentile = float((window <= window.iloc[-1]).mean() * 100)
    high_52w = index.tail(252).max()
    above20 = prices.gt(prices.rolling(20, min_periods=10).mean()).where(prices.notna())
    above50 = prices.gt(prices.rolling(50, min_periods=20).mean()).where(prices.notna())
    return MarketInternals(
        as_of=str(pd.Timestamp(prices.index[-1]).date()),
        source=source,
        index_above_sma50=_last_above(50, 20),
        index_above_sma200=_last_above(200, 120),
        index_return_1m=float(index.iloc[-1] / index.iloc[-22] - 1) * 100 if len(index) > 22 else None,
        index_drawdown_52w=float(index.iloc[-1] / high_52w - 1) * 100 if high_52w else None,
        breadth_above_sma20=float(above20.iloc[-1].mean() * 100) if len(prices) >= 10 else None,
        breadth_above_sma50=float(above50.iloc[-1].mean() * 100) if len(prices) >= 20 else None,
        vol_percentile=vol_percentile,
    )


class _Ledger:
    def __init__(self) -> None:
        self.score = 50.0
        self.reasons: list[dict[str, Any]] = []
        self.sector: dict[str, float] = {}
        self.commodity: dict[str, float] = {}
        self.usd_exporter = 0.0

    def add(self, factor: str, value: Any, points: float, effect: str) -> None:
        self.score += points
        self.reasons.append({"faktor": factor, "nilai": value, "poin": points, "efek": effect})

    def tilt(self, sector: str, points: float) -> None:
        self.sector[sector] = self.sector.get(sector, 0.0) + points


def assess_regime(
    internals: MarketInternals | MacroSnapshot | None,
    macro: MacroSnapshot | MarketInternals | None = None,
    month: int | None = None,
) -> RegimeAssessment:
    """Combine market internals and macro inputs into an auditable regime."""
    if isinstance(internals, MacroSnapshot) and (macro is None or isinstance(macro, MarketInternals)):
        internals, macro = macro, internals
    ledger = _Ledger()
    macro = macro or MacroSnapshot()
    month = month or pd.Timestamp.now(tz="Asia/Jakarta").month

    if internals is not None:
        if internals.index_above_sma50 is not None:
            ledger.add(
                "Indeks vs SMA50", "di atas" if internals.index_above_sma50 else "di bawah",
                8 if internals.index_above_sma50 else -8, "Tren menengah indeks",
            )
        if internals.index_above_sma200 is not None:
            ledger.add(
                "Indeks vs SMA200", "di atas" if internals.index_above_sma200 else "di bawah",
                6 if internals.index_above_sma200 else -6, "Tren primer indeks",
            )
        breadth = internals.breadth_above_sma50
        if breadth is not None:
            if breadth >= 60:
                ledger.add("Breadth > SMA50", f"{breadth:.0f}%", 8, "Partisipasi luas, reli sehat")
            elif breadth <= 40:
                ledger.add("Breadth > SMA50", f"{breadth:.0f}%", -8, "Partisipasi sempit, reli rapuh")
        vol = internals.vol_percentile
        if vol is not None:
            if vol >= 80:
                ledger.add("Volatilitas 20h (persentil 1th)", f"{vol:.0f}", -10, "Turbulensi tinggi")
            elif vol <= 30:
                ledger.add("Volatilitas 20h (persentil 1th)", f"{vol:.0f}", 4, "Pasar tenang")
        drawdown = internals.index_drawdown_52w
        if drawdown is not None and drawdown <= -15:
            ledger.add("Drawdown indeks dari puncak 52 minggu", f"{drawdown:.1f}%", -6, "Pasar dalam koreksi dalam")

    if macro.bi_rate_change_3m is not None:
        if macro.bi_rate_change_3m < 0:
            ledger.add("BI rate 3 bulan", f"{macro.bi_rate_change_3m:+.2f}pp", 6, "Siklus pelonggaran")
            ledger.tilt("Financial Services", 4)
            ledger.tilt("Real Estate", 8)
            ledger.tilt("Consumer Cyclical", 4)
        elif macro.bi_rate_change_3m > 0:
            ledger.add("BI rate 3 bulan", f"{macro.bi_rate_change_3m:+.2f}pp", -6, "Siklus pengetatan")
            ledger.tilt("Real Estate", -8)
            ledger.tilt("Consumer Cyclical", -4)
    if macro.inflation_yoy is not None:
        low, high = BI_INFLATION_TARGET
        if macro.inflation_yoy > high + 0.5:
            ledger.add("Inflasi YoY", f"{macro.inflation_yoy:.2f}%", -5, "Di atas target BI, risiko suku bunga naik")
            ledger.tilt("Consumer Defensive", -3)
            ledger.tilt("Consumer Cyclical", -4)
        elif low <= macro.inflation_yoy <= high:
            ledger.add("Inflasi YoY", f"{macro.inflation_yoy:.2f}%", 3, "Dalam target BI")
    if macro.idr_change_1m is not None:
        change = macro.idr_change_1m
        if change >= 4:
            ledger.add("USD/IDR 1 bulan", f"{change:+.2f}%", -12, "Rupiah melemah tajam, risiko outflow asing")
        elif change >= 2:
            ledger.add("USD/IDR 1 bulan", f"{change:+.2f}%", -8, "Rupiah melemah")
        elif change <= -1:
            ledger.add("USD/IDR 1 bulan", f"{change:+.2f}%", 4, "Rupiah menguat, inflow asing")
        if change >= 2:
            ledger.tilt("Financial Services", -8)
            ledger.tilt("Real Estate", -6)
            ledger.tilt("Consumer Defensive", -3)
            ledger.usd_exporter += 6
    if macro.yield_10y_change_1m_bp is not None:
        change_bp = macro.yield_10y_change_1m_bp
        if change_bp >= 30:
            ledger.add("Yield SBN 10Y 1 bulan", f"{change_bp:+.0f}bp", -6, "Biaya modal naik, valuasi tertekan")
            ledger.tilt("Financial Services", -4)
            ledger.tilt("Real Estate", -8)
            ledger.tilt("Industrials", -4)
        elif change_bp <= -30:
            ledger.add("Yield SBN 10Y 1 bulan", f"{change_bp:+.0f}bp", 5, "Biaya modal turun")
            ledger.tilt("Real Estate", 5)
            ledger.tilt("Financial Services", 3)
    if macro.pmi is not None:
        if macro.pmi >= 50:
            ledger.add("PMI manufaktur", f"{macro.pmi:.1f}", 3, "Ekspansi")
            ledger.tilt("Industrials", 3)
            ledger.tilt("Basic Materials", 2)
        elif macro.pmi < 48:
            ledger.add("PMI manufaktur", f"{macro.pmi:.1f}", -4, "Kontraksi")
            ledger.tilt("Industrials", -3)
    if macro.foreign_flow_20d_trn is not None:
        flow = macro.foreign_flow_20d_trn
        if flow >= 2:
            ledger.add("Net foreign 20 hari", f"Rp{flow:+.1f}T", 5, "Akumulasi asing")
            ledger.tilt("Financial Services", 3)
        elif flow <= -2:
            ledger.add("Net foreign 20 hari", f"Rp{flow:+.1f}T", -6, "Distribusi asing")
            ledger.tilt("Financial Services", -3)
    if macro.dxy_change_1m is not None and macro.dxy_change_1m >= 2:
        ledger.add("DXY 1 bulan", f"{macro.dxy_change_1m:+.2f}%", -4, "Dolar global menguat, tekanan EM")
    if macro.us10y_change_1m_bp is not None and macro.us10y_change_1m_bp >= 30:
        ledger.add("US 10Y 1 bulan", f"{macro.us10y_change_1m_bp:+.0f}bp", -3, "Tekanan yield global")
    for commodity in ("coal", "nickel", "gold", "oil", "cpo", "tin", "pulp"):
        change = getattr(macro, f"{commodity}_change_1m")
        if change is None:
            continue
        if change >= 8:
            ledger.commodity[commodity] = 8.0
        elif change <= -8:
            ledger.commodity[commodity] = -8.0
        elif abs(change) >= 4:
            ledger.commodity[commodity] = 4.0 if change > 0 else -4.0
        if commodity in ledger.commodity:
            ledger.reasons.append(
                {"faktor": f"Harga {commodity} 1 bulan", "nilai": f"{change:+.1f}%", "poin": 0,
                 "efek": f"Tilt emiten {commodity} {ledger.commodity[commodity]:+.0f}"}
            )
    if month in DIVIDEND_SEASON_MONTHS:
        ledger.tilt("Financial Services", 3)
        ledger.reasons.append(
            {"faktor": "Kalender", "nilai": "musim RUPS/dividen", "poin": 0,
             "efek": "Tilt bank +3 (cek cum-date)"}
        )

    score = float(np.clip(ledger.score, 0.0, 100.0))
    label = "RISK_ON" if score >= 60 else "RISK_OFF" if score <= 40 else "NEUTRAL"
    caps: list[str] = []
    if (
        internals is not None
        and internals.index_above_sma50 is False
        and internals.breadth_above_sma50 is not None
        and internals.breadth_above_sma50 < 40
        and label == "RISK_ON"
    ):
        label = "NEUTRAL"
        caps.append("Indeks < SMA50 dan breadth < 40%: label maksimal NEUTRAL")
    return RegimeAssessment(
        label=label,
        score=round(score, 2),
        reasons=ledger.reasons,
        sector_tilts=ledger.sector,
        commodity_tilts=ledger.commodity,
        usd_exporter_tilt=ledger.usd_exporter,
        internals=internals.to_dict() if internals is not None else {},
        macro=macro.to_dict(),
        caps=caps,
    )


MARKET_PROXIES = {
    "^JKSE": "IHSG",
    "IDR=X": "USD/IDR",
    "DX-Y.NYB": "DXY",
    "^TNX": "US 10Y",
    "BZ=F": "Brent",
    "GC=F": "Emas",
    "MTF=F": "Batubara Rotterdam",
}


def fetch_market_proxies(period: str = "2y") -> tuple[dict[str, float | None], pd.Series | None, dict[str, str]]:
    """Fetch free market proxies from Yahoo. Returns (macro fields, IHSG series, status).

    Network-dependent and best-effort: each failure is reported in ``status``
    instead of being silently replaced by a guess.
    """
    import yfinance as yf

    fields_out: dict[str, float | None] = {}
    status: dict[str, str] = {}
    benchmark = None
    for symbol, label in MARKET_PROXIES.items():
        try:
            history = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            close = history["Close"].dropna()
            if len(close) < 25:
                raise ValueError("histori terlalu pendek")
        except Exception as error:  # pragma: no cover - network
            status[label] = f"gagal: {error}"
            continue
        status[label] = f"ok ({close.index[-1].date()})"
        change_pct = float(close.iloc[-1] / close.iloc[-22] - 1) * 100
        if symbol == "^JKSE":
            benchmark = close
            benchmark.index = pd.to_datetime(benchmark.index).tz_localize(None)
        elif symbol == "IDR=X":
            fields_out["idr_change_1m"] = round(change_pct, 3)
        elif symbol == "DX-Y.NYB":
            fields_out["dxy_change_1m"] = round(change_pct, 3)
        elif symbol == "^TNX":
            fields_out["us10y_change_1m_bp"] = round(float(close.iloc[-1] - close.iloc[-22]) * 100, 1)
        elif symbol == "BZ=F":
            fields_out["oil_change_1m"] = round(change_pct, 3)
        elif symbol == "GC=F":
            fields_out["gold_change_1m"] = round(change_pct, 3)
        elif symbol == "MTF=F":
            fields_out["coal_change_1m"] = round(change_pct, 3)
    return fields_out, benchmark, status


# Convenience aliases for planner view and decision engine
compute_market_internals = market_internals_from_prices
classify_regime = assess_regime
fetch_regime_features = fetch_market_proxies

