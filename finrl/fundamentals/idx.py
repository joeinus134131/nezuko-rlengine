"""Fundamental scoring for IDX equities.

Design rules:

* Valuation uses *yields* (``1/PER``, ``1/PBV``) so loss-making or negative
  book companies naturally rank last instead of looking "cheap".
* Bank-specific leverage/liquidity ratios (DER, current ratio) are not applied
  to ``Financial Services`` because they are not comparable to non-banks.
* Missing data is never invented. It lowers ``completeness`` and the score is
  shrunk towards the neutral 50 in proportion to the missing information.
* Manual/CSV values (from annual reports, Stockbit, RTI, broker exports)
  override automatic values and are flagged as ``manual``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any
from typing import IO

import numpy as np
import pandas as pd

from finrl.analytics.scales import blended_score
from finrl.analytics.scales import weighted_mean

FUNDAMENTAL_COLUMNS = (
    "pe",
    "pb",
    "roe",
    "net_margin",
    "der",
    "eps_growth",
    "revenue_growth",
    "dividend_yield",
    "current_ratio",
    "fcf_yield",
)

# Yahoo sector names are used as canonical sector keys so automatic snapshots
# and this fallback map agree. Fallback covers LQ45 + SRI-KEHATI presets.
SECTOR_MAP: dict[str, str] = {
    **dict.fromkeys(("BBCA", "BBNI", "BBRI", "BBTN", "BMRI", "BRIS", "ARTO"), "Financial Services"),
    **dict.fromkeys(("ADRO", "AADI", "ITMG", "PTBA", "MEDC", "AKRA", "HRUM", "INDY"), "Energy"),
    **dict.fromkeys(
        ("ANTM", "INCO", "MDKA", "TINS", "INKP", "TKIM", "INTP", "SMGR", "BRPT", "TPIA", "MBMA", "NCKL"),
        "Basic Materials",
    ),
    **dict.fromkeys(
        ("CPIN", "JPFA", "GGRM", "HMSP", "ICBP", "INDF", "UNVR", "AALI", "AMRT", "MYOR"),
        "Consumer Defensive",
    ),
    **dict.fromkeys(("ACES", "ERAA", "MAPI", "PJAA", "AUTO"), "Consumer Cyclical"),
    **dict.fromkeys(("EXCL", "TLKM", "MNCN", "TBIG", "TOWR", "ISAT", "EMTK"), "Communication Services"),
    **dict.fromkeys(("ASII", "UNTR", "JSMR", "PTPP", "WIKA", "ADHI", "WSKT", "WTON"), "Industrials"),
    **dict.fromkeys(("BSDE", "PWON", "SMRA", "PPRO", "CTRA"), "Real Estate"),
    **dict.fromkeys(("KLBF", "MIKA", "SIDO", "HEAL"), "Healthcare"),
    **dict.fromkeys(("PGAS",), "Utilities"),
    **dict.fromkeys(("GOTO", "BUKA"), "Technology"),
}

BANK_SECTORS = {"Financial Services"}

# (worst, best) anchors for the absolute part of the blended score.
ABSOLUTE_RANGES: dict[str, tuple[float, float]] = {
    "earnings_yield": (0.0, 0.12),  # PER ~8x scores 100, loss scores 0
    "book_yield": (0.0, 1.0),  # PBV 1x scores 100
    "fcf_yield": (-0.02, 0.10),
    "roe": (0.0, 0.25),
    "net_margin": (0.0, 0.25),
    "der": (2.0, 0.0),
    "current_ratio": (0.8, 2.0),
    "eps_growth": (-0.20, 0.30),
    "revenue_growth": (-0.10, 0.20),
    "dividend_yield": (0.0, 0.07),
}

PILLARS: dict[str, tuple[str, ...]] = {
    "value": ("earnings_yield", "book_yield", "fcf_yield"),
    "quality": ("roe", "net_margin", "der", "current_ratio"),
    "growth": ("eps_growth", "revenue_growth"),
    "income": ("dividend_yield",),
}
PILLAR_WEIGHTS = {"value": 0.30, "quality": 0.35, "growth": 0.20, "income": 0.15}


def base_ticker(ticker: str) -> str:
    return str(ticker).upper().split(".")[0]


def sector_for(ticker: str, provided: str | None = None) -> str:
    if isinstance(provided, str) and provided.strip():
        return provided.strip()
    return SECTOR_MAP.get(base_ticker(ticker), "Unknown")


def _as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def normalize_yahoo_info(ticker: str, info: dict[str, Any]) -> dict[str, Any]:
    """Convert a ``yfinance.Ticker.info`` dict into canonical decimal fields."""
    der = _as_float(info.get("debtToEquity"))
    if np.isfinite(der):
        der = der / 100.0  # Yahoo reports D/E in percent (45.3 == 0.453x)
    dividend_yield = _as_float(info.get("dividendYield"))
    if np.isfinite(dividend_yield) and dividend_yield > 0.3:
        dividend_yield = dividend_yield / 100.0  # newer Yahoo payloads use percent
    if not np.isfinite(dividend_yield):
        dividend_yield = _as_float(info.get("trailingAnnualDividendYield"))
    market_cap = _as_float(info.get("marketCap"))
    free_cashflow = _as_float(info.get("freeCashflow"))
    fcf_yield = free_cashflow / market_cap if market_cap and np.isfinite(market_cap) else np.nan
    return {
        "ticker": ticker,
        "name": info.get("shortName") or info.get("longName"),
        "sector": sector_for(ticker, info.get("sector")),
        "industry": info.get("industry"),
        "market_cap": market_cap,
        "pe": _as_float(info.get("trailingPE")),
        "pb": _as_float(info.get("priceToBook")),
        "roe": _as_float(info.get("returnOnEquity")),
        "net_margin": _as_float(info.get("profitMargins")),
        "der": der,
        "eps_growth": _as_float(info.get("earningsGrowth")),
        "revenue_growth": _as_float(info.get("revenueGrowth")),
        "dividend_yield": dividend_yield,
        "current_ratio": _as_float(info.get("currentRatio")),
        "fcf_yield": fcf_yield,
        "target_mean_price": _as_float(info.get("targetMeanPrice")),
        "source": "yahoo",
    }


def fetch_yahoo_fundamentals(tickers: list[str], max_workers: int = 8) -> pd.DataFrame:
    """Download fundamental snapshots in parallel. Failures become empty rows."""
    import yfinance as yf

    def _one(ticker: str) -> dict[str, Any]:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        row = normalize_yahoo_info(ticker, info)
        if not info:
            row["source"] = "gagal"
        return row

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        rows = list(executor.map(_one, tickers))
    frame = pd.DataFrame(rows).set_index("ticker")
    frame["retrieved_at"] = pd.Timestamp.now(tz="Asia/Jakarta").isoformat()
    return frame


def read_fundamental_csv(source: str | IO[Any]) -> pd.DataFrame:
    """Read a manual fundamentals CSV.

    Required column: ``ticker``. Optional: any of :data:`FUNDAMENTAL_COLUMNS`
    plus ``sector``. Percent values may be written as ``12.5`` or ``0.125``;
    values above 1.5 for ratio-like percent fields are divided by 100.
    """
    frame = pd.read_csv(source)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "ticker" not in frame.columns:
        raise ValueError("CSV fundamental wajib memiliki kolom 'ticker'.")
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame["ticker"] = frame["ticker"].where(frame["ticker"].str.contains(r"\."), frame["ticker"] + ".JK")
    percent_like = ("roe", "net_margin", "eps_growth", "revenue_growth", "dividend_yield", "fcf_yield")
    for column in FUNDAMENTAL_COLUMNS:
        if column not in frame.columns:
            continue
        frame[column] = pd.to_numeric(
            frame[column].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if column in percent_like:
            frame[column] = frame[column].where(frame[column].abs() <= 1.5, frame[column] / 100.0)
    frame["source"] = "manual"
    return frame.set_index("ticker")


def merge_fundamentals(automatic: pd.DataFrame | None, manual: pd.DataFrame | None) -> pd.DataFrame:
    """Manual values override automatic ones cell-by-cell."""
    frames = [frame for frame in (automatic, manual) if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=[*FUNDAMENTAL_COLUMNS, "sector", "source"])
    if len(frames) == 1:
        return frames[0].copy()
    merged = manual.combine_first(automatic)
    overlap = manual.index.intersection(automatic.index)
    merged.loc[overlap, "source"] = "yahoo+manual"
    return merged


def score_fundamentals(frame: pd.DataFrame | None = None, tickers: list[str] | None = None) -> pd.DataFrame:
    """Return pillar scores, ``fundamental_score``, completeness, and quality flag."""
    if frame is None:
        frame = pd.DataFrame(index=tickers if tickers is not None else [])
    elif "ticker" in frame.columns:
        frame = frame.set_index("ticker")
    index = pd.Index(tickers if tickers is not None else frame.index, name="ticker")
    data = frame.reindex(index).copy() if not frame.empty else pd.DataFrame(index=index)
    for column in FUNDAMENTAL_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    provided_sector = data["sector"] if "sector" in data.columns else pd.Series(None, index=index)
    sectors = pd.Series(
        [sector_for(ticker, provided_sector.get(ticker)) for ticker in index], index=index
    )
    is_bank = sectors.isin(BANK_SECTORS)

    features = pd.DataFrame(index=index)
    features["earnings_yield"] = 1.0 / data["pe"].replace(0, np.nan)
    features["book_yield"] = 1.0 / data["pb"].replace(0, np.nan)
    for column in ("fcf_yield", "roe", "net_margin", "der", "current_ratio", "eps_growth", "revenue_growth", "dividend_yield"):
        features[column] = data[column]
    # Bank balance sheets are leverage by design; do not score DER/current ratio.
    features.loc[is_bank, ["der", "current_ratio"]] = np.nan
    applicable = pd.DataFrame(True, index=index, columns=features.columns)
    applicable.loc[is_bank, ["der", "current_ratio"]] = False

    scores = pd.DataFrame(index=index)
    group = sectors.where(sectors != "Unknown")
    for column, (worst, best) in ABSOLUTE_RANGES.items():
        scores[column] = blended_score(features[column], worst, best, groups=group)

    result = pd.DataFrame(index=index)
    result["sector"] = sectors
    for pillar, columns in PILLARS.items():
        result[f"{pillar}_score"] = scores[list(columns)].mean(axis=1, skipna=True)
    raw = weighted_mean(
        result[[f"{pillar}_score" for pillar in PILLARS]].rename(columns=lambda c: c.replace("_score", "")),
        PILLAR_WEIGHTS,
    )
    available = features.notna() & applicable
    completeness = available.sum(axis=1) / applicable.sum(axis=1).replace(0, np.nan)
    completeness = completeness.fillna(0.0)
    # Shrink towards neutral 50 proportionally to missing information.
    result["fundamental_raw"] = raw
    result["fundamental_score"] = (50.0 + (raw.fillna(50.0) - 50.0) * completeness).round(2)
    result["completeness"] = completeness.round(3)
    result["data_quality"] = np.select(
        [completeness >= 0.8, completeness >= 0.4, completeness > 0],
        ["lengkap", "sebagian", "minim"],
        default="kosong",
    )
    source = data["source"] if "source" in data.columns else pd.Series("-", index=index)
    result["source"] = source.fillna("-")
    for column in FUNDAMENTAL_COLUMNS:
        result[column] = data[column]
    for optional in ("name", "market_cap", "target_mean_price", "retrieved_at"):
        if optional in data.columns:
            result[optional] = data[optional]
    return result
