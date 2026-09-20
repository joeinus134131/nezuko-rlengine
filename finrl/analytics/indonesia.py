"""Vectorized analytics for large Indonesian equity universes.

The functions operate on FinRL's long OHLCV format and deliberately avoid
network access, so they are reusable in the dashboard, notebooks, and tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class IDXAnalysis:
    screener: pd.DataFrame
    correlation: pd.DataFrame
    drawdown: pd.DataFrame
    breadth: pd.DataFrame
    normalized_prices: pd.DataFrame


def _column(data: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in data.columns:
            return name
    raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(names)}")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def _max_drawdown(returns: pd.DataFrame) -> pd.Series:
    wealth = (1 + returns.fillna(0)).cumprod()
    return wealth.div(wealth.cummax()).sub(1).min()


def _period_return(prices: pd.DataFrame, periods: int) -> pd.Series:
    if len(prices) <= periods:
        return pd.Series(np.nan, index=prices.columns)
    return prices.iloc[-1].div(prices.iloc[-periods - 1]).sub(1)


def _percentile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, ascending=ascending, na_option="bottom") * 100


def build_idx_analysis(data: pd.DataFrame, risk_free_rate: float = 0.06) -> IDXAnalysis:
    """Calculate screening, risk, correlation, drawdown, and market breadth.

    Args:
        data: Long-form OHLCV frame containing ticker, time, close, and volume.
        risk_free_rate: Annual Indonesian risk-free assumption as a decimal.
    """
    if data.empty:
        raise ValueError("Data pasar kosong.")
    ticker_col = _column(data, "tic", "ticker", "symbol")
    date_col = _column(data, "timestamp", "date", "datetime")
    close_col = _column(data, "close", "Close")
    volume_col = _column(data, "volume", "Volume")

    frame = data[[date_col, ticker_col, close_col, volume_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[close_col] = pd.to_numeric(frame[close_col], errors="coerce")
    frame[volume_col] = pd.to_numeric(frame[volume_col], errors="coerce")
    frame = frame.dropna(subset=[date_col, ticker_col, close_col])
    frame = frame.sort_values([date_col, ticker_col])

    prices = frame.pivot_table(index=date_col, columns=ticker_col, values=close_col, aggfunc="last")
    volumes = frame.pivot_table(index=date_col, columns=ticker_col, values=volume_col, aggfunc="last")
    prices = prices.sort_index().ffill(limit=5)
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    observations = returns.count()

    annual_return = returns.mean() * TRADING_DAYS
    annual_volatility = returns.std() * np.sqrt(TRADING_DAYS)
    downside = returns.where(returns < 0).std() * np.sqrt(TRADING_DAYS)
    sharpe = _safe_divide(annual_return - risk_free_rate, annual_volatility)
    sortino = _safe_divide(annual_return - risk_free_rate, downside)
    var_95 = returns.quantile(0.05)
    cvar_95 = pd.Series(
        {ticker: returns[ticker][returns[ticker] <= var_95[ticker]].mean() for ticker in returns},
        dtype=float,
    )
    latest = prices.iloc[-1]
    sma20 = prices.rolling(20, min_periods=10).mean().iloc[-1]
    sma50 = prices.rolling(50, min_periods=20).mean().iloc[-1]
    sma200 = prices.rolling(200, min_periods=60).mean().iloc[-1]
    avg_volume = volumes.tail(20).mean()
    avg_value = (prices * volumes).tail(20).mean()
    liquidity_change = _safe_divide(volumes.tail(20).mean(), volumes.tail(60).mean()).sub(1)

    screener = pd.DataFrame(
        {
            "price": latest,
            "return_1m": _period_return(prices, 21),
            "return_3m": _period_return(prices, 63),
            "return_6m": _period_return(prices, 126),
            "return_12m": _period_return(prices, 252),
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": _max_drawdown(returns),
            "daily_var_95": var_95,
            "daily_cvar_95": cvar_95,
            "avg_volume_20d": avg_volume,
            "avg_value_idr_20d": avg_value,
            "liquidity_change": liquidity_change,
            "above_sma20": latest > sma20,
            "above_sma50": latest > sma50,
            "above_sma200": latest > sma200,
            "observations": observations,
        }
    )
    quality = _percentile_score(screener["sharpe"]) * 0.25
    momentum = _percentile_score(screener["return_6m"]) * 0.30
    trend = (
        screener[["above_sma20", "above_sma50", "above_sma200"]].sum(axis=1) / 3 * 100
    ) * 0.20
    defensive = _percentile_score(screener["annual_volatility"], ascending=False) * 0.15
    liquidity = _percentile_score(np.log1p(screener["avg_value_idr_20d"])) * 0.10
    screener["composite_score"] = quality + momentum + trend + defensive + liquidity
    screener.index.name = "ticker"
    screener = screener.sort_values("composite_score", ascending=False)

    wealth = (1 + returns.fillna(0)).cumprod()
    drawdown = wealth.div(wealth.cummax()).sub(1)
    normalized = prices.div(prices.apply(lambda series: series.dropna().iloc[0] if series.notna().any() else np.nan)) * 100
    daily_breadth = pd.DataFrame(index=prices.index)
    daily_breadth["advancers_pct"] = returns.gt(0).mean(axis=1) * 100
    daily_breadth["above_sma20_pct"] = prices.gt(prices.rolling(20, min_periods=10).mean()).mean(axis=1) * 100
    daily_breadth["above_sma50_pct"] = prices.gt(prices.rolling(50, min_periods=20).mean()).mean(axis=1) * 100
    daily_breadth.index.name = "date"

    return IDXAnalysis(
        screener=screener,
        correlation=returns.corr(min_periods=20),
        drawdown=drawdown,
        breadth=daily_breadth,
        normalized_prices=normalized,
    )
