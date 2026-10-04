"""Robustness guard for RL strategies and any equity curve.

Implements the checks that separate a real edge from a lucky backtest:

* full metric set (CAGR, Sharpe, Sortino, Calmar, VaR/CVaR, profit factor...)
* Probabilistic Sharpe Ratio — Bailey & Lopez de Prado (2012)
* Deflated Sharpe Ratio — Bailey & Lopez de Prado (2014), corrects for the
  number of configurations tried (selection bias / multiple testing)
* stationary block-bootstrap confidence interval of the Sharpe ratio
* walk-forward windows, walk-forward efficiency, multi-seed consistency
* return concentration ("was it one lucky spike?")
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def _returns(equity: pd.Series | list[float] | np.ndarray) -> pd.Series:
    series = pd.Series(equity, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return series.pct_change().dropna()


def performance_metrics(
    equity: pd.Series | list[float] | np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.06,
) -> dict[str, float]:
    """Compute the standard metric set from an equity curve (values, not returns)."""
    values = pd.Series(equity, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3 or values.iloc[0] <= 0:
        raise ValueError("Equity curve minimal 3 titik dan nilai awal > 0.")
    returns = values.pct_change().dropna()
    years = len(returns) / periods_per_year
    total_return = values.iloc[-1] / values.iloc[0] - 1
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1 if years > 0 and values.iloc[-1] > 0 else np.nan
    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = returns - rf_period
    sharpe = excess.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year) if returns.std(ddof=1) > 0 else np.nan
    downside = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)) * np.sqrt(periods_per_year)
    sortino = excess.mean() * periods_per_year / downside if downside > 0 else np.nan
    drawdown = values / values.cummax() - 1
    max_drawdown = drawdown.min()
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 and np.isfinite(cagr) else np.nan
    var95 = returns.quantile(0.05)
    cvar95 = returns[returns <= var95].mean()
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    profit_factor = gains / losses if losses > 0 else np.nan
    win_rate = (returns > 0).mean()
    average_win = returns[returns > 0].mean() if (returns > 0).any() else 0.0
    average_loss = returns[returns < 0].mean() if (returns < 0).any() else 0.0
    expectancy = win_rate * average_win + (1 - win_rate) * average_loss
    underwater = (drawdown < 0).astype(int)
    longest = int(underwater.groupby((underwater == 0).cumsum()).sum().max()) if len(underwater) else 0
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": float(max_drawdown),
        "longest_drawdown_periods": float(longest),
        "var95": float(var95),
        "cvar95": float(cvar95),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "expectancy": float(expectancy),
        "skew": float(stats.skew(returns, bias=False)) if len(returns) > 3 else np.nan,
        "kurtosis": float(stats.kurtosis(returns, fisher=False, bias=False)) if len(returns) > 3 else np.nan,
        "observations": float(len(returns)),
    }


def probabilistic_sharpe_ratio(
    returns: pd.Series | np.ndarray, sharpe_benchmark: float = 0.0
) -> float:
    """P(true per-period Sharpe > benchmark) given sample length, skew, kurtosis.

    ``sharpe_benchmark`` is expressed per period (not annualised).
    """
    sample = pd.Series(returns, dtype=float).dropna()
    n = len(sample)
    if n < 10 or sample.std(ddof=1) == 0:
        return float("nan")
    sharpe = sample.mean() / sample.std(ddof=1)
    skew = stats.skew(sample, bias=False)
    kurt = stats.kurtosis(sample, fisher=False, bias=False)
    denominator = np.sqrt(max(1e-12, 1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2))
    z = (sharpe - sharpe_benchmark) * np.sqrt(n - 1) / denominator
    return float(stats.norm.cdf(z))


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` skill-less strategies."""
    if n_trials <= 1 or sharpe_variance <= 0:
        return 0.0
    std = np.sqrt(sharpe_variance)
    return float(
        std
        * (
            (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
            + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )
    )


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray, n_trials: int, trial_sharpes: list[float] | None = None
) -> float:
    """PSR against the Sharpe you would expect from the best of ``n_trials`` random tries.

    ``trial_sharpes`` are per-period Sharpe ratios of all tried configurations
    (seeds, hyper-parameters, windows). If omitted, the variance is approximated
    by the sampling variance of the observed Sharpe (1/n).
    """
    sample = pd.Series(returns, dtype=float).dropna()
    if trial_sharpes and len(trial_sharpes) > 1:
        variance = float(np.var(trial_sharpes, ddof=1))
    else:
        variance = 1.0 / max(len(sample), 1)
    return probabilistic_sharpe_ratio(sample, expected_max_sharpe(max(n_trials, 1), variance))


def bootstrap_sharpe_ci(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = 252,
    n_boot: int = 1000,
    block: int = 10,
    alpha: float = 0.10,
    seed: int = 7,
) -> tuple[float, float]:
    """Stationary (moving-block) bootstrap CI for the annualised Sharpe ratio."""
    sample = np.asarray(pd.Series(returns, dtype=float).dropna())
    n = len(sample)
    if n < max(30, block * 2):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        resampled = np.concatenate([sample[s : s + block] for s in starts])[:n]
        std = resampled.std(ddof=1)
        sharpes[i] = resampled.mean() / std * np.sqrt(periods_per_year) if std > 0 else np.nan
    low, high = np.nanquantile(sharpes, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def return_concentration(equity: pd.Series | list[float], top_fraction: float = 0.01) -> float:
    """Share of total log-return contributed by the best ``top_fraction`` of days.

    Values above ~1.0 mean the strategy is negative without its best days —
    a classic sign of one lucky spike.
    """
    log_returns = np.log1p(_returns(equity))
    total = log_returns.sum()
    if len(log_returns) < 20 or total <= 0:
        return float("nan")
    top_n = max(1, int(len(log_returns) * top_fraction))
    return float(log_returns.nlargest(top_n).sum() / total)


def walk_forward_windows(
    start: str,
    end: str,
    train_years: int = 4,
    test_months: int = 6,
    step_months: int = 6,
) -> list[dict[str, str]]:
    """Rolling (train, test) windows. Test windows never overlap their training data."""
    windows: list[dict[str, str]] = []
    train_start = pd.Timestamp(start)
    final = pd.Timestamp(end)
    while True:
        train_end = train_start + pd.DateOffset(years=train_years) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if test_end > final:
            break
        windows.append(
            {
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
            }
        )
        train_start = train_start + pd.DateOffset(months=step_months)
    return windows


@dataclass
class GuardThresholds:
    min_observations: int = 60
    min_psr: float = 0.90
    min_dsr: float = 0.50
    min_window_pass_rate: float = 0.60
    max_seed_sharpe_std: float = 0.50
    max_concentration: float = 1.0  # >100%: negative without the best 1% of days
    min_wfe: float = 0.50
    max_drawdown: float = -0.35


# Alias for planner/decision package
BacktestGuardSettings = GuardThresholds


@dataclass
class GuardResult:
    status: str
    passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def evaluate_equity(
    equity: list[float] | pd.Series,
    benchmark: list[float] | pd.Series | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.06,
    n_trials: int = 1,
    trial_sharpes: list[float] | None = None,
    thresholds: GuardThresholds | None = None,
) -> GuardResult:
    """Single equity-curve guard (used by the planner for the active RL model)."""
    limits = thresholds or GuardThresholds()
    metrics = performance_metrics(equity, periods_per_year, risk_free_rate)
    returns = _returns(equity)
    rf_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    metrics["psr_vs_rf"] = probabilistic_sharpe_ratio(returns - rf_period, 0.0)
    metrics["dsr"] = deflated_sharpe_ratio(returns - rf_period, n_trials, trial_sharpes)
    metrics["sharpe_ci90_low"], metrics["sharpe_ci90_high"] = bootstrap_sharpe_ci(returns, periods_per_year)
    metrics["concentration_top1pct"] = return_concentration(equity)
    result = GuardResult(status="", metrics=metrics)

    def check(condition: bool | None, ok: str, bad: str, hard: bool = False) -> None:
        if condition is None:
            result.warnings.append(f"Tidak dapat dinilai: {bad}")
        elif condition:
            result.passed.append(ok)
        elif hard:
            result.failures.append(bad)
        else:
            result.warnings.append(bad)

    check(metrics["observations"] >= limits.min_observations,
          f"Observasi {metrics['observations']:.0f} memadai",
          f"Observasi hanya {metrics['observations']:.0f} (< {limits.min_observations})", hard=True)
    psr = metrics["psr_vs_rf"]
    check(None if not np.isfinite(psr) else psr >= limits.min_psr,
          f"PSR {psr:.2f}: edge di atas bunga bebas risiko signifikan",
          f"PSR {psr:.2f} < {limits.min_psr}: Sharpe belum signifikan secara statistik", hard=True)
    dsr = metrics["dsr"]
    if n_trials > 1:
        check(None if not np.isfinite(dsr) else dsr >= limits.min_dsr,
              f"DSR {dsr:.2f} lolos koreksi {n_trials} percobaan",
              f"DSR {dsr:.2f}: kemungkinan hasil terbaik dari banyak percobaan (overfit)", hard=True)
    concentration = metrics["concentration_top1pct"]
    check(None if not np.isfinite(concentration) else concentration <= limits.max_concentration,
          f"Tidak bergantung pada lonjakan tunggal (1% hari terbaik = {concentration:.0%} return)",
          f"Tanpa 1% hari terbaik strategi merugi ({concentration:.0%} return dari 1% hari terbaik)")
    check(metrics["max_drawdown"] >= limits.max_drawdown,
          f"Max drawdown {metrics['max_drawdown']:.1%} dalam batas",
          f"Max drawdown {metrics['max_drawdown']:.1%} melewati batas {limits.max_drawdown:.0%}")
    if benchmark is not None and len(pd.Series(benchmark).dropna()) > 2:
        bench_metrics = performance_metrics(benchmark, periods_per_year, risk_free_rate)
        metrics["benchmark_total_return"] = bench_metrics["total_return"]
        metrics["benchmark_sharpe"] = bench_metrics["sharpe"]
        check(metrics["sharpe"] > bench_metrics["sharpe"],
              "Sharpe di atas benchmark buy-and-hold",
              "Sharpe tidak mengalahkan benchmark buy-and-hold", hard=False)
    result.status = "DITOLAK" if result.failures else "WASPADA" if result.warnings else "LULUS"
    return result


def evaluate_walk_forward(
    runs: pd.DataFrame,
    thresholds: GuardThresholds | None = None,
) -> GuardResult:
    """Aggregate walk-forward x seed runs.

    ``runs`` columns: ``window``, ``seed``, ``oos_sharpe``, ``oos_return``,
    ``benchmark_return`` and optionally ``is_return`` (in-sample, annualised)
    and ``oos_annual_return`` for walk-forward efficiency.
    """
    limits = thresholds or GuardThresholds()
    if runs.empty:
        raise ValueError("Belum ada hasil walk-forward.")
    result = GuardResult(status="")
    per_window = runs.groupby("window").agg(
        sharpe_mean=("oos_sharpe", "mean"),
        sharpe_std=("oos_sharpe", "std"),
        return_mean=("oos_return", "mean"),
        benchmark=("benchmark_return", "mean"),
    )
    per_window["beat_benchmark"] = per_window["return_mean"] > per_window["benchmark"]
    pass_rate = float(per_window["beat_benchmark"].mean())
    seed_std = float(per_window["sharpe_std"].fillna(0).mean())
    result.metrics.update(
        windows=int(len(per_window)),
        seeds=int(runs["seed"].nunique()),
        window_pass_rate=pass_rate,
        mean_seed_sharpe_std=seed_std,
        mean_oos_sharpe=float(runs["oos_sharpe"].mean()),
        per_window=per_window.reset_index().to_dict(orient="records"),
    )
    if pass_rate >= limits.min_window_pass_rate:
        result.passed.append(f"Mengalahkan benchmark di {pass_rate:.0%} window")
    else:
        result.failures.append(f"Hanya mengalahkan benchmark di {pass_rate:.0%} window")
    if runs["seed"].nunique() < 3:
        result.warnings.append("Kurang dari 3 seed: konsistensi belum terbukti")
    elif seed_std <= limits.max_seed_sharpe_std:
        result.passed.append(f"Konsisten antar seed (std Sharpe {seed_std:.2f})")
    else:
        result.failures.append(f"Tidak stabil antar seed (std Sharpe {seed_std:.2f})")
    if {"is_return", "oos_annual_return"}.issubset(runs.columns):
        is_mean = runs["is_return"].mean()
        wfe = float(runs["oos_annual_return"].mean() / is_mean) if is_mean > 0 else float("nan")
        result.metrics["walk_forward_efficiency"] = wfe
        if np.isfinite(wfe) and wfe >= limits.min_wfe:
            result.passed.append(f"Walk-forward efficiency {wfe:.2f}")
        else:
            result.warnings.append(f"Walk-forward efficiency {wfe:.2f} rendah (indikasi overfit)")
    trial_sharpes = runs["oos_sharpe"].dropna().tolist()
    result.metrics["n_trials"] = len(trial_sharpes)
    result.status = "DITOLAK" if result.failures else "WASPADA" if result.warnings else "LULUS"
    return result


evaluate_equity_curve = evaluate_equity

