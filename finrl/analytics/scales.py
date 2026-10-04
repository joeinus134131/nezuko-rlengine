"""Score scaling helpers shared by the fundamental, regime, and decision layers.

Two complementary views are combined:

* **Cross-sectional (relative)** — percentile rank inside the universe or
  sector. Answers "is this stock better than its peers today?".
* **Absolute (time-series)** — a piecewise-linear map between an economically
  "bad" and "good" value. Answers "is this number good on its own?". This keeps
  a weak market from making its least-bad stock look excellent and keeps small
  universes (1-4 tickers) meaningful, where percentiles are degenerate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def linear_score(values: pd.Series, worst: float, best: float) -> pd.Series:
    """Map ``worst -> 0`` and ``best -> 100`` linearly, clipped to ``[0, 100]``.

    ``best`` may be smaller than ``worst`` for metrics where lower is better.
    NaN stays NaN so the caller can treat it as missing data, not as a score.
    """
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if best == worst:
        raise ValueError("best dan worst tidak boleh sama.")
    scaled = (numeric - worst) / (best - worst) * 100.0
    return scaled.clip(lower=0.0, upper=100.0)


def percentile_score(
    values: pd.Series, higher_is_better: bool = True, groups: pd.Series | None = None
) -> pd.Series:
    """Percentile rank 0-100. NaN stays NaN (missing data is not penalised here)."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    if groups is None:
        ranked = numeric.rank(pct=True, ascending=higher_is_better)
    else:
        ranked = numeric.groupby(groups).rank(pct=True, ascending=higher_is_better)
    return ranked * 100.0


def blended_score(
    values: pd.Series,
    worst: float,
    best: float,
    relative_weight: float = 0.5,
    min_cross_section: int = 5,
    groups: pd.Series | None = None,
    min_group_size: int = 3,
) -> pd.Series:
    """Blend percentile and absolute scores.

    * If fewer than ``min_cross_section`` valid observations exist, only the
      absolute score is used (percentiles of 1-4 names are not informative).
    * If ``groups`` (e.g. sector) is given, a group percentile is used for
      groups with at least ``min_group_size`` valid members, otherwise the
      universe percentile is used for that ticker.
    """
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    absolute = linear_score(numeric, worst, best)
    valid = int(numeric.notna().sum())
    if valid < min_cross_section:
        return absolute
    higher_is_better = best > worst
    relative = percentile_score(numeric, higher_is_better)
    if groups is not None:
        aligned = groups.reindex(numeric.index)
        group_sizes = numeric.groupby(aligned).transform("count")
        group_relative = percentile_score(numeric, higher_is_better, aligned)
        use_group = group_sizes.ge(min_group_size) & aligned.notna()
        relative = relative.where(~use_group, group_relative)
    weight = float(np.clip(relative_weight, 0.0, 1.0))
    return relative * weight + absolute * (1.0 - weight)


def weighted_mean(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Row-wise weighted mean that renormalises weights over non-missing columns."""
    columns = [column for column in weights if column in frame.columns]
    if not columns:
        return pd.Series(np.nan, index=frame.index)
    values = frame[columns].astype(float)
    weight_row = pd.Series({column: float(weights[column]) for column in columns})
    mask = values.notna()
    numerator = values.fillna(0.0).mul(weight_row, axis=1).sum(axis=1)
    denominator = mask.mul(weight_row, axis=1).sum(axis=1)
    return numerator.div(denominator.replace(0.0, np.nan))
