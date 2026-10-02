"""Pure validation helpers shared by the dashboard and tests."""
from __future__ import annotations

from datetime import date

import pandas as pd


def experiment_period_errors(
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    *,
    today: date | None = None,
) -> list[str]:
    """Return actionable chronology errors for an experiment split."""
    labels = {
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
    }
    parsed: dict[str, pd.Timestamp] = {}
    errors: list[str] = []
    for name, value in labels.items():
        try:
            parsed[name] = pd.Timestamp(value).normalize()
        except (TypeError, ValueError):
            errors.append(f"Tanggal {name} tidak valid: {value!r}.")
    if errors:
        return errors

    if parsed["train_start"] >= parsed["train_end"]:
        errors.append("Train mulai harus lebih awal daripada train selesai.")
    if parsed["train_end"] >= parsed["test_start"]:
        errors.append(
            "Periode train dan test tidak boleh tumpang tindih; "
            "test harus dimulai setelah train selesai."
        )
    if parsed["test_start"] >= parsed["test_end"]:
        errors.append("Test mulai harus lebih awal daripada test selesai.")

    today_ts = pd.Timestamp(today or date.today()).normalize()
    if parsed["test_end"] > today_ts:
        errors.append(
            f"Test selesai ({parsed['test_end'].date()}) tidak boleh melewati "
            f"hari ini ({today_ts.date()})."
        )
    return errors
