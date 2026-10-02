from __future__ import annotations

from datetime import date

from finrl.validation import experiment_period_errors


def test_experiment_period_accepts_strict_chronology() -> None:
    assert experiment_period_errors(
        "2020-01-01",
        "2022-12-31",
        "2023-01-01",
        "2024-12-31",
        today=date(2025, 1, 1),
    ) == []


def test_experiment_period_rejects_overlap_and_future() -> None:
    errors = experiment_period_errors(
        "2020-01-01",
        "2025-12-31",
        "2024-01-01",
        "2026-12-31",
        today=date(2026, 10, 2),
    )
    assert any("tumpang tindih" in error for error in errors)
    assert any("melewati hari ini" in error for error in errors)


def test_experiment_period_rejects_reversed_ranges() -> None:
    errors = experiment_period_errors(
        "2022-01-01",
        "2021-01-01",
        "2024-12-31",
        "2024-01-01",
        today=date(2025, 1, 1),
    )
    assert any("Train mulai" in error for error in errors)
    assert any("Test mulai" in error for error in errors)
