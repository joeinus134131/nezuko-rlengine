"""Verify the local FinRL runtime used by the dashboard."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "streamlit",
    "torch",
    "gymnasium",
    "gym",
    "elegantrl",
    "stable_baselines3",
    "ray",
    "yfinance",
    "stockstats",
    "talib",
    "alpaca",
    "alpaca_trade_api",
)


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    failures = []
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "installed")
            print(f"[OK] {module_name}: {version}")
        except Exception as error:
            failures.append((module_name, str(error)))
            print(f"[FAIL] {module_name}: {error}")

    for directory in ("datasets", "trained_models", "results", "tensorboard_log"):
        path = PROJECT_ROOT / directory
        path.mkdir(exist_ok=True)
        print(f"[OK] writable: {path.resolve()}")

    try:
        import finrl.dashboard  # noqa: F401

        print("[OK] finrl.dashboard import")
    except Exception as error:
        failures.append(("finrl.dashboard", str(error)))
        print(f"[FAIL] finrl.dashboard: {error}")

    if failures:
        print(f"Setup belum siap: {len(failures)} pemeriksaan gagal.")
        return 1
    print("Setup siap. Jalankan: .venv311/bin/streamlit run finrl/dashboard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
