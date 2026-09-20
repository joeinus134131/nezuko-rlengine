"""Scheduled IDX research monitor that sends source-attributed Telegram alerts."""
from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from finrl.analytics.indonesia import build_idx_analysis
from finrl.config import INDICATORS
from finrl.integrations.ai_router import chat_completion
from finrl.integrations.secure_store import load_secret
from finrl.integrations.telegram import send_telegram_message
from finrl.meta.data_processor import DataProcessor


@dataclass(frozen=True)
class MonitorConfig:
    tickers: list[str]
    chat_id: str
    schedule_mode: str = "daily"
    daily_time: str = "17:00"
    interval_minutes: int = 60
    lookback_days: int = 420
    top_n: int = 5
    risk_free_rate: float = 0.06
    use_ai_summary: bool = False
    ai_base_url: str = "http://localhost:20128/v1"
    ai_model: str = ""

    @classmethod
    def from_json(cls, path: str | Path) -> "MonitorConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


def _signal_label(row: pd.Series) -> str:
    trend_count = sum(bool(row.get(name, False)) for name in ("above_sma20", "above_sma50", "above_sma200"))
    score = float(row.get("composite_score", 0))
    if score >= 70 and trend_count >= 2:
        return "WATCH POSITIVE"
    if score < 40 or trend_count == 0:
        return "CAUTION"
    return "NEUTRAL / MONITOR"


def build_daily_report(config: MonitorConfig) -> str:
    end = pd.Timestamp.now(tz="Asia/Jakarta").normalize().tz_localize(None)
    start = end - timedelta(days=config.lookback_days)
    processor = DataProcessor("yahoofinance")
    data = processor.download_data(
        config.tickers,
        start.strftime("%Y-%m-%d"),
        (end + timedelta(days=1)).strftime("%Y-%m-%d"),
        "1D",
    )
    data = processor.clean_data(data)
    data = processor.add_technical_indicator(data, INDICATORS)
    analysis = build_idx_analysis(data, risk_free_rate=config.risk_free_rate)
    top = analysis.screener.head(config.top_n)
    breadth = analysis.breadth.iloc[-1]

    lines = [
        "FinRL IDX Daily Monitor",
        f"Waktu: {pd.Timestamp.now(tz='Asia/Jakarta').strftime('%Y-%m-%d %H:%M WIB')}",
        f"Universe: {len(analysis.screener)} ticker | Data: Yahoo Finance",
        (
            f"Breadth: advancers {breadth['advancers_pct']:.1f}% | "
            f">SMA20 {breadth['above_sma20_pct']:.1f}% | "
            f">SMA50 {breadth['above_sma50_pct']:.1f}%"
        ),
        "",
        "Top research watchlist:",
    ]
    for ticker, row in top.iterrows():
        lines.append(
            f"• {ticker} — {_signal_label(row)} | score {row['composite_score']:.1f} | "
            f"1M {row['return_1m'] * 100:.1f}% | 6M {row['return_6m'] * 100:.1f}% | "
            f"vol {row['annual_volatility'] * 100:.1f}% | DD {row['max_drawdown'] * 100:.1f}%"
        )
    lines.extend(
        [
            "",
            "Label adalah sinyal riset teknikal relatif, bukan rekomendasi beli/jual.",
            "Verifikasi data, fundamental, berita material, likuiditas, dan risiko pribadi.",
        ]
    )
    report = "\n".join(lines)

    if config.use_ai_summary and config.ai_model:
        try:
            api_key = load_secret("ai_router_api_key") or ""
            response = chat_completion(
                config.ai_base_url,
                api_key,
                config.ai_model,
                [
                    {
                        "role": "system",
                        "content": (
                            "Ringkas laporan monitoring IDX dalam Bahasa Indonesia. Jangan menambah "
                            "fakta baru atau memberi perintah beli/jual. Soroti risiko dan data yang "
                            "perlu diverifikasi. Maksimal 900 karakter."
                        ),
                    },
                    {"role": "user", "content": report},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            report += f"\n\nAI research note ({response.model}):\n{response.content}"
        except Exception as error:
            report += f"\n\nAI research note tidak tersedia: {error}"
    return report


def run_once(config: MonitorConfig) -> dict[str, Any]:
    token = load_secret("telegram_bot_token")
    if not token:
        raise RuntimeError("Telegram bot token belum tersimpan di secure vault.")
    return send_telegram_message(token, config.chat_id, build_daily_report(config))


def run_loop(config: MonitorConfig) -> None:
    running = True

    def stop(*_args):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_daily_date: str | None = None
    next_interval = datetime.now()
    while running:
        now = datetime.now()
        should_run = False
        if config.schedule_mode == "interval":
            should_run = now >= next_interval
            if should_run:
                next_interval = now + timedelta(minutes=max(5, config.interval_minutes))
        else:
            current_date = now.strftime("%Y-%m-%d")
            should_run = now.strftime("%H:%M") >= config.daily_time and last_daily_date != current_date
            if should_run:
                last_daily_date = current_date
        if should_run:
            try:
                run_once(config)
            except Exception as error:
                print(f"monitor error: {error}", flush=True)
        time.sleep(15)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = MonitorConfig.from_json(args.config)
    if args.once:
        run_once(config)
    else:
        run_loop(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
