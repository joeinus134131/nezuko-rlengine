"""FinRL processor backed by a configured licensed REST provider."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from finrl.integrations.provider_adapter import GenericRESTProvider
from finrl.integrations.provider_adapter import get_profile
from finrl.meta.data_processors.base_processor import BaseDataProcessor


class ConfiguredProviderProcessor(BaseDataProcessor):
    def __init__(self, profile_id: str, config_path: str):
        super().__init__()
        if not profile_id:
            raise ValueError("Pilih active licensed market-data provider terlebih dahulu.")
        profile = get_profile(Path(config_path), profile_id)
        if profile.kind not in {"market_data", "research"}:
            raise ValueError("Provider training harus bertipe market_data atau research.")
        self.adapter = GenericRESTProvider(profile)

    def download_data(self, ticker_list, start_date, end_date, time_interval):
        frames = [
            self.adapter.get_history(ticker, start_date, end_date, time_interval)
            for ticker in ticker_list
        ]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "tic"])

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"timestamp", "tic", "open", "high", "low", "close", "volume"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Data provider tidak lengkap: {', '.join(sorted(missing))}")
        clean = df.copy()
        clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True)
        numeric = ["open", "high", "low", "close", "volume"]
        clean[numeric] = clean[numeric].apply(pd.to_numeric, errors="coerce")
        clean = clean.dropna(subset=["timestamp", "tic", "close"])
        clean = clean.drop_duplicates(["timestamp", "tic"], keep="last")
        return clean.sort_values(["timestamp", "tic"]).reset_index(drop=True)

    def add_vix(self, df: pd.DataFrame) -> pd.DataFrame:
        raise ValueError("VIX belum tersedia pada generic licensed-provider adapter.")
