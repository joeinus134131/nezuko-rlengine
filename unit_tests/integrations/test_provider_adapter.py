from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from finrl.integrations.provider_adapter import GenericRESTProvider
from finrl.integrations.provider_adapter import ProviderProfile
from finrl.integrations.provider_adapter import get_profile
from finrl.integrations.provider_adapter import save_provider_store


def _profile(**overrides):
    values = {
        "id": "idx_feed",
        "name": "Licensed Test Feed",
        "kind": "market_data",
        "tier": "licensed",
        "base_url": "https://api.vendor.test",
        "quote_path": "/quotes/{symbol}",
        "history_path": "/history/{symbol}",
        "auth_type": "none",
        "quote_root": "data",
        "history_root": "data.bars",
    }
    values.update(overrides)
    return ProviderProfile(**values)


def test_symbol_normalization_only_removes_suffix():
    profile = _profile()
    assert profile.provider_symbol("bbca.jk") == "BBCA"
    assert profile.provider_symbol("NVDA") == "NVDA"


def test_quote_is_normalized_and_provenance_is_kept(monkeypatch):
    response = Mock(status_code=200)
    response.json.return_value = {
        "data": {
            "symbol": "BBCA",
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "last": 9550,
            "bid": 9525,
            "ask": 9550,
            "volume": 123456,
            "currency": "IDR",
            "exchange": "IDX",
        }
    }
    monkeypatch.setattr("finrl.integrations.provider_adapter.requests.get", lambda *a, **k: response)
    monkeypatch.setattr("finrl.integrations.provider_adapter.socket.getaddrinfo", lambda *a: [])

    quote = GenericRESTProvider(_profile()).get_quote("BBCA.JK")

    assert quote.provider_symbol == "BBCA"
    assert quote.last == 9550
    assert quote.tier == "licensed"
    assert quote.status == "fresh"


def test_history_maps_vendor_payload_to_finrl_ohlcv(monkeypatch):
    response = Mock(status_code=200)
    response.json.return_value = {
        "data": {"bars": [{
            "timestamp": "2026-09-18T09:00:00+07:00",
            "open": 9500, "high": 9600, "low": 9475,
            "close": 9575, "volume": 1000,
        }]}
    }
    monkeypatch.setattr("finrl.integrations.provider_adapter.requests.get", lambda *a, **k: response)
    monkeypatch.setattr("finrl.integrations.provider_adapter.socket.getaddrinfo", lambda *a: [])

    frame = GenericRESTProvider(_profile()).get_history(
        "BBCA.JK", "2026-09-18", "2026-09-19", "1D"
    )

    assert frame.columns.tolist() == [
        "timestamp", "tic", "open", "high", "low", "close", "volume"
    ]
    assert frame.loc[0, "tic"] == "BBCA.JK"
    assert frame.loc[0, "close"] == 9575


def test_provider_store_round_trip(tmp_path: Path):
    path = tmp_path / "providers.json"
    profile = _profile()
    save_provider_store(path, {
        "active_market_data": profile.id,
        "active_broker": "",
        "providers": [asdict(profile)],
    })

    loaded = get_profile(path, profile.id)

    assert loaded.name == profile.name
    assert "token" not in path.read_text(encoding="utf-8").lower()
