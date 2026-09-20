"""Vendor-neutral adapters for licensed market data and broker read APIs.

Provider profiles contain endpoints and response mappings, but never credentials.
Credentials are resolved from the operating-system vault by ``secret_name``.
"""
from __future__ import annotations

import ipaddress
import json
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

from finrl.integrations.secure_store import load_secret


PROVIDER_KINDS = ("market_data", "broker", "research")
PROVIDER_TIERS = ("licensed", "executable", "research")


def _extract(payload: Any, path: str, default: Any = None) -> Any:
    """Resolve a dotted path, including numeric list indexes."""
    if not path:
        return payload
    value = payload
    for part in path.split("."):
        try:
            value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return default
    return value


def _utc_timestamp(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Timestamp provider tidak valid: {value!r}")
    return timestamp


@dataclass
class ProviderProfile:
    id: str
    name: str
    kind: str
    tier: str
    base_url: str
    quote_path: str = ""
    history_path: str = ""
    account_path: str = ""
    auth_type: str = "bearer"
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer "
    secret_name: str = ""
    strip_suffix: str = ".JK"
    timeout_seconds: int = 15
    stale_after_seconds: int = 120
    quote_root: str = ""
    history_root: str = ""
    account_root: str = ""
    quote_mapping: dict[str, str] = field(default_factory=lambda: {
        "symbol": "symbol", "timestamp": "timestamp", "last": "last",
        "bid": "bid", "ask": "ask", "volume": "volume",
        "currency": "currency", "exchange": "exchange",
    })
    history_mapping: dict[str, str] = field(default_factory=lambda: {
        "timestamp": "timestamp", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "volume",
    })
    history_params: dict[str, str] = field(default_factory=lambda: {
        "start": "start", "end": "end", "interval": "interval",
    })

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProviderProfile":
        allowed = set(cls.__dataclass_fields__)
        profile = cls(**{key: value for key, value in payload.items() if key in allowed})
        profile.validate()
        return profile

    def validate(self) -> None:
        if not self.id or not self.id.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Provider ID hanya boleh berisi huruf, angka, '-' dan '_'.")
        if self.kind not in PROVIDER_KINDS or self.tier not in PROVIDER_TIERS:
            raise ValueError("Kind/tier provider tidak didukung.")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("Base URL provider harus berupa URL HTTP(S) yang valid.")
        if self.timeout_seconds < 1 or self.timeout_seconds > 120:
            raise ValueError("Timeout provider harus antara 1 dan 120 detik.")

    def provider_symbol(self, ticker: str) -> str:
        symbol = ticker.strip().upper()
        if self.strip_suffix and symbol.endswith(self.strip_suffix.upper()):
            symbol = symbol[: -len(self.strip_suffix)]
        return symbol


@dataclass
class NormalizedQuote:
    provider_id: str
    provider_name: str
    tier: str
    requested_ticker: str
    provider_symbol: str
    timestamp: str
    last: float
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    currency: str | None = None
    exchange: str | None = None
    age_seconds: float = 0.0
    status: str = "fresh"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class GenericRESTProvider:
    """Read-only REST adapter configured from a provider contract profile."""

    def __init__(self, profile: ProviderProfile, *, allow_private_hosts: bool = False):
        profile.validate()
        self.profile = profile
        self.allow_private_hosts = allow_private_hosts

    def _assert_remote_host(self) -> None:
        """Block accidental access to local metadata/services by default."""
        hostname = urlparse(self.profile.base_url).hostname
        if not hostname or self.allow_private_hosts:
            return
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(hostname, None)}
        except socket.gaierror:
            return  # requests will return the actionable DNS error
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise ConnectionError("Host lokal/private diblokir oleh pengaman adapter.")

    def _auth(self) -> tuple[dict[str, str], dict[str, str]]:
        headers = {"Accept": "application/json", "User-Agent": "FinRL-Workbench/1.0"}
        params: dict[str, str] = {}
        if self.profile.auth_type == "none":
            return headers, params
        token = load_secret(self.profile.secret_name) if self.profile.secret_name else None
        if not token:
            raise PermissionError(
                f"Secret '{self.profile.secret_name}' belum tersedia di credential vault."
            )
        if self.profile.auth_type in {"bearer", "header"}:
            headers[self.profile.auth_header] = f"{self.profile.auth_prefix}{token}"
        elif self.profile.auth_type == "query":
            params[self.profile.auth_header] = token
        else:
            raise ValueError(f"Auth type tidak didukung: {self.profile.auth_type}")
        return headers, params

    def _get(self, path: str, *, symbol: str = "", params: dict[str, Any] | None = None) -> Any:
        if not path:
            raise ValueError("Endpoint provider belum dikonfigurasi.")
        self._assert_remote_host()
        headers, auth_params = self._auth()
        request_params = {**auth_params, **(params or {})}
        endpoint = path.format(symbol=symbol)
        url = urljoin(self.profile.base_url.rstrip("/") + "/", endpoint.lstrip("/"))
        response = requests.get(
            url, headers=headers, params=request_params,
            timeout=self.profile.timeout_seconds,
        )
        if response.status_code >= 400:
            detail = response.text[:300].replace("\n", " ")
            raise ConnectionError(f"{self.profile.name} HTTP {response.status_code}: {detail}")
        try:
            return response.json()
        except ValueError as error:
            raise ValueError(f"{self.profile.name} tidak mengembalikan JSON valid.") from error

    def get_quote(self, ticker: str) -> NormalizedQuote:
        symbol = self.profile.provider_symbol(ticker)
        raw = self._get(self.profile.quote_path, symbol=symbol)
        payload = _extract(raw, self.profile.quote_root)
        mapping = self.profile.quote_mapping
        last = _extract(payload, mapping.get("last", "last"))
        if last is None:
            raise ValueError("Response quote tidak memiliki field last sesuai mapping.")
        observed = _utc_timestamp(_extract(payload, mapping.get("timestamp", "timestamp")))
        now = pd.Timestamp.now(tz="UTC")
        age = max(0.0, (now - observed).total_seconds())

        def optional_number(name: str) -> float | None:
            value = _extract(payload, mapping.get(name, name))
            return None if value in (None, "") else float(value)

        return NormalizedQuote(
            provider_id=self.profile.id, provider_name=self.profile.name,
            tier=self.profile.tier, requested_ticker=ticker,
            provider_symbol=str(_extract(payload, mapping.get("symbol", "symbol"), symbol)),
            timestamp=observed.isoformat(), last=float(last),
            bid=optional_number("bid"), ask=optional_number("ask"),
            volume=optional_number("volume"),
            currency=_extract(payload, mapping.get("currency", "currency")),
            exchange=_extract(payload, mapping.get("exchange", "exchange")),
            age_seconds=age,
            status="stale" if age > self.profile.stale_after_seconds else "fresh",
        )

    def get_history(
        self, ticker: str, start: str, end: str, interval: str
    ) -> pd.DataFrame:
        symbol = self.profile.provider_symbol(ticker)
        parameter_names = self.profile.history_params
        params = {
            parameter_names.get("start", "start"): start,
            parameter_names.get("end", "end"): end,
            parameter_names.get("interval", "interval"): interval,
        }
        raw = self._get(self.profile.history_path, symbol=symbol, params=params)
        records = _extract(raw, self.profile.history_root)
        if not isinstance(records, list):
            raise ValueError("History root harus menunjuk ke array bar OHLCV.")
        mapping = self.profile.history_mapping
        rows = []
        for record in records:
            rows.append({
                "timestamp": _utc_timestamp(_extract(record, mapping["timestamp"])),
                "tic": ticker.upper(),
                "open": float(_extract(record, mapping["open"])),
                "high": float(_extract(record, mapping["high"])),
                "low": float(_extract(record, mapping["low"])),
                "close": float(_extract(record, mapping["close"])),
                "volume": float(_extract(record, mapping["volume"], 0) or 0),
            })
        return pd.DataFrame(rows)

    def get_account(self) -> dict[str, Any]:
        if self.profile.kind != "broker":
            raise ValueError("Account endpoint hanya tersedia untuk profil broker.")
        raw = self._get(self.profile.account_path)
        payload = _extract(raw, self.profile.account_root)
        if not isinstance(payload, dict):
            raise ValueError("Account root harus menunjuk ke JSON object.")
        return payload


def load_provider_store(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"active_market_data": "", "active_broker": "", "providers": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("providers", []), list):
        raise ValueError("Format provider store tidak valid.")
    return payload


def save_provider_store(path: Path, payload: dict[str, Any]) -> None:
    """Validate profiles and replace the non-secret config atomically."""
    for item in payload.get("providers", []):
        ProviderProfile.from_dict(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def get_profile(path: Path, profile_id: str) -> ProviderProfile:
    store = load_provider_store(path)
    for item in store.get("providers", []):
        if item.get("id") == profile_id:
            return ProviderProfile.from_dict(item)
    raise ValueError(f"Provider profile tidak ditemukan: {profile_id}")


def yahoo_research_quote(ticker: str) -> NormalizedQuote:
    """Return an explicitly research-tier Yahoo snapshot for comparison only."""
    import yfinance as yf

    bars = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
    if bars.empty:
        raise ValueError(f"Yahoo tidak mengembalikan data untuk {ticker}.")
    if isinstance(bars.columns, pd.MultiIndex):
        bars.columns = bars.columns.get_level_values(0)
    row = bars.dropna(subset=["Close"]).iloc[-1]
    timestamp = _utc_timestamp(bars.dropna(subset=["Close"]).index[-1])
    age = max(0.0, (pd.Timestamp.now(tz="UTC") - timestamp).total_seconds())
    return NormalizedQuote(
        provider_id="yahoo_research", provider_name="Yahoo Finance (research)",
        tier="research", requested_ticker=ticker, provider_symbol=ticker,
        timestamp=timestamp.isoformat(), last=float(row["Close"]),
        volume=float(row.get("Volume", 0) or 0), age_seconds=age,
        status="research/delayed",
    )
