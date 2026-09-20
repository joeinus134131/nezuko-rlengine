"""Validation helpers for the Alpaca paper-trading connection."""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse, urlunparse


PAPER_HOST = "paper-api.alpaca.markets"
PAPER_BASE_URL = f"https://{PAPER_HOST}"


def normalize_alpaca_paper_url(value: str) -> str:
    """Return the SDK base URL, removing a mistakenly appended API version.

    ``alpaca_trade_api.REST(..., api_version='v2')`` appends ``/v2`` itself.
    Passing a base URL ending in ``/v2`` therefore creates ``/v2/v2``.
    """
    candidate = value.strip().rstrip("/")
    if not candidate:
        raise ValueError("Alpaca paper base URL wajib diisi.")
    parsed = urlparse(candidate)
    if parsed.scheme != "https" or parsed.hostname != PAPER_HOST:
        raise ValueError(
            f"Gunakan endpoint paper resmi {PAPER_BASE_URL}; endpoint live atau host lain ditolak."
        )
    path_parts = [part for part in parsed.path.split("/") if part]
    if path_parts and any(part.lower() != "v2" for part in path_parts):
        raise ValueError(
            "Alpaca paper base URL tidak boleh memuat path selain /v2. "
            f"Gunakan {PAPER_BASE_URL}."
        )
    return urlunparse(("https", PAPER_HOST, "", "", "", ""))


def validate_alpaca_paper_connection(
    api_key: str, api_secret: str, base_url: str
) -> tuple[str, Any]:
    """Perform a read-only account request and return normalized URL/account."""
    if not api_key.strip() or not api_secret.strip():
        raise ValueError("Alpaca paper API key dan secret wajib diisi.")
    normalized = normalize_alpaca_paper_url(base_url)
    import alpaca_trade_api as tradeapi

    client = tradeapi.REST(api_key.strip(), api_secret.strip(), normalized, "v2")
    try:
        account = client.get_account()
    except Exception as error:
        status = getattr(error, "status_code", None)
        message = str(error).lower()
        if status in {401, 403} or "unauthorized" in message or "forbidden" in message:
            raise PermissionError(
                "Kredensial Alpaca ditolak. Gunakan key/secret dari akun Paper, "
                "bukan akun Live, dan buat ulang secret bila nilainya tidak lagi terlihat."
            ) from error
        if status == 404 or "404" in message:
            raise ConnectionError(
                f"Endpoint Alpaca tidak ditemukan. Base URL yang benar: {PAPER_BASE_URL} "
                "(tanpa /v2)."
            ) from error
        raise ConnectionError(f"Koneksi read-only ke Alpaca gagal: {error}") from error
    return normalized, account
