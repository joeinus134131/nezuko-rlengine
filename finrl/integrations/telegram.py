"""Telegram Bot API notification client."""
from __future__ import annotations

from typing import Any

import requests


def send_telegram_message(
    bot_token: str,
    chat_id: str,
    text: str,
    parse_mode: str | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    if not bot_token or not chat_id:
        raise ValueError("Telegram bot token dan chat ID wajib diisi.")
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text[:4096],
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    response = requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json=payload,
        timeout=timeout,
    )
    try:
        result = response.json()
    except ValueError as error:
        raise ConnectionError(f"Telegram mengembalikan HTTP {response.status_code} non-JSON.") from error
    if not response.ok or not result.get("ok"):
        description = result.get("description", response.text[:300])
        raise ConnectionError(f"Telegram gagal ({response.status_code}): {description}")
    return result
