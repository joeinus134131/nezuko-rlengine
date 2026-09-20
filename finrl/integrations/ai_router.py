"""Small OpenAI-compatible client for 9Router, OpenRouter, and other gateways."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class RouterResponse:
    content: str
    model: str
    usage: dict[str, Any]


class RouterHTTPError(ConnectionError):
    """HTTP failure returned by the router or its upstream provider."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Router HTTP {status_code}: {detail[:500]}")


def normalize_base_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Base URL harus berupa URL http/https yang valid.")
    return value


def _headers(api_key: str) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-OpenRouter-Title": "FinRL Workbench",
    }
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return headers


def _request_json(
    url: str,
    api_key: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers=_headers(api_key),
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RouterHTTPError(error.code, detail) from error
    except urllib.error.URLError as error:
        raise ConnectionError(f"Router tidak dapat dihubungi: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise ConnectionError("Router mengembalikan respons non-JSON.") from error
    if not isinstance(result, dict):
        raise ConnectionError("Format respons router tidak dikenali.")
    return result


def list_models(base_url: str, api_key: str, timeout: int = 20) -> list[str]:
    result = _request_json(
        f"{normalize_base_url(base_url)}/models", api_key, timeout=timeout
    )
    models = result.get("data", [])
    identifiers = [item.get("id") for item in models if isinstance(item, dict)]
    return sorted({model for model in identifiers if isinstance(model, str) and model})


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 1800,
    timeout: int = 90,
) -> RouterResponse:
    if not model.strip():
        raise ValueError("Model router belum dipilih.")
    result = _request_json(
        f"{normalize_base_url(base_url)}/chat/completions",
        api_key,
        payload={
            "model": model.strip(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=timeout,
    )
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ConnectionError(f"Respons chat router tidak lengkap: {result}") from error
    if not isinstance(content, str):
        raise ConnectionError("Konten respons router bukan teks.")
    return RouterResponse(
        content=content,
        model=str(result.get("model") or model),
        usage=result.get("usage") if isinstance(result.get("usage"), dict) else {},
    )
