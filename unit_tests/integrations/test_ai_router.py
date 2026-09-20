from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from finrl.integrations.ai_router import chat_completion
from finrl.integrations.ai_router import list_models
from finrl.integrations.ai_router import normalize_base_url
from finrl.integrations.ai_router import RouterHTTPError


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_normalize_base_url() -> None:
    assert normalize_base_url("http://localhost:20128/v1/") == "http://localhost:20128/v1"
    with pytest.raises(ValueError):
        normalize_base_url("file:///tmp/router")


@patch("urllib.request.urlopen")
def test_list_models(mock_urlopen) -> None:
    mock_urlopen.return_value = FakeResponse(
        {"data": [{"id": "openai/model-a"}, {"id": "anthropic/model-b"}]}
    )
    assert list_models("http://localhost:20128/v1", "secret") == [
        "anthropic/model-b",
        "openai/model-a",
    ]


@patch("urllib.request.urlopen")
def test_chat_completion(mock_urlopen) -> None:
    mock_urlopen.return_value = FakeResponse(
        {
            "model": "router/selected",
            "choices": [{"message": {"content": "Analisis terstruktur"}}],
            "usage": {"total_tokens": 42},
        }
    )
    response = chat_completion(
        "https://router.example/v1",
        "secret",
        "auto",
        [{"role": "user", "content": "Analisis"}],
    )
    assert response.content == "Analisis terstruktur"
    assert response.model == "router/selected"
    assert response.usage["total_tokens"] == 42


def test_router_http_error_preserves_status_and_detail() -> None:
    error = RouterHTTPError(403, "provider denied access")
    assert error.status_code == 403
    assert error.detail == "provider denied access"
