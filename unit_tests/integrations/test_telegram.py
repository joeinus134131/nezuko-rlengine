from __future__ import annotations

from unittest.mock import patch

import pytest

from finrl.integrations.telegram import send_telegram_message


class FakeResponse:
    ok = True
    status_code = 200
    text = "ok"

    def json(self):
        return {"ok": True, "result": {"message_id": 1}}


@patch("requests.post")
def test_send_telegram_message(mock_post) -> None:
    mock_post.return_value = FakeResponse()
    result = send_telegram_message("token", "123", "hello")
    assert result["ok"] is True
    request = mock_post.call_args
    assert request.args[0].endswith("/bottoken/sendMessage")
    assert request.kwargs["json"]["chat_id"] == "123"


def test_send_telegram_message_requires_credentials() -> None:
    with pytest.raises(ValueError):
        send_telegram_message("", "", "hello")
