from unittest.mock import Mock

import pytest

from finrl.integrations.alpaca import PAPER_BASE_URL
from finrl.integrations.alpaca import normalize_alpaca_paper_url
from finrl.integrations.alpaca import validate_alpaca_paper_connection


@pytest.mark.parametrize(
    "value",
    [
        "https://paper-api.alpaca.markets",
        "https://paper-api.alpaca.markets/",
        "https://paper-api.alpaca.markets/v2",
        "https://paper-api.alpaca.markets/v2/v2/",
    ],
)
def test_normalize_paper_url_prevents_duplicate_v2(value):
    assert normalize_alpaca_paper_url(value) == PAPER_BASE_URL


def test_normalize_paper_url_rejects_live_endpoint():
    with pytest.raises(ValueError, match="endpoint paper resmi"):
        normalize_alpaca_paper_url("https://api.alpaca.markets")


def test_validate_connection_is_read_only(monkeypatch):
    account = Mock(status="ACTIVE")
    client = Mock()
    client.get_account.return_value = account
    rest = Mock(return_value=client)
    monkeypatch.setattr("alpaca_trade_api.REST", rest)

    normalized, result = validate_alpaca_paper_connection(
        "paper-key", "paper-secret", "https://paper-api.alpaca.markets/v2"
    )

    assert normalized == PAPER_BASE_URL
    assert result is account
    client.get_account.assert_called_once_with()
    rest.assert_called_once_with("paper-key", "paper-secret", PAPER_BASE_URL, "v2")


def test_validate_connection_explains_unauthorized(monkeypatch):
    client = Mock()
    client.get_account.side_effect = Exception("unauthorized")
    monkeypatch.setattr("alpaca_trade_api.REST", Mock(return_value=client))

    with pytest.raises(PermissionError, match="akun Paper"):
        validate_alpaca_paper_connection(
            "wrong", "wrong", "https://paper-api.alpaca.markets"
        )
