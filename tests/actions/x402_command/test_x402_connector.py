import base64
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from actions.x402_command.connector.x402 import (
    NETWORK_CONFIG,
    X402Config,
    X402Connector,
)
from actions.x402_command.interface import X402PaymentInput


@pytest.fixture
def config():
    return X402Config(
        private_key="0x" + "ab" * 32,
        x402_endpoint="https://example.com/x402",
        network="base-sepolia",
    )


@pytest.fixture
def connector(config):
    return X402Connector(config)


def _make_aiohttp_response(status, json_data=None, text_data=""):
    """Create a mock that works with aiohttp async context manager pattern."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.json = AsyncMock(return_value=json_data)
    mock_resp.text = AsyncMock(return_value=text_data)
    return mock_resp


def _make_mock_session(*responses):
    """Create a mock aiohttp session that returns responses in order."""
    mock_session = MagicMock()
    mock_session.closed = False

    call_count = 0

    @asynccontextmanager
    async def mock_post(*args, **kwargs):
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        yield responses[idx]

    mock_session.post = mock_post
    return mock_session


def test_initialization(connector):
    assert connector.config.x402_endpoint == "https://example.com/x402"
    assert connector.config.network == "base-sepolia"
    assert connector._session is None


def test_config_defaults():
    config = X402Config()
    assert config.private_key is None
    assert config.x402_endpoint is None
    assert config.network == "base-sepolia"


def test_network_config_contains_expected_networks():
    assert "base-sepolia" in NETWORK_CONFIG
    assert "base" in NETWORK_CONFIG
    assert "chain_id" in NETWORK_CONFIG["base-sepolia"]
    assert "usdc_address" in NETWORK_CONFIG["base-sepolia"]
    assert NETWORK_CONFIG["base-sepolia"]["chain_id"] == 84532
    assert NETWORK_CONFIG["base"]["chain_id"] == 8453


@pytest.mark.asyncio
async def test_connect_missing_endpoint():
    config = X402Config(
        private_key="0x" + "ab" * 32,
        x402_endpoint=None,
        network="base-sepolia",
    )
    conn = X402Connector(config)

    with patch("actions.x402_command.connector.x402.logging") as mock_log:
        await conn.connect(X402PaymentInput(action="test"))
        mock_log.error.assert_called_with("x402 endpoint is not configured")


@pytest.mark.asyncio
async def test_connect_missing_private_key():
    config = X402Config(
        private_key=None,
        x402_endpoint="https://example.com/x402",
        network="base-sepolia",
    )
    conn = X402Connector(config)

    with patch("actions.x402_command.connector.x402.logging") as mock_log:
        await conn.connect(X402PaymentInput(action="test"))
        mock_log.error.assert_called_with("Private key is not configured")


@pytest.mark.asyncio
async def test_connect_payment_requirements_failure(connector):
    mock_session = _make_mock_session()
    connector._session = mock_session

    with (
        patch.object(
            connector, "_get_session", new_callable=AsyncMock
        ) as mock_get_session,
        patch.object(
            connector, "_fetch_payment_requirements", new_callable=AsyncMock
        ) as mock_fetch,
    ):
        mock_get_session.return_value = mock_session
        mock_fetch.return_value = None
        await connector.connect(X402PaymentInput(action="test"))
        mock_fetch.assert_called_once_with(mock_session, "https://example.com/x402")


@pytest.mark.asyncio
async def test_connect_incomplete_requirements(connector):
    mock_session = _make_mock_session()

    with (
        patch.object(
            connector, "_get_session", new_callable=AsyncMock
        ) as mock_get_session,
        patch.object(
            connector, "_fetch_payment_requirements", new_callable=AsyncMock
        ) as mock_fetch,
        patch("actions.x402_command.connector.x402.logging") as mock_log,
    ):
        mock_get_session.return_value = mock_session
        mock_fetch.return_value = {"payTo": None, "maxAmountRequired": None}
        await connector.connect(X402PaymentInput(action="test"))
        mock_log.error.assert_called_with(
            "Incomplete payment requirements from endpoint"
        )


@pytest.mark.asyncio
async def test_connect_successful_payment(connector):
    mock_resp = _make_aiohttp_response(status=200)
    mock_session = _make_mock_session(mock_resp)

    requirements = {
        "payTo": "0x1234567890abcdef1234567890abcdef12345678",
        "maxAmountRequired": "10000",
        "maxTimeoutSeconds": 300,
        "network": "base-sepolia",
    }

    with (
        patch.object(
            connector, "_get_session", new_callable=AsyncMock
        ) as mock_get_session,
        patch.object(
            connector, "_fetch_payment_requirements", new_callable=AsyncMock
        ) as mock_fetch,
        patch.object(
            connector, "_build_payment_payload", return_value="encoded_payload"
        ),
        patch("actions.x402_command.connector.x402.logging") as mock_log,
    ):
        mock_get_session.return_value = mock_session
        mock_fetch.return_value = requirements
        await connector.connect(X402PaymentInput(action="buy coffee"))
        mock_log.info.assert_any_call("x402 payment successful: buy coffee")


@pytest.mark.asyncio
async def test_connect_failed_payment(connector):
    mock_resp = _make_aiohttp_response(status=500, text_data="Internal Server Error")
    mock_session = _make_mock_session(mock_resp)

    requirements = {
        "payTo": "0x1234567890abcdef1234567890abcdef12345678",
        "maxAmountRequired": "10000",
        "maxTimeoutSeconds": 300,
        "network": "base-sepolia",
    }

    with (
        patch.object(
            connector, "_get_session", new_callable=AsyncMock
        ) as mock_get_session,
        patch.object(
            connector, "_fetch_payment_requirements", new_callable=AsyncMock
        ) as mock_fetch,
        patch.object(
            connector, "_build_payment_payload", return_value="encoded_payload"
        ),
        patch("actions.x402_command.connector.x402.logging") as mock_log,
    ):
        mock_get_session.return_value = mock_session
        mock_fetch.return_value = requirements
        await connector.connect(X402PaymentInput(action="buy coffee"))
        mock_log.error.assert_any_call(
            "x402 payment failed with status 500: Internal Server Error"
        )


@pytest.mark.asyncio
async def test_fetch_payment_requirements_success(connector):
    json_data = {
        "accepts": [
            {
                "payTo": "0xabc",
                "maxAmountRequired": "1000",
                "maxTimeoutSeconds": 300,
                "network": "base-sepolia",
            }
        ]
    }
    mock_resp = _make_aiohttp_response(status=402, json_data=json_data)
    mock_session = _make_mock_session(mock_resp)

    result = await connector._fetch_payment_requirements(
        mock_session, "https://example.com/x402"
    )
    assert result is not None
    assert result["payTo"] == "0xabc"


@pytest.mark.asyncio
async def test_fetch_payment_requirements_non_402(connector):
    mock_resp = _make_aiohttp_response(status=200)
    mock_session = _make_mock_session(mock_resp)

    result = await connector._fetch_payment_requirements(
        mock_session, "https://example.com/x402"
    )
    assert result is None


@pytest.mark.asyncio
async def test_fetch_payment_requirements_empty_accepts(connector):
    json_data = {"accepts": []}
    mock_resp = _make_aiohttp_response(status=402, json_data=json_data)
    mock_session = _make_mock_session(mock_resp)

    result = await connector._fetch_payment_requirements(
        mock_session, "https://example.com/x402"
    )
    assert result is None


def test_build_payment_payload(connector):
    with patch("actions.x402_command.connector.x402.Account") as mock_account_cls:
        mock_account = MagicMock()
        mock_account.address = "0xSender"
        mock_account_cls.from_key.return_value = mock_account

        mock_signed = MagicMock()
        mock_signed.signature = b"\x01" * 65
        mock_account_cls.sign_typed_data.return_value = mock_signed

        result = connector._build_payment_payload(
            pay_to="0xRecipient",
            amount="10000",
            timeout_seconds=300,
        )

        decoded = json.loads(base64.b64decode(result))
        assert decoded["x402Version"] == 1
        assert decoded["scheme"] == "exact"
        assert decoded["network"] == "base-sepolia"
        assert "signature" in decoded["payload"]
        assert "authorization" in decoded["payload"]


def test_build_payment_payload_unsupported_network(connector):
    connector.config.network = "unsupported-network"
    with pytest.raises(ValueError, match="Unsupported network"):
        connector._build_payment_payload(
            pay_to="0xRecipient",
            amount="10000",
            timeout_seconds=300,
        )


def test_stop_no_session(connector):
    connector.stop()


def test_stop_with_closed_session(connector):
    mock_session = MagicMock()
    mock_session.closed = True
    connector._session = mock_session
    connector.stop()
