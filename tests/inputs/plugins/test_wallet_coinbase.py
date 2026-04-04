import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.plugins.wallet_coinbase import Message, WalletCoinbase, WalletCoinbaseConfig


def test_initialization_with_missing_wallet_address():
    """Missing COINBASE_WALLET_ADDRESS should fall back to a safe zero state."""
    with patch.dict(os.environ, {}, clear=True):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())
        assert wallet.cdp_client is None
        assert wallet.balance == 0.0
        assert wallet.balance_previous == 0.0
        assert wallet.asset_id == "eth"


def test_initialization_with_client_init_failure():
    """CDP client initialization failure should be handled gracefully."""
    env = {
        "COINBASE_WALLET_ADDRESS": "0x1234567890abcdef",
        "CDP_API_KEY_ID": "test_key_id",
        "CDP_API_KEY_SECRET": "test_secret",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch("inputs.plugins.wallet_coinbase.CdpClient") as mock_cdp_client_class,
    ):
        mock_cdp_client_class.side_effect = Exception("Network error")

        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        assert wallet.cdp_client is None
        assert wallet.balance == 0.0
        assert wallet.balance_previous == 0.0


def test_initialization_with_successful_client_init_default_asset():
    """Successful initialization should initialize CDP client using default asset_id 'eth'."""
    mock_client = MagicMock()

    env = {
        "COINBASE_WALLET_ADDRESS": "0x1234567890abcdef",
        "CDP_API_KEY_ID": "test_key_id",
        "CDP_API_KEY_SECRET": "test_secret",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch(
            "inputs.plugins.wallet_coinbase.CdpClient",
            return_value=mock_client,
        ) as mock_cdp_client_class,
    ):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        assert wallet.cdp_client == mock_client
        assert wallet.asset_id == "eth"
        assert wallet.network == "base"
        # Balance starts at 0.0 as sync fetch is not fully implemented
        assert wallet.balance == 0.0
        assert wallet.balance_previous == 0.0

        mock_cdp_client_class.assert_called_once_with(api_key_id="test_key_id", api_key_secret="test_secret")


def test_initialization_with_custom_asset_id():
    """Custom asset_id should be respected during initialization."""
    mock_client = MagicMock()

    config = WalletCoinbaseConfig(asset_id="btc", network="ethereum")

    env = {
        "COINBASE_WALLET_ADDRESS": "0x1234567890abcdef",
        "CDP_API_KEY_ID": "test_key_id",
        "CDP_API_KEY_SECRET": "test_secret",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch(
            "inputs.plugins.wallet_coinbase.CdpClient",
            return_value=mock_client,
        ),
    ):
        wallet = WalletCoinbase(config=config)

        assert wallet.asset_id == "btc"
        assert wallet.network == "ethereum"
        assert wallet.balance == 0.0
        assert wallet.balance_previous == 0.0


def test_initialization_without_api_keys_does_not_create_client():
    """
    If API key/secret are missing, CDP client should not be created.
    Initialization should still safely proceed.
    """
    env = {
        "COINBASE_WALLET_ADDRESS": "0x1234567890abcdef",
        # Intentionally omit API keys
    }
    with patch.dict(os.environ, env, clear=True):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        assert wallet.cdp_client is None
        assert wallet.balance == 0.0
        assert wallet.balance_previous == 0.0


@pytest.mark.asyncio
async def test_poll_with_no_client_returns_zero_delta():
    """_poll should return zero delta if CDP client is not initialized."""
    env = {
        "COINBASE_WALLET_ADDRESS": "0x1234567890abcdef",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch(
            "inputs.plugins.wallet_coinbase.asyncio.sleep",
            new=AsyncMock(return_value=None),
        ),
    ):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        result = await wallet._poll()

        assert result == [0.0, 0.0]


@pytest.mark.asyncio
async def test_poll_with_client_placeholder_implementation():
    """_poll should properly fetch balance and calculate delta."""
    mock_client = MagicMock()

    # Mock the async list_token_balances method
    mock_balance_result = MagicMock()
    mock_balance_result.balances = []
    mock_client.evm.list_token_balances = AsyncMock(return_value=mock_balance_result)

    env = {
        "COINBASE_WALLET_ADDRESS": "0x1234567890abcdef",
        "CDP_API_KEY_ID": "test_key_id",
        "CDP_API_KEY_SECRET": "test_secret",
    }
    with (
        patch.dict(os.environ, env, clear=True),
        patch(
            "inputs.plugins.wallet_coinbase.CdpClient",
            return_value=mock_client,
        ),
        patch(
            "inputs.plugins.wallet_coinbase.asyncio.sleep",
            new=AsyncMock(return_value=None),
        ),
    ):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())
        wallet.balance = 1.5
        wallet.balance_previous = 1.5

        result = await wallet._poll()

        # Returns zero balance as the mock returns empty balances list
        assert result == [0.0, -1.5]
        mock_client.evm.list_token_balances.assert_called_once_with(
            address="0x1234567890abcdef",
            network="base",
        )


@pytest.mark.asyncio
async def test_raw_to_text_positive_balance_change():
    """_raw_to_text should return Message for positive deltas."""
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("inputs.plugins.wallet_coinbase.time.time", return_value=1234.0),
    ):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        raw_input = [2.0, 0.5]
        result = await wallet._raw_to_text(raw_input)

    assert result is not None
    assert isinstance(result, Message)
    assert result.timestamp == 1234.0
    assert result.message == "0.50000"


@pytest.mark.asyncio
async def test_raw_to_text_zero_balance_change():
    """_raw_to_text should return None for zero deltas."""
    with patch.dict(os.environ, {}, clear=True):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        raw_input = [2.0, 0.0]
        result = await wallet._raw_to_text(raw_input)

    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_negative_balance_change():
    """_raw_to_text should return None for negative deltas."""
    with patch.dict(os.environ, {}, clear=True):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        raw_input = [2.0, -0.1]
        result = await wallet._raw_to_text(raw_input)

    assert result is None


def test_formatted_latest_buffer_with_multiple_transactions():
    """formatted_latest_buffer should sum messages, write IO, and clear buffer."""
    with patch.dict(os.environ, {}, clear=True):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

    wallet.io_provider = MagicMock()

    wallet.messages = [
        Message(timestamp=1000.0, message="0.5"),
        Message(timestamp=1001.0, message="0.3"),
        Message(timestamp=1002.0, message="0.2"),
    ]

    result = wallet.formatted_latest_buffer()

    assert result is not None
    assert "WalletCoinbase INPUT" in result
    assert "You just received 1.00000 ETH." in result

    wallet.io_provider.add_input.assert_called_once()
    assert len(wallet.messages) == 0


def test_formatted_latest_buffer_with_custom_asset_symbol():
    """Custom asset should appear in upper-case in formatted output."""
    config = WalletCoinbaseConfig(asset_id="btc")

    env = {
        "COINBASE_WALLET_ADDRESS": "0x1234567890abcdef",
        "CDP_API_KEY_ID": "test_key_id",
        "CDP_API_KEY_SECRET": "test_secret",
    }

    mock_client = MagicMock()

    with (
        patch.dict(os.environ, env, clear=True),
        patch(
            "inputs.plugins.wallet_coinbase.CdpClient",
            return_value=mock_client,
        ),
    ):
        wallet = WalletCoinbase(config=config)

    wallet.io_provider = MagicMock()

    wallet.messages = [
        Message(timestamp=1000.0, message="10.0"),
    ]

    result = wallet.formatted_latest_buffer()

    assert result is not None
    assert "You just received 10.00000 BTC." in result

    wallet.io_provider.add_input.assert_called_once()
    assert len(wallet.messages) == 0


def test_formatted_latest_buffer_with_empty_buffer():
    """Empty buffer should return None."""
    with patch.dict(os.environ, {}, clear=True):
        wallet = WalletCoinbase(config=WalletCoinbaseConfig())

    result = wallet.formatted_latest_buffer()
    assert result is None
