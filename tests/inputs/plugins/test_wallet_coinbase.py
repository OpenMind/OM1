# tests/inputs/plugins/test_wallet_coinbase.py

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock dependencies BEFORE importing WalletCoinbase to prevent
# network calls or loading heavy libraries (like zenoh/cdp).
if "cdp" not in sys.modules:
    sys.modules["cdp"] = MagicMock()
if "providers.io_provider" not in sys.modules:
    sys.modules["providers.io_provider"] = MagicMock()
    sys.modules["src.providers.io_provider"] = sys.modules["providers.io_provider"]

from inputs.plugins.wallet_coinbase import Message, WalletCoinbase, WalletCoinbaseConfig


class TestWalletCoinbase:
    """Unit tests for WalletCoinbase with fully isolated dependencies."""

    def test_initialization_with_missing_wallet_id(self):
        """Missing COINBASE_WALLET_ID should fall back to a safe zero state."""
        with patch.dict(os.environ, {}, clear=True):
            wallet = WalletCoinbase(config=WalletCoinbaseConfig())
            assert wallet.wallet is None
            assert wallet.balance == 0.0
            assert wallet.balance_previous == 0.0
            assert wallet.asset_id == "eth"

    def test_initialization_with_wallet_fetch_failure(self):
        """Wallet.fetch failure should be handled gracefully."""
        env = {
            "COINBASE_WALLET_ID": "test_wallet_id",
            "COINBASE_API_KEY": "k",
            "COINBASE_API_SECRET": "s",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("inputs.plugins.wallet_coinbase.Cdp.configure"):
                with patch("inputs.plugins.wallet_coinbase.Wallet.fetch") as mock_fetch:
                    mock_fetch.side_effect = Exception("Network error")

                    wallet = WalletCoinbase(config=WalletCoinbaseConfig())

                    assert wallet.wallet is None
                    assert wallet.balance == 0.0
                    assert wallet.balance_previous == 0.0

    def test_initialization_with_successful_wallet_fetch_default_asset(self):
        """Successful initialization should read balance using default asset_id 'eth'."""
        mock_wallet = MagicMock()
        mock_wallet.balance.return_value = "1.5"

        env = {
            "COINBASE_WALLET_ID": "test_wallet_id",
            "COINBASE_API_KEY": "k",
            "COINBASE_API_SECRET": "s",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "inputs.plugins.wallet_coinbase.Cdp.configure"
            ) as mock_configure:
                with patch(
                    "inputs.plugins.wallet_coinbase.Wallet.fetch",
                    return_value=mock_wallet,
                ):
                    wallet = WalletCoinbase(config=WalletCoinbaseConfig())

                    assert wallet.wallet == mock_wallet
                    assert wallet.asset_id == "eth"
                    assert wallet.balance == 1.5
                    assert wallet.balance_previous == 1.5

                    mock_configure.assert_called_once_with("k", "s")
                    mock_wallet.balance.assert_called_with("eth")

    def test_initialization_with_custom_asset_id(self):
        """Custom asset_id should be respected during initialization."""
        mock_wallet = MagicMock()
        mock_wallet.balance.return_value = "100.0"

        config = WalletCoinbaseConfig(asset_id="btc")

        env = {
            "COINBASE_WALLET_ID": "test_wallet_id",
            "COINBASE_API_KEY": "k",
            "COINBASE_API_SECRET": "s",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("inputs.plugins.wallet_coinbase.Cdp.configure"):
                with patch(
                    "inputs.plugins.wallet_coinbase.Wallet.fetch",
                    return_value=mock_wallet,
                ):
                    wallet = WalletCoinbase(config=config)

                    assert wallet.asset_id == "btc"
                    assert wallet.balance == 100.0
                    assert wallet.balance_previous == 100.0

                    mock_wallet.balance.assert_called_with("btc")

    def test_initialization_without_api_keys_does_not_call_configure(self):
        """
        If API key/secret are missing, Cdp.configure should not be called.
        Initialization should still safely proceed (with Wallet.fetch mocked).
        """
        mock_wallet = MagicMock()
        mock_wallet.balance.return_value = "3.0"

        env = {
            "COINBASE_WALLET_ID": "test_wallet_id",
            # Intentionally omit API key/secret
        }
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "inputs.plugins.wallet_coinbase.Cdp.configure"
            ) as mock_configure:
                with patch(
                    "inputs.plugins.wallet_coinbase.Wallet.fetch",
                    return_value=mock_wallet,
                ):
                    wallet = WalletCoinbase(config=WalletCoinbaseConfig())

                    assert wallet.wallet == mock_wallet
                    assert wallet.balance == 3.0
                    assert wallet.balance_previous == 3.0

                    mock_configure.assert_not_called()

    @pytest.mark.asyncio
    async def test_poll_with_wallet_refresh_failure_returns_zero_delta(self):
        """_poll should return zero delta if Wallet.fetch fails."""
        env = {
            "COINBASE_WALLET_ID": "test_wallet_id",
            "COINBASE_API_KEY": "k",
            "COINBASE_API_SECRET": "s",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("inputs.plugins.wallet_coinbase.Cdp.configure"):
                with patch("inputs.plugins.wallet_coinbase.Wallet.fetch") as mock_fetch:
                    mock_fetch.side_effect = Exception("Network error")

                    wallet = WalletCoinbase(config=WalletCoinbaseConfig())
                    # Avoid real sleep
                    with patch(
                        "inputs.plugins.wallet_coinbase.asyncio.sleep",
                        new=AsyncMock(return_value=None),
                    ):
                        result = await wallet._poll()

                    # Initialization already falls back to zeros.
                    assert result == [0.0, 0.0]

    @pytest.mark.asyncio
    async def test_poll_with_successful_wallet_refresh_calculates_delta(self):
        """_poll should update balance and compute correct delta on success."""
        mock_wallet = MagicMock()
        mock_wallet.balance.return_value = "2.0"

        env = {
            "COINBASE_WALLET_ID": "test_wallet_id",
            "COINBASE_API_KEY": "k",
            "COINBASE_API_SECRET": "s",
        }
        with patch.dict(os.environ, env, clear=True):
            with patch("inputs.plugins.wallet_coinbase.Cdp.configure"):
                with patch(
                    "inputs.plugins.wallet_coinbase.Wallet.fetch",
                    return_value=mock_wallet,
                ):
                    wallet = WalletCoinbase(config=WalletCoinbaseConfig())
                    wallet.balance_previous = 1.5

                    with patch(
                        "inputs.plugins.wallet_coinbase.asyncio.sleep",
                        new=AsyncMock(return_value=None),
                    ):
                        result = await wallet._poll()

                    assert result == [2.0, 0.5]
                    mock_wallet.balance.assert_called_with("eth")

    @pytest.mark.asyncio
    async def test_raw_to_text_positive_balance_change(self):
        """_raw_to_text should return Message for positive deltas."""
        with patch.dict(os.environ, {}, clear=True):
            wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        raw_input = [2.0, 0.5]

        with patch("inputs.plugins.wallet_coinbase.time.time", return_value=1234.0):
            result = await wallet._raw_to_text(raw_input)

        assert result is not None
        assert isinstance(result, Message)
        assert result.timestamp == 1234.0
        assert result.message == "0.50000"

    @pytest.mark.asyncio
    async def test_raw_to_text_zero_balance_change(self):
        """_raw_to_text should return None for zero deltas."""
        with patch.dict(os.environ, {}, clear=True):
            wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        raw_input = [2.0, 0.0]
        result = await wallet._raw_to_text(raw_input)

        assert result is None

    @pytest.mark.asyncio
    async def test_raw_to_text_negative_balance_change(self):
        """_raw_to_text should return None for negative deltas."""
        with patch.dict(os.environ, {}, clear=True):
            wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        raw_input = [2.0, -0.1]
        result = await wallet._raw_to_text(raw_input)

        assert result is None

    def test_formatted_latest_buffer_with_multiple_transactions(self):
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

    def test_formatted_latest_buffer_with_custom_asset_symbol(self):
        """Custom asset should appear in upper-case in formatted output."""
        config = WalletCoinbaseConfig(asset_id="btc")

        env = {
            "COINBASE_WALLET_ID": "test_wallet_id",
            "COINBASE_API_KEY": "k",
            "COINBASE_API_SECRET": "s",
        }

        mock_wallet = MagicMock()
        mock_wallet.balance.return_value = "0.0"

        with patch.dict(os.environ, env, clear=True):
            with patch("inputs.plugins.wallet_coinbase.Cdp.configure"):
                with patch(
                    "inputs.plugins.wallet_coinbase.Wallet.fetch",
                    return_value=mock_wallet,
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

    def test_formatted_latest_buffer_with_empty_buffer(self):
        """Empty buffer should return None."""
        with patch.dict(os.environ, {}, clear=True):
            wallet = WalletCoinbase(config=WalletCoinbaseConfig())

        result = wallet.formatted_latest_buffer()
        assert result is None
