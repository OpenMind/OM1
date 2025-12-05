"""
Test cases for WalletCoinbase input plugin.
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

# Mock cdp module before importing the source file that depends on it
mock_cdp = MagicMock()
sys.modules["cdp"] = mock_cdp

# Mock providers.io_provider to avoid importing zenoh and other deep dependencies
mock_io_provider_module = MagicMock()
sys.modules["providers.io_provider"] = mock_io_provider_module
# Also mock src.providers.io_provider in case it's imported that way elsewhere
sys.modules["src.providers.io_provider"] = mock_io_provider_module

import pytest

from inputs.base import SensorConfig
from inputs.plugins.wallet_coinbase import Message, WalletCoinbase


class TestWalletCoinbase:
    """Test cases for WalletCoinbase class."""

    def test_initialization_with_missing_wallet_id(self):
        """Test that initialization handles missing COINBASE_WALLET_ID gracefully."""
        with patch.dict(os.environ, {}, clear=True):
            wallet = WalletCoinbase()
            assert wallet.wallet is None
            assert wallet.balance == 0.0
            assert wallet.balance_previous == 0.0

    def test_initialization_with_wallet_fetch_failure(self):
        """Test that initialization handles wallet fetch failure gracefully."""
        with patch.dict(os.environ, {"COINBASE_WALLET_ID": "test_wallet_id"}):
            with patch("inputs.plugins.wallet_coinbase.Wallet.fetch") as mock_fetch:
                mock_fetch.side_effect = Exception("Network error")

                wallet = WalletCoinbase()
                assert wallet.wallet is None
                assert wallet.balance == 0.0
                assert wallet.balance_previous == 0.0

    def test_initialization_with_successful_wallet_fetch(self):
        """Test that initialization works correctly when wallet fetch succeeds."""
        mock_wallet = MagicMock()
        mock_wallet.balance.return_value = "1.5"

        with patch.dict(os.environ, {"COINBASE_WALLET_ID": "test_wallet_id"}):
            with patch(
                "inputs.plugins.wallet_coinbase.Wallet.fetch", return_value=mock_wallet
            ):
                with patch("inputs.plugins.wallet_coinbase.Cdp.configure"):
                    wallet = WalletCoinbase()
                    assert wallet.wallet == mock_wallet
                    assert wallet.balance == 1.5
                    assert wallet.balance_previous == 1.5
                    # Default asset should be 'eth'
                    mock_wallet.balance.assert_called_with("eth")

    def test_initialization_with_custom_asset_id(self):
        """Test that initialization works correctly with a custom asset_id."""
        mock_wallet = MagicMock()
        mock_wallet.balance.return_value = "100.0"

        # Create a config with a custom asset_id
        config = SensorConfig()
        config.asset_id = "eth"

        with patch.dict(os.environ, {"COINBASE_WALLET_ID": "test_wallet_id"}):
            with patch(
                "inputs.plugins.wallet_coinbase.Wallet.fetch", return_value=mock_wallet
            ):
                with patch("inputs.plugins.wallet_coinbase.Cdp.configure"):
                    wallet = WalletCoinbase(config=config)
                    assert wallet.asset_id == "eth"
                    assert wallet.balance == 100.0
                    # Should call balance with the custom asset id
                    mock_wallet.balance.assert_called_with("eth")

    @pytest.mark.asyncio
    async def test_poll_with_wallet_fetch_failure(self):
        """Test that _poll handles wallet fetch failure gracefully."""
        with patch.dict(os.environ, {"COINBASE_WALLET_ID": "test_wallet_id"}):
            with patch("inputs.plugins.wallet_coinbase.Wallet.fetch") as mock_fetch:
                mock_fetch.side_effect = Exception("Network error")

                wallet = WalletCoinbase()
                result = await wallet._poll()

                # Should return zero balance change when wallet refresh fails
                assert result == [0.0, 0.0]

    @pytest.mark.asyncio
    async def test_poll_with_successful_wallet_refresh(self):
        """Test that _poll works correctly when wallet refresh succeeds."""
        mock_wallet = MagicMock()
        mock_wallet.balance.return_value = "2.0"

        with patch.dict(os.environ, {"COINBASE_WALLET_ID": "test_wallet_id"}):
            with patch(
                "inputs.plugins.wallet_coinbase.Wallet.fetch", return_value=mock_wallet
            ):
                with patch("inputs.plugins.wallet_coinbase.Cdp.configure"):
                    wallet = WalletCoinbase()
                    wallet.balance_previous = 1.5  # Set previous balance

                    result = await wallet._poll()

                    # Should return current balance and balance change
                    assert result == [2.0, 0.5]

    def test_raw_to_text_with_positive_balance_change(self):
        """Test that _raw_to_text correctly handles positive balance changes."""
        wallet = WalletCoinbase()

        # Test with positive balance change
        raw_input = [2.0, 0.5]  # [current_balance, balance_change]

        result = asyncio.run(wallet._raw_to_text(raw_input))

        assert result is not None
        assert isinstance(result, Message)
        assert result.message == "0.50000"

    def test_raw_to_text_with_zero_balance_change(self):
        """Test that _raw_to_text returns None for zero balance change."""
        wallet = WalletCoinbase()

        # Test with zero balance change
        raw_input = [2.0, 0.0]  # [current_balance, balance_change]

        result = asyncio.run(wallet._raw_to_text(raw_input))

        assert result is None

    def test_formatted_latest_buffer_with_multiple_transactions(self):
        """Test that formatted_latest_buffer correctly combines multiple transactions."""
        wallet = WalletCoinbase()
        # Mock IO provider to avoid side effects
        wallet.io_provider = MagicMock()

        # Add multiple messages to the buffer
        wallet.messages = [
            Message(timestamp=1000.0, message="0.5"),
            Message(timestamp=1001.0, message="0.3"),
            Message(timestamp=1002.0, message="0.2"),
        ]

        result = wallet.formatted_latest_buffer()

        assert result is not None
        assert "You just received 1.00000 ETH." in result
        assert "WalletCoinbase INPUT" in result
        assert len(wallet.messages) == 0  # Buffer should be cleared

    def test_formatted_latest_buffer_with_custom_asset(self):
        """Test that formatted_latest_buffer correctly displays custom asset symbol."""
        config = SensorConfig()
        config.asset_id = "eth"
        wallet = WalletCoinbase(config=config)
        wallet.io_provider = MagicMock()

        wallet.messages = [
            Message(timestamp=1000.0, message="10.0"),
        ]

        result = wallet.formatted_latest_buffer()

        assert result is not None
        assert "You just received 10.00000 ETH." in result

    def test_formatted_latest_buffer_with_empty_buffer(self):
        """Test that formatted_latest_buffer returns None for empty buffer."""
        wallet = WalletCoinbase()

        result = wallet.formatted_latest_buffer()

        assert result is None