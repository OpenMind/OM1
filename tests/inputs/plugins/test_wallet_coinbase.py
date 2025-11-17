"""
Test cases for WalletCoinbase input plugin.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from inputs.plugins.wallet_coinbase import Message, WalletCoinbase


class TestWalletCoinbase:
    """Test cases for WalletCoinbase class."""

    def test_initialization_with_missing_wallet_id(self):
        """Test that initialization handles missing COINBASE_WALLET_ID gracefully."""
        with patch.dict(os.environ, {}, clear=True):
            wallet = WalletCoinbase()
            assert wallet.wallet is None
            assert wallet.ETH_balance == 0.0
            assert wallet.ETH_balance_previous == 0.0

    def test_initialization_with_wallet_fetch_failure(self):
        """Test that initialization handles wallet fetch failure gracefully."""
        with patch.dict(os.environ, {"COINBASE_WALLET_ID": "test_wallet_id"}):
            with patch("inputs.plugins.wallet_coinbase.Wallet.fetch") as mock_fetch:
                mock_fetch.side_effect = Exception("Network error")

                wallet = WalletCoinbase()
                assert wallet.wallet is None
                assert wallet.ETH_balance == 0.0
                assert wallet.ETH_balance_previous == 0.0

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
                    assert wallet.ETH_balance == 1.5
                    assert wallet.ETH_balance_previous == 1.5

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
                    wallet.ETH_balance_previous = 1.5  # Set previous balance

                    result = await wallet._poll()

                    # Should return current balance and balance change
                    assert result == [2.0, 0.5]

    def test_raw_to_text_with_positive_balance_change(self):
        """Test that _raw_to_text correctly handles positive balance changes."""
        wallet = WalletCoinbase()

        # Test with positive balance change
        raw_input = [2.0, 0.5]  # [current_balance, balance_change]

        # This is an async method, so we need to run it
        import asyncio

        result = asyncio.run(wallet._raw_to_text(raw_input))

        assert result is not None
        assert isinstance(result, Message)
        assert result.message == "0.50000"

    def test_raw_to_text_with_zero_balance_change(self):
        """Test that _raw_to_text returns None for zero balance change."""
        wallet = WalletCoinbase()

        # Test with zero balance change
        raw_input = [2.0, 0.0]  # [current_balance, balance_change]

        # This is an async method, so we need to run it
        import asyncio

        result = asyncio.run(wallet._raw_to_text(raw_input))

        assert result is None

    def test_formatted_latest_buffer_with_multiple_transactions(self):
        """Test that formatted_latest_buffer correctly combines multiple transactions."""
        wallet = WalletCoinbase()

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

    def test_formatted_latest_buffer_with_empty_buffer(self):
        """Test that formatted_latest_buffer returns None for empty buffer."""
        wallet = WalletCoinbase()

        result = wallet.formatted_latest_buffer()

        assert result is None
