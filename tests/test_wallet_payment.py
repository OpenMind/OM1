"""
Tests for Wallet Payment Action Plugin.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestWalletPaymentInterface:
    """Tests for WalletPayment interface classes."""

    def test_wallet_payment_input(self):
        """Test WalletPaymentInput dataclass."""
        from actions.wallet_payment.interface import WalletPaymentInput

        input_data = WalletPaymentInput(action="pay 0.001 ETH to 0x1234")
        assert input_data.action == "pay 0.001 ETH to 0x1234"

    def test_wallet_payment_interface(self):
        """Test WalletPayment interface class."""
        from actions.wallet_payment.interface import WalletPayment, WalletPaymentInput

        input_data = WalletPaymentInput(action="pay 0.001 ETH to 0x1234")
        output_data = WalletPaymentInput(action="pay 0.001 ETH to 0x1234")

        interface = WalletPayment(input=input_data, output=output_data)
        assert interface.input.action == "pay 0.001 ETH to 0x1234"


class TestWalletPaymentConfig:
    """Tests for WalletPaymentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from actions.wallet_payment.connector.coinbase import WalletPaymentConfig

        config = WalletPaymentConfig()
        assert config.asset_id == "eth"
        assert config.network == "base-sepolia"

    def test_custom_config(self):
        """Test custom configuration values."""
        from actions.wallet_payment.connector.coinbase import WalletPaymentConfig

        config = WalletPaymentConfig(asset_id="usdc", network="mainnet")
        assert config.asset_id == "usdc"
        assert config.network == "mainnet"


class TestCoinbaseWalletConnector:
    """Tests for CoinbaseWalletConnector."""

    def test_parse_payment_command_standard_format(self):
        """Test parsing standard payment command format."""
        from actions.wallet_payment.connector.coinbase import (
            CoinbaseWalletConnector,
            WalletPaymentConfig,
        )

        with patch.dict(
            "os.environ",
            {
                "COINBASE_API_KEY": "test",
                "COINBASE_API_SECRET": "test",
                "COINBASE_WALLET_ID": "test",
            },
        ):
            with patch("actions.wallet_payment.connector.coinbase.Cdp"):
                with patch("actions.wallet_payment.connector.coinbase.Wallet"):
                    connector = CoinbaseWalletConnector(WalletPaymentConfig())

                    amount, asset, recipient = connector._parse_payment_command(
                        "pay 0.001 ETH to 0x1234"
                    )
                    assert amount == "0.001"
                    assert asset == "eth"
                    assert recipient == "0x1234"

    def test_parse_payment_command_send_format(self):
        """Test parsing send command format."""
        from actions.wallet_payment.connector.coinbase import (
            CoinbaseWalletConnector,
            WalletPaymentConfig,
        )

        with patch.dict(
            "os.environ",
            {
                "COINBASE_API_KEY": "test",
                "COINBASE_API_SECRET": "test",
                "COINBASE_WALLET_ID": "test",
            },
        ):
            with patch("actions.wallet_payment.connector.coinbase.Cdp"):
                with patch("actions.wallet_payment.connector.coinbase.Wallet"):
                    connector = CoinbaseWalletConnector(WalletPaymentConfig())

                    amount, asset, recipient = connector._parse_payment_command(
                        "send 0.5 usdc 0xabcd"
                    )
                    assert amount == "0.5"
                    assert asset == "usdc"
                    assert recipient == "0xabcd"

    def test_parse_payment_command_without_to(self):
        """Test parsing command without 'to' keyword."""
        from actions.wallet_payment.connector.coinbase import (
            CoinbaseWalletConnector,
            WalletPaymentConfig,
        )

        with patch.dict(
            "os.environ",
            {
                "COINBASE_API_KEY": "test",
                "COINBASE_API_SECRET": "test",
                "COINBASE_WALLET_ID": "test",
            },
        ):
            with patch("actions.wallet_payment.connector.coinbase.Cdp"):
                with patch("actions.wallet_payment.connector.coinbase.Wallet"):
                    connector = CoinbaseWalletConnector(WalletPaymentConfig())

                    amount, asset, recipient = connector._parse_payment_command(
                        "pay 1.0 eth 0x9999"
                    )
                    assert amount == "1.0"
                    assert asset == "eth"
                    assert recipient == "0x9999"

    def test_parse_payment_command_invalid(self):
        """Test parsing invalid command."""
        from actions.wallet_payment.connector.coinbase import (
            CoinbaseWalletConnector,
            WalletPaymentConfig,
        )

        with patch.dict(
            "os.environ",
            {
                "COINBASE_API_KEY": "test",
                "COINBASE_API_SECRET": "test",
                "COINBASE_WALLET_ID": "test",
            },
        ):
            with patch("actions.wallet_payment.connector.coinbase.Cdp"):
                with patch("actions.wallet_payment.connector.coinbase.Wallet"):
                    connector = CoinbaseWalletConnector(WalletPaymentConfig())

                    amount, asset, recipient = connector._parse_payment_command(
                        "invalid command"
                    )
                    assert amount is None
                    assert asset is None
                    assert recipient is None

    @pytest.mark.asyncio
    async def test_connect_invalid_command(self):
        """Test connect with invalid payment command."""
        from actions.wallet_payment.connector.coinbase import (
            CoinbaseWalletConnector,
            WalletPaymentConfig,
        )
        from actions.wallet_payment.interface import WalletPaymentInput

        with patch.dict(
            "os.environ",
            {
                "COINBASE_API_KEY": "test",
                "COINBASE_API_SECRET": "test",
                "COINBASE_WALLET_ID": "test",
            },
        ):
            with patch("actions.wallet_payment.connector.coinbase.Cdp"):
                with patch("actions.wallet_payment.connector.coinbase.Wallet"):
                    connector = CoinbaseWalletConnector(WalletPaymentConfig())

                    input_data = WalletPaymentInput(action="invalid command")
                    # Should not raise, just logs error
                    await connector.connect(input_data)

    @pytest.mark.asyncio
    async def test_connect_no_wallet(self):
        """Test connect when wallet is not initialized."""
        from actions.wallet_payment.connector.coinbase import (
            CoinbaseWalletConnector,
            WalletPaymentConfig,
        )
        from actions.wallet_payment.interface import WalletPaymentInput

        with patch.dict(
            "os.environ",
            {
                "COINBASE_API_KEY": "test",
                "COINBASE_API_SECRET": "test",
                "COINBASE_WALLET_ID": "test",
            },
        ):
            with patch("actions.wallet_payment.connector.coinbase.Cdp"):
                with patch(
                    "actions.wallet_payment.connector.coinbase.Wallet"
                ) as mock_wallet:
                    mock_wallet.fetch.side_effect = Exception("Wallet not found")
                    connector = CoinbaseWalletConnector(WalletPaymentConfig())
                    connector.wallet = None

                    input_data = WalletPaymentInput(action="pay 0.001 ETH to 0x1234")
                    # Should not raise, just logs error
                    await connector.connect(input_data)

    @pytest.mark.asyncio
    async def test_connect_successful_payment(self):
        """Test successful payment execution."""
        from actions.wallet_payment.connector.coinbase import (
            CoinbaseWalletConnector,
            WalletPaymentConfig,
        )
        from actions.wallet_payment.interface import WalletPaymentInput

        with patch.dict(
            "os.environ",
            {
                "COINBASE_API_KEY": "test",
                "COINBASE_API_SECRET": "test",
                "COINBASE_WALLET_ID": "test",
                "COINBASE_WALLET_SEED": "test_seed",
            },
        ):
            with patch("actions.wallet_payment.connector.coinbase.Cdp"):
                mock_transfer = MagicMock()
                mock_transfer.transaction_hash = "0xabc123def456"
                mock_transfer.wait.return_value = None

                mock_wallet = MagicMock()
                mock_wallet.balance.return_value = 1.0  # Sufficient balance
                mock_wallet.transfer.return_value = mock_transfer

                with patch(
                    "actions.wallet_payment.connector.coinbase.Wallet"
                ) as wallet_class:
                    wallet_class.import_data.return_value = mock_wallet

                    with patch(
                        "actions.wallet_payment.connector.coinbase.WalletData"
                    ) as wallet_data_class:
                        mock_wallet_data = MagicMock()
                        wallet_data_class.from_dict.return_value = mock_wallet_data

                        connector = CoinbaseWalletConnector(WalletPaymentConfig())
                        connector.wallet = mock_wallet
                        connector.wallet_seed = "test_seed"

                        input_data = WalletPaymentInput(
                            action="pay 0.001 ETH to 0x1234567890abcdef"
                        )
                        # Should complete without error
                        await connector.connect(input_data)

                        # Verify transfer was called
                        mock_wallet.transfer.assert_called_once()


class TestWalletPaymentLoading:
    """Tests for loading wallet payment action via action loader."""

    def test_action_interface_docstring(self):
        """Test that WalletPayment has proper docstring."""
        from actions.wallet_payment.interface import WalletPayment

        assert WalletPayment.__doc__ is not None
        assert "payment" in WalletPayment.__doc__.lower()

    def test_action_input_has_action_field(self):
        """Test that WalletPaymentInput has required action field."""
        import typing as T

        from actions.wallet_payment.interface import WalletPaymentInput

        hints = T.get_type_hints(WalletPaymentInput)
        assert "action" in hints


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
