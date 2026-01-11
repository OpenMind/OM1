import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock providers before importing connector
sys.modules["providers"] = MagicMock()
sys.modules["providers.io_provider"] = MagicMock()

from actions.coinbase_payment.connector.cdp_api import (  # noqa: E402
    CoinbasePaymentConfig,
    CoinbasePaymentConnector,
    extract_payment_details,
    split_multi_payments,
)


class TestExtractPaymentDetails:
    """Tests for extract_payment_details function."""

    def test_extract_usdc_amount(self):
        """Test extracting USDC payment details."""
        amount, asset, address = extract_payment_details("send 5 usdc to 0x1234567890123456789012345678901234567890")
        assert amount == 5.0
        assert asset == "usdc"
        assert address == "0x1234567890123456789012345678901234567890"

    def test_extract_eth_amount(self):
        """Test extracting ETH payment details."""
        amount, asset, address = extract_payment_details("send 0.01 eth")
        assert amount == 0.01
        assert asset == "eth"
        assert address is None

    def test_extract_decimal_amount(self):
        """Test extracting decimal amounts."""
        amount, asset, _ = extract_payment_details("pay 0.001 usdc")
        assert amount == 0.001
        assert asset == "usdc"

    def test_extract_no_amount(self):
        """Test when no amount present."""
        amount, asset, address = extract_payment_details("check balance")
        assert amount is None
        assert asset is None
        assert address is None

    def test_extract_various_tokens(self):
        """Test various token names."""
        for token in ["usdt", "usdbc", "dai", "weth"]:
            amount, asset, _ = extract_payment_details(f"send 10 {token}")
            assert amount == 10.0
            assert asset == token


class TestSplitMultiPayments:
    """Tests for split_multi_payments function."""

    def test_single_payment(self):
        """Test single payment not split."""
        parts = split_multi_payments("send 5 usdc to john")
        assert len(parts) == 1
        assert "send 5 usdc" in parts[0]

    def test_two_payments_with_and(self):
        """Test two payments connected with 'and'."""
        parts = split_multi_payments("send 5 usdc to john and pay 10 usdt to coffee")
        assert len(parts) == 2
        assert "send 5 usdc" in parts[0]
        assert "pay 10 usdt" in parts[1]

    def test_payments_with_comma(self):
        """Test payments separated by comma."""
        parts = split_multi_payments("send 5 usdc, pay 3 eth")
        assert len(parts) == 2


class TestCoinbasePaymentConnector:
    """Tests for CoinbasePaymentConnector class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config with test values."""
        return CoinbasePaymentConfig(
            api_key_id="test_key_id",
            api_key_secret="test_secret",
            wallet_secret="test_wallet",
            account_address="0x1234567890123456789012345678901234567890",
            chain="base",
            testnet=True,
            destination_address="0x0000000000000000000000000000000000000001",
            contacts={
                "john": "0xaaaa000000000000000000000000000000000001",
                "coffee": "0xbbbb000000000000000000000000000000000002",
            },
            max_amounts={"usdc": 50.0, "eth": 0.01},
            payment_cooldown=30.0,
        )

    @pytest.fixture
    def mock_config_blocked_addresses(self):
        """Create mock config with blocked addresses (no whitelist)."""
        return CoinbasePaymentConfig(
            api_key_id="test_key_id",
            api_key_secret="test_secret",
            wallet_secret="test_wallet",
            account_address="0x1234567890123456789012345678901234567890",
            chain="base",
            testnet=True,
            destination_address="0x0000000000000000000000000000000000000001",
            contacts={
                "john": "0xaaaa000000000000000000000000000000000001",
            },
            blocked_addresses=["0xbad0000000000000000000000000000000000bad"],  # 40 hex chars
            whitelist_only=False,
        )

    @pytest.fixture
    def mock_config_whitelist(self):
        """Create mock config with whitelist mode enabled."""
        return CoinbasePaymentConfig(
            api_key_id="test_key_id",
            api_key_secret="test_secret",
            wallet_secret="test_wallet",
            account_address="0x1234567890123456789012345678901234567890",
            chain="base",
            testnet=True,
            destination_address="0x0000000000000000000000000000000000000001",
            contacts={
                "john": "0xaaaa000000000000000000000000000000000001",
            },
            whitelist_only=True,
        )

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_resolve_contact_known(self, mock_io, mock_config):
        """Test resolving known contact."""
        connector = CoinbasePaymentConnector(mock_config)
        address, has_unknown = connector._resolve_contact("send 5 usdc to john")
        assert address == "0xaaaa000000000000000000000000000000000001"
        assert has_unknown is False

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_resolve_contact_unknown(self, mock_io, mock_config):
        """Test detecting unknown contact."""
        connector = CoinbasePaymentConnector(mock_config)
        address, has_unknown = connector._resolve_contact("send 5 usdc to faruk")
        assert address is None
        assert has_unknown is True

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_resolve_contact_no_name(self, mock_io, mock_config):
        """Test when no contact name mentioned."""
        connector = CoinbasePaymentConnector(mock_config)
        address, has_unknown = connector._resolve_contact("send 5 usdc")
        assert address is None
        assert has_unknown is False

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_parse_action_balance(self, mock_io, mock_config):
        """Test parsing balance query."""
        connector = CoinbasePaymentConnector(mock_config)
        action_type, amount, asset, address, error = connector._parse_action("check balance")
        assert action_type == "balance"

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_parse_action_send_to_contact(self, mock_io, mock_config):
        """Test parsing send to contact."""
        connector = CoinbasePaymentConnector(mock_config)
        action_type, amount, asset, address, error = connector._parse_action("send 5 usdc to john")
        assert action_type == "send"
        assert amount == 5.0
        assert asset == "usdc"
        assert address == "0xaaaa000000000000000000000000000000000001"
        assert error is None

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_parse_action_no_amount(self, mock_io, mock_config):
        """Test that payment without amount returns helpful error."""
        connector = CoinbasePaymentConnector(mock_config)
        # "pay internet bill" - no amount specified
        action_type, _, _, _, error = connector._parse_action("pay internet bill")
        assert action_type == "error"
        assert "specify amount" in error.lower()

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_parse_action_zero_amount(self, mock_io, mock_config):
        """Test that zero amounts don't trigger payment."""
        connector = CoinbasePaymentConnector(mock_config)
        # Zero amount (0.0 is falsy) doesn't trigger send action
        action_type, amount, _, _, _ = connector._parse_action("send 0 usdc")
        # Amount 0 is treated as no valid amount found - now returns error
        assert action_type == "error" or amount is None

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_parse_action_invalid_token(self, mock_io, mock_config):
        """Test that invalid token returns error with available tokens."""
        connector = CoinbasePaymentConnector(mock_config)
        action_type, _, _, _, error = connector._parse_action("send 5 xyz to john")
        assert action_type == "error"
        assert "not available" in error.lower()
        assert "USDC" in error  # Should show available tokens

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_parse_action_exceeds_max(self, mock_io, mock_config):
        """Test that amounts exceeding max are blocked."""
        connector = CoinbasePaymentConnector(mock_config)
        action_type, _, _, _, error = connector._parse_action("send 100 usdc to john")
        assert action_type == "error"
        assert "exceeds limit" in error

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_parse_action_unknown_contact_blocked(self, mock_io, mock_config):
        """Test that unknown contacts are blocked."""
        connector = CoinbasePaymentConnector(mock_config)
        action_type, _, _, _, error = connector._parse_action("send 5 usdc to faruk")
        assert action_type == "error"
        assert "Unknown contact" in error

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_parse_action_blocked_address(self, mock_io, mock_config_blocked_addresses):
        """Test that blocked addresses are rejected."""
        connector = CoinbasePaymentConnector(mock_config_blocked_addresses)
        action_type, _, _, _, error = connector._parse_action(
            "send 5 usdc to 0xbad0000000000000000000000000000000000bad"
        )
        assert action_type == "error"
        assert "blocked" in error.lower()

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_parse_action_whitelist_mode(self, mock_io, mock_config_whitelist):
        """Test whitelist mode blocks non-contacts."""
        connector = CoinbasePaymentConnector(mock_config_whitelist)
        action_type, _, _, _, error = connector._parse_action(
            "send 5 usdc to 0x9999000000000000000000000000000000000009"
        )
        assert action_type == "error"
        assert "Whitelist mode" in error

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_available_tokens_display(self, mock_io, mock_config):
        """Test available tokens are listed correctly."""
        connector = CoinbasePaymentConnector(mock_config)
        assert "ETH" in connector.available_tokens_display
        assert "USDC" in connector.available_tokens_display

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_network_configuration(self, mock_io, mock_config):
        """Test network configuration is loaded correctly."""
        connector = CoinbasePaymentConnector(mock_config)
        assert connector.cdp_network == "base-sepolia"
        assert connector.chain == "base"
        assert connector.is_testnet is True

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_invalid_chain_raises(self, mock_io):
        """Test invalid chain raises ValueError."""
        config = CoinbasePaymentConfig(
            api_key_id="test",
            api_key_secret="test",
            wallet_secret="test",
            chain="invalid_chain",
        )
        with pytest.raises(ValueError) as exc_info:
            CoinbasePaymentConnector(config)
        assert "Unsupported chain" in str(exc_info.value)
