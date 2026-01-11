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


class TestNotificationSystem:
    """Tests for notification/event system."""

    @pytest.fixture
    def mock_config_ha_event(self):
        """Config with ha_event notification mode."""
        return CoinbasePaymentConfig(
            api_key_id="test_key_id",
            api_key_secret="test_secret",
            wallet_secret="test_wallet",
            account_address="0x1234567890123456789012345678901234567890",
            chain="base",
            testnet=True,
            ha_url="http://192.168.0.15:8123",
            ha_token="test_token",
            notification_mode="ha_event",
            ha_event_prefix="om1",
        )

    @pytest.fixture
    def mock_config_webhook(self):
        """Config with webhook notification mode."""
        return CoinbasePaymentConfig(
            api_key_id="test_key_id",
            api_key_secret="test_secret",
            wallet_secret="test_wallet",
            account_address="0x1234567890123456789012345678901234567890",
            chain="base",
            testnet=True,
            notification_mode="webhook",
            webhook_url="http://localhost:9999/webhook",
        )

    @pytest.fixture
    def mock_config_none(self):
        """Config with no notifications."""
        return CoinbasePaymentConfig(
            api_key_id="test_key_id",
            api_key_secret="test_secret",
            wallet_secret="test_wallet",
            account_address="0x1234567890123456789012345678901234567890",
            chain="base",
            testnet=True,
            notification_mode="none",
        )

    @pytest.fixture
    def mock_config_custom_prefix(self):
        """Config with custom HA event prefix."""
        return CoinbasePaymentConfig(
            api_key_id="test_key_id",
            api_key_secret="test_secret",
            wallet_secret="test_wallet",
            account_address="0x1234567890123456789012345678901234567890",
            chain="base",
            testnet=True,
            ha_url="http://192.168.0.15:8123",
            ha_token="test_token",
            notification_mode="ha_event",
            ha_event_prefix="myrobot",
        )

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_notification_mode_ha_event(self, mock_io, mock_config_ha_event):
        """Test ha_event notification mode is configured correctly."""
        connector = CoinbasePaymentConnector(mock_config_ha_event)
        assert connector.notification_mode == "ha_event"
        assert connector.ha_url == "http://192.168.0.15:8123"
        assert connector.ha_token == "test_token"
        assert connector.ha_event_prefix == "om1"

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_notification_mode_webhook(self, mock_io, mock_config_webhook):
        """Test webhook notification mode is configured correctly."""
        connector = CoinbasePaymentConnector(mock_config_webhook)
        assert connector.notification_mode == "webhook"
        assert connector.webhook_url == "http://localhost:9999/webhook"

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_notification_mode_none(self, mock_io, mock_config_none):
        """Test none notification mode disables notifications."""
        connector = CoinbasePaymentConnector(mock_config_none)
        assert connector.notification_mode == "none"

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_custom_ha_event_prefix(self, mock_io, mock_config_custom_prefix):
        """Test custom HA event prefix is applied."""
        connector = CoinbasePaymentConnector(mock_config_custom_prefix)
        assert connector.ha_event_prefix == "myrobot"

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_default_notification_mode(self, mock_io):
        """Test default notification mode is ha_event."""
        config = CoinbasePaymentConfig(
            api_key_id="test",
            api_key_secret="test",
            wallet_secret="test",
            chain="base",
            testnet=True,
        )
        connector = CoinbasePaymentConnector(config)
        assert connector.notification_mode == "ha_event"

    @patch("actions.coinbase_payment.connector.cdp_api.CDP_AVAILABLE", True)
    @patch("actions.coinbase_payment.connector.cdp_api.IOProvider")
    def test_default_ha_event_prefix(self, mock_io):
        """Test default HA event prefix is om1."""
        config = CoinbasePaymentConfig(
            api_key_id="test",
            api_key_secret="test",
            wallet_secret="test",
            chain="base",
            testnet=True,
        )
        connector = CoinbasePaymentConnector(config)
        assert connector.ha_event_prefix == "om1"
