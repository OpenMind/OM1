"""
Tests for Order Action Plugin with Integrated Payment.
Supports ETH and USDC payments.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestOrderInterface:
    """Tests for Order interface classes."""

    def test_order_input(self):
        """Test OrderInput dataclass."""
        from actions.order.interface import OrderInput

        input_data = OrderInput(action="order 2 pizzas")
        assert input_data.action == "order 2 pizzas"

    def test_order_output_eth(self):
        """Test OrderOutput dataclass with ETH payment."""
        from actions.order.interface import OrderOutput

        output_data = OrderOutput(
            order_id="ORD-12345678",
            item="pizza",
            quantity=2,
            price_usd=30.00,
            price_crypto=0.01,
            payment_asset="eth",
            status="paid",
            message="Order confirmed and paid",
            tx_hash="0xabc123",
        )
        assert output_data.order_id == "ORD-12345678"
        assert output_data.item == "pizza"
        assert output_data.quantity == 2
        assert output_data.price_usd == 30.00
        assert output_data.price_crypto == 0.01
        assert output_data.payment_asset == "eth"
        assert output_data.status == "paid"
        assert output_data.tx_hash == "0xabc123"

    def test_order_output_usdc(self):
        """Test OrderOutput dataclass with USDC payment."""
        from actions.order.interface import OrderOutput

        output_data = OrderOutput(
            order_id="ORD-12345678",
            item="coffee",
            quantity=1,
            price_usd=5.00,
            price_crypto=5.00,
            payment_asset="usdc",
            status="paid",
            message="Order confirmed and paid",
            tx_hash="0xdef456",
        )
        assert output_data.price_crypto == 5.00
        assert output_data.payment_asset == "usdc"

    def test_order_output_without_tx_hash(self):
        """Test OrderOutput without transaction hash."""
        from actions.order.interface import OrderOutput

        output_data = OrderOutput(
            order_id="ORD-12345678",
            item="pizza",
            quantity=1,
            price_usd=15.00,
            price_crypto=0.005,
            payment_asset="eth",
            status="confirmed",
            message="Order confirmed",
        )
        assert output_data.tx_hash is None

    def test_order_interface(self):
        """Test Order interface class."""
        from actions.order.interface import Order, OrderInput, OrderOutput

        input_data = OrderInput(action="order pizza")
        output_data = OrderOutput(
            order_id="ORD-12345678",
            item="pizza",
            quantity=1,
            price_usd=15.00,
            price_crypto=0.005,
            payment_asset="eth",
            status="confirmed",
            message="Order confirmed",
        )

        interface = Order(input=input_data, output=output_data)
        assert interface.input.action == "order pizza"
        assert interface.output is not None
        assert interface.output.item == "pizza"


class TestOrderConfig:
    """Tests for OrderConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from actions.order.connector.mock import OrderConfig

        config = OrderConfig()
        assert config.currency == "USD"
        assert config.auto_confirm is True
        assert config.payment_enabled is True
        assert config.payment_asset == "eth"
        assert config.merchant_address == ""
        assert config.use_dynamic_price is True
        assert config.fallback_price_usd == 3000.0

    def test_custom_config_eth(self):
        """Test custom configuration with ETH."""
        from actions.order.connector.mock import OrderConfig

        config = OrderConfig(
            currency="EUR",
            auto_confirm=False,
            payment_enabled=True,
            payment_asset="eth",
            merchant_address="0x1234",
            use_dynamic_price=False,
            fallback_price_usd=2500.0,
        )
        assert config.currency == "EUR"
        assert config.auto_confirm is False
        assert config.payment_asset == "eth"
        assert config.use_dynamic_price is False
        assert config.fallback_price_usd == 2500.0

    def test_custom_config_usdc(self):
        """Test custom configuration with USDC."""
        from actions.order.connector.mock import OrderConfig

        config = OrderConfig(
            payment_enabled=True,
            payment_asset="usdc",
            merchant_address="0x5678",
        )
        assert config.payment_asset == "usdc"


class TestMockOrderConnector:
    """Tests for MockOrderConnector."""

    def test_parse_order_command_with_quantity(self):
        """Test parsing order command with quantity."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig

        config = OrderConfig(payment_enabled=False)
        connector = MockOrderConnector(config)

        item, quantity = connector._parse_order_command("order 2 pizzas")
        assert item == "pizza"
        assert quantity == 2

    def test_parse_order_command_with_a(self):
        """Test parsing order command with 'a' article."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig

        config = OrderConfig(payment_enabled=False)
        connector = MockOrderConnector(config)

        item, quantity = connector._parse_order_command("order a pizza")
        assert item == "pizza"
        assert quantity == 1

    def test_parse_order_command_simple(self):
        """Test parsing simple order command."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig

        config = OrderConfig(payment_enabled=False)
        connector = MockOrderConnector(config)

        item, quantity = connector._parse_order_command("order burger")
        assert item == "burger"
        assert quantity == 1

    def test_parse_order_command_invalid(self):
        """Test parsing invalid command."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig

        config = OrderConfig(payment_enabled=False)
        connector = MockOrderConnector(config)

        item, quantity = connector._parse_order_command("invalid command")
        assert item is None
        assert quantity is None

    def test_get_price_known_item(self):
        """Test getting price for known item."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig

        config = OrderConfig(payment_enabled=False)
        connector = MockOrderConnector(config)

        price = connector._get_price("pizza")
        assert price == 15.00

    def test_get_price_unknown_item(self):
        """Test getting price for unknown item."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig

        config = OrderConfig(payment_enabled=False)
        connector = MockOrderConnector(config)

        price = connector._get_price("unknown_item")
        assert price == 10.00

    def test_get_fallback_price_eth(self):
        """Test fallback price for ETH."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig

        config = OrderConfig(payment_enabled=False, fallback_price_usd=3000.0)
        connector = MockOrderConnector(config)

        price = connector._get_fallback_price("eth")
        assert price == 3000.0

    def test_get_fallback_price_usdc(self):
        """Test fallback price for USDC (always 1.0)."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig

        config = OrderConfig(payment_enabled=False, fallback_price_usd=3000.0)
        connector = MockOrderConnector(config)

        price = connector._get_fallback_price("usdc")
        assert price == 1.0

    @pytest.mark.asyncio
    async def test_usd_to_crypto_eth(self):
        """Test USD to ETH conversion with fallback price."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig

        config = OrderConfig(
            payment_enabled=False,
            payment_asset="eth",
            use_dynamic_price=False,
            fallback_price_usd=3000.0,
        )
        connector = MockOrderConnector(config)

        eth = await connector._usd_to_crypto(30.0, "eth")
        assert eth == 0.01  # $30 / $3000 = 0.01 ETH

    @pytest.mark.asyncio
    async def test_usd_to_crypto_usdc(self):
        """Test USD to USDC conversion (1:1)."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig

        config = OrderConfig(
            payment_enabled=False,
            payment_asset="usdc",
            use_dynamic_price=False,
        )
        connector = MockOrderConnector(config)

        usdc = await connector._usd_to_crypto(30.0, "usdc")
        assert usdc == 30.0  # $30 / $1 = 30 USDC

    @pytest.mark.asyncio
    async def test_get_crypto_price_fallback(self):
        """Test crypto price fallback when dynamic pricing is disabled."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig

        config = OrderConfig(
            payment_enabled=False, use_dynamic_price=False, fallback_price_usd=2500.0
        )
        connector = MockOrderConnector(config)

        price = await connector._get_crypto_price_usd("eth")
        assert price == 2500.0

    @pytest.mark.asyncio
    async def test_connect_successful_order_eth(self):
        """Test successful order with ETH payment asset."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig
        from actions.order.interface import OrderInput

        config = OrderConfig(
            payment_enabled=False,
            payment_asset="eth",
            use_dynamic_price=False,
            fallback_price_usd=3000.0,
        )
        connector = MockOrderConnector(config)
        input_data = OrderInput(action="order 2 pizzas")

        result = await connector.connect(input_data)

        assert result is not None
        assert result.order_id.startswith("ORD-")
        assert result.item == "pizza"
        assert result.quantity == 2
        assert result.price_usd == 30.00
        assert result.price_crypto == 0.01  # $30 / $3000
        assert result.payment_asset == "eth"
        assert result.status == "confirmed"
        assert result.tx_hash is None

    @pytest.mark.asyncio
    async def test_connect_successful_order_usdc(self):
        """Test successful order with USDC payment asset."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig
        from actions.order.interface import OrderInput

        config = OrderConfig(
            payment_enabled=False,
            payment_asset="usdc",
            use_dynamic_price=False,
        )
        connector = MockOrderConnector(config)
        input_data = OrderInput(action="order coffee")

        result = await connector.connect(input_data)

        assert result is not None
        assert result.item == "coffee"
        assert result.price_usd == 5.00
        assert result.price_crypto == 5.00  # $5 / $1 = 5 USDC
        assert result.payment_asset == "usdc"

    @pytest.mark.asyncio
    async def test_connect_single_item_order(self):
        """Test order with single item."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig
        from actions.order.interface import OrderInput

        config = OrderConfig(
            payment_enabled=False, use_dynamic_price=False, fallback_price_usd=3000.0
        )
        connector = MockOrderConnector(config)
        input_data = OrderInput(action="order burger")

        result = await connector.connect(input_data)

        assert result is not None
        assert result.item == "burger"
        assert result.quantity == 1
        assert result.price_usd == 10.00
        assert result.status == "confirmed"

    @pytest.mark.asyncio
    async def test_connect_invalid_command(self):
        """Test order with invalid command."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig
        from actions.order.interface import OrderInput

        config = OrderConfig(payment_enabled=False, use_dynamic_price=False)
        connector = MockOrderConnector(config)
        input_data = OrderInput(action="invalid command")

        result = await connector.connect(input_data)

        assert result is not None
        assert result.status == "failed"
        assert "Could not parse" in result.message

    @pytest.mark.asyncio
    async def test_connect_pending_status(self):
        """Test order with auto_confirm disabled."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig
        from actions.order.interface import OrderInput

        config = OrderConfig(
            auto_confirm=False, payment_enabled=False, use_dynamic_price=False
        )
        connector = MockOrderConnector(config)
        input_data = OrderInput(action="order pizza")

        result = await connector.connect(input_data)

        assert result is not None
        assert result.status == "pending"

    @pytest.mark.asyncio
    async def test_connect_with_payment_success(self):
        """Test order with successful payment."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig
        from actions.order.interface import OrderInput

        with patch.dict(
            "os.environ",
            {
                "COINBASE_API_KEY": "test",
                "COINBASE_API_SECRET": "test",
                "COINBASE_WALLET_ID": "test",
                "COINBASE_WALLET_SEED": "test_seed",
            },
        ):
            with patch("cdp.Cdp"):
                mock_transfer = MagicMock()
                mock_transfer.transaction_hash = "0xabc123def456"
                mock_transfer.wait.return_value = None

                mock_wallet = MagicMock()
                mock_wallet.balance.return_value = 1.0
                mock_wallet.transfer.return_value = mock_transfer

                with patch("cdp.Wallet") as wallet_class:
                    with patch("cdp.WalletData") as wallet_data_class:
                        mock_wallet_data = MagicMock()
                        wallet_data_class.from_dict.return_value = mock_wallet_data
                        wallet_class.import_data.return_value = mock_wallet

                        config = OrderConfig(
                            payment_enabled=True,
                            payment_asset="eth",
                            merchant_address="0x1234567890abcdef",
                            use_dynamic_price=False,
                            fallback_price_usd=3000.0,
                        )
                        connector = MockOrderConnector(config)
                        connector.wallet = mock_wallet
                        connector.wallet_seed = "test_seed"
                        connector.wallet_id = "test"

                        input_data = OrderInput(action="order pizza")
                        result = await connector.connect(input_data)

                        assert result is not None
                        assert result.status == "paid"
                        assert result.tx_hash == "0xabc123def456"
                        assert result.price_usd == 15.00
                        assert result.price_crypto == 0.005  # $15 / $3000
                        assert result.payment_asset == "eth"

    @pytest.mark.asyncio
    async def test_connect_payment_no_merchant_address(self):
        """Test order with payment enabled but no merchant address."""
        from actions.order.connector.mock import MockOrderConnector, OrderConfig
        from actions.order.interface import OrderInput

        config = OrderConfig(
            payment_enabled=True, merchant_address="", use_dynamic_price=False
        )
        connector = MockOrderConnector(config)
        input_data = OrderInput(action="order pizza")

        result = await connector.connect(input_data)

        assert result is not None
        assert result.status == "confirmed"
        assert result.tx_hash is None


class TestOrderLoading:
    """Tests for loading order action."""

    def test_action_interface_docstring(self):
        """Test that Order has proper docstring."""
        from actions.order.interface import Order

        assert Order.__doc__ is not None
        assert "order" in Order.__doc__.lower()

    def test_action_input_has_action_field(self):
        """Test that OrderInput has required action field."""
        import typing as T

        from actions.order.interface import OrderInput

        hints = T.get_type_hints(OrderInput)
        assert "action" in hints

    def test_action_output_has_payment_fields(self):
        """Test that OrderOutput has payment-related fields."""
        import typing as T

        from actions.order.interface import OrderOutput

        hints = T.get_type_hints(OrderOutput)
        assert "price_usd" in hints
        assert "price_crypto" in hints
        assert "payment_asset" in hints
        assert "tx_hash" in hints


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
