"""
Mock Order Connector with Integrated Payment.

Simulates order placement and processes payment via Coinbase wallet.
Uses real-time crypto prices from CoinGecko API for USD conversion.
Supports ETH and USDC payments.
In production, this would connect to a real e-commerce API.
"""

import asyncio
import logging
import os
import re
import uuid
from typing import Optional, Tuple

import aiohttp
from dotenv import load_dotenv

from actions.base import ActionConfig, ActionConnector
from actions.order.interface import OrderInput, OrderOutput

# CoinGecko API endpoint for crypto prices
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/simple/price"

# CoinGecko IDs for supported assets
COINGECKO_IDS = {
    "eth": "ethereum",
    "usdc": "usd-coin",
}

# Default fallback prices
DEFAULT_PRICES = {
    "eth": 3000.0,
    "usdc": 1.0,
}


# Mock product catalog with prices in USD
PRODUCT_CATALOG = {
    "pizza": 15.00,
    "burger": 10.00,
    "coffee": 5.00,
    "sandwich": 8.00,
    "salad": 12.00,
    "soda": 3.00,
    "fries": 4.00,
    "ice cream": 6.00,
    "cake": 20.00,
    "beer": 7.00,
}

# Default price for unknown items
DEFAULT_PRICE = 10.00


class OrderConfig(ActionConfig):
    """
    Configuration for Mock Order Connector.

    Parameters
    ----------
    currency : str
        Currency for prices (default: "USD").
    auto_confirm : bool
        Automatically confirm orders (default: True).
    payment_enabled : bool
        Enable automatic payment after order (default: True).
    payment_asset : str
        Crypto asset for payment: "eth" or "usdc" (default: "eth").
    merchant_address : str
        Wallet address to send payment to.
    use_dynamic_price : bool
        Fetch real-time crypto prices from CoinGecko API (default: True).
    fallback_price_usd : float
        Fallback price if API fails (default: 3000.0 for ETH, 1.0 for USDC).
    network : str
        Blockchain network (default: "base-sepolia").
    """

    currency: str = "USD"
    auto_confirm: bool = True
    payment_enabled: bool = True
    payment_asset: str = "eth"
    merchant_address: str = ""
    use_dynamic_price: bool = True
    fallback_price_usd: float = 3000.0
    network: str = "base-sepolia"


class MockOrderConnector(ActionConnector[OrderConfig, OrderInput]):
    """
    Mock connector for order placement with integrated payment.

    This connector simulates an order system and processes payment
    via Coinbase CDP wallet. It demonstrates the full workflow:
    1. User places order
    2. Order is confirmed with price
    3. Payment is automatically processed via crypto wallet
    4. Transaction confirmation is returned

    In a real implementation, this would connect to:
    - Restaurant APIs (DoorDash, UberEats, etc.)
    - E-commerce platforms (Amazon, Shopify, etc.)
    - Custom order management systems
    """

    def __init__(self, config: OrderConfig):
        """Initialize the mock order connector with wallet integration."""
        super().__init__(config)
        self.orders: dict = {}
        self.wallet = None
        self.wallet_id = None
        self.wallet_seed = None

        load_dotenv()

        # Initialize wallet if payment is enabled
        if config.payment_enabled:
            self._init_wallet()

        logging.info("Mock order connector initialized")

    def _init_wallet(self) -> None:
        """Initialize Coinbase wallet for payments."""
        try:
            from cdp import Cdp, Wallet, WalletData

            api_key = os.environ.get("COINBASE_API_KEY")
            api_secret = os.environ.get("COINBASE_API_SECRET")
            self.wallet_id = os.environ.get("COINBASE_WALLET_ID")
            self.wallet_seed = os.environ.get("COINBASE_WALLET_SEED")

            if not api_key or not api_secret:
                logging.warning(
                    "COINBASE credentials not set, payment will be disabled"
                )
                return

            Cdp.configure(api_key, api_secret)

            if self.wallet_id and self.wallet_seed:
                wallet_data = WalletData.from_dict(
                    {"wallet_id": self.wallet_id, "seed": self.wallet_seed}
                )
                self.wallet = Wallet.import_data(wallet_data)
                logging.info(f"Wallet connected for payments: {self.wallet_id}")
            else:
                logging.warning("COINBASE_WALLET_ID/SEED not set, payment disabled")

        except ImportError:
            logging.warning("CDP SDK not installed, payment will be disabled")
        except Exception as e:
            logging.error(f"Failed to initialize wallet: {e}")

    def _parse_order_command(self, command: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Parse order command string.

        Supports formats:
        - "order pizza"
        - "order 2 pizzas"
        - "order a coffee"
        - "order 3 burgers"

        Parameters
        ----------
        command : str
            The order command string

        Returns
        -------
        Tuple[Optional[str], Optional[int]]
            (item, quantity) or (None, None) if parsing fails
        """
        command = command.strip().lower()

        # Pattern: [order] [quantity] [item]
        # Matches: "order 2 pizzas", "order a pizza", "order pizza", "2 pizzas", "a pizza"
        patterns = [
            r"(?:order\s+)?(\d+)\s+(\w+?)s?$",  # [order] 2 pizzas
            r"(?:order\s+)?(?:a|an)\s+(\w+?)s?$",  # [order] a pizza
            r"(?:order\s+)?(\w+?)s?$",  # [order] pizza
        ]

        for pattern in patterns:
            match = re.match(pattern, command, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    # Pattern with quantity
                    quantity = int(groups[0])
                    item = groups[1]
                else:
                    # Pattern without quantity (default to 1)
                    quantity = 1
                    item = groups[0]
                return item, quantity

        return None, None

    def _get_price(self, item: str) -> float:
        """Get price for an item from the catalog."""
        return PRODUCT_CATALOG.get(item.lower(), DEFAULT_PRICE)

    def _get_fallback_price(self, asset: str) -> float:
        """Get fallback price for an asset."""
        if asset.lower() == "usdc":
            return 1.0  # USDC is pegged to USD
        return self.config.fallback_price_usd

    async def _get_crypto_price_usd(self, asset: str) -> float:
        """
        Fetch real-time crypto price from CoinGecko API.

        Parameters
        ----------
        asset : str
            Crypto asset: "eth" or "usdc"

        Returns
        -------
        float
            Current price in USD, or fallback value if API fails.
        """
        asset = asset.lower()
        fallback_price = self._get_fallback_price(asset)

        if not self.config.use_dynamic_price:
            return fallback_price

        coingecko_id = COINGECKO_IDS.get(asset)
        if not coingecko_id:
            logging.warning(f"Unknown asset {asset}, using fallback price")
            return fallback_price

        try:
            async with aiohttp.ClientSession() as session:
                params = {"ids": coingecko_id, "vs_currencies": "usd"}
                async with session.get(
                    COINGECKO_API_URL, params=params, timeout=10
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = data.get(coingecko_id, {}).get("usd")
                        if price:
                            logging.info(f"Fetched {asset.upper()} price: ${price:.2f}")
                            return float(price)

            logging.warning(
                f"Failed to get {asset.upper()} price from API, using fallback"
            )
            return fallback_price

        except asyncio.TimeoutError:
            logging.warning("CoinGecko API timeout, using fallback price")
            return fallback_price
        except Exception as e:
            logging.warning(f"Error fetching crypto price: {e}, using fallback")
            return fallback_price

    async def _usd_to_crypto(self, usd_amount: float, asset: str) -> float:
        """Convert USD to crypto using real-time or fallback price."""
        crypto_price = await self._get_crypto_price_usd(asset)
        return usd_amount / crypto_price

    async def _process_payment(
        self, amount: float, asset: str, merchant_address: str
    ) -> Optional[str]:
        """
        Process payment via Coinbase wallet.

        Parameters
        ----------
        amount : float
            Amount to send in crypto
        asset : str
            Crypto asset: "eth" or "usdc"
        merchant_address : str
            Recipient wallet address

        Returns
        -------
        Optional[str]
            Transaction hash if successful, None otherwise
        """
        if not self.wallet:
            logging.error("Wallet not initialized, cannot process payment")
            return None

        asset = asset.lower()

        try:
            from cdp import Wallet, WalletData

            # Refresh wallet with signing capability
            if self.wallet_seed and self.wallet_id:
                wallet_data = WalletData.from_dict(
                    {"wallet_id": self.wallet_id, "seed": self.wallet_seed}
                )
                wallet = Wallet.import_data(wallet_data)
            elif self.wallet_id:
                wallet = Wallet.fetch(self.wallet_id)
            else:
                logging.error("Wallet ID not available")
                return None

            if wallet is None:
                logging.error("Failed to fetch wallet")
                return None

            # Check balance
            balance = float(wallet.balance(asset))
            if balance < amount:
                logging.error(
                    f"Insufficient {asset.upper()} balance: {balance} < {amount}"
                )
                return None

            # Execute transfer
            logging.info(
                f"Sending {amount:.6f} {asset.upper()} to merchant {merchant_address}"
            )
            transfer = wallet.transfer(
                amount=str(amount),
                asset_id=asset,
                destination=merchant_address,
            )

            # Wait for confirmation
            transfer.wait()
            tx_hash = transfer.transaction_hash

            logging.info(f"Payment successful! TX: {tx_hash}")
            return tx_hash

        except Exception as e:
            logging.error(f"Payment failed: {e}")
            return None

    async def connect(self, output_interface: OrderInput) -> Optional[OrderOutput]:  # type: ignore[override]
        """
        Process an order command with integrated payment.

        Full workflow:
        1. Parse order command
        2. Calculate price
        3. Process payment (if enabled) in ETH or USDC
        4. Return order confirmation with transaction

        Parameters
        ----------
        output_interface : OrderInput
            The order input containing the command

        Returns
        -------
        Optional[OrderOutput]
            Order details with payment info if successful
        """
        command = output_interface.action
        logging.info(f"Processing order command: {command}")

        # Get payment asset from config
        payment_asset = self.config.payment_asset.lower()

        # Parse the order command
        item, quantity = self._parse_order_command(command)

        if item is None or quantity is None:
            logging.error(f"Invalid order command format: {command}")
            return OrderOutput(
                order_id="",
                item="",
                quantity=0,
                price_usd=0.0,
                price_crypto=0.0,
                payment_asset=payment_asset,
                status="failed",
                message=f"Could not parse order command: {command}",
                tx_hash=None,
            )

        # Calculate price (with real-time crypto conversion)
        unit_price = self._get_price(item)
        total_price_usd = unit_price * quantity
        total_price_crypto = await self._usd_to_crypto(total_price_usd, payment_asset)

        # Generate order ID
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"

        logging.info(
            f"Order: {quantity}x {item} = ${total_price_usd:.2f} "
            f"({total_price_crypto:.6f} {payment_asset.upper()})"
        )

        # Process payment if enabled
        tx_hash = None
        status = "confirmed" if self.config.auto_confirm else "pending"
        message = f"Order confirmed: {quantity}x {item} for ${total_price_usd:.2f}"

        if self.config.payment_enabled and self.config.merchant_address:
            logging.info(f"Processing payment in {payment_asset.upper()}...")
            tx_hash = await self._process_payment(
                total_price_crypto, payment_asset, self.config.merchant_address
            )

            if tx_hash:
                status = "paid"
                message = (
                    f"Order confirmed and paid: {quantity}x {item} for "
                    f"${total_price_usd:.2f} ({total_price_crypto:.6f} {payment_asset.upper()}). "
                    f"TX: {tx_hash}"
                )
            else:
                status = "payment_failed"
                message = (
                    f"Order confirmed but payment failed: {quantity}x {item} "
                    f"for ${total_price_usd:.2f}"
                )
        elif self.config.payment_enabled and not self.config.merchant_address:
            logging.warning("Payment enabled but no merchant_address configured")

        # Create order output
        order = OrderOutput(
            order_id=order_id,
            item=item,
            quantity=quantity,
            price_usd=total_price_usd,
            price_crypto=total_price_crypto,
            payment_asset=payment_asset,
            status=status,
            message=message,
            tx_hash=tx_hash,
        )
        self.orders[order_id] = order

        logging.info(f"Order completed: {order_id} - Status: {status}")

        return order
