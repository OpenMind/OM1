import asyncio
import logging
import os
import time
from typing import List, Optional

from cdp import CdpClient
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class WalletCoinbaseConfig(SensorConfig):
    """
    Configuration for Wallet Coinbase Sensor.

    Parameters
    ----------
    asset_id : str
        Asset ID to query.
    wallet_address : str, optional
        Wallet address to monitor. If not provided, uses COINBASE_WALLET_ADDRESS env var.
    network : str
        Network to use (e.g., 'base', 'base-sepolia', 'ethereum'). Defaults to 'base'.
    """

    asset_id: str = Field(default="eth", description="Asset ID to query")
    wallet_address: Optional[str] = Field(default=None, description="Wallet address to monitor")
    network: str = Field(default="base", description="Network to use")


class WalletCoinbase(FuserInput[WalletCoinbaseConfig, List[float]]):
    """
    Queries current balance of the configured asset and reports a balance increase.

    Note: This uses the CDP SDK 1.x API. The wallet must be created and managed
    outside this sensor. This sensor only monitors balance changes.
    """

    def __init__(self, config: WalletCoinbaseConfig):
        """
        Initialize the WalletCoinbase input handler.

        Sets up the required providers and buffers for handling Coinbase wallet data.
        Fetches the initial wallet balance.

        Parameters
        ----------
        config : WalletCoinbaseConfig
            Configuration for the sensor input, specifying the asset ID to query.
        """
        super().__init__(config)

        self.asset_id = self.config.asset_id
        self.network = self.config.network

        # Track IO
        self.io_provider = IOProvider()
        self.messages: List[Message] = []

        self.POLL_INTERVAL = 0.5  # seconds between blockchain data updates
        self.COINBASE_WALLET_ADDRESS = self.config.wallet_address or os.environ.get("COINBASE_WALLET_ADDRESS")
        if self.COINBASE_WALLET_ADDRESS:
            logging.info(f"Coinbase wallet address configured: {self.COINBASE_WALLET_ADDRESS}")
        else:
            logging.warning("COINBASE_WALLET_ADDRESS environment variable not set")

        # Initialize CDP Client
        API_KEY_ID = os.environ.get("CDP_API_KEY_ID")
        API_KEY_SECRET = os.environ.get("CDP_API_KEY_SECRET")

        if not API_KEY_ID or not API_KEY_SECRET:
            logging.error("CDP_API_KEY_ID and CDP_API_KEY_SECRET environment variables are not set")
            self.cdp_client = None
        else:
            try:
                self.cdp_client = CdpClient(api_key_id=API_KEY_ID, api_key_secret=API_KEY_SECRET)
                logging.info("CDP Client initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing CDP Client: {e}")
                self.cdp_client = None

        # Initialize balance tracking
        self.balance = 0.0
        self.balance_previous = 0.0

        logging.info("Testing: WalletCoinbase: Initialized")

    async def _fetch_balance(self) -> float:
        """
        Fetch current balance asynchronously using CDP SDK 1.x.

        Returns
        -------
        float
            Current balance of the specified asset
        """
        if not self.cdp_client or not self.COINBASE_WALLET_ADDRESS:
            logging.warning("CDP client or wallet address not configured")
            return 0.0

        try:
            # Query token balances for the address on the specified network
            balances_result = await self.cdp_client.evm.list_token_balances(
                address=self.COINBASE_WALLET_ADDRESS,
                network=self.network,
            )

            for balance_item in balances_result.balances:
                token_symbol = getattr(balance_item, "symbol", "").lower()
                if token_symbol == self.asset_id.lower():
                    amount = getattr(balance_item, "amount", "0")
                    decimals = getattr(balance_item, "decimals", 18)
                    balance_value = float(amount) / (10**decimals)
                    logging.debug(
                        f"Balance for {self.asset_id}: {balance_value} " f"(raw: {amount}, decimals: {decimals})"
                    )
                    return balance_value

            logging.debug(
                f"Asset {self.asset_id} not found in token balances. "
                f"Available tokens: {[getattr(b, 'symbol', 'unknown') for b in balances_result.balances]}"
            )
            return 0.0

        except AttributeError as e:
            logging.error(f"CDP SDK API structure mismatch. The API may have changed: {e}")
            return 0.0
        except Exception as e:
            logging.error(f"Error fetching token balance: {e}")
            return 0.0

    async def _poll(self) -> List[float]:
        """
        Poll for Coinbase Wallet balance updates.

        Returns
        -------
        List[float]
            [current_balance, balance_change]
        """
        await asyncio.sleep(self.POLL_INTERVAL)

        try:
            if not self.cdp_client or not self.COINBASE_WALLET_ADDRESS:
                logging.debug("CDP client or wallet address not configured")
                return [self.balance, 0.0]

            new_balance = await self._fetch_balance()

            balance_change = new_balance - self.balance_previous

            self.balance = new_balance
            self.balance_previous = new_balance

            if balance_change != 0:
                logging.info(
                    f"WalletCoinbase: Balance updated to {self.balance} {self.asset_id} "
                    f"(change: {balance_change:+.5f})"
                )

        except Exception as e:
            logging.error(f"Error refreshing wallet data: {e}")
            balance_change = 0.0

        return [self.balance, balance_change]

    async def _raw_to_text(self, raw_input: List[float]) -> Optional[Message]:
        """
        Convert balance data to human-readable message.

        Parameters
        ----------
        raw_input : List[float]
            [current_balance, balance_change]

        Returns
        -------
        Message
            Timestamped status or transaction notification
        """
        balance_change = raw_input[1]

        message = ""

        if balance_change > 0:
            message = f"{balance_change:.5f}"
            logging.info(f"\n\nWalletCoinbase balance change: {message}")
        else:
            return None

        logging.debug(f"WalletCoinbase: {message}")
        return Message(timestamp=time.time(), message=message)

    async def raw_to_text(self, raw_input: List[float]):
        """
        Process balance update and manage message buffer.

        Parameters
        ----------
        raw_input : List[float]
            Raw balance data
        """
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Format and clear the buffer contents. If there are multiple transactions,
        combine them into a single message.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        transaction_sum = 0

        # all the messages, by definition, are non-zero
        for message in self.messages:
            transaction_sum += float(message.message)

        last_message = self.messages[-1]
        result_message = Message(
            timestamp=last_message.timestamp,
            message=f"You just received {transaction_sum:.5f} {self.asset_id.upper()}.",
        )

        result = f"""
{self.__class__.__name__} INPUT
// START
{result_message.message}
// END
"""

        self.io_provider.add_input(self.__class__.__name__, result_message.message, result_message.timestamp)
        self.messages = []
        return result
