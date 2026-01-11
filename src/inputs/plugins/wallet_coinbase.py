import asyncio
import logging
import os
import time
from typing import List, Optional

from cdp import Cdp, Wallet
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
    """

    asset_id: str = Field(default="eth", description="Asset ID to query")


class WalletCoinbase(FuserInput[WalletCoinbaseConfig, List[float]]):
    """
    Queries current balance of the configured asset and reports a balance increase.
    """

    def __init__(self, config: WalletCoinbaseConfig):
        super().__init__(config)

        self.asset_id = self.config.asset_id

        # Track IO
        self.io_provider = IOProvider()
        self.messages: List[Message] = []

        self.POLL_INTERVAL = 0.5  # seconds between blockchain data updates
        self.COINBASE_WALLET_ID = os.environ.get("COINBASE_WALLET_ID")
        if self.COINBASE_WALLET_ID:
            logging.info("Coinbase wallet ID configured successfully")
        else:
            logging.warning("COINBASE_WALLET_ID environment variable not set")

        # Initialize Wallet
        # TODO(Kyle): Create Wallet if the wallet ID is not found
        # TODO(Kyle): Support importing other wallets, following https://docs.cdp.coinbase.com/mpc-wallet/docs/wallets#importing-a-wallet
        API_KEY = os.environ.get("COINBASE_API_KEY")
        API_SECRET = os.environ.get("COINBASE_API_SECRET")
        if not API_KEY or not API_SECRET:
            logging.error(
                "COINBASE_API_KEY or COINBASE_API_SECRET environment variable is not set"
            )
        else:
            Cdp.configure(API_KEY, API_SECRET)

        try:
            # fetch wallet data
            if not self.COINBASE_WALLET_ID:
                raise ValueError("COINBASE_WALLET_ID environment variable is not set")

            self.wallet = Wallet.fetch(self.COINBASE_WALLET_ID)
            logging.info(f"Wallet: {self.wallet}")

            self.balance = float(self.wallet.balance(self.asset_id))
            self.balance_previous = self.balance
        except Exception as e:
            logging.error(f"Error fetching Coinbase Wallet data: {e}")
            self.wallet = None
            self.balance = 0.0
            self.balance_previous = 0.0

        logging.info("Testing: WalletCoinbase: Initialized")

    async def _poll(self) -> List[float]:
        """
        Poll for Coinbase Wallet balance updates.

        Returns
        -------
        List[float]
            [current_balance, balance_change]
        """
        await asyncio.sleep(self.POLL_INTERVAL)

        # randomly simulate ETH inbound transfers for debugging purposes
        # if random.randint(0, 10) > 7:
        #     faucet_transaction = self.wallet.faucet(asset_id='eth')
        #     faucet_transaction.wait()
        #     logging.info(f"WalletCoinbase: Faucet transaction: {faucet_transaction}")

        try:
            self.wallet = Wallet.fetch(self.COINBASE_WALLET_ID)  # type: ignore
            logging.info(
                f"WalletCoinbase: Wallet refreshed: {self.wallet.balance(self.asset_id)}, the current balance is {self.balance}"
            )
            self.balance = float(self.wallet.balance(self.asset_id))
            balance_change = self.balance - self.balance_previous
            self.balance_previous = self.balance
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

        self.io_provider.add_input(
            self.__class__.__name__, result_message.message, result_message.timestamp
        )
        self.messages = []
        return result
