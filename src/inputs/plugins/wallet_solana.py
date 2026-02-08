import asyncio
import logging
import os
import time
from typing import List, Optional

from solana.rpc.api import Client
from solders.pubkey import Pubkey

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

LAMPORTS_PER_SOL = 1_000_000_000


class WalletSolana(FuserInput[SensorConfig, List[float]]):
    """
    Solana wallet monitor that tracks SOL balance changes.

    Queries the Solana blockchain for account balance updates and reports
    incoming transactions.

    Raises
    ------
    ValueError
        If SOLANA_ADDRESS is not set or is invalid
    Exception
        If connection to Solana network fails
    """

    def __init__(self, config: SensorConfig):
        """Initialize WalletSolana instance."""
        super().__init__(config)

        self.io_provider = IOProvider()

        self.SOL_balance: float = 0
        self.SOL_balance_previous: float = 0
        self.balance_sol: float = 0
        self.balance_change: float = 0

        self.messages: list[Message] = []

        self.RPC_URL = os.environ.get("SOLANA_RPC_URL", "https://api.devnet.solana.com")
        self.POLL_INTERVAL = 4

        address = os.environ.get("SOLANA_ADDRESS")
        if not address:
            raise ValueError("SOLANA_ADDRESS environment variable is required")

        try:
            self.pubkey = Pubkey.from_string(address)
        except Exception as e:
            raise ValueError(f"Invalid Solana address: {address}") from e

        self.ACCOUNT_ADDRESS = address
        logging.debug(f"Using {self.ACCOUNT_ADDRESS} as the Solana wallet address")

        self.client = Client(self.RPC_URL)
        try:
            self.client.get_balance(self.pubkey)
        except Exception as e:
            raise Exception(f"Failed to connect to Solana RPC at {self.RPC_URL}") from e

        logging.info("WalletSolana: Initialized")

    async def _poll(self) -> List[float]:
        """
        Poll for Solana SOL balance updates.

        Returns
        -------
        List[float]
            [current_balance, balance_change]
        """
        await asyncio.sleep(self.POLL_INTERVAL)

        try:
            result = self.client.get_balance(self.pubkey)
            lamports = result.value
            self.balance_sol = lamports / LAMPORTS_PER_SOL

            logging.debug(
                f"Solana balance: {self.balance_sol:.9f} SOL "
                f"(address: {self.ACCOUNT_ADDRESS})"
            )

            self.SOL_balance = self.balance_sol
            self.balance_change = self.SOL_balance - self.SOL_balance_previous
            self.SOL_balance_previous = self.SOL_balance

        except Exception as e:
            logging.error(f"Error fetching Solana balance: {e}")

        return [self.SOL_balance, self.balance_change]

    async def _raw_to_text(self, raw_input: List[float]) -> Optional[Message]:
        """
        Convert balance data to human-readable message.

        Parameters
        ----------
        raw_input : List[float]
            [current_balance, balance_change]

        Returns
        -------
        Optional[Message]
            Timestamped transaction notification or None
        """
        balance_change = raw_input[1]

        if balance_change > 0:
            message = f"You just received {balance_change:.9f} SOL."
            logging.debug(f"WalletSolana: {message}")
            return Message(timestamp=time.time(), message=message)

        return None

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
        Format and clear the latest buffer contents.

        Returns
        -------
        Optional[str]
            Formatted string of buffer contents or None if buffer is empty
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = f"""
{self.__class__.__name__} INPUT
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        self.messages = []
        return result
