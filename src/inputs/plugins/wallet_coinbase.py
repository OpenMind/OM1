import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from cdp import Cdp, Wallet
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider

# Default path for storing wallet seed data
DEFAULT_WALLET_SEED_FILE = os.path.expanduser("~/.om1/coinbase_wallet_seed.json")


class WalletCoinbaseConfig(SensorConfig):
    """
    Configuration for Wallet Coinbase Sensor.

    Parameters
    ----------
    asset_id : str
        Asset ID to query.
    network_id : str
        Network ID for wallet creation (default: base-sepolia for testnet).
    wallet_seed_file : str
        Path to file storing wallet seed for persistence.
    """

    asset_id: str = Field(default="eth", description="Asset ID to query")
    network_id: str = Field(
        default="base-sepolia",
        description="Network ID for wallet creation (e.g., base-sepolia, base-mainnet)",
    )
    wallet_seed_file: str = Field(
        default=DEFAULT_WALLET_SEED_FILE,
        description="Path to file storing wallet seed for persistence",
    )


class WalletCoinbase(FuserInput[WalletCoinbaseConfig, List[float]]):
    """
    Queries current balance of the configured asset and reports a balance increase.
    """

    def __init__(self, config: WalletCoinbaseConfig):
        """
        Initialize the WalletCoinbase input handler.

        Sets up the required providers and buffers for handling Coinbase wallet data.
        Fetches the initial wallet balance. If no wallet ID is provided, creates a new
        wallet and saves it for future use.

        Parameters
        ----------
        config : WalletCoinbaseConfig
            Configuration for the sensor input, specifying the asset ID to query.
        """
        super().__init__(config)

        self.asset_id = self.config.asset_id
        self.network_id = self.config.network_id
        self.wallet_seed_file = self.config.wallet_seed_file

        # Track IO
        self.io_provider = IOProvider()
        self.messages: List[Message] = []

        self.POLL_INTERVAL = 0.5  # seconds between blockchain data updates
        self.COINBASE_WALLET_ID = os.environ.get("COINBASE_WALLET_ID")
        if self.COINBASE_WALLET_ID:
            logging.info("Coinbase wallet ID configured successfully")
        else:
            logging.warning(
                "COINBASE_WALLET_ID environment variable not set, will attempt to create or load wallet"
            )

        # Initialize CDP SDK
        API_KEY = os.environ.get("COINBASE_API_KEY")
        API_SECRET = os.environ.get("COINBASE_API_SECRET")
        if not API_KEY or not API_SECRET:
            logging.error(
                "COINBASE_API_KEY or COINBASE_API_SECRET environment variable is not set"
            )
            self.wallet = None
            self.balance = 0.0
            self.balance_previous = 0.0
            return

        Cdp.configure(API_KEY, API_SECRET)

        try:
            self.wallet = self._get_or_create_wallet()
            if self.wallet:
                self.COINBASE_WALLET_ID = self.wallet.id
                logging.info(f"Wallet initialized: {self.wallet}")
                self.balance = float(self.wallet.balance(self.asset_id))
                self.balance_previous = self.balance
            else:
                self.balance = 0.0
                self.balance_previous = 0.0
        except Exception as e:
            logging.error(f"Error initializing Coinbase Wallet: {e}")
            self.wallet = None
            self.balance = 0.0
            self.balance_previous = 0.0

        logging.info("WalletCoinbase: Initialized")

    def _get_or_create_wallet(self) -> Optional[Wallet]:
        """
        Get existing wallet or create a new one if not found.

        Priority:
        1. Use COINBASE_WALLET_ID environment variable if set
        2. Load wallet from saved seed file if exists
        3. Create new wallet and save seed for future use

        Returns
        -------
        Optional[Wallet]
            The wallet instance, or None if creation/fetch failed.
        """
        # Priority 1: Try to fetch wallet using environment variable
        if self.COINBASE_WALLET_ID:
            try:
                wallet = Wallet.fetch(self.COINBASE_WALLET_ID)
                logging.info(f"Fetched existing wallet: {wallet.id}")

                # Try to load seed if available for signing capabilities
                self._load_wallet_seed(wallet)
                return wallet
            except Exception as e:
                logging.warning(
                    f"Could not fetch wallet with ID {self.COINBASE_WALLET_ID}: {e}"
                )

        # Priority 2: Try to load wallet from saved seed file
        wallet = self._load_wallet_from_seed_file()
        if wallet:
            return wallet

        # Priority 3: Create new wallet
        return self._create_new_wallet()

    def _load_wallet_from_seed_file(self) -> Optional[Wallet]:
        """
        Load wallet from saved seed file.

        Returns
        -------
        Optional[Wallet]
            The wallet instance if loaded successfully, None otherwise.
        """
        try:
            if not os.path.exists(self.wallet_seed_file):
                return None

            with open(self.wallet_seed_file, "r") as f:
                wallet_data = json.load(f)

            wallet_id = wallet_data.get("wallet_id")
            seed = wallet_data.get("seed")

            if not wallet_id:
                logging.warning("Wallet seed file exists but missing wallet_id")
                return None

            wallet = Wallet.fetch(wallet_id)

            # Load seed for signing capabilities
            if seed:
                wallet.load_seed(self.wallet_seed_file)
                logging.info(f"Loaded wallet with signing capabilities: {wallet.id}")
            else:
                logging.info(f"Loaded wallet (view-only): {wallet.id}")

            return wallet
        except Exception as e:
            logging.warning(f"Could not load wallet from seed file: {e}")
            return None

    def _load_wallet_seed(self, wallet: Wallet) -> None:
        """
        Attempt to load seed for an existing wallet to enable signing.

        Parameters
        ----------
        wallet : Wallet
            The wallet to load seed for.
        """
        try:
            if os.path.exists(self.wallet_seed_file):
                with open(self.wallet_seed_file, "r") as f:
                    wallet_data = json.load(f)

                if wallet_data.get("wallet_id") == wallet.id and wallet_data.get(
                    "seed"
                ):
                    wallet.load_seed(self.wallet_seed_file)
                    logging.info(f"Loaded seed for wallet: {wallet.id}")
        except Exception as e:
            logging.debug(f"Could not load seed for wallet: {e}")

    def _create_new_wallet(self) -> Optional[Wallet]:
        """
        Create a new wallet and save its seed for future use.

        Returns
        -------
        Optional[Wallet]
            The newly created wallet, or None if creation failed.
        """
        try:
            logging.info(f"Creating new wallet on network: {self.network_id}")
            wallet = Wallet.create(network_id=self.network_id)

            # Save wallet data for future use
            self._save_wallet_seed(wallet)

            logging.info(f"Created new wallet: {wallet.id}")
            logging.info(
                f"Default address: {wallet.default_address.address_id if wallet.default_address else 'N/A'}"
            )

            # Set environment variable for current session
            os.environ["COINBASE_WALLET_ID"] = wallet.id

            return wallet
        except Exception as e:
            logging.error(f"Failed to create new wallet: {e}")
            return None

    def _save_wallet_seed(self, wallet: Wallet) -> None:
        """
        Save wallet seed to file for future use.

        Parameters
        ----------
        wallet : Wallet
            The wallet to save.
        """
        try:
            # Ensure directory exists
            seed_dir = os.path.dirname(self.wallet_seed_file)
            if seed_dir:
                Path(seed_dir).mkdir(parents=True, exist_ok=True)

            # Export and save wallet data
            wallet_data = wallet.export_data()
            save_data = {
                "wallet_id": wallet_data.wallet_id,
                "seed": wallet_data.seed,
                "network_id": wallet_data.network_id,
            }

            with open(self.wallet_seed_file, "w") as f:
                json.dump(save_data, f, indent=2)

            # Set restrictive permissions (owner read/write only)
            os.chmod(self.wallet_seed_file, 0o600)

            logging.info(f"Wallet seed saved to: {self.wallet_seed_file}")
        except Exception as e:
            logging.error(f"Failed to save wallet seed: {e}")

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
