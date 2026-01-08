"""
Coinbase Wallet Payment Connector.

Handles crypto payments via Coinbase CDP SDK.
"""

import logging
import os
import re
from typing import Optional, Tuple

from cdp import Cdp, Wallet, WalletData
from dotenv import load_dotenv

from actions.base import ActionConfig, ActionConnector
from actions.wallet_payment.interface import WalletPaymentInput


class WalletPaymentConfig(ActionConfig):
    """
    Configuration for Wallet Payment Connector.

    Parameters
    ----------
    asset_id : str
        Default asset for payments (default: "eth").
    network : str
        Blockchain network (default: "base-sepolia" for testnet).
    """

    asset_id: str = "eth"
    network: str = "base-sepolia"


class CoinbaseWalletConnector(ActionConnector[WalletPaymentConfig, WalletPaymentInput]):
    """
    Connector for sending crypto payments via Coinbase CDP SDK.

    This connector integrates with Coinbase's MPC wallet system to
    enable secure, programmatic crypto payments.

    Environment Variables Required:
    - COINBASE_API_KEY: Coinbase CDP API key
    - COINBASE_API_SECRET: Coinbase CDP API secret
    - COINBASE_WALLET_ID: Wallet ID for sending payments
    - COINBASE_WALLET_SEED: Wallet seed for signing transactions
    """

    def __init__(self, config: WalletPaymentConfig):
        """Initialize the Coinbase wallet connector."""
        super().__init__(config)

        load_dotenv()

        self.wallet: Optional[Wallet] = None
        self.wallet_id = os.environ.get("COINBASE_WALLET_ID")
        self.wallet_seed = os.environ.get("COINBASE_WALLET_SEED")

        # Initialize Coinbase CDP SDK
        api_key = os.environ.get("COINBASE_API_KEY")
        api_secret = os.environ.get("COINBASE_API_SECRET")

        if not api_key or not api_secret:
            logging.error(
                "COINBASE_API_KEY or COINBASE_API_SECRET environment variable is not set"
            )
            return

        try:
            Cdp.configure(api_key, api_secret)

            if self.wallet_id and self.wallet_seed:
                # Import wallet with seed for signing capability
                wallet_data = WalletData.from_dict(
                    {"wallet_id": self.wallet_id, "seed": self.wallet_seed}
                )
                self.wallet = Wallet.import_data(wallet_data)
                logging.info(
                    f"Coinbase wallet connected with signing: {self.wallet_id}"
                )
            elif self.wallet_id:
                self.wallet = Wallet.fetch(self.wallet_id)
                logging.warning("COINBASE_WALLET_SEED not set, signing will fail")
            else:
                logging.warning("COINBASE_WALLET_ID not set, payments will fail")

        except Exception as e:
            logging.error(f"Failed to initialize Coinbase wallet: {e}")

    def _parse_payment_command(
        self, command: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse payment command string.

        Supports formats:
        - "pay 0.001 ETH to 0x1234..."
        - "pay 0.01 eth to friend.eth"
        - "send 0.001 ETH 0x1234..."

        Parameters
        ----------
        command : str
            The payment command string

        Returns
        -------
        Tuple[Optional[str], Optional[str], Optional[str]]
            (amount, asset, recipient) or (None, None, None) if parsing fails
        """
        command = command.strip().lower()

        # Pattern: pay/send <amount> <asset> to <recipient>
        pattern = r"(?:pay|send)\s+([\d.]+)\s*(\w+)\s+(?:to\s+)?(\S+)"
        match = re.match(pattern, command, re.IGNORECASE)

        if match:
            amount = match.group(1)
            asset = match.group(2).lower()
            recipient = match.group(3)
            return amount, asset, recipient

        return None, None, None

    async def connect(self, output_interface: WalletPaymentInput) -> None:
        """
        Execute a crypto payment.

        Parameters
        ----------
        output_interface : WalletPaymentInput
            The payment input containing the command
        """
        command = output_interface.action
        logging.info(f"Processing payment command: {command}")

        # Parse the payment command
        amount, asset, recipient = self._parse_payment_command(command)

        if amount is None or asset is None or recipient is None:
            logging.error(f"Invalid payment command format: {command}")
            return

        # Validate wallet is available
        if not self.wallet:
            logging.error("Wallet not initialized. Check COINBASE credentials.")
            return

        try:
            # Refresh wallet to get latest state with signing capability
            if self.wallet_seed and self.wallet_id:
                wallet_data = WalletData.from_dict(
                    {"wallet_id": self.wallet_id, "seed": self.wallet_seed}
                )
                wallet = Wallet.import_data(wallet_data)
            elif self.wallet_id:
                wallet = Wallet.fetch(self.wallet_id)
            else:
                logging.error("Wallet ID not available")
                return
            if wallet is None:
                logging.error("Failed to fetch wallet")
                return
            self.wallet = wallet

            # Check balance
            balance = float(wallet.balance(asset))
            if balance < float(amount):
                logging.error(
                    f"Insufficient {asset.upper()} balance: {balance} < {amount}"
                )
                return

            # Execute the transfer
            logging.info(f"Sending {amount} {asset.upper()} to {recipient}")
            transfer = wallet.transfer(
                amount=amount,
                asset_id=asset,
                destination=recipient,
            )

            # Wait for transaction to be confirmed
            transfer.wait()

            tx_hash = transfer.transaction_hash
            logging.info(f"Payment successful! Transaction hash: {tx_hash}")

        except Exception as e:
            logging.error(f"Payment failed: {str(e)}")
