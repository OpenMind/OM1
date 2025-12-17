import logging
import os
from typing import Any, Optional

from .base import ProviderBase


class CoinbaseProvider(ProviderBase):
    """
    Coinbase provider with lazy import of the Coinbase SDK (cdp).
    Call init() before connect() to load and configure the SDK.
    """

    name = "coinbase"

    def __init__(self, wallet_id: Optional[str] = None):
        self.wallet_id = wallet_id or os.environ.get("COINBASE_WALLET_ID")
        self.wallet: Any = None
        self._configured = False
        self._Wallet: Optional[Any] = None  # reference to Wallet class after init()

    def init(self, config: Optional[dict] = None) -> None:
        """
        Initialize Coinbase SDK and configure credentials.
        This performs a lazy import of the 'cdp' package so importing
        providers module does not require cdp to be installed.
        """
        try:
            # lazy import to avoid heavy dependency at import time
            from cdp import Cdp, Wallet  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Coinbase SDK (cdp) is not available. Install it or provide a test stub."
            ) from e

        api_key = os.environ.get("COINBASE_API_KEY")
        api_secret = os.environ.get("COINBASE_API_SECRET")

        if not api_key or not api_secret:
            raise RuntimeError("COINBASE_API_KEY/COINBASE_API_SECRET not set")

        Cdp.configure(api_key, api_secret)
        self._configured = True
        self._Wallet = Wallet

    def connect(self, **kwargs) -> bool:
        """
        Connect to the configured Coinbase wallet. Must call init() first.
        """
        if not self._configured:
            logging.error("CoinbaseProvider: init() not called")
            return False

        if not self.wallet_id:
            logging.error("CoinbaseProvider: COINBASE_WALLET_ID not set")
            return False

        if self._Wallet is None:
            logging.error("CoinbaseProvider: Wallet class not loaded")
            return False

        try:
            # use stored Wallet class reference from init()
            self.wallet = self._Wallet.fetch(self.wallet_id)
            logging.info(f"CoinbaseProvider: connected to wallet {self.wallet_id}")
            return True
        except Exception as e:
            logging.error(f"CoinbaseProvider.connect error: {e}")
            return False
            return False

    def disconnect(self) -> None:
        self.wallet = None
        self._Wallet = None
        self._configured = False

    def get_balance(self, asset: str) -> float:
        if not self.wallet:
            raise RuntimeError("CoinbaseProvider: wallet not connected")
        return float(self.wallet.balance(asset))

    def sign_message(self, message: str) -> str:
        """
        Signing flow for Coinbase MPC wallets is SDK-specific.
        Not implemented here — implement using the cdp SDK if required.
        """
        raise NotImplementedError("CoinbaseProvider.sign_message is not implemented")
