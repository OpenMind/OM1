from typing import Any, Dict, Optional

from eth_account import Account
from eth_account.messages import encode_defunct

from .base import ProviderBase


class MetaMaskProvider(ProviderBase):
    name = "metamask"

    def init(self, config: Optional[Dict[str, Any]] = None) -> None:
        return None

    def connect(self, **kwargs) -> bool:
        return True

    def disconnect(self) -> None:
        return None

    def get_balance(self, asset: str) -> float:
        raise NotImplementedError("MetaMaskProvider: use on-chain RPC to fetch balance")

    def sign_message(self, message: str) -> str:
        raise NotImplementedError("MetaMaskProvider cannot sign server-side")

    @staticmethod
    def verify_signature(address: str, message: str, signature: str) -> bool:
        msg = encode_defunct(text=message)
        recovered = Account.recover_message(msg, signature=signature)
        return recovered.lower() == address.lower()
