"""
Wallet Payment Action Interface.

Defines the input/output interface for crypto wallet payments.
"""

from dataclasses import dataclass

from actions.base import Interface


@dataclass
class WalletPaymentInput:
    """
    Input interface for Wallet Payment action.

    Parameters
    ----------
    action : str
        Payment command in format: "pay <amount> <asset> to <recipient>"
        Examples:
        - "pay 0.001 ETH to 0x1234..."
        - "pay 0.01 eth to friend.eth"
    """

    action: str


@dataclass
class WalletPayment(Interface[WalletPaymentInput, WalletPaymentInput]):
    """
    Wallet Payment action interface.

    This action allows the OM1 agent to send crypto payments
    via a connected wallet (e.g., Coinbase).

    Example LLM output:
    ```
    WalletPayment: 'pay 0.001 ETH to 0x742d35Cc6634C0532925a3b844Bc9e7595f6D1E2'
    ```
    """

    input: WalletPaymentInput
    output: WalletPaymentInput
