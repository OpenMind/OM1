from dataclasses import dataclass

from actions.base import Interface


@dataclass
class CoinbasePaymentInput:
    """Input for Coinbase payment actions."""

    action: str


@dataclass
class CoinbasePaymentOutput:
    """Output from Coinbase payment actions."""

    status: str
    tx_hash: str = ""
    amount: str = ""


@dataclass
class CoinbasePayment(Interface[CoinbasePaymentInput, CoinbasePaymentOutput]):
    """
    Send cryptocurrency payments via Coinbase CDP.
    Supports: ETH, USDC, USDT, USDbC, DAI, WETH (depends on network).
    Commands: 'send 5 usdc to john', 'pay coffee 10 usdt', 'balance'.
    """

    input: CoinbasePaymentInput
    output: CoinbasePaymentOutput
