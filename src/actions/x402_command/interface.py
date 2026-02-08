from dataclasses import dataclass

from actions.base import Interface


@dataclass
class X402PaymentInput:
    """
    Input interface for the x402 payment action.

    Parameters
    ----------
    action : str
        The message or command to send along with the x402 payment.
    """

    action: str


@dataclass
class X402Payment(Interface[X402PaymentInput, X402PaymentInput]):
    """
    This action sends a message to an x402-protected endpoint by completing
    an on-chain USDC payment. The agent can use this to pay for services,
    information, or any resource gated behind the x402 payment protocol.
    """

    input: X402PaymentInput
    output: X402PaymentInput
