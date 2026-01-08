"""
Order Action Interface.

Defines the input/output interface for placing orders via smart assistants
with integrated crypto wallet payment.
"""

from dataclasses import dataclass
from typing import Optional

from actions.base import Interface


@dataclass
class OrderInput:
    """
    Input interface for Order action.

    Parameters
    ----------
    action : str
        Order command in format: "order <item>" or "order <quantity> <item>"
        Examples:
        - "order pizza"
        - "order 2 coffees"
        - "order a burger"
    """

    action: str


@dataclass
class OrderOutput:
    """
    Output interface for Order action with payment details.

    Parameters
    ----------
    order_id : str
        Unique identifier for the order.
    item : str
        The item ordered.
    quantity : int
        Number of items ordered.
    price_usd : float
        Total price in USD.
    price_crypto : float
        Total price in crypto (ETH or USDC).
    payment_asset : str
        Crypto asset used for payment ("eth" or "usdc").
    status : str
        Order status (pending, confirmed, paid, payment_failed, failed).
    message : str
        Human-readable status message.
    tx_hash : Optional[str]
        Blockchain transaction hash if payment was processed.
    """

    order_id: str
    item: str
    quantity: int
    price_usd: float
    price_crypto: float
    payment_asset: str
    status: str
    message: str
    tx_hash: Optional[str] = None


@dataclass
class Order(Interface[OrderInput, OrderOutput]):
    """
    Order action interface with integrated payment.

    This action allows the OM1 agent to place orders via smart assistants
    and automatically process payment via crypto wallet.

    Full workflow:
    1. User says "order 2 pizzas"
    2. Order is created with price calculation
    3. Payment is sent to merchant via crypto wallet
    4. Confirmation with transaction hash is returned

    Example LLM output:
    ```
    order: 'order 2 pizzas'
    speak: {{'sentence': 'Ordering 2 pizzas and processing payment'}}
    ```
    """

    input: OrderInput
    output: Optional[OrderOutput] = None  # type: ignore[assignment]
