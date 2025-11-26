from typing import Optional
from pydantic import BaseModel
from actions.base import Interface


class PizzaOrderInput(BaseModel):
    """
    Interface payload for pizza ordering + blockchain verification.
    Sent from the LLM → connector.
    """
    # main selector (LLM decides which step)
    action: str            # create_order, verify_payment, check_order

    # fields used when creating the order
    item: Optional[str] = None
    amount: Optional[float] = None
    wallet_address: Optional[str] = None

    # fields used when verifying payment
    order_id: Optional[str] = None
    payment_hash: Optional[str] = None


class PizzaOrder(Interface[PizzaOrderInput, PizzaOrderInput]):
    """Interface definition for pizza_order action."""
    T_input = PizzaOrderInput
    T_output = PizzaOrderInput
