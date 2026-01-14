from dataclasses import dataclass
import typing as T
from actions.base import Interface


@dataclass
class BalanceInput:
    """Input for balance check action."""
    action: str = "check"


@dataclass  
class BalanceOutput:
    """Output with wallet balance information."""
    eth_balance: str = ""
    usdc_balance: str = ""
    address: str = ""


class CheckBalance(Interface[BalanceInput, BalanceOutput]):
    """Check wallet balance (ETH and USDC)."""
    input: BalanceInput
    output: BalanceOutput
