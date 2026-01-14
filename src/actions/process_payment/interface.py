from dataclasses import dataclass, field
from typing import Optional, List
from actions.base import Interface

@dataclass
class PaymentInput:
    """
    Enhanced input interface for payment action.
    
    Supports:
    - Dynamic recipient addresses (Ethereum addresses or ENS names)
    - Multiple currencies (ETH, USDC, DAI)
    - Split payments to multiple recipients
    - Safety validations
    """
    action: str = "pay"
    amount: str = "0.001"  # Numeric amount only
    currency: str = "ETH"  # ETH, USDC, or DAI
    recipient: str = ""  # Ethereum address or ENS name (e.g., "vitalik.eth")
    recipients: List[str] = field(default_factory=list)  # For split payments
    confirmation_required: bool = True
    
    # Internal fields set during validation
    resolved_address: Optional[str] = None
    tx_hash: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def __repr__(self):
        if self.tx_hash:
            explorer_url = f"https://sepolia.basescan.org/tx/{self.tx_hash}"
            recipient_display = self.resolved_address[:6] + "..." + self.resolved_address[-4:] if self.resolved_address else self.recipient
            return (
                f"\n💳 [WALLET] PAYMENT SUCCESSFUL\n"
                f"   Amount: {self.amount} {self.currency}\n"
                f"   To: {recipient_display}\n"
                f"   Tx Hash: {self.tx_hash[:10]}...\n"
                f"   Explorer: {explorer_url}\n"
                f"   Status: ✅ CONFIRMED ON-CHAIN (Base Sepolia)\n"
            )
        else:
            return (
                f"\n💳 [WALLET] PREPARING PAYMENT\n"
                f"   Amount: {self.amount} {self.currency}\n"
                f"   To: {self.recipient}\n"
                f"   Status: 🔄 VALIDATING...\n"
            )

class ProcessPayment(Interface):
    """Process Payment action interface."""
    input: PaymentInput
    output: PaymentInput
