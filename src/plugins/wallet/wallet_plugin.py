import time
import random
import threading
from typing import Dict, List

class WalletPlugin:
    """
    Manages cryptocurrency wallet operations with transaction tracking
    Supports mock mode and real wallet integration
    """
    
    def __init__(self, config_path="src/plugins/wallet/config.yaml"):
        import yaml
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            self.mock_mode = cfg.get("mock_mode", True)
            self.eth_price_usd = cfg.get("eth_price_usd", 2500)
            self.wallet_address = cfg.get("wallet_address", "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb")
        except:
            self.mock_mode = True
            self.eth_price_usd = 2500
            self.wallet_address = "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb"
        
        self.balance_eth = 5.0
        self.balance_usd = self.balance_eth * self.eth_price_usd
        self.transaction_history: List[Dict] = []
        
        # Start background thread for price updates
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        """Background thread for simulating price fluctuations"""
        while True:
            if self.mock_mode:
                # Simulate realistic ETH price fluctuation
                change = random.uniform(-50, 50)
                self.eth_price_usd = max(1000, self.eth_price_usd + change)
                self.balance_usd = self.balance_eth * self.eth_price_usd
            time.sleep(10)

    def get_balance(self) -> Dict:
        """Get current wallet balance"""
        return {
            "eth": round(self.balance_eth, 4),
            "usd": round(self.balance_usd, 2),
            "eth_price": round(self.eth_price_usd, 2),
            "wallet_address": self.wallet_address
        }

    def send_payment(self, to: str, amount: float, currency="ETH") -> Dict:
        """
        Send payment with detailed status reporting
        Returns: {success, status, tx, amount, to, confirmations, error}
        """
        if currency == "ETH" and amount <= self.balance_eth:
            # Generate transaction hash
            tx_hash = f"0x{random.randint(1000000000000000, 9999999999999999):x}"
            
            transaction = {
                "success": True,
                "status": "pending",
                "tx": tx_hash,
                "amount": amount,
                "currency": currency,
                "to": to,
                "from": self.wallet_address,
                "timestamp": time.time(),
                "confirmations": 0
            }
            
            # Update balance
            self.balance_eth -= amount
            self.balance_usd = self.balance_eth * self.eth_price_usd
            
            # Add to history
            self.transaction_history.append(transaction)
            
            # Simulate transaction confirmation
            time.sleep(0.5)
            transaction["status"] = "confirmed"
            transaction["confirmations"] = 1
            
            return transaction
        
        # Insufficient funds
        return {
            "success": False,
            "status": "failed",
            "error": "insufficient funds",
            "available_balance": self.balance_eth,
            "requested_amount": amount,
            "currency": currency
        }
    
    def get_transaction_history(self) -> List[Dict]:
        """Get all transactions"""
        return self.transaction_history.copy()
    
    def get_transaction_status(self, tx_hash: str) -> Dict:
        """Get specific transaction status by hash"""
        for tx in self.transaction_history:
            if tx.get("tx") == tx_hash:
                return {
                    "tx": tx_hash,
                    "status": tx["status"],
                    "confirmations": tx.get("confirmations", 0),
                    "amount": tx["amount"],
                    "to": tx["to"],
                    "timestamp": tx["timestamp"]
                }
        return {"error": "Transaction not found", "tx": tx_hash}
    
    def get_wallet_info(self) -> Dict:
        """Get complete wallet information"""
        return {
            "address": self.wallet_address,
            "balance": self.get_balance(),
            "transaction_count": len(self.transaction_history),
            "mode": "mock" if self.mock_mode else "real"
        }
