import time, random, threading
from typing import Dict

class WalletPlugin:
    def __init__(self):
        self.balance_eth = 5.0
        self.balance_usd = self.balance_eth * 2500
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            time.sleep(10)

    def get_balance(self) -> Dict:
        return {"eth": round(self.balance_eth, 4), "usd": round(self.balance_usd, 2)}

    def send_payment(self, to: str, amount: float, currency="ETH") -> Dict:
        if currency == "ETH" and amount <= self.balance_eth:
            self.balance_eth -= amount
            self.balance_usd = self.balance_eth * 2500
            tx = f"0x{random.randint(1000000000, 9999999999):x}"
            return {"success": True, "tx": tx, "amount": amount, "to": to}
        return {"success": False, "error": "insufficient funds"}
