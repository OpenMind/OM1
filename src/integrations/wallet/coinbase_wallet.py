class CoinbaseWallet:
    """
    Mock Coinbase wallet for bounty demonstration.
    Real API can be plugged in later.
    """

    def pay(self, amount: float, currency: str = "USDC"):
        if amount <= 0:
            return {
                "status": "failed",
                "reason": "Invalid amount",
            }

        return {
            "status": "success",
            "amount": amount,
            "currency": currency,
            "tx_hash": "0xDEMO_TX_HASH",
        }
