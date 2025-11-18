import unittest
from src.plugins.wallet.wallet_plugin import WalletPlugin

class TestWallet(unittest.TestCase):
    def setUp(self):
        self.wallet = WalletPlugin()

    def test_get_balance(self):
        balance = self.wallet.get_balance()
        self.assertIn("eth", balance)
        self.assertIn("usd", balance)
        self.assertGreater(balance["eth"], 0)

    def test_send_payment_success(self):
        result = self.wallet.send_payment(to="Lau90eth", amount=0.05)
        self.assertTrue(result["success"])
        self.assertIn("tx", result)

    def test_send_payment_insufficient(self):
        result = self.wallet.send_payment(to="Lau90eth", amount=1000)
        self.assertFalse(result["success"])

if __name__ == '__main__':
    unittest.main(verbosity=2)
