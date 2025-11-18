from src.plugins.wallet.wallet_plugin import WalletPlugin
import time

w = WalletPlugin()
print("Bounty #367 – Smart Assistant + Wallet Payments Demo (mock mode)\n")
print(f"Initial balance: {w.get_balance()['eth']} ETH = {w.get_balance()['usd']} USD\n")

time.sleep(3)
print('Voice command received: "Alexa, pay 0.05 ETH to Lau90eth"')
result = w.send_payment(to="Lau90eth", amount=0.05)
print(f"Transaction successful!")
print(f"Tx hash: {result['tx']}")
print(f"Amount sent: {result['amount']} ETH to {result['to']}\n")

time.sleep(3)
print(f"Final balance: {w.get_balance()['eth']} ETH = {w.get_balance()['usd']} USD")
print("Payment completed – ready for real wallet integration")
