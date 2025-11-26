#!/usr/bin/env python3
"""Create a real wallet for pizza payment demo"""

from eth_account import Account
import secrets

# Generate new wallet
priv = secrets.token_hex(32)
private_key = "0x" + priv
account = Account.from_key(private_key)

print("=" * 60)
print("🍕 DEMO PIZZA PAYMENT WALLET CREATED")
print("=" * 60)
print(f"\n📍 Wallet Address: {account.address}")
print(f"🔐 Private Key: {private_key}")
print("\n⚠️  IMPORTANT:")
print("   - This is a REAL wallet address on Ethereum blockchain")
print("   - Keep the private key SECRET (for demo only)")
print("   - Works on Sepolia + Mainnet + Scroll + Base")
print("\n💡 For Bounty Submission:")
print("   1. Use this wallet address in your backend (payment receiver)")
print("   2. Fund it with small testnet USDT/USDC or ETH if needed")
print("   3. Show confirmed pizza payments on block explorer")
print("   4. Include screenshots in PR")
print("=" * 60)
