---
title: Coinbase Hackathon Feb 1. 2025
Description: "Coinbase Hackathon Quadruped Configuration"
---

The `Frenchie` quadruped has support for a Coinbase wallet. The dog's Sepolia Base Testnet account is `0xbB568962CF24d4CeBBf5d48308aCdAE873B93202`. You can check `Frenchie's` account balances using Coinbase's block explorer:

```
https://sepolia.basescan.org/address/0xbB568962CF24d4CeBBf5d48308aCdAE873B93202
```

To send Sepolia Base ETH to `Frenchie`, you can set up a Coinbase wallet on your phone or on [Google Chrome](https://chromewebstore.google.com/detail/coinbase-wallet-extension/hnfanknocfeofbddgcijnmhnfnkdnaad?pli=1). Then request Sepolia Base ETH from a friend or the [Coinbase developer faucet](https://docs.base.org/docs/tools/network-faucets/). Finally, send some Sepolia Base ETH to `Fenchie`, and see what she does!

## How it works

The agent tracks the balance of testnet ETH in a Coinbase wallet and sends a message when there is a new transaction. The agent can be instructed via the prompt to express appreciation for receiving tokens. See `/config/coinbase.json` for an example of how this is done:

```bash
"system_prompt": "
...
You like receiving ETH. If you receive an ETH transaction, show your appreciation through actions and speech.
...
4. If there is a new ETH transaction, you might:
	Move: 'shake paw'
	Speak: {{'sentence': 'Thank you I really appreciate the ETH you just sent.'}}
	Face: 'smile'
...",
```

The Coinbase wallet currently supports Base Sepolia and Base Mainnet networks. The Coinbase Wallet integration requires the following environment variables:

- `COINBASE_WALLET_ID`: The ID for the Coinbase Wallet.
- `COINBASE_API_KEY`: The API key for the Coinbase Project API.
- `COINBASE_API_SECRET`: The API secret for the Coinbase Project API.

The `API_KEY` and `API_SECRET` are generated from the [Coinbase Developer Portal](https://portal.cdp.coinbase.com) by navigating to the "API Keys" tab and then clicking "Create API Key". If you don't already have a Developer-Managed Wallet, you can create one by following [these instructions](https://docs.cdp.coinbase.com/mpc-wallet/docs/quickstart#creating-a-wallet) with the API key and secret you just created. Then, you can get a Wallet ID from the created wallet.

These keys are all strings and should look like this:

```bash
COINBASE_WALLET_ID="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
COINBASE_API_KEY="organizations/your-org-id/apiKeys/your-api-key-id"
COINBASE_API_SECRET="-----BEGIN EC PRIVATE KEY-----\nyour-api-key-private-key\n-----END EC PRIVATE KEY-----\n"
```

## Coinbase Hackathon Quadruped Configuration

Follow the instructions in `/robotics/unitree_go2_quadruped.md`. Then, run

```bash
uv run src/run.py cb_hackathon
```
