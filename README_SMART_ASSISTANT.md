# Smart Assistant + Wallet Payments Integration

> Bounty #367 Submission - OM1 Framework

## Implementation Summary

This project enables OM1 to:

1. **Communicate with Home Assistant** via REST API to trigger actions and scripts
2. **Process crypto payments** through Web3 (ETH/USDC on Base Sepolia)
3. **Report transaction status** with TX hash and blockchain explorer link

## Demonstrated Workflow

```
User Request → OM1 → Home Assistant API → Payment Transaction → Confirmation
```

**Example**: When a user requests an order, OM1 triggers a Home Assistant script and processes the associated payment, returning the transaction hash for verification.

---

## Technical Implementation

### Components

| Component | Technology |
|-----------|------------|
| Smart Assistant | Home Assistant (REST API) |
| Crypto Wallet | Web3.py + Base Sepolia |
| LLM | OpenMind API |
| Input | Console (extensible to voice) |

### Actions

- `iot_control` - Triggers Home Assistant services and scripts
- `process_payment` - Executes ETH/USDC transactions with address validation
- `speak` - Provides user feedback

### Payment Features

- Send to Ethereum addresses or ENS names
- Balance validation before transactions
- Gas estimation and safety checks
- Transaction confirmation with explorer link

---

## Setup

```bash
git clone https://github.com/cutepawss/OM1.git
cd OM1
uv sync
cp env.example .env
# Configure: OM_API_KEY, ETH_PRIVATE_KEY, HOME_ASSISTANT_URL, HOME_ASSISTANT_TOKEN
python src/run.py smart_assistant
```

---

## Files

| Path | Description |
|------|-------------|
| `config/smart_assistant.json5` | Agent configuration |
| `src/actions/process_payment/` | Payment logic |
| `src/actions/iot_control/` | Home Assistant connector |

---

## Bounty Submission

- **Home Assistant Integrated**: Home Assistant (Docker)  
- **Post Link**: https://x.com/minion_btc/status/2011567224855630114  
- **Demo Video**: https://www.youtube.com/watch?v=O5WX-LpWhLY  
- **Notes**: Real blockchain transactions, real HA API calls, documented setup
