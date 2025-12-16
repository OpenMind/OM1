# Wallet Providers

Modular wallet provider abstraction following the project structure pattern.

## Structure
```
src/providers/
├── base.py                  # Abstract provider interface
├── coinbase_provider.py     # Coinbase CDP wallet implementation
└── metamask_provider.py     # MetaMask signature verification
```

## Usage

### Coinbase Provider
```python
from providers.coinbase_provider import CoinbaseProvider

provider = CoinbaseProvider(wallet_id="your-wallet-id")
provider.init()  # Lazy-loads CDP SDK
provider.connect()
balance = provider.get_balance("eth")
```

### MetaMask Provider
```python
from providers.metamask_provider import MetaMaskProvider

# Verify signatures from frontend
is_valid = MetaMaskProvider.verify_signature(
    address="0x...",
    message="Sign this message",
    signature="0x..."
)
```

## Integration with Frontend

The TypeScript demo in `contrib/multi-wallet/` provides a reference implementation
for connecting to these providers via HTTP API.

## Tests

Run provider tests:
```bash
pytest tests/test_providers_coinbase.py
pytest tests/test_providers_metamask.py
```
