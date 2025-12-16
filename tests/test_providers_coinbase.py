from src.providers.coinbase_provider import CoinbaseProvider


class DummyWallet:
    @staticmethod
    def fetch(_id):
        return type("W", (), {"balance": staticmethod(lambda asset: "0.12345")})()


def test_coinbase_provider_connect(monkeypatch):
    # create provider instance and simulate that init() already configured SDK
    p = CoinbaseProvider(wallet_id="dummy")
    p.init = lambda config=None: None  # keep noop for safety
    p._configured = True  # simulate successful init()
    p._Wallet = DummyWallet  # provide Wallet class used by connect()
    assert p.connect() is True
    assert p.get_balance("eth") == 0.12345
