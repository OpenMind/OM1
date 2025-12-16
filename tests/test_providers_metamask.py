from providers.metamask import MetaMaskProvider
from providers.metamask import MetaMaskProvider as MP


def test_metamask_provider_connect():
    p = MetaMaskProvider()
    p.init = lambda config=None: None  # skip heavy init
    assert p.connect() is True


def test_metamask_verify_signature(monkeypatch):
    # stub Account.recover_message to return same address
    monkeypatch.setattr(
        "providers.metamask.Account",
        type(
            "A",
            (),
            {
                "recover_message": staticmethod(
                    lambda msg, signature=None: "0xabcDEF0000000000000000000000000000000000"
                )
            },
        ),
    )
    addr = "0xabcDEF0000000000000000000000000000000000"
    message = "hello"
    signature = "0xdeadbeef"  # stub value (we only rely on recover_message above)
    assert MP.verify_signature(addr, message, signature) is True
