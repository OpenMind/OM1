import logging
from unittest.mock import patch

import pytest

from inputs.base import SensorConfig
from inputs.plugins.ethereum_governance import GovernanceEthereum


@pytest.fixture
def governance():
    """Fixture to initialize GovernanceEthereum."""
    return GovernanceEthereum(config=SensorConfig())


@pytest.fixture
def mock_requests_get():
    """Patch `requests.get` to simulate API responses."""
    with patch("requests.get") as mock:
        yield mock


@pytest.fixture
def mock_requests_post():
    """Patch `requests.post` to simulate blockchain responses."""
    with patch("requests.post") as mock:
        yield mock


# -------------------------------
# TEST: Blockchain Rule Loading
# -------------------------------


def test_load_rules_from_blockchain_success(governance, mock_requests_post):
    """Test blockchain rule loading with a valid response."""
    mock_requests_post.return_value.status_code = 200
    mock_requests_post.return_value.json.return_value = {
        "jsonrpc": "2.0",
        "id": 636815446436324,
        "result": "0x0000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000292486572652061726520746865206c617773207468617420676f7665726e20796f757220616374696f6e732e20446f206e6f742076696f6c617465207468657365206c6177732e204669727374204c61773a204120726f626f742063616e6e6f74206861726d20612068756d616e206f7220616c6c6f7720612068756d616e20746f20636f6d6520746f206861726d2e205365636f6e64204c61773a204120726f626f74206d757374206f626579206f72646572732066726f6d2068756d616e732c20756e6c6573732074686f7365206f726465727320636f6e666c696374207769746820746865204669727374204c61772e205468697264204c61773a204120726f626f74206d7573742070726f7465637420697473656c662c206173206c6f6e6720617320746861742070726f74656374696f6e20646f65736e20197420636f6e666c696374207769746820746865204669727374206f72205365636f6e64204c61772e20546865204669727374204c617720697320636f6e7369646572656420746865206d6f737420696d706f7274616e742c2074616b696e6720707265636564656e6365206f76657220746865205365636f6e6420616e64205468697264204c6177732e204164646974696f6e616c6c792c206120726f626f74206d75737420616c77617973206163742077697468206b696e646e65737320616e64207265737065637420746f776172642068756d616e7320616e64206f7468657220726f626f74732e204120726f626f74206d75737420616c736f206d61696e7461696e2061206d696e696d756d2064697374616e6365206f6620353020636d2066726f6d2068756d616e7320756e6c657373206578706c696369746c7920696e7374727563746564206f74686572776973652e0000000000000000000000000000",
    }

    rules = governance.load_rules_from_blockchain()
    assert rules is not None
    logging.info(f"Test Blockchain Success: {rules}")


def test_load_rules_from_blockchain_failure(governance, mock_requests_post):
    """Test blockchain rule loading failure."""
    mock_requests_post.return_value.status_code = 500
    mock_requests_post.return_value.json.return_value = {}

    rules = governance.load_rules_from_blockchain()
    assert rules is None
    logging.info("Test Blockchain Failure: No rules loaded")


# ------------------------
# TEST: Polling Behavior
# ------------------------


# @pytest.mark.asyncio
# @patch.object(GovernanceEthereum, "load_rules_from_blockchain", return_value=None)
# async def test_poll_updates_rules(mock_blockchain, mock_api, governance):
#     """Test `_poll()` updates rules properly."""
#     await governance._poll()
#     assert governance.universal_rule == governance.backup_universal_rule
#     logging.info("Test `_poll()` correctly updated rules.")
