import asyncio
from unittest.mock import patch

import pytest

from inputs.base import SensorConfig
from inputs.plugins.ethereum_governance import GovernanceEthereum


class MockResponse:
    """Mock aiohttp response."""

    def __init__(self, status: int, json_data: dict):
        self.status = status
        self._json_data = json_data

    async def json(self):
        return self._json_data


class MockClientSession:
    """Mock aiohttp.ClientSession."""

    def __init__(self, response: MockResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, *args, **kwargs):
        return MockPostContext(self._response)


class MockPostContext:
    """Mock async context manager for aiohttp post."""

    def __init__(self, response: MockResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def governance():
    return GovernanceEthereum(config=SensorConfig())


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_success(governance):
    mock_response = MockResponse(
        status=200,
        json_data={
            "jsonrpc": "2.0",
            "id": 636815446436324,
            "result": "0x0000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000292486572652061726520746865206c617773207468617420676f7665726e20796f757220616374696f6e732e20446f206e6f742076696f6c617465207468657365206c6177732e204669727374204c61773a204120726f626f742063616e6e6f74206861726d20612068756d616e206f7220616c6c6f7720612068756d616e20746f20636f6d6520746f206861726d2e205365636f6e64204c61773a204120726f626f74206d757374206f626579206f72646572732066726f6d2068756d616e732c20756e6c6573732074686f7365206f726465727320636f6e666c696374207769746820746865204669727374204c61772e205468697264204c61773a204120726f626f74206d7573742070726f7465637420697473656c662c206173206c6f6e6720617320746861742070726f74656374696f6e20646f65736e20197420636f6e666c696374207769746820746865204669727374206f72205365636f6e64204c61772e20546865204669727374204c617720697320636f6e7369646572656420746865206d6f737420696d706f7274616e742c2074616b696e6720707265636564656e6365206f76657220746865205365636f6e6420616e64205468697264204c6177732e204164646974696f6e616c6c792c206120726f626f74206d75737420616c77617973206163742077697468206b696e646e65737320616e64207265737065637420746f776172642068756d616e7320616e64206f7468657220726f626f74732e204120726f626f74206d75737420616c736f206d61696e7461696e2061206d696e696d756d2064697374616e6365206f6620353020636d2066726f6d2068756d616e7320756e6c657373206578706c696369746c7920696e7374727563746564206f74686572776973652e0000000000000000000000000000",
        },
    )

    with patch(
        "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
        return_value=MockClientSession(mock_response),
    ):
        rules = await governance.load_rules_from_blockchain()

    assert rules is not None
    assert "robot" in rules.lower()


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_failure(governance):
    mock_response = MockResponse(status=500, json_data={})

    with patch(
        "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
        return_value=MockClientSession(mock_response),
    ):
        rules = await governance.load_rules_from_blockchain()

    assert rules is None


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_empty_result(governance):
    mock_response = MockResponse(
        status=200,
        json_data={"jsonrpc": "2.0", "id": 1, "result": None},
    )

    with patch(
        "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
        return_value=MockClientSession(mock_response),
    ):
        rules = await governance.load_rules_from_blockchain()

    assert rules is None


@pytest.mark.asyncio
async def test_poll_returns_rules(governance):
    mock_response = MockResponse(
        status=200,
        json_data={
            "jsonrpc": "2.0",
            "id": 1,
            "result": "0x"
            + "0" * 64  # offset
            + "0" * 64  # padding
            + "0" * 64  # padding
            + "0000000000000000000000000000000000000000000000000000000000000005"  # length
            + "48656c6c6f"
            + "0" * 54,  # "Hello"
        },
    )

    with patch(
        "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
        return_value=MockClientSession(mock_response),
    ):
        governance.POLL_INTERVAL = 0.01  # Speed up test
        result = await governance._poll()

    assert result is not None
    assert "Hello" in result


def test_governance_initialization():
    governance = GovernanceEthereum(config=SensorConfig())

    assert governance.rpc_url == "https://holesky.drpc.org"
    assert governance.contract_address == "0xe706b7e30e378b89c7b2ee7bfd8ce2b91959d695"
    assert governance.function_selector == "0x1db3d5ff"
    assert governance.POLL_INTERVAL == 5.0
    assert governance.universal_rule is None


def test_decode_eth_response_valid():
    governance = GovernanceEthereum(config=SensorConfig())

    # Encoded "Hello" string
    hex_response = (
        "0x"
        + "0" * 64  # offset
        + "0" * 64  # padding
        + "0" * 64  # padding
        + "0000000000000000000000000000000000000000000000000000000000000005"  # length=5
        + "48656c6c6f"
        + "0" * 54  # "Hello" + padding
    )

    result = governance.decode_eth_response(hex_response)
    assert result == "Hello"


def test_decode_eth_response_invalid():
    governance = GovernanceEthereum(config=SensorConfig())

    result = governance.decode_eth_response("invalid_hex")
    assert result is None


class MockClientSessionWithDelay:
    """Mock aiohttp.ClientSession with configurable delay."""

    def __init__(self, delay: float = 0.5):
        self.delay = delay

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, *args, **kwargs):
        return MockPostContextWithDelay(self.delay)


class MockPostContextWithDelay:
    """Mock async context manager for aiohttp post with delay."""

    def __init__(self, delay: float):
        self.delay = delay

    async def __aenter__(self):
        await asyncio.sleep(self.delay)
        return MockResponse(
            status=200,
            json_data={
                "jsonrpc": "2.0",
                "id": 1,
                "result": "0x"
                + "0" * 64
                + "0" * 64
                + "0" * 64
                + "0000000000000000000000000000000000000000000000000000000000000005"
                + "48656c6c6f"
                + "0" * 54,
            },
        )

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_load_rules_is_non_blocking():
    import time

    results = []
    call_start = None
    call_end = None

    async def concurrent_task():
        nonlocal call_start, call_end
        for i in range(10):
            if call_start and call_end is None:
                results.append(time.time())
            await asyncio.sleep(0.1)

    async def governance_call():
        nonlocal call_start, call_end
        governance = GovernanceEthereum(config=SensorConfig())

        with patch(
            "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
            return_value=MockClientSessionWithDelay(delay=0.5),
        ):
            call_start = time.time()
            await governance.load_rules_from_blockchain()
            call_end = time.time()

    await asyncio.gather(governance_call(), concurrent_task())

    ticks_during_call = len(results)

    assert ticks_during_call >= 3, (
        f"Expected at least 3 concurrent ticks during HTTP call, got {ticks_during_call}. "
        f"This indicates the HTTP call is blocking the event loop."
    )


@pytest.mark.asyncio
async def test_poll_is_non_blocking():
    import time

    results = []
    poll_start = None
    poll_end = None

    async def concurrent_ticker():
        for i in range(20):
            if poll_start and poll_end is None:
                results.append(time.time())
            await asyncio.sleep(0.05)

    async def poll_call():
        nonlocal poll_start, poll_end
        governance = GovernanceEthereum(config=SensorConfig())

        with patch(
            "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
            return_value=MockClientSessionWithDelay(delay=0.5),
        ):
            governance.POLL_INTERVAL = 0.1
            poll_start = time.time()
            await governance._poll()
            poll_end = time.time()

    await asyncio.gather(poll_call(), concurrent_ticker())

    ticks_during_poll = len(results)

    assert ticks_during_poll >= 8, (
        f"Expected at least 8 ticks during _poll(), got {ticks_during_poll}. "
        f"This indicates _poll() is blocking the event loop."
    )


class MockClientSessionWithError:
    """Mock aiohttp.ClientSession that raises errors."""

    def __init__(self, error_type: str):
        self.error_type = error_type

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, *args, **kwargs):
        return MockPostContextWithError(self.error_type)


class MockPostContextWithError:
    """Mock context manager that raises errors."""

    def __init__(self, error_type: str):
        self.error_type = error_type

    async def __aenter__(self):
        import aiohttp

        if self.error_type == "client_error":
            raise aiohttp.ClientError("Connection failed")
        elif self.error_type == "timeout":
            raise asyncio.TimeoutError()
        elif self.error_type == "generic":
            raise Exception("Generic error")
        return MockResponse(status=200, json_data={})

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
async def test_load_rules_handles_client_error():
    """Test that load_rules_from_blockchain handles aiohttp.ClientError."""
    governance = GovernanceEthereum(config=SensorConfig())

    with patch(
        "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
        return_value=MockClientSessionWithError("client_error"),
    ):
        result = await governance.load_rules_from_blockchain()

    assert result is None


@pytest.mark.asyncio
async def test_load_rules_handles_timeout():
    """Test that load_rules_from_blockchain handles asyncio.TimeoutError."""
    governance = GovernanceEthereum(config=SensorConfig())

    with patch(
        "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
        return_value=MockClientSessionWithError("timeout"),
    ):
        result = await governance.load_rules_from_blockchain()

    assert result is None


@pytest.mark.asyncio
async def test_load_rules_handles_generic_error():
    """Test that load_rules_from_blockchain handles generic exceptions."""
    governance = GovernanceEthereum(config=SensorConfig())

    with patch(
        "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
        return_value=MockClientSessionWithError("generic"),
    ):
        result = await governance.load_rules_from_blockchain()

    assert result is None


@pytest.mark.asyncio
async def test_poll_handles_exception():
    """Test that _poll handles exceptions from load_rules_from_blockchain."""
    governance = GovernanceEthereum(config=SensorConfig())
    governance.POLL_INTERVAL = 0.01

    with patch.object(
        governance,
        "load_rules_from_blockchain",
        side_effect=Exception("Test error"),
    ):
        result = await governance._poll()

    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    """Test that _raw_to_text returns None when given None input."""
    governance = GovernanceEthereum(config=SensorConfig())

    result = await governance._raw_to_text(None)
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_input():
    """Test that _raw_to_text converts valid input to Message."""
    governance = GovernanceEthereum(config=SensorConfig())

    result = await governance._raw_to_text("Test rules")
    assert result is not None
    assert result.message == "Test rules"
    assert result.timestamp > 0


@pytest.mark.asyncio
async def test_raw_to_text_buffer_management():
    """Test that raw_to_text manages message buffer correctly."""
    governance = GovernanceEthereum(config=SensorConfig())

    await governance.raw_to_text("First rule")
    assert len(governance.messages) == 1
    assert governance.messages[0].message == "First rule"

    await governance.raw_to_text("First rule")
    assert len(governance.messages) == 1

    await governance.raw_to_text("Second rule")
    assert len(governance.messages) == 2
    assert governance.messages[1].message == "Second rule"


@pytest.mark.asyncio
async def test_raw_to_text_with_none_input():
    """Test that raw_to_text with None input does not modify messages."""
    governance = GovernanceEthereum(config=SensorConfig())

    await governance.raw_to_text(None)
    assert len(governance.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test that formatted_latest_buffer returns None when messages is empty."""
    governance = GovernanceEthereum(config=SensorConfig())

    result = governance.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_with_message():
    """Test that formatted_latest_buffer returns correctly formatted string."""
    from inputs.base import Message

    governance = GovernanceEthereum(config=SensorConfig())
    governance.messages = [Message(timestamp=12345.0, message="Test governance rule")]

    with patch.object(governance.io_provider, "add_input") as mock_add_input:
        result = governance.formatted_latest_buffer()

        assert result is not None
        assert "Universal Laws" in result
        assert "Test governance rule" in result
        assert "// START" in result
        assert "// END" in result
        mock_add_input.assert_called_once()


def test_decode_eth_response_too_short():
    """Test that decode_eth_response handles too-short hex responses gracefully."""
    governance = GovernanceEthereum(config=SensorConfig())

    # Hex response shorter than 128 bytes (required for string_length read at bytes 96-128)
    short_hex = "0x" + "00" * 50  # Only 50 bytes, less than required 128

    result = governance.decode_eth_response(short_hex)
    # Should return None or empty string, not raise IndexError
    assert result is None or result == ""


def test_decode_eth_response_with_control_characters():
    """Test that decode_eth_response correctly strips unwanted control characters."""
    governance = GovernanceEthereum(config=SensorConfig())

    # Build hex with control character \x19 embedded in "Hello\x19World"
    # String: "Hello\x19World" = 11 bytes
    # Hex: 48656c6c6f19576f726c64
    hex_response = (
        "0x"
        + "0" * 64  # offset
        + "0" * 64  # padding
        + "0" * 64  # padding
        + "000000000000000000000000000000000000000000000000000000000000000b"  # length=11
        + "48656c6c6f19576f726c64"  # "Hello\x19World"
        + "0" * 42  # padding to fill 32-byte slot
    )

    result = governance.decode_eth_response(hex_response)
    # Control character \x19 should be stripped
    assert result == "HelloWorld"
    assert "\x19" not in result


def test_decode_eth_response_without_0x_prefix():
    """Test that decode_eth_response works with hex strings missing '0x' prefix."""
    governance = GovernanceEthereum(config=SensorConfig())

    # Same as valid test but without "0x" prefix
    hex_response = (
        "0" * 64  # offset
        + "0" * 64  # padding
        + "0" * 64  # padding
        + "0000000000000000000000000000000000000000000000000000000000000005"  # length=5
        + "48656c6c6f"
        + "0" * 54  # "Hello" + padding
    )

    result = governance.decode_eth_response(hex_response)
    assert result == "Hello"


@pytest.mark.asyncio
async def test_load_rules_missing_result_key():
    """Test that load_rules_from_blockchain handles missing 'result' key in response."""
    governance = GovernanceEthereum(config=SensorConfig())

    mock_response = MockResponse(
        status=200,
        json_data={"jsonrpc": "2.0", "id": 1, "error": "Some error"},  # No "result" key
    )

    with patch(
        "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
        return_value=MockClientSession(mock_response),
    ):
        rules = await governance.load_rules_from_blockchain()

    assert rules is None


def test_formatted_latest_buffer_does_not_clear_messages():
    """Test that formatted_latest_buffer does not clear messages after formatting."""
    from inputs.base import Message

    governance = GovernanceEthereum(config=SensorConfig())
    governance.messages = [
        Message(timestamp=12345.0, message="First rule"),
        Message(timestamp=12346.0, message="Second rule"),
    ]

    with patch.object(governance.io_provider, "add_input"):
        governance.formatted_latest_buffer()

    assert len(governance.messages) == 2
    assert governance.messages[0].message == "First rule"
    assert governance.messages[1].message == "Second rule"
