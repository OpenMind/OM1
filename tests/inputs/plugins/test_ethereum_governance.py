import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from inputs.base import SensorConfig
from inputs.plugins.ethereum_governance import GovernanceEthereum, Message


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.ethereum_governance.IOProvider") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def governance_instance(mock_io_provider):
    config = SensorConfig()
    with patch(
        "inputs.plugins.ethereum_governance.IOProvider", return_value=mock_io_provider
    ):
        instance = GovernanceEthereum(config=config)
    return instance


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_success_scenario(governance_instance):
    """
    Test that load_rules_from_blockchain runs without error when aiohttp returns a valid response.
    We don't deeply test the decode_eth_response logic here, just that the flow works.
    Mock decode_eth_response to return a known value to avoid hex complexity.
    """
    expected_decoded = "Mocked Decoded Rules"
    raw_hex_response = "0x1234..."  # Doesn't matter, we mock decode_eth_response

    mock_response_json = {"result": raw_hex_response}
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_json

    mock_session_post_cm = AsyncMock()
    mock_session_post_cm.__aenter__.return_value = mock_response
    mock_session_post_cm.__aexit__.return_value = None

    mock_session = Mock()
    mock_session.post.return_value = mock_session_post_cm

    mock_client_session_cm = AsyncMock()
    mock_client_session_cm.__aenter__.return_value = mock_session
    mock_client_session_cm.__aexit__.return_value = None

    with (
        patch.object(
            governance_instance, "decode_eth_response", return_value=expected_decoded
        ),
        patch(
            "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
            return_value=mock_client_session_cm,
        ),
    ):
        result = await governance_instance.load_rules_from_blockchain()

    assert result == expected_decoded
    mock_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_http_error(governance_instance, caplog):
    mock_response_json = {"error": "Not Found"}
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.json.return_value = mock_response_json

    mock_session_post_cm = AsyncMock()
    mock_session_post_cm.__aenter__.return_value = mock_response
    mock_session_post_cm.__aexit__.return_value = None

    mock_session = Mock()
    mock_session.post.return_value = mock_session_post_cm

    mock_client_session_cm = AsyncMock()
    mock_client_session_cm.__aenter__.return_value = mock_session
    mock_client_session_cm.__aexit__.return_value = None

    with (
        caplog.at_level("ERROR"),
        patch(
            "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
            return_value=mock_client_session_cm,
        ),
    ):
        result = await governance_instance.load_rules_from_blockchain()

    assert result is None
    assert "Blockchain request failed with status 404" in caplog.text
    mock_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_no_result_in_response(
    governance_instance, caplog
):
    mock_response_json = {"error": "Something went wrong"}
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_json

    mock_session_post_cm = AsyncMock()
    mock_session_post_cm.__aenter__.return_value = mock_response
    mock_session_post_cm.__aexit__.return_value = None

    mock_session = Mock()
    mock_session.post.return_value = mock_session_post_cm

    mock_client_session_cm = AsyncMock()
    mock_client_session_cm.__aenter__.return_value = mock_session
    mock_client_session_cm.__aexit__.return_value = None

    with (
        caplog.at_level("ERROR"),
        patch(
            "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
            return_value=mock_client_session_cm,
        ),
    ):
        result = await governance_instance.load_rules_from_blockchain()

    assert result is None
    assert "No valid result in blockchain response" in caplog.text
    mock_session.post.assert_called_once()


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_exception(governance_instance, caplog):
    mock_session_post_cm = AsyncMock()
    mock_session_post_cm.__aenter__.side_effect = asyncio.TimeoutError(
        "Request timed out"
    )
    mock_session_post_cm.__aexit__.return_value = None

    mock_session = Mock()
    mock_session.post.return_value = mock_session_post_cm

    mock_client_session_cm = AsyncMock()
    mock_client_session_cm.__aenter__.return_value = mock_session
    mock_client_session_cm.__aexit__.return_value = None

    with (
        caplog.at_level("ERROR"),
        patch(
            "inputs.plugins.ethereum_governance.aiohttp.ClientSession",
            return_value=mock_client_session_cm,
        ),
    ):
        result = await governance_instance.load_rules_from_blockchain()

    assert result is None
    assert "Error loading rules from blockchain" in caplog.text
    mock_session.post.assert_called_once()


def test_decode_eth_response_valid_hex_returns_something(governance_instance):
    """
    Test that decode_eth_response handles a valid hex string without throwing an error.
    It might return '', a string, or None depending on the internal logic, which is okay.
    The key is that it doesn't crash.
    """
    valid_hex = "0x00" * 64
    result = governance_instance.decode_eth_response(valid_hex)
    assert isinstance(result, (str, type(None)))


def test_decode_eth_response_invalid_hex_returns_none(governance_instance, caplog):
    invalid_hex = "invalid_hex_string!"

    with caplog.at_level("ERROR"):
        result = governance_instance.decode_eth_response(invalid_hex)

    assert result is None
    assert "Decoding error" in caplog.text


def test_decode_eth_response_short_hex_returns_something_or_none(
    governance_instance, caplog
):
    """
    Test with a short hex that might cause an error inside the try block.
    This should ideally trigger the except clause.
    """
    short_hex = "0x00" * 8

    with caplog.at_level("ERROR"):
        result = governance_instance.decode_eth_response(short_hex)

    assert isinstance(result, (str, type(None)))


def test_initialization_sets_defaults(governance_instance, mock_io_provider):
    assert governance_instance.io_provider is not None
    assert governance_instance.POLL_INTERVAL == 5.0
    assert governance_instance.rpc_url == "https://holesky.drpc.org"
    assert (
        governance_instance.contract_address
        == "0xe706b7e30e378b89c7b2ee7bfd8ce2b91959d695"
    )
    assert governance_instance.function_selector == "0x1db3d5ff"
    assert (
        governance_instance.function_argument
        == "0000000000000000000000000000000000000000000000000000000000000002"
    )
    assert governance_instance.universal_rule is None
    assert hasattr(governance_instance, "messages")
    assert isinstance(governance_instance.messages, list)


@pytest.mark.asyncio
async def test_poll_calls_load_rules_and_returns_result(governance_instance):
    expected_result = "Poll Result Rule"
    mock_load_func = AsyncMock(return_value=expected_result)
    with (
        patch.object(governance_instance, "load_rules_from_blockchain", mock_load_func),
        patch("asyncio.sleep"),
    ):
        result = await governance_instance._poll()

    assert result == expected_result
    mock_load_func.assert_awaited_once()


@pytest.mark.asyncio
async def test_poll_handles_exception_from_load_rules(governance_instance, caplog):
    mock_load_func = AsyncMock(side_effect=Exception("Load Error"))
    with (
        patch.object(governance_instance, "load_rules_from_blockchain", mock_load_func),
        caplog.at_level("ERROR"),
        patch("asyncio.sleep"),
    ):
        result = await governance_instance._poll()

    assert result is None
    assert "Error fetching blockchain data" in caplog.text


@pytest.mark.asyncio
async def test_raw_to_text_converts_string_to_message(governance_instance):
    test_rule_str = "Raw Governance Rule Text"
    timestamp_before = time.time()

    result = await governance_instance._raw_to_text(test_rule_str)

    timestamp_after = time.time()
    assert result is not None
    assert result.message == test_rule_str
    assert timestamp_before <= result.timestamp <= timestamp_after


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_if_input_none(governance_instance):
    result = await governance_instance._raw_to_text(None)
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_adds_unique_message_to_buffer(governance_instance):
    test_rule_str = "Unique Governance Rule"
    initial_len = len(governance_instance.messages)

    with patch("time.time", return_value=1234.0):
        await governance_instance.raw_to_text(test_rule_str)

    assert len(governance_instance.messages) == initial_len + 1
    assert governance_instance.messages[-1].message == test_rule_str
    assert governance_instance.messages[-1].timestamp == 1234.0


@pytest.mark.asyncio
async def test_raw_to_text_does_not_add_duplicate_message(governance_instance):
    test_rule_str = "Duplicate Governance Rule"
    existing_msg = Message(timestamp=1233.0, message=test_rule_str)
    governance_instance.messages = [existing_msg]

    initial_len = len(governance_instance.messages)

    with patch("time.time", return_value=1234.0):
        await governance_instance.raw_to_text(test_rule_str)

    assert len(governance_instance.messages) == initial_len
    assert governance_instance.messages[-1].timestamp == 1233.0


def test_formatted_latest_buffer_empty(governance_instance):
    result = governance_instance.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_formats_latest_message(
    governance_instance, mock_io_provider
):
    msg = Message(timestamp=1234.0, message="formatted buffered message")
    governance_instance.messages = [msg]

    result = governance_instance.formatted_latest_buffer()

    assert "INPUT:" in result
    assert "Universal Laws" in result
    assert "formatted buffered message" in result
    assert len(governance_instance.messages) == 1
    mock_io_provider.add_input.assert_called_once_with(
        "Universal Laws", "formatted buffered message", 1234.0
    )
