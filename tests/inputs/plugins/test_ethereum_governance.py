import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from web3.exceptions import BadFunctionCallOutput, ContractLogicError

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
        with patch("inputs.plugins.ethereum_governance.Web3") as mock_web3:
            mock_w3_instance = Mock()
            mock_web3.return_value = mock_w3_instance

            mock_contract = Mock()
            mock_w3_instance.eth.contract.return_value = mock_contract
            mock_w3_instance.to_checksum_address.return_value = (
                "0xe706b7e30e378b89c7b2ee7bfd8ce2b91959d695"
            )

            instance = GovernanceEthereum(config=config)
            instance.contract = mock_contract

    return instance


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_success_scenario(governance_instance):
    expected_decoded = "Rule 1: Test rule\nRule 2: Another rule"
    governance_instance.contract.functions.getRuleSet.return_value.call.return_value = (
        expected_decoded
    )

    result = await governance_instance.load_rules_from_blockchain()

    assert result == expected_decoded
    governance_instance.contract.functions.getRuleSet.assert_called_once_with(2)


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_with_control_characters(governance_instance):
    raw_rules = "Rule 1\x19: Test\x00 rule\nRule 2: Another rule"
    expected_cleaned = "Rule 1: Test rule\nRule 2: Another rule"
    governance_instance.contract.functions.getRuleSet.return_value.call.return_value = (
        raw_rules
    )

    result = await governance_instance.load_rules_from_blockchain()

    assert result == expected_cleaned


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_http_error(governance_instance, caplog):
    governance_instance.contract.functions.getRuleSet.return_value.call.side_effect = (
        BadFunctionCallOutput("Call failed")
    )

    with caplog.at_level("ERROR"):
        result = await governance_instance.load_rules_from_blockchain()

    assert result is None
    assert "Blockchain function call failed" in caplog.text


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_no_result_in_response(
    governance_instance, caplog
):
    governance_instance.contract.functions.getRuleSet.return_value.call.return_value = (
        None
    )

    with caplog.at_level("WARNING"):
        result = await governance_instance.load_rules_from_blockchain()

    assert result is None
    assert "Contract function returned None" in caplog.text


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_exception(governance_instance, caplog):
    governance_instance.contract.functions.getRuleSet.return_value.call.side_effect = (
        ContractLogicError("Contract error")
    )

    with caplog.at_level("ERROR"):
        result = await governance_instance.load_rules_from_blockchain()

    assert result is None
    assert "Smart contract logic error" in caplog.text


@pytest.mark.asyncio
async def test_load_rules_handles_generic_error(governance_instance, caplog):
    governance_instance.contract.functions.getRuleSet.return_value.call.side_effect = (
        Exception("Network error")
    )

    with caplog.at_level("ERROR"):
        result = await governance_instance.load_rules_from_blockchain()

    assert result is None
    assert "General error calling blockchain function" in caplog.text


@pytest.mark.asyncio
async def test_load_rules_with_tabs_and_newlines_preserved(governance_instance):
    raw_rules = "Rule 1:\tIndented\nRule 2:\r\nNew line"
    expected_cleaned = "Rule 1:\tIndented\nRule 2:\r\nNew line"
    governance_instance.contract.functions.getRuleSet.return_value.call.return_value = (
        raw_rules
    )

    result = await governance_instance.load_rules_from_blockchain()

    assert result == expected_cleaned
    assert "\t" in result
    assert "\n" in result


def test_initialization_sets_defaults(governance_instance, mock_io_provider):
    assert governance_instance.io_provider is not None
    assert governance_instance.POLL_INTERVAL == 5.0
    assert governance_instance.rpc_url == "https://holesky.drpc.org"
    assert (
        governance_instance.contract_address
        == "0xe706b7e30e378b89c7b2ee7bfd8ce2b91959d695"
    )
    assert governance_instance.rule_set_version == 2
    assert governance_instance.universal_rule is None
    assert hasattr(governance_instance, "messages")
    assert isinstance(governance_instance.messages, list)


def test_initialization_contract_error_raises_exception():
    config = SensorConfig()

    with patch("inputs.plugins.ethereum_governance.IOProvider"):
        with patch("inputs.plugins.ethereum_governance.Web3") as mock_web3:
            mock_w3_instance = Mock()
            mock_web3.return_value = mock_w3_instance
            mock_w3_instance.to_checksum_address.return_value = (
                "0xe706b7e30e378b89c7b2ee7bfd8ce2b91959d695"
            )
            mock_w3_instance.eth.contract.side_effect = Exception(
                "Contract init failed"
            )

            with pytest.raises(Exception, match="Contract init failed"):
                GovernanceEthereum(config=config)


def test_rule_set_estimator(governance_instance):
    assert governance_instance.rule_set_estimator() == 2
    governance_instance.rule_set_version = 5
    assert governance_instance.rule_set_estimator() == 5


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


@pytest.mark.asyncio
async def test_raw_to_text_adds_first_message_to_empty_buffer(governance_instance):
    test_rule_str = "First Governance Rule"
    governance_instance.messages = []

    with patch("time.time", return_value=1234.0):
        await governance_instance.raw_to_text(test_rule_str)

    assert len(governance_instance.messages) == 1
    assert governance_instance.messages[0].message == test_rule_str
    assert governance_instance.messages[0].timestamp == 1234.0


@pytest.mark.asyncio
async def test_raw_to_text_adds_different_message_to_existing_buffer(
    governance_instance,
):
    first_message = Message(timestamp=1000.0, message="First Rule")
    governance_instance.messages = [first_message]
    different_rule_str = "Second Different Rule"

    with patch("time.time", return_value=2000.0):
        await governance_instance.raw_to_text(different_rule_str)

    assert len(governance_instance.messages) == 2
    assert governance_instance.messages[0].message == "First Rule"
    assert governance_instance.messages[0].timestamp == 1000.0
    assert governance_instance.messages[1].message == "Second Different Rule"
    assert governance_instance.messages[1].timestamp == 2000.0


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


def test_formatted_latest_buffer_with_multiple_messages(
    governance_instance, mock_io_provider
):
    msg1 = Message(timestamp=1000.0, message="old message")
    msg2 = Message(timestamp=2000.0, message="newer message")
    msg3 = Message(timestamp=3000.0, message="latest message")
    governance_instance.messages = [msg1, msg2, msg3]

    result = governance_instance.formatted_latest_buffer()

    assert "INPUT:" in result
    assert "Universal Laws" in result
    assert "latest message" in result
    assert "old message" not in result
    assert "newer message" not in result
    assert len(governance_instance.messages) == 3
    mock_io_provider.add_input.assert_called_once_with(
        "Universal Laws", "latest message", 3000.0
    )


@pytest.mark.asyncio
async def test_full_integration_flow(governance_instance, mock_io_provider):
    expected_rules = "Rule A: Test\nRule B: Another"
    governance_instance.contract.functions.getRuleSet.return_value.call.return_value = (
        expected_rules
    )

    with patch("asyncio.sleep"):
        raw_result = await governance_instance._poll()
    assert raw_result == expected_rules

    with patch("time.time", return_value=5000.0):
        await governance_instance.raw_to_text(raw_result)
    assert len(governance_instance.messages) == 1
    assert governance_instance.messages[0].message == expected_rules

    formatted = governance_instance.formatted_latest_buffer()
    assert formatted is not None
    assert "Universal Laws" in formatted
    assert expected_rules in formatted
    mock_io_provider.add_input.assert_called_once_with(
        "Universal Laws", expected_rules, 5000.0
    )
