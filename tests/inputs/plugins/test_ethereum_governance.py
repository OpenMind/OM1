import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest

from inputs.base import SensorConfig
from inputs.plugins.ethereum_governance import GovernanceEthereum, Message


class MockResponse:
    def __init__(self, status: int, json_data: dict):
        self.status = status
        self._json_data = json_data

    async def json(self):
        return self._json_data


class MockClientSession:
    def __init__(self, response: MockResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, *args, **kwargs):
        return MockPostContext(self._response)


class MockPostContext:
    def __init__(self, response: MockResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def governance():
    config = SensorConfig()
    with (
        patch("inputs.plugins.ethereum_governance.Web3"),
        patch("inputs.plugins.ethereum_governance.HTTPProvider"),
    ):
        instance = GovernanceEthereum(config=config)
        instance.contract = Mock()
    return instance


@pytest.mark.asyncio
async def test_poll_returns_rules(governance):
    expected_rules = "Hello World"
    mock_load_func = AsyncMock(return_value=expected_rules)
    with (
        patch.object(governance, "load_rules_from_blockchain", mock_load_func),
        patch("asyncio.sleep"),
    ):
        governance.POLL_INTERVAL = 0.01
        result = await governance._poll()

    assert result == expected_rules
    mock_load_func.assert_awaited_once()


@pytest.mark.asyncio
async def test_poll_handles_load_failure(governance):
    mock_load_func = AsyncMock(return_value=None)
    with (
        patch.object(governance, "load_rules_from_blockchain", mock_load_func),
        patch("asyncio.sleep"),
    ):
        governance.POLL_INTERVAL = 0.01
        result = await governance._poll()

    assert result is None
    mock_load_func.assert_awaited_once()


@pytest.mark.asyncio
async def test_poll_handles_exception_from_load(governance, caplog):
    mock_load_func = AsyncMock(side_effect=Exception("Test error"))
    with (
        patch.object(governance, "load_rules_from_blockchain", mock_load_func),
        caplog.at_level("ERROR"),
        patch("asyncio.sleep"),
    ):
        governance.POLL_INTERVAL = 0.01
        result = await governance._poll()

    assert result is None
    assert "Error fetching blockchain data" in caplog.text


@pytest.mark.asyncio
async def test_raw_to_text_with_none():
    governance = GovernanceEthereum(config=SensorConfig())
    with (
        patch("inputs.plugins.ethereum_governance.Web3"),
        patch("inputs.plugins.ethereum_governance.HTTPProvider"),
    ):
        result = await governance._raw_to_text(None)
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_with_valid_input():
    governance = GovernanceEthereum(config=SensorConfig())
    with (
        patch("inputs.plugins.ethereum_governance.Web3"),
        patch("inputs.plugins.ethereum_governance.HTTPProvider"),
    ):
        result = await governance._raw_to_text("Test rules")
    assert result is not None
    assert result.message == "Test rules"
    assert result.timestamp > 0


@pytest.mark.asyncio
async def test_raw_to_text_buffer_management(governance):
    await governance.raw_to_text("First rule")
    assert len(governance.messages) == 1
    assert governance.messages[0].message == "First rule"

    await governance.raw_to_text("First rule")
    assert len(governance.messages) == 1

    await governance.raw_to_text("Second rule")
    assert len(governance.messages) == 2
    assert governance.messages[1].message == "Second rule"


@pytest.mark.asyncio
async def test_raw_to_text_with_none_input(governance):
    initial_len = len(governance.messages)

    await governance.raw_to_text(None)
    assert len(governance.messages) == initial_len


def test_formatted_latest_buffer_empty(governance):
    governance.messages = []
    result = governance.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_with_message(governance):
    msg = Message(timestamp=12345.0, message="Test governance rule")
    governance.messages = [msg]

    with patch.object(governance.io_provider, "add_input") as mock_add_input:
        result = governance.formatted_latest_buffer()

        assert result is not None
        assert "Universal Laws" in result
        assert "Test governance rule" in result
        assert "// START" in result
        assert "// END" in result
        mock_add_input.assert_called_once_with(
            "Universal Laws", "Test governance rule", 12345.0
        )


def test_formatted_latest_buffer_does_not_clear_messages(governance):
    msg1 = Message(timestamp=12345.0, message="First rule")
    msg2 = Message(timestamp=12346.0, message="Second rule")
    governance.messages = [msg1, msg2]

    with patch.object(governance.io_provider, "add_input"):
        governance.formatted_latest_buffer()

    assert len(governance.messages) == 2
    assert governance.messages[0].message == "First rule"
    assert governance.messages[1].message == "Second rule"


def test_governance_initialization():
    config = SensorConfig()
    expected_address = "0xe706b7e30e378b89c7b2ee7bfd8ce2b91959d695"
    expected_checksummed_address = expected_address.upper()

    with (
        patch("inputs.plugins.ethereum_governance.Web3") as mock_w3_constructor,
        patch("inputs.plugins.ethereum_governance.HTTPProvider"),
    ):
        mock_w3_instance = Mock()
        mock_contract_instance = Mock()

        mock_w3_instance.to_checksum_address.return_value = expected_checksummed_address

        mock_w3_constructor.return_value = mock_w3_instance
        mock_w3_instance.eth.contract.return_value = mock_contract_instance

        governance = GovernanceEthereum(config=config)

        assert governance.rpc_url == "https://holesky.drpc.org"
        assert governance.contract_address == expected_address
        assert governance.POLL_INTERVAL == 5.0
        assert governance.universal_rule is None
        assert governance.rule_set_version == 2

        mock_w3_constructor.assert_called_once()
        mock_w3_instance.eth.contract.assert_called_once()
        call_kwargs = mock_w3_instance.eth.contract.call_args.kwargs
        assert call_kwargs.get("address") == expected_checksummed_address
        assert "abi" in call_kwargs
        assert governance.contract == mock_contract_instance

        mock_w3_instance.to_checksum_address.assert_called_once_with(expected_address)


@pytest.fixture
def mock_web3_components():
    with (
        patch("inputs.plugins.ethereum_governance.Web3") as mock_w3_constructor,
        patch("inputs.plugins.ethereum_governance.HTTPProvider"),
    ):
        mock_w3_instance = Mock()
        mock_contract_instance = Mock()
        mock_w3_constructor.return_value = mock_w3_instance
        mock_w3_instance.eth.contract.return_value = mock_contract_instance
        mock_w3_instance.to_checksum_address = lambda addr: addr.upper()

        yield {
            "w3_mock": mock_w3_instance,
            "contract_mock": mock_contract_instance,
            "w3_constructor": mock_w3_constructor,
        }


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_success(mock_web3_components):
    config = SensorConfig()
    expected_rules = "Example Governance Rules From Contract"
    expected_version = 2
    mock_web3_components[
        "contract_mock"
    ].functions.getRuleSet.return_value.call.return_value = expected_rules

    with patch("inputs.plugins.ethereum_governance.HTTPProvider"):
        governance = GovernanceEthereum(config=config)

    result = await governance.load_rules_from_blockchain()

    assert result == expected_rules
    mock_web3_components["contract_mock"].functions.getRuleSet.assert_called_once_with(
        expected_version
    )
    mock_web3_components[
        "contract_mock"
    ].functions.getRuleSet.return_value.call.assert_called_once()


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_handles_none_return(
    mock_web3_components, caplog
):
    config = SensorConfig()
    mock_web3_components[
        "contract_mock"
    ].functions.getRuleSet.return_value.call.return_value = None

    with (
        patch("inputs.plugins.ethereum_governance.HTTPProvider"),
        caplog.at_level(logging.WARNING),
    ):
        governance = GovernanceEthereum(config=config)

    with caplog.at_level(logging.WARNING):
        result = await governance.load_rules_from_blockchain()

    assert result is None
    assert "Contract function returned None." in caplog.text


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_handles_bad_function_call(
    mock_web3_components, caplog
):
    from web3.exceptions import BadFunctionCallOutput

    config = SensorConfig()
    error_msg = "Function does not exist or bad args"
    mock_web3_components[
        "contract_mock"
    ].functions.getRuleSet.return_value.call.side_effect = BadFunctionCallOutput(
        error_msg
    )

    with (
        patch("inputs.plugins.ethereum_governance.HTTPProvider"),
        caplog.at_level(logging.ERROR),
    ):
        governance = GovernanceEthereum(config=config)

    result = await governance.load_rules_from_blockchain()

    assert result is None
    assert "Blockchain function call failed (BadFunctionCallOutput)" in caplog.text


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_handles_contract_logic_error(
    mock_web3_components, caplog
):
    from web3.exceptions import ContractLogicError

    config = SensorConfig()
    error_msg = "Internal contract error"
    mock_web3_components[
        "contract_mock"
    ].functions.getRuleSet.return_value.call.side_effect = ContractLogicError(error_msg)

    with (
        patch("inputs.plugins.ethereum_governance.HTTPProvider"),
        caplog.at_level(logging.ERROR),
    ):
        governance = GovernanceEthereum(config=config)

    result = await governance.load_rules_from_blockchain()

    assert result is None
    assert "Smart contract logic error during call" in caplog.text


@pytest.mark.asyncio
async def test_load_rules_from_blockchain_handles_general_exception(
    mock_web3_components, caplog
):
    config = SensorConfig()
    error_msg = "Some other error occurred"
    mock_web3_components[
        "contract_mock"
    ].functions.getRuleSet.return_value.call.side_effect = Exception(error_msg)

    with (
        patch("inputs.plugins.ethereum_governance.HTTPProvider"),
        caplog.at_level(logging.ERROR),
    ):
        governance = GovernanceEthereum(config=config)

    result = await governance.load_rules_from_blockchain()

    assert result is None
    assert "General error calling blockchain function via web3.py" in caplog.text
