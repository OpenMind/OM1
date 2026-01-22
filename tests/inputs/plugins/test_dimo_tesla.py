import sys
from unittest.mock import Mock, patch

import pytest

if "dimo" not in sys.modules:
    sys.modules["dimo"] = Mock()
    sys.modules["dimo.DIMO"] = Mock()
    sys.modules["dimo.auth"] = Mock()
    sys.modules["dimo.token_exchange"] = Mock()
    sys.modules["dimo.query"] = Mock()

from inputs.plugins.dimo_tesla import DIMOTesla, DIMOTeslaConfig


@pytest.fixture
def mock_dimo_lib():
    with patch.dict("sys.modules", sys.modules):
        mock_dimo_cls = sys.modules["dimo.DIMO"]
        mock_dimo_instance = Mock()
        mock_dimo_cls.return_value = mock_dimo_instance

        mock_auth = Mock()
        mock_token_exchange = Mock()
        mock_dimo_instance.auth = mock_auth
        mock_dimo_instance.token_exchange = mock_token_exchange

        mock_auth.get_dev_jwt = Mock(return_value={"access_token": "fake_dev_jwt"})
        mock_token_exchange.exchange = Mock(return_value={"token": "fake_vehicle_jwt"})

        mock_dimo_instance.query = Mock()

        yield {
            "cls": mock_dimo_cls,
            "instance": mock_dimo_instance,
            "auth": mock_auth,
            "token_exchange": mock_token_exchange,
            "query": mock_dimo_instance.query,
        }


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.dimo_tesla.IOProvider") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def dimo_tesla_instance_with_creds(mock_dimo_lib, mock_io_provider):
    config = DIMOTeslaConfig(
        client_id="test_client_id",
        domain="test_domain.com",
        private_key="test_private_key",
        token_id=12345,
    )
    with (
        patch("inputs.plugins.dimo_tesla.DIMO", mock_dimo_lib["cls"]),
        patch("inputs.plugins.dimo_tesla.IOProvider", return_value=mock_io_provider),
    ):
        instance = DIMOTesla(config=config)
    return instance


@pytest.fixture
def dimo_tesla_instance_without_creds(mock_io_provider):
    config = DIMOTeslaConfig()
    with patch("inputs.plugins.dimo_tesla.IOProvider", return_value=mock_io_provider):
        instance = DIMOTesla(config=config)
    return instance


def test_initialization_creates_providers_and_buffers(dimo_tesla_instance_with_creds):
    assert dimo_tesla_instance_with_creds.io_provider is not None
    assert hasattr(dimo_tesla_instance_with_creds, "messages")
    assert isinstance(dimo_tesla_instance_with_creds.messages, list)
    assert hasattr(dimo_tesla_instance_with_creds, "message_buffer")
    assert dimo_tesla_instance_with_creds.token_id == 12345
    assert dimo_tesla_instance_with_creds.vehicle_jwt is not None


def test_initialization_aborts_without_credentials(caplog, mock_io_provider):
    config = DIMOTeslaConfig()
    with caplog.at_level("INFO"):
        with patch(
            "inputs.plugins.dimo_tesla.IOProvider", return_value=mock_io_provider
        ):
            instance = DIMOTesla(config=config)
    assert "You did not provide credentials" in caplog.text
    assert instance.vehicle_jwt is None


def test_initialization_aborts_without_token_id(
    caplog, mock_io_provider, mock_dimo_lib
):
    config = DIMOTeslaConfig(
        client_id="test_client_id",
        domain="test_domain.com",
        private_key="test_private_key",
    )
    with caplog.at_level("INFO"):
        with (
            patch("inputs.plugins.dimo_tesla.DIMO", mock_dimo_lib["cls"]),
            patch(
                "inputs.plugins.dimo_tesla.IOProvider", return_value=mock_io_provider
            ),
        ):
            instance = DIMOTesla(config=config)
    assert "You did not provide a token_id" in caplog.text
    assert instance.vehicle_jwt is None


@pytest.mark.asyncio
async def test_poll_returns_none_if_no_vehicle_jwt(dimo_tesla_instance_without_creds):
    dimo_tesla_instance_without_creds.vehicle_jwt = None
    result = await dimo_tesla_instance_without_creds._poll()
    assert result is None


@pytest.mark.asyncio
async def test_poll_refreshes_expired_token_and_queries_api(
    dimo_tesla_instance_with_creds, mock_dimo_lib
):
    import time

    dimo_tesla_instance_with_creds.vehicle_jwt = "old_jwt"
    dimo_tesla_instance_with_creds.vehicle_jwt_expires = time.time() - 1
    dimo_tesla_instance_with_creds.dev_jwt = "existing_dev_jwt"

    mock_dimo_lib["token_exchange"].exchange.reset_mock()

    mock_dimo_lib["token_exchange"].exchange.return_value = {"token": "refreshed_jwt"}

    mock_query_response = {
        "data": {
            "signalsLatest": {
                "powertrainTransmissionTravelledDistance": {
                    "timestamp": "t",
                    "value": 15000,
                },
                "exteriorAirTemperature": {"timestamp": "t", "value": 22.5},
                "speed": {"timestamp": "t", "value": 0},
                "powertrainRange": {"timestamp": "t", "value": 350},
                "currentLocationLatitude": {"timestamp": "t", "value": 37.7749},
                "currentLocationLongitude": {"timestamp": "t", "value": -122.4194},
            }
        }
    }
    mock_dimo_lib["query"].return_value = mock_query_response

    result = await dimo_tesla_instance_with_creds._poll()

    mock_dimo_lib["token_exchange"].exchange.assert_called_once_with(
        developer_jwt="existing_dev_jwt", token_id=12345
    )
    mock_dimo_lib["query"].assert_called_once()
    assert "15000" in result
    assert dimo_tesla_instance_with_creds.vehicle_jwt == "refreshed_jwt"


@pytest.mark.asyncio
async def test_poll_queries_api_and_returns_data_string(
    dimo_tesla_instance_with_creds, mock_dimo_lib
):
    import time

    dimo_tesla_instance_with_creds.vehicle_jwt = "valid_jwt"
    dimo_tesla_instance_with_creds.vehicle_jwt_expires = time.time() + 3600

    mock_dimo_lib["token_exchange"].exchange.reset_mock()

    mock_query_response = {
        "data": {
            "signalsLatest": {
                "powertrainTransmissionTravelledDistance": {
                    "timestamp": "t",
                    "value": 20000,
                },
                "exteriorAirTemperature": {"timestamp": "t", "value": 18.0},
                "speed": {"timestamp": "t", "value": 45},
                "powertrainRange": {"timestamp": "t", "value": 300},
                "currentLocationLatitude": {"timestamp": "t", "value": 40.7128},
                "currentLocationLongitude": {"timestamp": "t", "value": -74.0060},
            }
        }
    }
    mock_dimo_lib["query"].return_value = mock_query_response

    result = await dimo_tesla_instance_with_creds._poll()

    mock_dimo_lib["token_exchange"].exchange.assert_not_called()
    mock_dimo_lib["query"].assert_called_once()
    assert "20000" in result


@pytest.mark.asyncio
async def test_poll_handles_query_error_gracefully(
    dimo_tesla_instance_with_creds, mock_dimo_lib
):
    import time

    dimo_tesla_instance_with_creds.vehicle_jwt = "valid_jwt"
    dimo_tesla_instance_with_creds.vehicle_jwt_expires = time.time() + 3600

    mock_query_response = {}  # Invalid structure for parsing
    mock_dimo_lib["query"].return_value = mock_query_response

    result = await dimo_tesla_instance_with_creds._poll()

    assert result is None
    mock_dimo_lib["query"].assert_called_once()


@pytest.mark.asyncio
async def test_raw_to_text_converts_string_to_message(dimo_tesla_instance_with_creds):
    import time

    test_data_str = "Sample Tesla Data String"
    timestamp_before = time.time()

    result = await dimo_tesla_instance_with_creds._raw_to_text(test_data_str)

    timestamp_after = time.time()
    assert result is not None
    assert result.message == test_data_str
    assert timestamp_before <= result.timestamp <= timestamp_after


@pytest.mark.asyncio
async def test_raw_to_text_returns_none_if_input_none(dimo_tesla_instance_with_creds):
    result = await dimo_tesla_instance_with_creds._raw_to_text(None)
    assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_adds_unique_message_to_buffer(
    dimo_tesla_instance_with_creds,
):
    test_data_str = "Unique Tesla Data"
    initial_len = len(dimo_tesla_instance_with_creds.messages)

    with patch("time.time", return_value=1234.0):
        await dimo_tesla_instance_with_creds.raw_to_text(test_data_str)

    assert len(dimo_tesla_instance_with_creds.messages) == initial_len + 1
    assert dimo_tesla_instance_with_creds.messages[-1].message == test_data_str
    assert dimo_tesla_instance_with_creds.messages[-1].timestamp == 1234.0


@pytest.mark.asyncio
async def test_raw_to_text_does_not_add_duplicate_message(
    dimo_tesla_instance_with_creds,
):
    from inputs.plugins.dimo_tesla import Message

    test_data_str = "Duplicate Tesla Data"
    existing_msg = Message(timestamp=1233.0, message=test_data_str)
    dimo_tesla_instance_with_creds.messages = [existing_msg]

    initial_len = len(dimo_tesla_instance_with_creds.messages)

    with patch("time.time", return_value=1234.0):
        await dimo_tesla_instance_with_creds.raw_to_text(test_data_str)

    assert len(dimo_tesla_instance_with_creds.messages) == initial_len
    assert dimo_tesla_instance_with_creds.messages[-1].timestamp == 1233.0


def test_formatted_latest_buffer_empty(dimo_tesla_instance_with_creds):
    result = dimo_tesla_instance_with_creds.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_formats_and_clears_latest_message(
    dimo_tesla_instance_with_creds, mock_io_provider
):
    from inputs.plugins.dimo_tesla import Message

    msg = Message(timestamp=1234.0, message="buffered message")
    dimo_tesla_instance_with_creds.messages = [msg]

    result = dimo_tesla_instance_with_creds.formatted_latest_buffer()

    assert "INPUT:" in result
    assert "buffered message" in result
    mock_io_provider.add_input.assert_called_once_with(
        "Tesla Data", "buffered message", 1234.0
    )
    assert len(dimo_tesla_instance_with_creds.messages) == 1
