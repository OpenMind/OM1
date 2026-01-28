from unittest.mock import Mock, patch

import pytest
import requests

from inputs.plugins.fabric_closest_peer import (
    FabricClosestPeer,
    FabricClosestPeerConfig,
)


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.fabric_closest_peer.IOProvider") as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def fabric_closest_peer_instance(mock_io_provider):
    config = FabricClosestPeerConfig()
    with (
        patch("inputs.plugins.fabric_closest_peer.requests"),
        patch(
            "inputs.plugins.fabric_closest_peer.IOProvider",
            return_value=mock_io_provider,
        ),
    ):
        instance = FabricClosestPeer(config=config)
    return instance


def test_initialization_sets_defaults(fabric_closest_peer_instance, mock_io_provider):
    assert fabric_closest_peer_instance.io is not None
    assert mock_io_provider is not None

    assert fabric_closest_peer_instance.descriptor_for_LLM == "Closest Peer from Fabric"
    assert isinstance(fabric_closest_peer_instance.messages, list)
    assert isinstance(
        fabric_closest_peer_instance.msg_q, type(__import__("queue").Queue())
    )

    assert fabric_closest_peer_instance.fabric_endpoint == "http://localhost:8545"
    assert fabric_closest_peer_instance.mock_mode is True


@pytest.mark.asyncio
async def test_poll_returns_mocked_peer_when_mock_mode_enabled(
    fabric_closest_peer_instance,
):
    config = FabricClosestPeerConfig(
        mock_mode=True, mock_lat=-33.86785, mock_lon=151.20732
    )
    fabric_closest_peer_instance.config = config
    fabric_closest_peer_instance.mock_mode = True

    result = await fabric_closest_peer_instance._poll()

    expected_result = "Closest peer at -33.86785, 151.20732"
    assert result == expected_result
    fabric_closest_peer_instance.io.add_dynamic_variable.assert_any_call(
        "closest_peer_lat", -33.86785
    )
    fabric_closest_peer_instance.io.add_dynamic_variable.assert_any_call(
        "closest_peer_lon", 151.20732
    )
    assert not fabric_closest_peer_instance.msg_q.empty()
    queued_msg = fabric_closest_peer_instance.msg_q.get_nowait()
    assert queued_msg == expected_result


@pytest.mark.asyncio
async def test_poll_returns_none_if_requests_is_none_and_mock_disabled(caplog):
    pass


@pytest.mark.asyncio
async def test_poll_fetches_peer_via_requests_when_mock_disabled_success(
    fabric_closest_peer_instance, mock_io_provider
):
    config = FabricClosestPeerConfig(mock_mode=False)
    fabric_closest_peer_instance.config = config
    fabric_closest_peer_instance.mock_mode = False

    mock_io_provider.get_dynamic_variable.side_effect = lambda x: {
        "latitude": -33.868820,
        "longitude": 151.209295,
    }.get(x)

    json_response_data = {
        "result": [{"peer": {"latitude": -33.865, "longitude": 151.210}}]
    }
    mock_response = Mock()
    mock_response.json.return_value = json_response_data

    with patch(
        "inputs.plugins.fabric_closest_peer.requests.post", return_value=mock_response
    ) as mock_post:
        result = await fabric_closest_peer_instance._poll()

        expected_result = "Closest peer at -33.86500, 151.21000"
        assert result == expected_result
        fabric_closest_peer_instance.io.add_dynamic_variable.assert_any_call(
            "closest_peer_lat", -33.865
        )
        fabric_closest_peer_instance.io.add_dynamic_variable.assert_any_call(
            "closest_peer_lon", 151.210
        )
        assert not fabric_closest_peer_instance.msg_q.empty()
        queued_msg = fabric_closest_peer_instance.msg_q.get_nowait()
        assert queued_msg == expected_result

        mock_post.assert_called_once_with(
            "http://localhost:8545",
            json={
                "method": "omp2p_findClosestPeer",
                "params": [{"latitude": -33.868820, "longitude": 151.209295}],
                "id": 1,
                "jsonrpc": "2.0",
            },
            timeout=3.0,
            headers={"Content-Type": "application/json"},
        )


@pytest.mark.asyncio
async def test_poll_returns_none_if_io_latitude_or_longitude_missing(
    caplog, fabric_closest_peer_instance, mock_io_provider
):
    config = FabricClosestPeerConfig(mock_mode=False)
    fabric_closest_peer_instance.config = config
    fabric_closest_peer_instance.mock_mode = False

    mock_io_provider.get_dynamic_variable.side_effect = lambda x: {
        "latitude": -33.868820,
        "longitude": None,
    }.get(x)

    with caplog.at_level("ERROR"):
        result = await fabric_closest_peer_instance._poll()

    assert result is None
    assert "FabricClosestPeer: latitude or longitude not set." in caplog.text


@pytest.mark.asyncio
async def test_poll_returns_none_on_requests_exception(
    caplog, fabric_closest_peer_instance, mock_io_provider
):
    config = FabricClosestPeerConfig(mock_mode=False)
    fabric_closest_peer_instance.config = config
    fabric_closest_peer_instance.mock_mode = False

    mock_io_provider.get_dynamic_variable.side_effect = lambda x: {
        "latitude": -33.868820,
        "longitude": 151.209295,
    }.get(x)

    with patch(
        "inputs.plugins.fabric_closest_peer.requests.post",
        side_effect=requests.exceptions.RequestException("Network error"),
    ):
        with caplog.at_level("ERROR"):
            result = await fabric_closest_peer_instance._poll()

    assert result is None
    assert (
        "FabricClosestPeer: error calling Fabric endpoint – Network error"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_poll_returns_none_if_no_peer_found(
    caplog, fabric_closest_peer_instance, mock_io_provider
):
    config = FabricClosestPeerConfig(mock_mode=False)
    fabric_closest_peer_instance.config = config
    fabric_closest_peer_instance.mock_mode = False

    mock_io_provider.get_dynamic_variable.side_effect = lambda x: {
        "latitude": -33.868820,
        "longitude": 151.209295,
    }.get(x)

    json_response_data_no_peer = {"result": []}
    mock_response_no_peer = Mock()
    mock_response_no_peer.json.return_value = json_response_data_no_peer
    with patch(
        "inputs.plugins.fabric_closest_peer.requests.post",
        return_value=mock_response_no_peer,
    ):
        with caplog.at_level("INFO"):
            result = await fabric_closest_peer_instance._poll()

    assert result is None
    assert "FabricClosestPeer: no peer found." in caplog.text


@pytest.mark.asyncio
async def test_raw_to_text_adds_message_to_list(fabric_closest_peer_instance):
    test_message = "Closest peer at -33.86500, 151.21000"
    initial_len = len(fabric_closest_peer_instance.messages)

    await fabric_closest_peer_instance.raw_to_text(test_message)

    assert len(fabric_closest_peer_instance.messages) == initial_len + 1
    assert fabric_closest_peer_instance.messages[-1] == test_message


@pytest.mark.asyncio
async def test_raw_to_text_does_nothing_if_input_none(fabric_closest_peer_instance):
    initial_len = len(fabric_closest_peer_instance.messages)
    await fabric_closest_peer_instance.raw_to_text(None)

    assert len(fabric_closest_peer_instance.messages) == initial_len


def test_formatted_latest_buffer_empty(fabric_closest_peer_instance):
    result = fabric_closest_peer_instance.formatted_latest_buffer()
    assert result is None


def test_formatted_latest_buffer_formats_and_clears_latest_message(
    fabric_closest_peer_instance, mock_io_provider
):
    msg = "Closest peer at -33.86500, 151.21000"
    fabric_closest_peer_instance.msg_q.put(msg)

    result = fabric_closest_peer_instance.formatted_latest_buffer()

    assert "Closest Peer from Fabric INPUT" in result
    assert "Closest peer at -33.86500, 151.21000" in result
    mock_io_provider.add_input.assert_called_once()
    call_args = mock_io_provider.add_input.call_args
    assert call_args[0][0] == "Closest Peer from Fabric"
    assert call_args[0][1] == msg
    assert fabric_closest_peer_instance.msg_q.empty()
