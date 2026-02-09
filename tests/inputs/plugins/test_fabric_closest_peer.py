from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.plugins.fabric_closest_peer import (
    FabricClosestPeer,
    FabricClosestPeerConfig,
)


@pytest.fixture
def mock_io_provider():
    with patch("inputs.plugins.fabric_closest_peer.IOProvider") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def config_mock_mode():
    """Config with mock mode enabled."""
    return FabricClosestPeerConfig(
        fabric_endpoint="http://test.endpoint",
        mock_mode=True,
        mock_lat=40.12345,
        mock_lon=-74.98765,
    )


@pytest.fixture
def config_real_mode():
    """Config with mock mode disabled."""
    return FabricClosestPeerConfig(
        fabric_endpoint="http://test.endpoint",
        mock_mode=False,
    )


@pytest.fixture
def peer_mock_mode(config_mock_mode, mock_io_provider):
    """FabricClosestPeer instance with mock mode."""
    return FabricClosestPeer(config_mock_mode)


@pytest.fixture
def peer_real_mode(config_real_mode, mock_io_provider):
    """FabricClosestPeer instance with real mode."""
    return FabricClosestPeer(config_real_mode)


def test_init(peer_mock_mode):
    """Test initialization sets correct attributes."""
    assert peer_mock_mode.fabric_endpoint == "http://test.endpoint"
    assert peer_mock_mode.mock_mode is True
    assert peer_mock_mode.descriptor_for_LLM == "Closest Peer from Fabric"


@pytest.mark.asyncio
async def test_poll_mock_mode(peer_mock_mode):
    """Test _poll returns mock coordinates in mock mode."""
    peer_mock_mode.config.mock_lat = 40.12345
    peer_mock_mode.config.mock_lon = -74.98765

    result = await peer_mock_mode._poll()

    assert result == "Closest peer at 40.12345, -74.98765"


@pytest.mark.asyncio
async def test_poll_uses_aiohttp_not_requests(peer_real_mode):
    """Test that _poll uses aiohttp for non-blocking HTTP."""
    mock_response = MagicMock()
    mock_response.json = AsyncMock(
        return_value={"result": [{"peer": {"latitude": 40.0, "longitude": -74.0}}]}
    )

    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post_cm)

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("inputs.plugins.fabric_closest_peer.aiohttp.ClientTimeout"),
        patch(
            "inputs.plugins.fabric_closest_peer.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ),
    ):
        peer_real_mode.io.get_dynamic_variable = MagicMock(
            side_effect=lambda key: 40.0 if key == "latitude" else -74.0
        )

        result = await peer_real_mode._poll()

        mock_session.post.assert_called_once()
        assert result == "Closest peer at 40.00000, -74.00000"


@pytest.mark.asyncio
async def test_poll_passes_correct_json_rpc_params(peer_real_mode):
    """Test that _poll sends correct JSON-RPC parameters."""
    mock_response = MagicMock()
    mock_response.json = AsyncMock(
        return_value={"result": [{"peer": {"latitude": 40.0, "longitude": -74.0}}]}
    )

    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post_cm)

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("inputs.plugins.fabric_closest_peer.aiohttp.ClientTimeout"),
        patch(
            "inputs.plugins.fabric_closest_peer.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ),
    ):
        peer_real_mode.io.get_dynamic_variable = MagicMock(
            side_effect=lambda key: 40.5 if key == "latitude" else -74.5
        )

        await peer_real_mode._poll()

        call_kwargs = mock_session.post.call_args[1]
        assert call_kwargs["json"]["method"] == "omp2p_findClosestPeer"
        assert call_kwargs["json"]["params"][0]["latitude"] == 40.5
        assert call_kwargs["json"]["params"][0]["longitude"] == -74.5
        assert call_kwargs["headers"]["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_poll_returns_none_when_no_gps(peer_real_mode):
    """Test _poll returns None when GPS coordinates are not available."""
    peer_real_mode.io.get_dynamic_variable = MagicMock(return_value=None)

    result = await peer_real_mode._poll()

    assert result is None


@pytest.mark.asyncio
async def test_poll_returns_none_when_no_peer_found(peer_real_mode):
    """Test _poll returns None when no peer is found in response."""
    mock_response = MagicMock()
    mock_response.json = AsyncMock(return_value={"result": [{}]})

    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_post_cm)

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("inputs.plugins.fabric_closest_peer.aiohttp.ClientTimeout"),
        patch(
            "inputs.plugins.fabric_closest_peer.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ),
    ):
        peer_real_mode.io.get_dynamic_variable = MagicMock(
            side_effect=lambda key: 40.0 if key == "latitude" else -74.0
        )

        result = await peer_real_mode._poll()

        assert result is None


@pytest.mark.asyncio
async def test_poll_handles_exception(peer_real_mode, caplog):
    """Test _poll handles exceptions gracefully."""
    with patch(
        "inputs.plugins.fabric_closest_peer.aiohttp.ClientSession",
        side_effect=Exception("Network error"),
    ):
        peer_real_mode.io.get_dynamic_variable = MagicMock(
            side_effect=lambda key: 40.0 if key == "latitude" else -74.0
        )

        result = await peer_real_mode._poll()

        assert result is None
        assert "error calling Fabric endpoint" in caplog.text


@pytest.mark.asyncio
async def test_raw_to_text_appends_message(peer_mock_mode):
    """Test raw_to_text appends message to messages list."""
    await peer_mock_mode.raw_to_text("test message")

    assert len(peer_mock_mode.messages) == 1
    assert peer_mock_mode.messages[0] == "test message"


@pytest.mark.asyncio
async def test_raw_to_text_ignores_none(peer_mock_mode):
    """Test raw_to_text ignores None input."""
    await peer_mock_mode.raw_to_text(None)

    assert len(peer_mock_mode.messages) == 0


def test_formatted_latest_buffer_returns_formatted_message(peer_mock_mode):
    """Test formatted_latest_buffer returns correctly formatted output."""
    peer_mock_mode.msg_q.put("Closest peer at 40.12345, -74.98765")

    result = peer_mock_mode.formatted_latest_buffer()

    assert "Closest Peer from Fabric INPUT" in result
    assert "// START" in result
    assert "Closest peer at 40.12345, -74.98765" in result
    assert "// END" in result
    peer_mock_mode.io.add_input.assert_called_once()


def test_formatted_latest_buffer_returns_none_when_empty(peer_mock_mode):
    """Test formatted_latest_buffer returns None when queue is empty."""
    result = peer_mock_mode.formatted_latest_buffer()

    assert result is None
