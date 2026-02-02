"""Unit tests for slam_hook module."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from aiohttp import ClientError, ClientTimeout

from hooks.slam_hook import start_slam_hook, stop_slam_hook


def create_mock_response(status, json_data=None, json_error=None):
    """Helper to create a mock aiohttp response."""
    mock_resp = MagicMock()
    mock_resp.status = status

    if json_error:
        mock_resp.json = AsyncMock(side_effect=json_error)
    else:
        mock_resp.json = AsyncMock(return_value=json_data)

    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    return mock_resp


@pytest.fixture
def mock_elevenlabs_provider():
    """Mock ElevenLabsTTSProvider."""
    with patch("hooks.slam_hook.ElevenLabsTTSProvider") as mock:
        provider_instance = Mock()
        provider_instance.add_pending_message = Mock()
        mock.return_value = provider_instance
        yield provider_instance


class TestStartSlamHook:
    """Tests for start_slam_hook function."""

    @pytest.mark.asyncio
    async def test_start_slam_success_default_params(self):
        """Test successful SLAM start with default parameters."""
        context = {}
        expected_response = {"message": "SLAM started successfully"}

        mock_response = create_mock_response(200, expected_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await start_slam_hook(context)

        assert result["status"] == "success"
        assert result["message"] == "SLAM process initiated"
        assert result["response"] == expected_response

    @pytest.mark.asyncio
    async def test_start_slam_success_custom_base_url(self):
        """Test successful SLAM start with custom base URL."""
        context = {"base_url": "http://robot.local:7000"}
        expected_response = {"message": "SLAM running"}

        mock_response = create_mock_response(200, expected_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await start_slam_hook(context)

        # Verify the URL was constructed correctly
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://robot.local:7000/start/slam"

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_start_slam_http_error_with_json(self):
        """Test SLAM start with HTTP error response containing JSON."""
        context = {}
        error_response = {"message": "SLAM initialization failed"}

        mock_response = create_mock_response(500, error_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match=r"Failed to start SLAM"):
                await start_slam_hook(context)

    @pytest.mark.asyncio
    async def test_start_slam_http_error_no_json(self):
        """Test SLAM start with HTTP error and no JSON response."""
        context = {}

        mock_response = create_mock_response(503, json_error=Exception("Invalid JSON"))

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match=r"Failed to start SLAM"):
                await start_slam_hook(context)

    @pytest.mark.asyncio
    async def test_start_slam_client_error(self):
        """Test SLAM start with client connection error."""
        context = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Connection timeout"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match=r"Error calling SLAM API"):
                await start_slam_hook(context)

    @pytest.mark.asyncio
    async def test_start_slam_timeout_configured(self):
        """Test that SLAM start uses correct timeout configuration."""
        context = {}
        expected_response = {"message": "Success"}

        mock_response = create_mock_response(200, expected_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await start_slam_hook(context)

        # Verify timeout is set
        call_args = mock_session.post.call_args
        assert "timeout" in call_args[1]
        timeout = call_args[1]["timeout"]
        assert isinstance(timeout, ClientTimeout)


class TestStopSlamHook:
    """Tests for stop_slam_hook function."""

    @pytest.mark.asyncio
    async def test_stop_slam_success_default_params(self, mock_elevenlabs_provider):
        """Test successful SLAM stop with default parameters."""
        context = {}
        save_response = {"message": "Map saved"}
        stop_response = {"message": "SLAM stopped"}

        mock_save_response = create_mock_response(200, save_response)
        mock_stop_response = create_mock_response(200, stop_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=[mock_save_response, mock_stop_response]
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await stop_slam_hook(context)

        assert result["status"] == "success"
        assert result["message"] == "SLAM process stopped"
        assert result["response"] == stop_response

        mock_elevenlabs_provider.add_pending_message.assert_called_once_with(
            "Map has been saved successfully."
        )

    @pytest.mark.asyncio
    async def test_stop_slam_success_custom_params(self, mock_elevenlabs_provider):
        """Test successful SLAM stop with custom parameters."""
        context = {"base_url": "http://custom.robot:6000", "map_name": "my_custom_map"}
        save_response = {"message": "Custom map saved"}
        stop_response = {"message": "SLAM stopped"}

        mock_save_response = create_mock_response(200, save_response)
        mock_stop_response = create_mock_response(200, stop_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=[mock_save_response, mock_stop_response]
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await stop_slam_hook(context)

        calls = mock_session.post.call_args_list
        assert len(calls) == 2

        save_call = calls[0]
        assert save_call[0][0] == "http://custom.robot:6000/maps/save"
        assert save_call[1]["json"]["map_name"] == "my_custom_map"

        stop_call = calls[1]
        assert stop_call[0][0] == "http://custom.robot:6000/stop/slam"

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_stop_slam_save_map_fails(self, mock_elevenlabs_provider):
        """Test SLAM stop when map save fails."""
        context = {}
        error_response = {"message": "Failed to save map"}

        mock_save_response = create_mock_response(500, error_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_save_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match=r"Failed to save SLAM map"):
                await stop_slam_hook(context)

    @pytest.mark.asyncio
    async def test_stop_slam_save_map_no_json(self, mock_elevenlabs_provider):
        """Test SLAM stop when map save fails with no JSON."""
        context = {}

        mock_save_response = create_mock_response(500, json_error=Exception("No JSON"))

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_save_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            # The exception is logged but then continues to stop SLAM
            # But actually looking at the code, it raises an exception
            with pytest.raises(Exception):
                await stop_slam_hook(context)

    @pytest.mark.asyncio
    async def test_stop_slam_stop_fails_after_save(self, mock_elevenlabs_provider):
        """Test SLAM stop when stop operation fails after successful save."""
        context = {}
        save_response = {"message": "Map saved"}
        error_response = {"message": "Failed to stop SLAM"}

        mock_save_response = create_mock_response(200, save_response)
        mock_stop_response = create_mock_response(500, error_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=[mock_save_response, mock_stop_response]
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match=r"Failed to stop SLAM"):
                await stop_slam_hook(context)

    @pytest.mark.asyncio
    async def test_stop_slam_client_error(self, mock_elevenlabs_provider):
        """Test SLAM stop with client connection error."""
        context = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Network unreachable"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match=r"Error calling SLAM API"):
                await stop_slam_hook(context)

    @pytest.mark.asyncio
    async def test_stop_slam_timeouts_configured(self, mock_elevenlabs_provider):
        """Test that SLAM stop uses correct timeout configurations."""
        context = {}
        save_response = {"message": "Map saved"}
        stop_response = {"message": "SLAM stopped"}

        mock_save_response = create_mock_response(200, save_response)
        mock_stop_response = create_mock_response(200, stop_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(
            side_effect=[mock_save_response, mock_stop_response]
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await stop_slam_hook(context)

        calls = mock_session.post.call_args_list
        for call in calls:
            assert "timeout" in call[1]
            timeout = call[1]["timeout"]
            assert isinstance(timeout, ClientTimeout)
