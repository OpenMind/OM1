"""Unit tests for nav2_hook module."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from aiohttp import ClientError, ClientTimeout

from hooks.nav2_hook import start_nav2_hook, stop_nav2_hook


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
    with patch("hooks.nav2_hook.ElevenLabsTTSProvider") as mock:
        provider_instance = Mock()
        provider_instance.add_pending_message = Mock()
        mock.return_value = provider_instance
        yield provider_instance


class TestStartNav2Hook:
    """Tests for start_nav2_hook function."""

    @pytest.mark.asyncio
    async def test_start_nav2_success_default_params(self, mock_elevenlabs_provider):
        """Test successful Nav2 start with default parameters."""
        context = {}
        expected_response = {"message": "Nav2 started successfully"}

        mock_response = create_mock_response(200, expected_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await start_nav2_hook(context)

        assert result["status"] == "success"
        assert result["message"] == "Nav2 process initiated"
        assert result["response"] == expected_response
        mock_elevenlabs_provider.add_pending_message.assert_called_once_with(
            "Navigation system has started successfully."
        )

    @pytest.mark.asyncio
    async def test_start_nav2_success_custom_params(self, mock_elevenlabs_provider):
        """Test successful Nav2 start with custom parameters."""
        context = {"base_url": "http://robot.local:8080", "map_name": "custom_map"}
        expected_response = {"message": "Started with custom map"}

        mock_response = create_mock_response(200, expected_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await start_nav2_hook(context)

        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://robot.local:8080/start/nav2"
        assert call_args[1]["json"]["map_name"] == "custom_map"

        assert result["status"] == "success"
        assert result["response"] == expected_response

    @pytest.mark.asyncio
    async def test_start_nav2_http_error_response(self, mock_elevenlabs_provider):
        """Test Nav2 start with HTTP error response."""
        context = {}
        error_response = {"message": "Service unavailable"}

        mock_response = create_mock_response(503, error_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(
                Exception, match="Failed to start Nav2: Service unavailable"
            ):
                await start_nav2_hook(context)

    @pytest.mark.asyncio
    async def test_start_nav2_http_error_no_json(self, mock_elevenlabs_provider):
        """Test Nav2 start with HTTP error and no JSON response."""
        context = {}

        mock_response = create_mock_response(500, json_error=Exception("No JSON"))

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match="Failed to start Nav2: Unknown error"):
                await start_nav2_hook(context)

    @pytest.mark.asyncio
    async def test_start_nav2_client_error(self, mock_elevenlabs_provider):
        """Test Nav2 start with client connection error."""
        context = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(
                Exception, match="Error calling Nav2 API: Connection refused"
            ):
                await start_nav2_hook(context)

    @pytest.mark.asyncio
    async def test_start_nav2_timeout_configured(self, mock_elevenlabs_provider):
        """Test that Nav2 start uses correct timeout configuration."""
        context = {}
        expected_response = {"message": "Success"}

        mock_response = create_mock_response(200, expected_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await start_nav2_hook(context)

        call_args = mock_session.post.call_args
        assert "timeout" in call_args[1]
        timeout = call_args[1]["timeout"]
        assert isinstance(timeout, ClientTimeout)


class TestStopNav2Hook:
    """Tests for stop_nav2_hook function."""

    @pytest.mark.asyncio
    async def test_stop_nav2_success_default_params(self):
        """Test successful Nav2 stop with default parameters."""
        context = {}
        expected_response = {"message": "Nav2 stopped successfully"}

        mock_response = create_mock_response(200, expected_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await stop_nav2_hook(context)

        assert result["status"] == "success"
        assert result["message"] == "Nav2 process stopped"
        assert result["response"] == expected_response

    @pytest.mark.asyncio
    async def test_stop_nav2_success_custom_base_url(self):
        """Test successful Nav2 stop with custom base URL."""
        context = {"base_url": "http://custom.server:9000"}
        expected_response = {"message": "Stopped"}

        mock_response = create_mock_response(200, expected_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await stop_nav2_hook(context)

        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://custom.server:9000/stop/nav2"

        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_stop_nav2_http_error(self):
        """Test Nav2 stop with HTTP error response."""
        context = {}
        error_response = {"message": "Failed to stop"}

        mock_response = create_mock_response(400, error_response)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(Exception, match="Failed to stop Nav2: Failed to stop"):
                await stop_nav2_hook(context)

    @pytest.mark.asyncio
    async def test_stop_nav2_client_error(self):
        """Test Nav2 stop with client connection error."""
        context = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Network error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(
                Exception, match="Error calling Nav2 stop API: Network error"
            ):
                await stop_nav2_hook(context)
