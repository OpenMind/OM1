from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from aiohttp import ClientError, ClientTimeout

from hooks.person_follow_hook import (
    FOLLOWING_BASE_URL,
    VISION_BASE_URL,
    set_mode_hook,
    start_person_follow_hook,
    stop_person_follow_hook,
    switch_person_follow_hook,
)


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
    with patch("hooks.person_follow_hook.ElevenLabsTTSProvider") as mock:
        provider_instance = Mock()
        provider_instance.add_pending_message = Mock()
        mock.return_value = provider_instance
        yield provider_instance


class TestStartPersonFollowHook:
    """Tests for start_person_follow_hook function."""

    @pytest.mark.asyncio
    async def test_start_person_follow_success_first_attempt(self, mock_elevenlabs_provider):
        """Test successful person follow start on first attempt."""
        context = {}

        # Mock enroll response
        mock_enroll_response = create_mock_response(200)

        # Mock status response showing tracking
        mock_status_response = create_mock_response(200, {"is_tracked": True})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_enroll_response)
        mock_session.get = MagicMock(return_value=mock_status_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await start_person_follow_hook(context)

        assert result["status"] == "success"
        assert result["message"] == "Person enrolled and tracking"
        assert result["is_tracked"] is True

        mock_elevenlabs_provider.add_pending_message.assert_called_once_with("I see you! I'll follow you now.")

    @pytest.mark.asyncio
    async def test_start_person_follow_success_after_retries(self, mock_elevenlabs_provider):
        """Test successful person follow start after multiple attempts."""
        context = {"max_retries": 3}

        # First two enroll attempts fail, third succeeds
        mock_enroll_fail = create_mock_response(500)
        mock_enroll_success = create_mock_response(200)

        # Status shows not tracked initially, then tracked
        mock_status_not_tracked = create_mock_response(200, {"is_tracked": False})
        mock_status_tracked = create_mock_response(200, {"is_tracked": True})

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=[mock_enroll_fail, mock_enroll_fail, mock_enroll_success])
        mock_session.get = MagicMock(side_effect=[mock_status_not_tracked, mock_status_tracked])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await start_person_follow_hook(context)

        assert result["status"] == "success"
        assert result["is_tracked"] is True

    @pytest.mark.asyncio
    async def test_start_person_follow_enrolled_not_tracking(self, mock_elevenlabs_provider):
        """Test person follow enrolled but not yet tracking."""
        context = {"max_retries": 2, "enroll_timeout": 1.0}

        # Enroll succeeds but person never detected
        mock_enroll_response = create_mock_response(200)
        mock_status_response = create_mock_response(200, {"is_tracked": False})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_enroll_response)
        mock_session.get = MagicMock(return_value=mock_status_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await start_person_follow_hook(context)

        assert result["status"] == "success"
        assert result["message"] == "Enrolled but awaiting person detection"
        assert result["is_tracked"] is False

        mock_elevenlabs_provider.add_pending_message.assert_called_once_with(
            "Person following mode activated. Please stand in front of me."
        )

    @pytest.mark.asyncio
    async def test_start_person_follow_custom_base_url(self, mock_elevenlabs_provider):
        """Test person follow with custom base URL."""
        context = {"base_url": "http://custom.robot:9000"}

        mock_enroll_response = create_mock_response(200)
        mock_status_response = create_mock_response(200, {"is_tracked": True})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_enroll_response)
        mock_session.get = MagicMock(return_value=mock_status_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await start_person_follow_hook(context)

        # Verify correct URLs were called
        enroll_call = mock_session.post.call_args_list[0]
        assert enroll_call[0][0] == "http://custom.robot:9000/enroll"

        status_call = mock_session.get.call_args_list[0]
        assert status_call[0][0] == "http://custom.robot:9000/status"

        assert result["is_tracked"] is True

    @pytest.mark.asyncio
    async def test_start_person_follow_enroll_client_error(self, mock_elevenlabs_provider):
        """Test person follow when enroll encounters client error on all attempts."""
        context = {"max_retries": 2}

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Connection failed"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await start_person_follow_hook(context)

        # Should return success but not tracking
        assert result["status"] == "success"
        assert result["is_tracked"] is False

    @pytest.mark.asyncio
    async def test_start_person_follow_status_poll_error(self, mock_elevenlabs_provider):
        """Test person follow when status polling encounters errors."""
        context = {}

        mock_enroll_response = create_mock_response(200)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_enroll_response)
        mock_session.get = MagicMock(side_effect=ClientError("Status unavailable"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await start_person_follow_hook(context)

        assert result["status"] == "success"
        assert result["is_tracked"] is False

    @pytest.mark.asyncio
    async def test_start_person_follow_connection_error(self, mock_elevenlabs_provider):
        """Test person follow with persistent connection error."""
        context = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Network unreachable"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(side_effect=ClientError("Network unreachable"))

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await start_person_follow_hook(context)

        assert result["status"] == "error"
        assert "Connection error" in result["message"]

        assert mock_elevenlabs_provider.add_pending_message.call_count == 2
        mock_elevenlabs_provider.add_pending_message.assert_any_call(
            "I couldn't connect to the person following system."
        )

    @pytest.mark.asyncio
    async def test_start_person_follow_default_constants(self, mock_elevenlabs_provider):
        """Test person follow uses correct default constants."""
        context = {}

        mock_enroll_response = create_mock_response(200)
        mock_status_response = create_mock_response(200, {"is_tracked": False})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_enroll_response)
        mock_session.get = MagicMock(return_value=mock_status_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await start_person_follow_hook(context)

        enroll_call = mock_session.post.call_args_list[0]
        assert VISION_BASE_URL in enroll_call[0][0]

    @pytest.mark.asyncio
    async def test_start_person_follow_timeout_configuration(self, mock_elevenlabs_provider):
        """Test that person follow uses correct timeout configuration."""
        context = {}

        mock_enroll_response = create_mock_response(200)
        mock_status_response = create_mock_response(200, {"is_tracked": True})

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_enroll_response)
        mock_session.get = MagicMock(return_value=mock_status_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await start_person_follow_hook(context)

        # Verify timeouts are configured
        enroll_call = mock_session.post.call_args
        assert "timeout" in enroll_call[1]

        status_call = mock_session.get.call_args
        assert "timeout" in status_call[1]


class TestStopPersonFollowHook:
    """Tests for stop_person_follow_hook function."""

    @pytest.mark.asyncio
    async def test_stop_person_follow_success_default_url(self):
        """Test successful person follow stop with default URL."""
        context = {}

        mock_response = create_mock_response(200)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await stop_person_follow_hook(context)

        assert result["status"] == "success"
        assert result["message"] == "Person tracking stopped"

        call_args = mock_session.post.call_args
        assert call_args[0][0] == f"{VISION_BASE_URL}/clear"

    @pytest.mark.asyncio
    async def test_stop_person_follow_success_custom_url(self):
        """Test successful person follow stop with custom URL."""
        context = {"base_url": "http://robot.custom:8888"}

        mock_response = create_mock_response(200)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await stop_person_follow_hook(context)

        assert result["status"] == "success"

        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://robot.custom:8888/clear"

    @pytest.mark.asyncio
    async def test_stop_person_follow_http_error(self):
        """Test person follow stop with HTTP error response."""
        context = {}

        mock_response = create_mock_response(500)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await stop_person_follow_hook(context)

        assert result["status"] == "error"
        assert result["message"] == "Clear failed"

    @pytest.mark.asyncio
    async def test_stop_person_follow_client_error(self):
        """Test person follow stop with client connection error."""
        context = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Connection lost"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await stop_person_follow_hook(context)

        assert result["status"] == "error"
        assert "Connection error" in result["message"]
        assert "Connection lost" in result["message"]

    @pytest.mark.asyncio
    async def test_stop_person_follow_timeout_configured(self):
        """Test that person follow stop uses correct timeout."""
        context = {}

        mock_response = create_mock_response(200)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await stop_person_follow_hook(context)

        call_args = mock_session.post.call_args
        assert "timeout" in call_args[1]
        timeout = call_args[1]["timeout"]
        assert isinstance(timeout, ClientTimeout)


class TestSwitchPersonFollowHook:
    """Tests for switch_person_follow_hook function."""

    @pytest.mark.asyncio
    async def test_switch_person_follow_success_default_url(self, mock_elevenlabs_provider):
        """Test successful person follow switch with default URL."""
        context = {}

        mock_response = create_mock_response(200)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await switch_person_follow_hook(context)

        assert result["status"] == "success"
        assert result["message"] == "Person tracking target switched"

        call_args = mock_session.post.call_args
        assert call_args[0][0] == f"{VISION_BASE_URL}/switch"

        mock_elevenlabs_provider.add_pending_message.assert_called_once_with("I'll follow a new person now.")

    @pytest.mark.asyncio
    async def test_switch_person_follow_success_custom_url(self, mock_elevenlabs_provider):
        """Test successful person follow switch with custom URL."""
        context = {"base_url": "http://robot.local:7777"}

        mock_response = create_mock_response(200)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await switch_person_follow_hook(context)

        assert result["status"] == "success"
        assert result["message"] == "Person tracking target switched"

        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://robot.local:7777/switch"

        mock_elevenlabs_provider.add_pending_message.assert_called_once_with("I'll follow a new person now.")

    @pytest.mark.asyncio
    async def test_switch_person_follow_http_error_404(self, mock_elevenlabs_provider):
        """Test person follow switch with 404 error response."""
        context = {}

        mock_response = create_mock_response(404)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await switch_person_follow_hook(context)

        assert result["status"] == "error"
        assert "Switch failed with status 404" in result["message"]

        mock_elevenlabs_provider.add_pending_message.assert_called_once_with("I couldn't switch to a new person.")

    @pytest.mark.asyncio
    async def test_switch_person_follow_http_error_500(self, mock_elevenlabs_provider):
        """Test person follow switch with 500 error response."""
        context = {}

        mock_response = create_mock_response(500)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await switch_person_follow_hook(context)

        assert result["status"] == "error"
        assert "Switch failed with status 500" in result["message"]

    @pytest.mark.asyncio
    async def test_switch_person_follow_client_error(self, mock_elevenlabs_provider):
        """Test person follow switch with client connection error."""
        context = {}

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Network timeout"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await switch_person_follow_hook(context)

        assert result["status"] == "error"
        assert "Connection error" in result["message"]
        assert "Network timeout" in result["message"]

        mock_elevenlabs_provider.add_pending_message.assert_called_once_with("I couldn't connect to switch the person.")

    @pytest.mark.asyncio
    async def test_switch_person_follow_timeout_configured(self, mock_elevenlabs_provider):
        """Test that person follow switch uses correct timeout configuration."""
        context = {}

        mock_response = create_mock_response(200)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await switch_person_follow_hook(context)

        call_args = mock_session.post.call_args
        assert "timeout" in call_args[1]
        timeout = call_args[1]["timeout"]
        assert isinstance(timeout, ClientTimeout)


class TestSetModeHook:
    """Tests for set_mode_hook function."""

    @pytest.mark.asyncio
    async def test_set_mode_success_greeting_mode(self):
        """Test successful mode set to 'greeting' with default URL."""
        context = {"mode": "greeting"}

        mock_response = create_mock_response(200)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await set_mode_hook(context)

        assert result["status"] == "success"
        assert result["message"] == "Mode set to greeting"

        call_args = mock_session.post.call_args
        assert call_args[0][0] == f"{FOLLOWING_BASE_URL}/command"

        # Verify payload
        assert "json" in call_args[1]
        payload = call_args[1]["json"]
        assert payload["cmd"] == "set_mode"
        assert payload["mode"] == "greeting"

    @pytest.mark.asyncio
    async def test_set_mode_success_following_mode(self):
        """Test successful mode set to 'following'."""
        context = {"mode": "following"}

        mock_response = create_mock_response(200)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await set_mode_hook(context)

        assert result["status"] == "success"
        assert result["message"] == "Mode set to following"

        # Verify payload contains correct mode
        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        assert payload["mode"] == "following"

    @pytest.mark.asyncio
    async def test_set_mode_custom_base_url(self):
        """Test mode set with custom base URL."""
        context = {"mode": "greeting", "base_url": "http://custom.robot:3000"}

        mock_response = create_mock_response(200)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await set_mode_hook(context)

        assert result["status"] == "success"

        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://custom.robot:3000/command"

    @pytest.mark.asyncio
    async def test_set_mode_http_error_400(self):
        """Test mode set with 400 error response."""
        context = {"mode": "invalid_mode"}

        mock_response = create_mock_response(400)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await set_mode_hook(context)

        assert result["status"] == "error"
        assert "Set mode failed with status 400" in result["message"]

    @pytest.mark.asyncio
    async def test_set_mode_http_error_500(self):
        """Test mode set with 500 error response."""
        context = {"mode": "greeting"}

        mock_response = create_mock_response(500)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await set_mode_hook(context)

        assert result["status"] == "error"
        assert "Set mode failed with status 500" in result["message"]

    @pytest.mark.asyncio
    async def test_set_mode_client_error(self):
        """Test mode set with client connection error."""
        context = {"mode": "greeting"}

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await set_mode_hook(context)

        assert result["status"] == "error"
        assert "Connection error" in result["message"]
        assert "Connection refused" in result["message"]

    @pytest.mark.asyncio
    async def test_set_mode_timeout_configured(self):
        """Test that mode set uses correct timeout configuration."""
        context = {"mode": "greeting"}

        mock_response = create_mock_response(200)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await set_mode_hook(context)

        call_args = mock_session.post.call_args
        assert "timeout" in call_args[1]
        timeout = call_args[1]["timeout"]
        assert isinstance(timeout, ClientTimeout)

    @pytest.mark.asyncio
    async def test_set_mode_headers_configured(self):
        """Test that mode set includes correct headers."""
        context = {"mode": "greeting"}

        mock_response = create_mock_response(200)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await set_mode_hook(context)

        call_args = mock_session.post.call_args
        assert "headers" in call_args[1]
        headers = call_args[1]["headers"]
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_set_mode_custom_mode_value(self):
        """Test mode set with a custom mode value."""
        context = {"mode": "custom_patrol"}

        mock_response = create_mock_response(200)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await set_mode_hook(context)

        assert result["status"] == "success"
        assert result["message"] == "Mode set to custom_patrol"

        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        assert payload["mode"] == "custom_patrol"
