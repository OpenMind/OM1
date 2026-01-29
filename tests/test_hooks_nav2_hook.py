"""
Unit tests for the Nav2 hooks module (src/hooks/nav2_hook.py).
Tests the start_nav2_hook and stop_nav2_hook functions.
"""

import sys
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import aiohttp
import pytest

# --- Setup path *before* importing from src ---
current_file_dir = Path(__file__).resolve().parent
project_root = current_file_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
# ------------------------------------------------

from src.hooks.nav2_hook import start_nav2_hook, stop_nav2_hook  # noqa: E402


# --- Helper Functions ---
def create_mock_session_with_response(status, json_data):
    """Helper to create a mocked session with a specific response."""
    mock_response = AsyncMock()
    mock_response.status = status
    mock_response.json = AsyncMock(return_value=json_data)

    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session_instance = MagicMock()
    mock_session_instance.post.return_value = mock_post_cm

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session_instance)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    return mock_session_cm, mock_session_instance


def create_mock_session_with_client_error(error_msg):
    """Helper to create a mocked session that raises aiohttp.ClientError."""
    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__.side_effect = aiohttp.ClientError(error_msg)
    mock_post_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session_instance = MagicMock()
    mock_session_instance.post.return_value = mock_post_cm

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session_instance)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    return mock_session_cm, mock_session_instance


# --- Tests ---
class TestStartNav2Hook:

    @pytest.mark.asyncio
    async def test_start_nav2_hook_success(self):
        context = {"base_url": "http://localhost:5000", "map_name": "test_map"}
        mock_session_cm, mock_session_instance = create_mock_session_with_response(
            200, {"message": "Started successfully"}
        )

        with patch(
            "src.hooks.nav2_hook.aiohttp.ClientSession", return_value=mock_session_cm
        ):
            with patch("src.hooks.nav2_hook.ElevenLabsTTSProvider") as mock_tts_class:
                mock_tts_instance = MagicMock()
                mock_tts_class.return_value = mock_tts_instance

                with patch("src.hooks.nav2_hook.logging") as mock_log:
                    result = await start_nav2_hook(context)

                    mock_session_instance.post.assert_called_once_with(
                        "http://localhost:5000/start/nav2",
                        json={"map_name": "test_map"},
                        headers={"Content-Type": "application/json"},
                        timeout=ANY,
                    )
                    mock_log.info.assert_called_once_with(
                        "Nav2 started successfully: Started successfully"
                    )
                    mock_tts_instance.add_pending_message.assert_called_once_with(
                        "Navigation system has started successfully."
                    )

                    expected_result = {
                        "status": "success",
                        "message": "Nav2 process initiated",
                        "response": {"message": "Started successfully"},
                    }
                    assert result == expected_result

    @pytest.mark.asyncio
    async def test_start_nav2_hook_http_error(self):
        context = {"base_url": "http://localhost:5000", "map_name": "test_map"}
        mock_session_cm, mock_session_instance = create_mock_session_with_response(
            500, {"message": "Internal Server Error"}
        )

        with patch(
            "src.hooks.nav2_hook.aiohttp.ClientSession", return_value=mock_session_cm
        ):
            with patch("src.hooks.nav2_hook.ElevenLabsTTSProvider"):
                with patch("src.hooks.nav2_hook.logging") as mock_log:
                    with pytest.raises(Exception) as exc_info:
                        await start_nav2_hook(context)

                    assert "Failed to start Nav2: Internal Server Error" in str(
                        exc_info.value
                    )
                    mock_log.error.assert_called_once_with(
                        "Failed to start Nav2: Internal Server Error"
                    )

    @pytest.mark.asyncio
    async def test_start_nav2_hook_client_error(self):
        context = {"base_url": "http://localhost:5000", "map_name": "test_map"}
        mock_session_cm, mock_session_instance = create_mock_session_with_client_error(
            "Connection error"
        )

        with patch(
            "src.hooks.nav2_hook.aiohttp.ClientSession", return_value=mock_session_cm
        ):
            with patch("src.hooks.nav2_hook.ElevenLabsTTSProvider"):
                with patch("src.hooks.nav2_hook.logging") as mock_log:
                    with pytest.raises(Exception) as exc_info:
                        await start_nav2_hook(context)

                    assert "Error calling Nav2 API" in str(exc_info.value)
                    assert "Connection error" in str(exc_info.value)
                    mock_log.error.assert_called_once_with(
                        "Error calling Nav2 API: Connection error"
                    )


class TestStopNav2Hook:

    @pytest.mark.asyncio
    async def test_stop_nav2_hook_success(self):
        context = {
            "base_url": "http://localhost:5000",
        }
        mock_session_cm, mock_session_instance = create_mock_session_with_response(
            200, {"message": "Stopped successfully"}
        )

        with patch(
            "src.hooks.nav2_hook.aiohttp.ClientSession", return_value=mock_session_cm
        ):
            with patch("src.hooks.nav2_hook.logging") as mock_log:
                result = await stop_nav2_hook(context)

                mock_session_instance.post.assert_called_once_with(
                    "http://localhost:5000/stop/nav2",
                    headers={"Content-Type": "application/json"},
                    timeout=ANY,
                )
                mock_log.info.assert_called_once_with(
                    "Nav2 started successfully: Stopped successfully"
                )

                expected_result = {
                    "status": "success",
                    "message": "Nav2 process initiated",
                    "response": {"message": "Stopped successfully"},
                }
                assert result == expected_result

    @pytest.mark.asyncio
    async def test_stop_nav2_hook_http_error(self):
        context = {
            "base_url": "http://localhost:5000",
        }
        mock_session_cm, mock_session_instance = create_mock_session_with_response(
            500, {"message": "Internal Server Error"}
        )

        with patch(
            "src.hooks.nav2_hook.aiohttp.ClientSession", return_value=mock_session_cm
        ):
            with patch("src.hooks.nav2_hook.logging") as mock_log:
                with pytest.raises(Exception) as exc_info:
                    await stop_nav2_hook(context)

                assert "Failed to start Nav2: Internal Server Error" in str(
                    exc_info.value
                )
                mock_log.error.assert_called_once_with(
                    "Failed to start Nav2: Internal Server Error"
                )

    @pytest.mark.asyncio
    async def test_stop_nav2_hook_client_error(self):
        context = {
            "base_url": "http://localhost:5000",
        }
        mock_session_cm, mock_session_instance = create_mock_session_with_client_error(
            "Connection error"
        )

        with patch(
            "src.hooks.nav2_hook.aiohttp.ClientSession", return_value=mock_session_cm
        ):
            with patch("src.hooks.nav2_hook.logging") as mock_log:
                with pytest.raises(Exception) as exc_info:
                    await stop_nav2_hook(context)

                assert "Error calling Nav2 API" in str(exc_info.value)
                assert "Connection error" in str(exc_info.value)
                mock_log.error.assert_called_once_with(
                    "Error calling Nav2 API: Connection error"
                )
