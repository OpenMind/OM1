"""
Unit tests for the SLAM hooks module (src/hooks/slam_hook.py).
Tests the start_slam_hook and stop_slam_hook functions.
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

from src.hooks.slam_hook import start_slam_hook, stop_slam_hook  # noqa: E402


def create_mock_session(responses):
    """Helper to create a mocked session with given responses."""
    mock_session = MagicMock()
    post_cms = []

    for status, json_data in responses:
        mock_resp = AsyncMock()
        mock_resp.status = status
        mock_resp.json = AsyncMock(return_value=json_data)

        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        post_cms.append(mock_cm)

    mock_session.post.side_effect = post_cms
    return mock_session


class TestStartSLAMHook:

    @pytest.mark.asyncio
    async def test_start_slam_hook_success(self):
        context = {"base_url": "http://localhost:5000"}
        mock_session = create_mock_session([(200, {"message": "Started successfully"})])

        with patch("src.hooks.slam_hook.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = mock_session
            mock_cls.return_value.__aexit__.return_value = None

            with patch("src.hooks.slam_hook.logging") as mock_log:
                result = await start_slam_hook(context)

                mock_session.post.assert_called_once_with(
                    "http://localhost:5000/start/slam",
                    headers={"Content-Type": "application/json"},
                    timeout=ANY,
                )
                mock_log.info.assert_called_once_with(
                    "SLAM started successfully: Started successfully"
                )

                expected = {
                    "status": "success",
                    "message": "SLAM process initiated",
                    "response": {"message": "Started successfully"},
                }
                assert result == expected

    @pytest.mark.asyncio
    async def test_start_slam_hook_http_error(self):
        context = {"base_url": "http://localhost:5000"}
        mock_session = create_mock_session(
            [(500, {"message": "Internal Server Error"})]
        )

        with patch("src.hooks.slam_hook.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = mock_session
            mock_cls.return_value.__aexit__.return_value = None

            with patch("src.hooks.slam_hook.logging") as mock_log:
                with pytest.raises(Exception) as exc_info:
                    await start_slam_hook(context)

                assert "Failed to start SLAM: Internal Server Error" in str(
                    exc_info.value
                )
                mock_log.error.assert_called_once_with(
                    "Failed to start SLAM: Internal Server Error"
                )

    @pytest.mark.asyncio
    async def test_start_slam_hook_client_error(self):
        context = {"base_url": "http://localhost:5000"}

        with patch("src.hooks.slam_hook.aiohttp.ClientSession") as mock_cls:
            mock_session = MagicMock()
            mock_session.post.side_effect = aiohttp.ClientError("Connection error")
            mock_cls.return_value.__aenter__.return_value = mock_session
            mock_cls.return_value.__aexit__.return_value = None

            with patch("src.hooks.slam_hook.logging") as mock_log:
                with pytest.raises(Exception) as exc_info:
                    await start_slam_hook(context)

                assert "Error calling SLAM API" in str(exc_info.value)
                assert "Connection error" in str(exc_info.value)
                mock_log.error.assert_called_once_with(
                    "Error calling SLAM API: Connection error"
                )


class TestStopSLAMHook:

    @pytest.mark.asyncio
    async def test_stop_slam_hook_success(self):
        context = {"base_url": "http://localhost:5000", "map_name": "test_map"}
        # Save success, then stop success
        mock_session = create_mock_session(
            [
                (200, {"message": "Map saved successfully"}),
                (200, {"message": "SLAM stopped successfully"}),
            ]
        )

        with patch("src.hooks.slam_hook.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = mock_session
            mock_cls.return_value.__aexit__.return_value = None

            with patch("src.hooks.slam_hook.ElevenLabsTTSProvider") as mock_tts_class:
                mock_tts_instance = MagicMock()
                mock_tts_class.return_value = mock_tts_instance

                with patch("src.hooks.slam_hook.logging") as mock_log:
                    result = await stop_slam_hook(context)

                    assert mock_session.post.call_count == 2
                    mock_session.post.assert_any_call(
                        "http://localhost:5000/maps/save",
                        json={"map_name": "test_map"},
                        headers={"Content-Type": "application/json"},
                        timeout=ANY,
                    )
                    mock_session.post.assert_any_call(
                        "http://localhost:5000/stop/slam",
                        headers={"Content-Type": "application/json"},
                        timeout=ANY,
                    )

                    mock_log.info.assert_any_call(
                        "SLAM map saved successfully: Map saved successfully"
                    )
                    mock_log.info.assert_any_call(
                        "SLAM stopped successfully: SLAM stopped successfully"
                    )

                    mock_tts_instance.add_pending_message.assert_called_once_with(
                        "Map has been saved successfully."
                    )

                    expected = {
                        "status": "success",
                        "message": "SLAM process stopped",
                        "response": {"message": "SLAM stopped successfully"},
                    }
                    assert result == expected

    @pytest.mark.asyncio
    async def test_stop_slam_hook_save_failure(self):
        context = {"base_url": "http://localhost:5000", "map_name": "test_map"}
        mock_session = create_mock_session([(500, {"message": "Save failed"})])

        with patch("src.hooks.slam_hook.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = mock_session
            mock_cls.return_value.__aexit__.return_value = None

            with patch("src.hooks.slam_hook.ElevenLabsTTSProvider"):
                with patch("src.hooks.slam_hook.logging") as mock_log:
                    with pytest.raises(Exception) as exc_info:
                        await stop_slam_hook(context)

                    assert mock_session.post.call_count == 1
                    mock_session.post.assert_called_once_with(
                        "http://localhost:5000/maps/save",
                        json={"map_name": "test_map"},
                        headers={"Content-Type": "application/json"},
                        timeout=ANY,
                    )

                    assert "Failed to save SLAM map: Save failed" in str(exc_info.value)
                    mock_log.error.assert_called_once_with(
                        "Failed to save SLAM map: Save failed"
                    )

    @pytest.mark.asyncio
    async def test_stop_slam_hook_stop_failure(self):
        context = {"base_url": "http://localhost:5000", "map_name": "test_map"}
        # Save success, then stop failure
        mock_session = create_mock_session(
            [
                (200, {"message": "Map saved successfully"}),
                (500, {"message": "Stop failed"}),
            ]
        )

        with patch("src.hooks.slam_hook.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = mock_session
            mock_cls.return_value.__aexit__.return_value = None

            with patch("src.hooks.slam_hook.ElevenLabsTTSProvider") as mock_tts_class:
                mock_tts_instance = MagicMock()
                mock_tts_class.return_value = mock_tts_instance

                with patch("src.hooks.slam_hook.logging") as mock_log:
                    with pytest.raises(Exception) as exc_info:
                        await stop_slam_hook(context)

                    assert mock_session.post.call_count == 2
                    mock_log.info.assert_any_call(
                        "SLAM map saved successfully: Map saved successfully"
                    )

                    assert "Failed to stop SLAM: Stop failed" in str(exc_info.value)
                    mock_log.error.assert_called_once_with(
                        "Failed to stop SLAM: Stop failed"
                    )

                    mock_tts_instance.add_pending_message.assert_called_once_with(
                        "Map has been saved successfully."
                    )

    @pytest.mark.asyncio
    async def test_stop_slam_hook_client_error(self):
        context = {"base_url": "http://localhost:5000", "map_name": "test_map"}

        with patch("src.hooks.slam_hook.aiohttp.ClientSession") as mock_cls:
            mock_session = MagicMock()
            mock_session.post.side_effect = aiohttp.ClientError("Connection error")
            mock_cls.return_value.__aenter__.return_value = mock_session
            mock_cls.return_value.__aexit__.return_value = None

            with patch("src.hooks.slam_hook.ElevenLabsTTSProvider"):
                with patch("src.hooks.slam_hook.logging") as mock_log:
                    with pytest.raises(Exception) as exc_info:
                        await stop_slam_hook(context)

                    assert "Error calling SLAM API" in str(exc_info.value)
                    assert "Connection error" in str(exc_info.value)
                    mock_log.error.assert_called_once_with(
                        "Error calling SLAM API: Connection error"
                    )
