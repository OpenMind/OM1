"""
Unit tests for the Person Follow hooks module (src/hooks/person_follow_hook.py).
Tests the start_person_follow_hook and stop_person_follow_hook functions.
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

from src.hooks.person_follow_hook import (  # noqa: E402
    start_person_follow_hook,
    stop_person_follow_hook,
)


def create_mock_session(post_response=None, get_responses=None):
    """Helper to create a mocked session with given responses."""
    if get_responses is None:
        get_responses = []

    mock_session = MagicMock()

    # Mock post response
    mock_post_response = AsyncMock()
    mock_post_response.status = post_response or 200

    mock_post_cm = MagicMock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_post_response)
    mock_post_cm.__aexit__ = AsyncMock(return_value=None)
    mock_session.post.return_value = mock_post_cm

    # Mock get responses
    get_cms = []
    for status, json_data in get_responses:
        mock_get_response = AsyncMock()
        mock_get_response.status = status
        mock_get_response.json = AsyncMock(return_value=json_data)

        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_get_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)
        get_cms.append(mock_get_cm)

    if get_cms:
        mock_session.get.side_effect = get_cms
    else:
        # Default get response
        mock_get_response = AsyncMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(return_value={})

        mock_get_cm = MagicMock()
        mock_get_cm.__aenter__ = AsyncMock(return_value=mock_get_response)
        mock_get_cm.__aexit__ = AsyncMock(return_value=None)
        mock_session.get.return_value = mock_get_cm

    return mock_session


class TestStartPersonFollowHook:

    @pytest.mark.asyncio
    async def test_start_person_follow_hook_success_on_first_attempt(self):
        context = {
            "person_follow_base_url": "http://localhost:8080",
            "enroll_timeout": 3.0,
            "max_retries": 5,
        }
        mock_session = create_mock_session(get_responses=[(200, {"is_tracked": True})])

        with patch("src.hooks.person_follow_hook.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = mock_session
            mock_cls.return_value.__aexit__.return_value = None

            with patch(
                "src.hooks.person_follow_hook.ElevenLabsTTSProvider"
            ) as mock_tts_class:
                mock_tts_instance = MagicMock()
                mock_tts_class.return_value = mock_tts_instance

                with (
                    patch("src.hooks.person_follow_hook.logging") as mock_log,
                    patch("src.hooks.person_follow_hook.asyncio.sleep") as mock_sleep,
                ):

                    result = await start_person_follow_hook(context)

                    mock_session.post.assert_called_once_with(
                        "http://localhost:8080/enroll", timeout=ANY
                    )
                    mock_session.get.assert_called_once_with(
                        "http://localhost:8080/status", timeout=ANY
                    )
                    mock_log.info.assert_any_call("Person Follow: Tracking started")
                    mock_tts_instance.add_pending_message.assert_called_once_with(
                        "I see you! I'll follow you now."
                    )

                    expected = {
                        "status": "success",
                        "message": "Person enrolled and tracking",
                        "is_tracked": True,
                    }
                    assert result == expected
                    mock_sleep.assert_called_once_with(0.5)

    @pytest.mark.asyncio
    async def test_start_person_follow_hook_success_after_retry(self):
        context = {
            "person_follow_base_url": "http://localhost:8080",
            "enroll_timeout": 1.0,
            "max_retries": 5,
        }
        # Not tracked -> tracked
        mock_session = create_mock_session(
            get_responses=[(200, {"is_tracked": False}), (200, {"is_tracked": True})]
        )

        with patch("src.hooks.person_follow_hook.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = mock_session
            mock_cls.return_value.__aexit__.return_value = None

            with patch(
                "src.hooks.person_follow_hook.ElevenLabsTTSProvider"
            ) as mock_tts_class:
                mock_tts_instance = MagicMock()
                mock_tts_class.return_value = mock_tts_instance

                with (
                    patch("src.hooks.person_follow_hook.logging") as mock_log,
                    patch("src.hooks.person_follow_hook.asyncio.sleep"),
                ):

                    result = await start_person_follow_hook(context)

                    assert mock_session.get.call_count >= 2
                    mock_log.info.assert_any_call("Person Follow: Tracking started")
                    mock_tts_instance.add_pending_message.assert_called_once_with(
                        "I see you! I'll follow you now."
                    )

                    expected = {
                        "status": "success",
                        "message": "Person enrolled and tracking",
                        "is_tracked": True,
                    }
                    assert result == expected

    @pytest.mark.asyncio
    async def test_start_person_follow_hook_max_retries_exceeded(self):
        context = {
            "person_follow_base_url": "http://localhost:8080",
            "enroll_timeout": 0.1,
            "max_retries": 2,
        }
        mock_session = create_mock_session(get_responses=[(200, {"is_tracked": False})])

        with patch("src.hooks.person_follow_hook.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = mock_session
            mock_cls.return_value.__aexit__.return_value = None

            with patch(
                "src.hooks.person_follow_hook.ElevenLabsTTSProvider"
            ) as mock_tts_class:
                mock_tts_instance = MagicMock()
                mock_tts_class.return_value = mock_tts_instance

                with (
                    patch("src.hooks.person_follow_hook.logging") as mock_log,
                    patch("src.hooks.person_follow_hook.asyncio.sleep"),
                ):

                    result = await start_person_follow_hook(context)

                    assert mock_session.post.call_count == 2  # max_retries
                    mock_log.info.assert_any_call(
                        "Person Follow: Awaiting person detection"
                    )
                    mock_tts_instance.add_pending_message.assert_called_once_with(
                        "Person following mode activated. Please stand in front of me."
                    )

                    expected = {
                        "status": "success",
                        "message": "Enrolled but awaiting person detection",
                        "is_tracked": False,
                    }
                    assert result == expected

    @pytest.mark.asyncio
    async def test_start_person_follow_hook_connection_error(self):
        context = {
            "person_follow_base_url": "http://localhost:8080",
            "enroll_timeout": 3.0,
            "max_retries": 5,
        }

        with patch("src.hooks.person_follow_hook.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__.side_effect = aiohttp.ClientError(
                "Connection error"
            )
            mock_cls.return_value.__aexit__.return_value = None

            with patch(
                "src.hooks.person_follow_hook.ElevenLabsTTSProvider"
            ) as mock_tts_class:
                mock_tts_instance = MagicMock()
                mock_tts_class.return_value = mock_tts_instance

                with (
                    patch("src.hooks.person_follow_hook.logging") as mock_log,
                    patch("src.hooks.person_follow_hook.asyncio.sleep"),
                ):

                    result = await start_person_follow_hook(context)

                    mock_log.error.assert_called_once_with(
                        "Person Follow: Connection error: Connection error"
                    )
                    mock_tts_instance.add_pending_message.assert_called_once_with(
                        "I couldn't connect to the person following system."
                    )

                    expected = {
                        "status": "error",
                        "message": "Connection error: Connection error",
                    }
                    assert result == expected


class TestStopPersonFollowHook:

    @pytest.mark.asyncio
    async def test_stop_person_follow_hook_success(self):
        context = {"person_follow_base_url": "http://localhost:8080"}
        mock_response = AsyncMock()
        mock_response.status = 200

        mock_post_cm = MagicMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_cm

        with patch("src.hooks.person_follow_hook.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = mock_session
            mock_cls.return_value.__aexit__.return_value = None

            with patch("src.hooks.person_follow_hook.logging") as mock_log:
                result = await stop_person_follow_hook(context)

                mock_session.post.assert_called_once_with(
                    "http://localhost:8080/clear", timeout=ANY
                )
                mock_log.info.assert_any_call("Person Follow: Cleared successfully")

                expected = {"status": "success", "message": "Person tracking stopped"}
                assert result == expected

    @pytest.mark.asyncio
    async def test_stop_person_follow_hook_failure(self):
        context = {"person_follow_base_url": "http://localhost:8080"}
        mock_response = AsyncMock()
        mock_response.status = 500

        mock_post_cm = MagicMock()
        mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_post_cm.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_cm

        with patch("src.hooks.person_follow_hook.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = mock_session
            mock_cls.return_value.__aexit__.return_value = None

            with patch("src.hooks.person_follow_hook.logging") as mock_log:
                result = await stop_person_follow_hook(context)

                mock_session.post.assert_called_once_with(
                    "http://localhost:8080/clear", timeout=ANY
                )
                mock_log.error.assert_any_call("Person Follow: Failed to clear")

                expected = {"status": "error", "message": "Clear failed"}
                assert result == expected

    @pytest.mark.asyncio
    async def test_stop_person_follow_hook_connection_error(self):
        context = {"person_follow_base_url": "http://localhost:8080"}

        mock_session = MagicMock()
        mock_session.post.side_effect = aiohttp.ClientError("Connection error")

        with patch("src.hooks.person_follow_hook.aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__.return_value = mock_session
            mock_cls.return_value.__aexit__.return_value = None

            with patch("src.hooks.person_follow_hook.logging") as mock_log:
                result = await stop_person_follow_hook(context)

                mock_session.post.assert_called_once_with(
                    "http://localhost:8080/clear", timeout=ANY
                )
                mock_log.error.assert_any_call(
                    "Person Follow: Clear error: Connection error"
                )

                expected = {
                    "status": "error",
                    "message": "Connection error: Connection error",
                }
                assert result == expected
