from unittest.mock import Mock, patch

import pytest

from actions.selfie.connector.selfie import SelfieConfig, SelfieConnector
from actions.selfie.interface import SelfieInput


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies."""

    with (
        patch("actions.selfie.connector.selfie.ElevenLabsTTSProvider") as mock_tts,
        patch("actions.selfie.connector.selfie.IOProvider") as mock_io,
    ):
        mock_tts_instance = Mock()
        mock_tts.return_value = mock_tts_instance

        mock_io_instance = Mock()
        mock_io.return_value = mock_io_instance

        yield mock_tts_instance, mock_io_instance


@pytest.fixture
def connector(mock_dependencies):
    """Create SelfieConnector with mocked dependencies."""
    config = SelfieConfig()
    return SelfieConnector(config)


class TestSelfieConfig:
    """Test SelfieConfig configuration."""

    def test_default_config(self):
        config = SelfieConfig()
        assert config.face_http_base_url == "http://127.0.0.1:6793"
        assert config.face_recent_sec == 1.0
        assert config.poll_ms == 200
        assert config.timeout_sec == 15
        assert config.http_timeout_sec == 5.0

    def test_custom_config(self):
        config = SelfieConfig(
            face_http_base_url="http://custom:9999",
            face_recent_sec=2.0,
            poll_ms=500,
            timeout_sec=30,
            http_timeout_sec=10.0,
        )
        assert config.face_http_base_url == "http://custom:9999"
        assert config.face_recent_sec == 2.0
        assert config.poll_ms == 500
        assert config.timeout_sec == 30
        assert config.http_timeout_sec == 10.0


class TestSelfieConnectorInit:
    """Test SelfieConnector initialization."""

    def test_init(self, connector, mock_dependencies):
        assert connector.base_url == "http://127.0.0.1:6793"
        assert connector.recent_sec == 1.0
        assert connector.poll_ms == 200
        assert connector.default_timeout == 15
        assert connector.http_timeout == 5.0


class TestSelfieConnectorHelpers:
    """Test helper methods."""

    def test_post_json_success(self, connector, mock_dependencies):
        with patch("actions.selfie.connector.selfie.requests") as mock_requests:
            mock_response = Mock()
            mock_response.json.return_value = {"ok": True}
            mock_requests.post.return_value = mock_response

            result = connector._post_json("/selfie", {"id": "wendy"})
            assert result == {"ok": True}
            mock_requests.post.assert_called_once_with(
                "http://127.0.0.1:6793/selfie",
                json={"id": "wendy"},
                timeout=5.0,
            )

    def test_post_json_failure(self, connector, mock_dependencies):
        with patch("actions.selfie.connector.selfie.requests") as mock_requests:
            mock_requests.post.side_effect = Exception("Connection refused")
            result = connector._post_json("/selfie", {"id": "test"})
            assert result is None

    def test_get_config(self, connector, mock_dependencies):
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {"config": {"blur": True}}
            result = connector._get_config()
            assert result == {"config": {"blur": True}}
            mock_post.assert_called_once_with("/config", {"get": True})

    def test_get_config_none(self, connector, mock_dependencies):
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = None
            result = connector._get_config()
            assert result == {}

    def test_set_blur(self, connector, mock_dependencies):
        with patch.object(connector, "_post_json") as mock_post:
            connector._set_blur(True)
            mock_post.assert_called_once_with("/config", {"set": {"blur": True}})

    def test_who_snapshot(self, connector, mock_dependencies):
        with patch.object(connector, "_post_json") as mock_post:
            mock_post.return_value = {"now": ["wendy"], "unknown_now": 0}
            result = connector._who_snapshot()
            assert result == {"now": ["wendy"], "unknown_now": 0}

    def test_wait_single_face_success(self, connector, mock_dependencies):
        with (
            patch.object(connector, "_who_snapshot") as mock_who,
            patch.object(connector, "sleep"),
        ):
            mock_who.return_value = {"now": ["wendy"], "unknown_now": 0}
            result = connector._wait_single_face(5)
            assert result is True

    def test_wait_single_face_timeout(self, connector, mock_dependencies):
        with (
            patch.object(connector, "_who_snapshot") as mock_who,
            patch.object(connector, "sleep"),
        ):
            mock_who.return_value = {"now": [], "unknown_now": 0}
            result = connector._wait_single_face(1)
            assert result is False

    def test_write_status(self, connector, mock_dependencies):
        _, mock_io = mock_dependencies
        connector._write_status("ok id=wendy")
        mock_io.add_input.assert_called_once()


class TestSelfieConnectorConnect:
    """Test connect method."""

    @pytest.mark.asyncio
    async def test_connect_empty_name(self, connector, mock_dependencies):
        """Test connect with empty name returns early."""
        _, mock_io = mock_dependencies
        selfie_input = SelfieInput(action="")
        with patch("actions.selfie.connector.selfie.logging") as mock_logging:
            await connector.connect(selfie_input)
            mock_logging.error.assert_called_with(
                "Selfie requires a non-empty `id` (e.g., 'wendy')."
            )

    @pytest.mark.asyncio
    async def test_connect_successful_enrollment(self, connector, mock_dependencies):
        """Test successful selfie enrollment."""
        mock_tts, mock_io = mock_dependencies
        selfie_input = SelfieInput(action="wendy", timeout_sec=5)

        with (
            patch.object(
                connector, "_get_config", return_value={"config": {"blur": True}}
            ),
            patch.object(connector, "_set_blur") as mock_blur,
            patch.object(connector, "_wait_single_face", return_value=True),
            patch.object(
                connector, "_post_json", return_value={"ok": True}
            ) as mock_post,
        ):
            await connector.connect(selfie_input)
            mock_post.assert_called_with("/selfie", {"id": "wendy"})
            mock_tts.add_pending_message.assert_called_once_with(
                "Woof! Woof! I remember you, wendy! You are now enrolled."
            )
            assert mock_blur.call_count == 2  # once off, once restore

    @pytest.mark.asyncio
    async def test_connect_no_face_detected(self, connector, mock_dependencies):
        """Test connect when no face is detected."""
        mock_tts, mock_io = mock_dependencies
        selfie_input = SelfieInput(action="wendy", timeout_sec=5)

        with (
            patch.object(
                connector, "_get_config", return_value={"config": {"blur": False}}
            ),
            patch.object(connector, "_set_blur"),
            patch.object(connector, "_wait_single_face", return_value=False),
            patch.object(
                connector,
                "_who_snapshot",
                return_value={"now": [], "unknown_now": 0},
            ),
        ):
            await connector.connect(selfie_input)
            mock_tts.add_pending_message.assert_called_once()
            assert "0 faces" in mock_tts.add_pending_message.call_args[0][0]

    @pytest.mark.asyncio
    async def test_connect_selfie_api_fails(self, connector, mock_dependencies):
        """Test connect when selfie API returns non-ok."""
        mock_tts, mock_io = mock_dependencies
        selfie_input = SelfieInput(action="wendy", timeout_sec=5)

        with (
            patch.object(
                connector, "_get_config", return_value={"config": {"blur": True}}
            ),
            patch.object(connector, "_set_blur"),
            patch.object(connector, "_wait_single_face", return_value=True),
            patch.object(connector, "_post_json", return_value={"ok": False}),
        ):
            await connector.connect(selfie_input)
            mock_tts.add_pending_message.assert_called_once_with(
                "Woof! Woof! I couldn't see you clearly. Please try again."
            )
