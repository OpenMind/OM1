from unittest.mock import Mock, patch

import pytest

from actions.speak.connector.ub_tts import UbTtsConfig, UbTtsConnector
from actions.speak.interface import SpeakInput


@pytest.fixture
def mock_zenoh_session():
    """Mock zenoh session creation and its methods."""
    mock_session = Mock()
    with patch(
        "actions.speak.connector.ub_tts.open_zenoh_session",
        return_value=mock_session,
    ):
        yield mock_session


@pytest.fixture
def mock_tts_provider():
    """Mock the UbTtsProvider."""
    with patch("actions.speak.connector.ub_tts.UbTtsProvider") as mock_cls:
        mock_instance = Mock()
        mock_cls.return_value = mock_instance
        yield {"cls": mock_cls, "instance": mock_instance}


@pytest.fixture
def connector(mock_zenoh_session, mock_tts_provider):
    """Create a UbTtsConnector with mocked dependencies."""
    config = UbTtsConfig(
        robot_ip="192.168.1.100", base_url="http://192.168.1.100:9090/v1/"
    )
    return UbTtsConnector(config)


class TestUbTtsConfig:
    """Test UbTtsConfig configuration."""

    def test_default_config(self):
        """Test default config values."""
        config = UbTtsConfig()
        assert config.robot_ip is None

    def test_custom_config(self):
        """Test custom config values."""
        config = UbTtsConfig(
            robot_ip="10.0.0.1",
            base_url="http://10.0.0.1:8080/v1/",
        )
        assert config.robot_ip == "10.0.0.1"
        assert config.base_url == "http://10.0.0.1:8080/v1/"


class TestUbTtsConnector:
    """Test the UB TTS connector."""

    def test_init_creates_zenoh_session(self, mock_zenoh_session, mock_tts_provider):
        """Test initialization opens zenoh session and declares subscriber/publisher."""
        config = UbTtsConfig(
            robot_ip="192.168.1.1", base_url="http://192.168.1.1:9090/v1/"
        )
        connector = UbTtsConnector(config)

        mock_zenoh_session.declare_subscriber.assert_called_once_with(
            "om/tts/request", connector._zenoh_tts_status_request
        )
        mock_zenoh_session.declare_publisher.assert_called_once_with("om/tts/response")

    def test_init_creates_tts_provider(self, mock_zenoh_session, mock_tts_provider):
        """Test initialization creates UbTtsProvider with correct URL."""
        config = UbTtsConfig(
            robot_ip="192.168.1.1", base_url="http://192.168.1.1:9090/v1/"
        )
        UbTtsConnector(config)

        mock_tts_provider["cls"].assert_called_once_with(
            url="http://192.168.1.1:9090/v1/voice/tts"
        )

    def test_init_zenoh_failure(self, mock_tts_provider):
        """Test initialization handles zenoh session failure gracefully."""
        with patch(
            "actions.speak.connector.ub_tts.open_zenoh_session",
            side_effect=Exception("Connection refused"),
        ):
            config = UbTtsConfig(
                robot_ip="192.168.1.1", base_url="http://192.168.1.1:9090/v1/"
            )
            connector = UbTtsConnector(config)
            assert connector.session is None

    @pytest.mark.asyncio
    async def test_connect_tts_enabled(self, connector, mock_tts_provider):
        """Test connect passes message to TTS provider when enabled."""
        connector.tts_enabled = True
        speak_input = SpeakInput(action="Hello!")

        with patch("actions.speak.connector.ub_tts.time") as mock_time:
            mock_time.time.return_value = 1000
            await connector.connect(speak_input)

        mock_tts_provider["instance"].adding_pending_message.assert_called_once_with(
            message="Hello!",
            interrupt=True,
            timestamp=1000,
        )

    @pytest.mark.asyncio
    async def test_connect_tts_disabled(self, connector, mock_tts_provider):
        """Test connect skips when TTS is disabled."""
        connector.tts_enabled = False
        speak_input = SpeakInput(action="Hello!")

        await connector.connect(speak_input)

        mock_tts_provider["instance"].adding_pending_message.assert_not_called()

    def test_zenoh_tts_status_request_read(self, connector, mock_zenoh_session):
        """Test status request with code 2 (read status) responds with current state."""
        connector.tts_enabled = True

        mock_data = Mock()
        mock_tts_status = Mock()
        mock_tts_status.code = 2
        mock_tts_status.request_id = "req-001"
        mock_tts_status.header.frame_id = "frame-001"

        with (
            patch(
                "actions.speak.connector.ub_tts.TTSStatusRequest"
            ) as mock_request_cls,
            patch(
                "actions.speak.connector.ub_tts.TTSStatusResponse"
            ) as mock_response_cls,
            patch("actions.speak.connector.ub_tts.prepare_header") as mock_header,
            patch("actions.speak.connector.ub_tts.String") as mock_string,
        ):
            mock_request_cls.deserialize.return_value = mock_tts_status
            mock_header.return_value = "prepared_header"

            connector._zenoh_tts_status_request(mock_data)

            mock_header.assert_called_once_with("frame-001")
            mock_string.assert_called_with(data="TTS Enabled")
            call_kwargs = mock_response_cls.call_args[1]
            assert call_kwargs["header"] == "prepared_header"
            assert call_kwargs["request_id"] == "req-001"
            assert call_kwargs["code"] == 1

    def test_zenoh_tts_status_request_enable(self, connector, mock_zenoh_session):
        """Test status request with code 1 enables TTS."""
        connector.tts_enabled = False

        mock_data = Mock()
        mock_tts_status = Mock()
        mock_tts_status.code = 1
        mock_tts_status.request_id = "req-002"
        mock_tts_status.header.frame_id = "frame-002"

        with (
            patch(
                "actions.speak.connector.ub_tts.TTSStatusRequest"
            ) as mock_request_cls,
            patch("actions.speak.connector.ub_tts.TTSStatusResponse"),
            patch("actions.speak.connector.ub_tts.prepare_header"),
            patch("actions.speak.connector.ub_tts.String"),
        ):
            mock_request_cls.deserialize.return_value = mock_tts_status

            connector._zenoh_tts_status_request(mock_data)

            assert connector.tts_enabled is True

    def test_zenoh_tts_status_request_disable(self, connector, mock_zenoh_session):
        """Test status request with code 0 disables TTS."""
        connector.tts_enabled = True

        mock_data = Mock()
        mock_tts_status = Mock()
        mock_tts_status.code = 0
        mock_tts_status.request_id = "req-003"
        mock_tts_status.header.frame_id = "frame-003"

        with (
            patch(
                "actions.speak.connector.ub_tts.TTSStatusRequest"
            ) as mock_request_cls,
            patch("actions.speak.connector.ub_tts.TTSStatusResponse"),
            patch("actions.speak.connector.ub_tts.prepare_header"),
            patch("actions.speak.connector.ub_tts.String"),
        ):
            mock_request_cls.deserialize.return_value = mock_tts_status

            connector._zenoh_tts_status_request(mock_data)

            assert connector.tts_enabled is False

    def test_stop_closes_session_and_tts(
        self, connector, mock_zenoh_session, mock_tts_provider
    ):
        """Test stop closes zenoh session and stops TTS provider."""
        connector.stop()

        mock_zenoh_session.close.assert_called_once()
        mock_tts_provider["instance"].stop.assert_called_once()

    def test_stop_no_session(self, mock_tts_provider):
        """Test stop handles missing session gracefully."""
        with patch(
            "actions.speak.connector.ub_tts.open_zenoh_session",
            side_effect=Exception("Connection refused"),
        ):
            config = UbTtsConfig(
                robot_ip="192.168.1.1", base_url="http://192.168.1.1:9090/v1/"
            )
            connector = UbTtsConnector(config)
            connector.stop()
            # Should not raise - session is None
            mock_tts_provider["instance"].stop.assert_called_once()
