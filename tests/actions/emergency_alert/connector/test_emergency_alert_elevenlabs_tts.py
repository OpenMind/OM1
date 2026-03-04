import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from actions.emergency_alert.interface import EmergencyAlertInput

sys.modules["om1_speech"] = MagicMock()

from actions.emergency_alert.connector.elevenlabs_tts import (  # noqa: E402
    EmergencyAlertElevenLabsTTSConnector,
    SpeakElevenLabsTTSConfig,
)


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for connector initialization."""

    with (
        patch("actions.emergency_alert.connector.elevenlabs_tts.IOProvider") as mock_io,
        patch(
            "actions.emergency_alert.connector.elevenlabs_tts.open_zenoh_session"
        ) as mock_zenoh_session,
        patch(
            "actions.emergency_alert.connector.elevenlabs_tts.ASRRTSPProvider"
        ) as mock_asr,
        patch(
            "actions.emergency_alert.connector.elevenlabs_tts.ElevenLabsTTSProvider"
        ) as mock_tts,
        patch(
            "actions.emergency_alert.connector.elevenlabs_tts.TeleopsConversationProvider"
        ) as mock_conv,
        patch("actions.emergency_alert.connector.elevenlabs_tts.prepare_header"),
        patch(
            "actions.emergency_alert.connector.elevenlabs_tts.AudioStatus"
        ) as mock_audio_status,
        patch("actions.emergency_alert.connector.elevenlabs_tts.String"),
    ):
        mock_io_instance = Mock()
        mock_io.return_value = mock_io_instance

        mock_session = Mock()
        mock_zenoh_session.return_value = mock_session

        mock_asr_instance = Mock()
        mock_asr.return_value = mock_asr_instance

        mock_tts_instance = Mock()
        mock_tts.return_value = mock_tts_instance

        mock_conv_instance = Mock()
        mock_conv.return_value = mock_conv_instance

        mock_audio_status_instance = Mock()
        mock_audio_status.return_value = mock_audio_status_instance
        mock_audio_status.STATUS_MIC.UNKNOWN.value = 0
        mock_audio_status.STATUS_SPEAKER.READY.value = 1
        mock_audio_status.STATUS_SPEAKER.ACTIVE.value = 2

        yield {
            "io": mock_io_instance,
            "session": mock_session,
            "asr": mock_asr_instance,
            "tts": mock_tts_instance,
            "conv": mock_conv_instance,
            "audio_status": mock_audio_status,
        }


@pytest.fixture
def connector(mock_dependencies):
    """Create connector with mocked dependencies."""
    config = SpeakElevenLabsTTSConfig()
    return EmergencyAlertElevenLabsTTSConnector(config)


class TestSpeakElevenLabsTTSConfig:
    """Test SpeakElevenLabsTTSConfig configuration."""

    def test_default_config(self):
        config = SpeakElevenLabsTTSConfig()
        assert config.elevenlabs_api_key is None
        assert config.voice_id == "JBFqnCBsd6RMkjVDRZzb"
        assert config.model_id == "eleven_flash_v2_5"
        assert config.output_format == "pcm_16000"
        assert config.silence_rate == 0

    def test_custom_config(self):
        config = SpeakElevenLabsTTSConfig(
            elevenlabs_api_key="test_key",
            voice_id="custom_voice",
            model_id="custom_model",
            output_format="pcm_16000",
            silence_rate=3,
        )
        assert config.elevenlabs_api_key == "test_key"
        assert config.voice_id == "custom_voice"
        assert config.model_id == "custom_model"
        assert config.output_format == "pcm_16000"
        assert config.silence_rate == 3


class TestEmergencyAlertConnectorInit:
    """Test EmergencyAlertElevenLabsTTSConnector initialization."""

    def test_init(self, connector, mock_dependencies):
        assert connector.tts_enabled is True
        assert connector.audio_topic == "robot/status/audio"
        assert connector.tts_status_request_topic == "om/tts/request"
        mock_dependencies["tts"].start.assert_called_once()
        mock_dependencies["session"].declare_publisher.assert_called_once()

    def test_init_zenoh_error(self):
        """Test initialization when Zenoh session fails."""
        with (
            patch("actions.emergency_alert.connector.elevenlabs_tts.IOProvider"),
            patch(
                "actions.emergency_alert.connector.elevenlabs_tts.open_zenoh_session"
            ) as mock_zenoh_session,
            patch("actions.emergency_alert.connector.elevenlabs_tts.ASRRTSPProvider"),
            patch(
                "actions.emergency_alert.connector.elevenlabs_tts.ElevenLabsTTSProvider"
            ) as mock_tts,
            patch(
                "actions.emergency_alert.connector.elevenlabs_tts.TeleopsConversationProvider"
            ),
            patch("actions.emergency_alert.connector.elevenlabs_tts.prepare_header"),
            patch(
                "actions.emergency_alert.connector.elevenlabs_tts.AudioStatus"
            ) as mock_audio_status,
            patch("actions.emergency_alert.connector.elevenlabs_tts.String"),
            patch(
                "actions.emergency_alert.connector.elevenlabs_tts.logging"
            ) as mock_logging,
        ):
            mock_zenoh_session.side_effect = Exception("Zenoh connection failed")
            mock_audio_status.STATUS_MIC.UNKNOWN.value = 0
            mock_audio_status.STATUS_SPEAKER.READY.value = 1
            mock_tts_instance = Mock()
            mock_tts.return_value = mock_tts_instance

            config = SpeakElevenLabsTTSConfig()
            connector = EmergencyAlertElevenLabsTTSConnector(config)

            assert connector.session is None
            mock_logging.error.assert_called()


class TestEmergencyAlertConnectorConnect:
    """Test connect method."""

    @pytest.mark.asyncio
    async def test_connect_tts_disabled(self, connector, mock_dependencies):
        """Test connect when TTS is disabled."""
        connector.tts_enabled = False
        alert_input = EmergencyAlertInput(action="Fire!")
        with patch(
            "actions.emergency_alert.connector.elevenlabs_tts.logging"
        ) as mock_logging:
            await connector.connect(alert_input)
            mock_logging.info.assert_any_call("TTS is disabled, skipping TTS action")

    @pytest.mark.asyncio
    async def test_connect_with_audio_pub(self, connector, mock_dependencies):
        """Test connect publishes audio status when audio_pub exists."""
        mock_dependencies["tts"].create_pending_message.return_value = {"text": "Help!"}
        mock_dependencies["tts"].get_pending_message_count.return_value = 0
        mock_dependencies["io"].llm_prompt = None

        alert_input = EmergencyAlertInput(action="Help!")
        await connector.connect(alert_input)

        mock_dependencies["tts"].create_pending_message.assert_called_once_with("Help!")
        connector.audio_pub.put.assert_called()

    @pytest.mark.asyncio
    async def test_connect_too_many_pending(self, connector, mock_dependencies):
        """Test connect skips when too many pending messages."""
        mock_dependencies["tts"].create_pending_message.return_value = {
            "text": "Alert!"
        }
        mock_dependencies["tts"].get_pending_message_count.return_value = 5
        mock_dependencies["io"].llm_prompt = None

        alert_input = EmergencyAlertInput(action="Alert!")
        with patch(
            "actions.emergency_alert.connector.elevenlabs_tts.logging"
        ) as mock_logging:
            await connector.connect(alert_input)
            mock_logging.warning.assert_any_call(
                "Too many pending TTS messages, skipping adding new message"
            )

    @pytest.mark.asyncio
    async def test_connect_stores_conversation(self, connector, mock_dependencies):
        """Test connect stores robot message when Voice input detected."""
        mock_dependencies["tts"].create_pending_message.return_value = {
            "text": "Emergency!"
        }
        mock_dependencies["tts"].get_pending_message_count.return_value = 0
        mock_dependencies["io"].llm_prompt = "INPUT: Voice says help"

        alert_input = EmergencyAlertInput(action="Emergency!")
        await connector.connect(alert_input)

        mock_dependencies["conv"].store_robot_message.assert_called_once_with(
            "Emergency!"
        )

    @pytest.mark.asyncio
    async def test_connect_no_audio_pub_uses_tts(self, connector, mock_dependencies):
        """Test connect uses TTS directly when no audio_pub."""
        connector.audio_pub = None
        mock_dependencies["tts"].create_pending_message.return_value = {"text": "Fire!"}
        mock_dependencies["tts"].get_pending_message_count.return_value = 0
        mock_dependencies["io"].llm_prompt = None

        alert_input = EmergencyAlertInput(action="Fire!")
        await connector.connect(alert_input)

        mock_dependencies["tts"].register_tts_state_callback.assert_called_once()
        mock_dependencies["tts"].add_pending_message.assert_called_once()


class TestZenohTTSStatusRequest:
    """Test _zenoh_tts_status_request callback."""

    def test_enable_tts(self, connector, mock_dependencies):
        """Test enabling TTS via Zenoh message."""
        connector.tts_enabled = False

        mock_data = Mock()
        mock_data.payload.to_bytes.return_value = b"test"
        mock_tts_status = Mock()
        mock_tts_status.code = 1
        mock_dependencies["audio_status"].return_value = mock_tts_status

        with patch(
            "actions.emergency_alert.connector.elevenlabs_tts.TTSStatusRequest"
        ) as mock_tts_req:
            mock_tts_req.deserialize.return_value = mock_tts_status
            connector._zenoh_tts_status_request(mock_data)

        assert connector.tts_enabled is True

    def test_disable_tts(self, connector, mock_dependencies):
        """Test disabling TTS via Zenoh message."""
        connector.tts_enabled = True

        mock_data = Mock()
        mock_data.payload.to_bytes.return_value = b"test"
        mock_tts_status = Mock()
        mock_tts_status.code = 0

        with patch(
            "actions.emergency_alert.connector.elevenlabs_tts.TTSStatusRequest"
        ) as mock_tts_req:
            mock_tts_req.deserialize.return_value = mock_tts_status
            connector._zenoh_tts_status_request(mock_data)

        assert connector.tts_enabled is False
