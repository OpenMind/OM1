import sys
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from actions.speak.connector.kokoro_tts import (  # noqa: E402
    SpeakKokoroTTSConfig,
    SpeakKokoroTTSConnector,
)
from actions.speak.interface import SpeakInput  # noqa: E402
from zenoh_msgs import AudioStatus, String  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def mock_zenoh_module():
    """Mock the zenoh module before any imports."""
    mock_zenoh = MagicMock()
    mock_zenoh.Sample = MagicMock
    sys.modules["zenoh"] = mock_zenoh
    yield mock_zenoh
    if "zenoh" in sys.modules:
        del sys.modules["zenoh"]


@pytest.fixture(autouse=True, scope="session")
def mock_om1_speech_module():
    """Mock the om1_speech module before any imports."""
    mock_om1_speech = MagicMock()
    mock_om1_speech.AudioOutputLiveStream = MagicMock()
    sys.modules["om1_speech"] = mock_om1_speech
    yield mock_om1_speech
    if "om1_speech" in sys.modules:
        del sys.modules["om1_speech"]


@pytest.fixture
def default_config():
    """Create a default config for testing."""
    return SpeakKokoroTTSConfig()


@pytest.fixture
def custom_config():
    """Create a custom config for testing."""
    return SpeakKokoroTTSConfig(
        voice_id="custom_voice",
        model_id="custom_model",
        output_format="wav",
        rate=48000,
        enable_tts_interrupt=True,
        silence_rate=2,
        api_key="test_api_key",  # type: ignore
    )


@pytest.fixture
def speak_input():
    """Create a SpeakInput instance for testing."""
    return SpeakInput(action="Hello, world!")


@pytest.fixture
def mock_zenoh_session():
    """Create a mock Zenoh session."""
    session = Mock()
    session.declare_publisher.return_value = Mock()
    session.declare_subscriber.return_value = Mock()
    session.close = Mock()
    return session


@pytest.fixture(autouse=True)
def reset_mocks(mock_om1_speech_module, mock_zenoh_module):
    """Reset all mock objects between tests."""
    mock_om1_speech_module.AudioOutputLiveStream.reset_mock()
    mock_om1_speech_module.AudioOutputLiveStream.return_value = MagicMock()
    mock_zenoh_module.reset_mock()
    yield


class TestSpeakKokoroTTSConfig:
    """Test the configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpeakKokoroTTSConfig()

        assert config.voice_id == "af_bella"
        assert config.model_id == "kokoro"
        assert config.output_format == "pcm"
        assert config.rate == 24000
        assert config.enable_tts_interrupt is False
        assert config.silence_rate == 0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SpeakKokoroTTSConfig(
            voice_id="test_voice",
            model_id="test_model",
            output_format="wav",
            rate=48000,
            enable_tts_interrupt=True,
            silence_rate=5,
        )

        assert config.voice_id == "test_voice"
        assert config.model_id == "test_model"
        assert config.output_format == "wav"
        assert config.rate == 48000
        assert config.enable_tts_interrupt is True
        assert config.silence_rate == 5


class TestSpeakKokoroTTSConnector:
    """Test the Kokoro TTS connector."""

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    def test_init_with_default_config(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test initialization with default configuration."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakKokoroTTSConnector(default_config)

        mock_open_zenoh_session.assert_called_once()
        assert mock_zenoh_session.declare_publisher.call_count == 2
        assert mock_zenoh_session.declare_subscriber.call_count == 2

        mock_tts_provider.assert_called_once_with(
            url="http://127.0.0.1:8880/v1",
            api_key=None,
            voice_id="af_bella",
            model_id="kokoro",
            output_format="pcm",
            rate=24000,
            enable_tts_interrupt=False,
        )

        mock_tts_instance.start.assert_called_once()
        mock_tts_instance.configure.assert_called_once()

        assert connector.silence_rate == 0
        assert connector.silence_counter == 0
        assert connector.tts_enabled is True
        assert connector.session == mock_zenoh_session

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    def test_init_with_custom_config(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        custom_config,
        mock_zenoh_session,
    ):
        """Test initialization with custom configuration."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakKokoroTTSConnector(custom_config)

        mock_tts_provider.assert_called_once_with(
            url="http://127.0.0.1:8880/v1",
            api_key="test_api_key",
            voice_id="custom_voice",
            model_id="custom_model",
            output_format="wav",
            rate=48000,
            enable_tts_interrupt=True,
        )

        assert connector.silence_rate == 2

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    def test_init_zenoh_failure(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
    ):
        """Test initialization when Zenoh session fails to open."""
        mock_open_zenoh_session.side_effect = Exception("Zenoh connection failed")

        connector = SpeakKokoroTTSConnector(default_config)

        assert connector.session is None
        assert connector.audio_pub is None

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    @pytest.mark.asyncio
    async def test_connect_tts_enabled(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
        speak_input,
    ):
        """Test connect method when TTS is enabled."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_instance.create_pending_message.return_value = {
            "id": "test_id",
            "text": "Hello, world!",
        }
        mock_tts_provider.return_value = mock_tts_instance
        mock_audio_pub = Mock()
        mock_zenoh_session.declare_publisher.return_value = mock_audio_pub

        mock_io_instance = Mock()
        mock_io_instance.llm_prompt = "INPUT: Voice: Hello"
        mock_io_provider.return_value = mock_io_instance

        mock_conversation_instance = Mock()
        mock_conversation_provider.return_value = mock_conversation_instance

        connector = SpeakKokoroTTSConnector(default_config)
        connector.io_provider = mock_io_instance
        connector.conversation_provider = mock_conversation_instance

        await connector.connect(speak_input)

        mock_tts_instance.create_pending_message.assert_called_once_with(
            "Hello, world!"
        )
        mock_conversation_instance.store_robot_message.assert_called_once_with(
            "Hello, world!"
        )
        mock_audio_pub.put.assert_called()

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    @pytest.mark.asyncio
    async def test_connect_tts_disabled(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
        speak_input,
    ):
        """Test connect method when TTS is disabled."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakKokoroTTSConnector(default_config)
        connector.tts_enabled = False

        await connector.connect(speak_input)

        mock_tts_instance.create_pending_message.assert_not_called()

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    @pytest.mark.asyncio
    async def test_connect_silence_rate_skip(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        mock_zenoh_session,
        speak_input,
    ):
        """Test connect method with silence rate causing skip."""
        config = SpeakKokoroTTSConfig(silence_rate=2)
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_instance.create_pending_message.return_value = {
            "id": "test_id",
            "text": "Hello, world!",
        }
        mock_tts_provider.return_value = mock_tts_instance

        mock_io_instance = Mock()
        mock_io_instance.llm_prompt = "INPUT: Text: Hello"
        mock_io_provider.return_value = mock_io_instance

        connector = SpeakKokoroTTSConnector(config)
        connector.io_provider = mock_io_instance

        await connector.connect(speak_input)
        assert connector.silence_counter == 1
        mock_tts_instance.create_pending_message.assert_not_called()

        await connector.connect(speak_input)
        assert connector.silence_counter == 2
        mock_tts_instance.create_pending_message.assert_not_called()

        await connector.connect(speak_input)
        assert connector.silence_counter == 0
        mock_tts_instance.create_pending_message.assert_called_once()

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    @pytest.mark.asyncio
    async def test_connect_without_audio_publisher(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
        speak_input,
    ):
        """Test connect method when audio publisher is None."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_instance.create_pending_message.return_value = {
            "id": "test_id",
            "text": "Hello, world!",
        }
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakKokoroTTSConnector(default_config)
        connector.audio_pub = None

        await connector.connect(speak_input)

        mock_tts_instance.add_pending_message.assert_called_once_with(
            {"id": "test_id", "text": "Hello, world!"}
        )

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    def test_zenoh_audio_message(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test processing of Zenoh audio status messages."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_provider.return_value = Mock()

        connector = SpeakKokoroTTSConnector(default_config)

        mock_sample = Mock()
        mock_audio_status = Mock()
        mock_sample.payload.to_bytes.return_value = b"test_data"

        with patch(
            "actions.speak.connector.kokoro_tts.AudioStatus"
        ) as mock_audio_status_class:
            mock_audio_status_class.deserialize.return_value = mock_audio_status

            connector.zenoh_audio_message(mock_sample)

            mock_audio_status_class.deserialize.assert_called_once_with(b"test_data")
            assert connector.audio_status == mock_audio_status

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    def test_zenoh_tts_status_request_enable(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test TTS status request to enable TTS."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_provider.return_value = Mock()
        mock_response_pub = Mock()

        connector = SpeakKokoroTTSConnector(default_config)
        connector._zenoh_tts_status_response_pub = mock_response_pub
        connector.tts_enabled = False

        mock_sample = Mock()
        mock_sample.payload.to_bytes.return_value = b"test_data"

        mock_header = Mock()
        mock_header.frame_id = "test_frame"

        mock_tts_status = Mock()
        mock_tts_status.code = 1  # Enable TTS
        mock_tts_status.request_id = String("test_request_id")
        mock_tts_status.header = mock_header

        with patch(
            "actions.speak.connector.kokoro_tts.TTSStatusRequest"
        ) as mock_request_class:
            with patch("actions.speak.connector.kokoro_tts.TTSStatusResponse"):
                mock_request_class.deserialize.return_value = mock_tts_status

                connector._zenoh_tts_status_request(mock_sample)

                assert connector.tts_enabled is True
                mock_response_pub.put.assert_called_once()

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    def test_zenoh_tts_status_request_disable(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test TTS status request to disable TTS."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_provider.return_value = Mock()
        mock_response_pub = Mock()

        connector = SpeakKokoroTTSConnector(default_config)
        connector._zenoh_tts_status_response_pub = mock_response_pub
        connector.tts_enabled = True

        mock_sample = Mock()
        mock_sample.payload.to_bytes.return_value = b"test_data"

        mock_header = Mock()
        mock_header.frame_id = "test_frame"

        mock_tts_status = Mock()
        mock_tts_status.code = 0  # Disable TTS
        mock_tts_status.request_id = String("test_request_id")
        mock_tts_status.header = mock_header

        with patch(
            "actions.speak.connector.kokoro_tts.TTSStatusRequest"
        ) as mock_request_class:
            with patch("actions.speak.connector.kokoro_tts.TTSStatusResponse"):
                mock_request_class.deserialize.return_value = mock_tts_status

                connector._zenoh_tts_status_request(mock_sample)

                assert connector.tts_enabled is False
                mock_response_pub.put.assert_called_once()

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    def test_zenoh_tts_status_request_read(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test TTS status request to read current status."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_provider.return_value = Mock()
        mock_response_pub = Mock()

        connector = SpeakKokoroTTSConnector(default_config)
        connector._zenoh_tts_status_response_pub = mock_response_pub
        connector.tts_enabled = True

        mock_sample = Mock()
        mock_sample.payload.to_bytes.return_value = b"test_data"

        mock_header = Mock()
        mock_header.frame_id = "test_frame"

        mock_tts_status = Mock()
        mock_tts_status.code = 2  # Read status
        mock_tts_status.request_id = String("test_request_id")
        mock_tts_status.header = mock_header

        with patch(
            "actions.speak.connector.kokoro_tts.TTSStatusRequest"
        ) as mock_request_class:
            with patch("actions.speak.connector.kokoro_tts.TTSStatusResponse"):
                mock_request_class.deserialize.return_value = mock_tts_status

                connector._zenoh_tts_status_request(mock_sample)

                assert connector.tts_enabled is True
                mock_response_pub.put.assert_called_once()

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    def test_stop(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test stopping the connector."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakKokoroTTSConnector(default_config)

        connector.stop()

        mock_zenoh_session.close.assert_called_once()
        mock_tts_instance.stop.assert_called_once()

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    def test_stop_no_session(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
    ):
        """Test stopping the connector when session is None."""
        mock_open_zenoh_session.side_effect = Exception("Failed to open session")
        mock_tts_instance = Mock()
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakKokoroTTSConnector(default_config)
        connector.stop()

        mock_tts_instance.stop.assert_called_once()

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    def test_stop_no_tts(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test stopping the connector when TTS is None."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts_provider.return_value = mock_tts_instance

        connector = SpeakKokoroTTSConnector(default_config)
        connector.tts = None  # type: ignore

        connector.stop()

        mock_zenoh_session.close.assert_called_once()

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    def test_last_voice_command_time_initialization(
        self,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test that last_voice_command_time is initialized."""
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_provider.return_value = Mock()

        start_time = time.time()
        connector = SpeakKokoroTTSConnector(default_config)
        end_time = time.time()

        assert start_time <= connector.last_voice_command_time <= end_time

    @patch("actions.speak.connector.kokoro_tts.open_zenoh_session")
    @patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider")
    @patch("actions.speak.connector.kokoro_tts.IOProvider")
    @patch("actions.speak.connector.kokoro_tts.TeleopsConversationProvider")
    @patch("actions.speak.connector.kokoro_tts.uuid4")
    def test_audio_status_initialization(
        self,
        mock_uuid4,
        mock_conversation_provider,
        mock_io_provider,
        mock_tts_provider,
        mock_open_zenoh_session,
        default_config,
        mock_zenoh_session,
    ):
        """Test that audio status is properly initialized."""
        mock_uuid4.return_value = "test-uuid"
        mock_open_zenoh_session.return_value = mock_zenoh_session
        mock_tts_provider.return_value = Mock()

        with patch(
            "actions.speak.connector.kokoro_tts.prepare_header"
        ) as mock_prepare_header:
            mock_prepare_header.return_value = "test-header"

            connector = SpeakKokoroTTSConnector(default_config)

            assert connector.audio_status is not None
            assert (
                connector.audio_status.status_speaker
                == AudioStatus.STATUS_SPEAKER.READY.value
            )
            mock_prepare_header.assert_called_with("test-uuid")


if __name__ == "__main__":
    pytest.main([__file__])
