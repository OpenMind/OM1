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


@pytest.fixture
def mock_zenoh_sample():
    """Create a mock Zenoh sample."""
    sample = Mock()
    sample.payload.to_bytes.return_value = b"test_data"
    return sample


@pytest.fixture
def mock_tts_status_header():
    """Create a mock TTS status header."""
    header = Mock()
    header.frame_id = "test_frame"
    return header


@pytest.fixture(autouse=True)
def reset_mocks(mock_om1_speech_module, mock_zenoh_module):
    """Reset all mock objects between tests."""
    mock_om1_speech_module.AudioOutputLiveStream.reset_mock()
    mock_om1_speech_module.AudioOutputLiveStream.return_value = MagicMock()
    mock_zenoh_module.reset_mock()
    yield


@pytest.fixture
def common_mocks(mock_zenoh_session):
    """Provide commonly used mocks for connector tests."""
    with (
        patch(
            "actions.speak.connector.kokoro_tts.open_zenoh_session"
        ) as mock_open_zenoh,
        patch("actions.speak.connector.kokoro_tts.KokoroTTSProvider") as mock_tts,
        patch("actions.speak.connector.kokoro_tts.IOProvider") as mock_io,
        patch(
            "actions.speak.connector.kokoro_tts.TeleopsConversationProvider"
        ) as mock_conv,
    ):

        mock_open_zenoh.return_value = mock_zenoh_session
        mock_tts_instance = Mock()
        mock_tts.return_value = mock_tts_instance

        yield {
            "open_zenoh_session": mock_open_zenoh,
            "tts_provider": mock_tts,
            "tts_instance": mock_tts_instance,
            "io_provider": mock_io,
            "conversation_provider": mock_conv,
            "zenoh_session": mock_zenoh_session,
        }


def create_tts_status_mock(code, request_id="test_request_id", frame_id="test_frame"):
    """Helper to create TTS status mock objects."""
    header = Mock()
    header.frame_id = frame_id

    tts_status = Mock()
    tts_status.code = code
    tts_status.request_id = String(request_id)
    tts_status.header = header

    return tts_status


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

    def test_init_with_default_config(self, default_config, common_mocks):
        """Test initialization with default configuration."""
        connector = SpeakKokoroTTSConnector(default_config)

        common_mocks["open_zenoh_session"].assert_called_once()
        assert common_mocks["zenoh_session"].declare_publisher.call_count == 2
        assert common_mocks["zenoh_session"].declare_subscriber.call_count == 2

        common_mocks["tts_provider"].assert_called_once_with(
            url="http://127.0.0.1:8880/v1",
            api_key=None,
            voice_id="af_bella",
            model_id="kokoro",
            output_format="pcm",
            rate=24000,
            enable_tts_interrupt=False,
        )

        common_mocks["tts_instance"].start.assert_called_once()
        common_mocks["tts_instance"].configure.assert_called_once()

        assert connector.silence_rate == 0
        assert connector.silence_counter == 0
        assert connector.tts_enabled is True
        assert connector.session == common_mocks["zenoh_session"]

    def test_init_with_custom_config(self, custom_config, common_mocks):
        """Test initialization with custom configuration."""
        connector = SpeakKokoroTTSConnector(custom_config)

        common_mocks["tts_provider"].assert_called_once_with(
            url="http://127.0.0.1:8880/v1",
            api_key="test_api_key",
            voice_id="custom_voice",
            model_id="custom_model",
            output_format="wav",
            rate=48000,
            enable_tts_interrupt=True,
        )

        assert connector.silence_rate == 2

    def test_init_zenoh_failure(self, default_config, common_mocks):
        """Test initialization when Zenoh session fails to open."""
        common_mocks["open_zenoh_session"].side_effect = Exception(
            "Zenoh connection failed"
        )

        connector = SpeakKokoroTTSConnector(default_config)

        assert connector.session is None
        assert connector.audio_pub is None

    @pytest.mark.asyncio
    async def test_connect_tts_enabled(self, default_config, common_mocks, speak_input):
        """Test connect method when TTS is enabled."""
        common_mocks["tts_instance"].create_pending_message.return_value = {
            "id": "test_id",
            "text": "Hello, world!",
        }
        mock_audio_pub = Mock()
        common_mocks["zenoh_session"].declare_publisher.return_value = mock_audio_pub

        mock_io_instance = Mock()
        mock_io_instance.llm_prompt = "INPUT: Voice: Hello"
        common_mocks["io_provider"].return_value = mock_io_instance

        mock_conversation_instance = Mock()
        common_mocks["conversation_provider"].return_value = mock_conversation_instance

        connector = SpeakKokoroTTSConnector(default_config)
        connector.io_provider = mock_io_instance
        connector.conversation_provider = mock_conversation_instance

        await connector.connect(speak_input)

        common_mocks["tts_instance"].create_pending_message.assert_called_once_with(
            "Hello, world!"
        )
        mock_conversation_instance.store_robot_message.assert_called_once_with(
            "Hello, world!"
        )
        mock_audio_pub.put.assert_called()

    @pytest.mark.asyncio
    async def test_connect_tts_disabled(
        self, default_config, common_mocks, speak_input
    ):
        """Test connect method when TTS is disabled."""
        connector = SpeakKokoroTTSConnector(default_config)
        connector.tts_enabled = False

        await connector.connect(speak_input)

        common_mocks["tts_instance"].create_pending_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_silence_rate_skip(self, common_mocks, speak_input):
        """Test connect method with silence rate causing skip."""
        config = SpeakKokoroTTSConfig(silence_rate=2)
        common_mocks["tts_instance"].create_pending_message.return_value = {
            "id": "test_id",
            "text": "Hello, world!",
        }

        mock_io_instance = Mock()
        mock_io_instance.llm_prompt = "INPUT: Text: Hello"
        common_mocks["io_provider"].return_value = mock_io_instance

        connector = SpeakKokoroTTSConnector(config)
        connector.io_provider = mock_io_instance

        await connector.connect(speak_input)
        assert connector.silence_counter == 1
        common_mocks["tts_instance"].create_pending_message.assert_not_called()

        await connector.connect(speak_input)
        assert connector.silence_counter == 2
        common_mocks["tts_instance"].create_pending_message.assert_not_called()

        await connector.connect(speak_input)
        assert connector.silence_counter == 0
        common_mocks["tts_instance"].create_pending_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_without_audio_publisher(
        self, default_config, common_mocks, speak_input
    ):
        """Test connect method when audio publisher is None."""
        common_mocks["tts_instance"].create_pending_message.return_value = {
            "id": "test_id",
            "text": "Hello, world!",
        }

        connector = SpeakKokoroTTSConnector(default_config)
        connector.audio_pub = None

        await connector.connect(speak_input)

        common_mocks["tts_instance"].add_pending_message.assert_called_once_with(
            {"id": "test_id", "text": "Hello, world!"}
        )

    def test_zenoh_audio_message(self, default_config, common_mocks, mock_zenoh_sample):
        """Test processing of Zenoh audio status messages."""
        connector = SpeakKokoroTTSConnector(default_config)

        mock_audio_status = Mock()

        with patch(
            "actions.speak.connector.kokoro_tts.AudioStatus"
        ) as mock_audio_status_class:
            mock_audio_status_class.deserialize.return_value = mock_audio_status

            connector.zenoh_audio_message(mock_zenoh_sample)

            mock_audio_status_class.deserialize.assert_called_once_with(b"test_data")
            assert connector.audio_status == mock_audio_status

    def test_zenoh_audio_message_error_handling(
        self, default_config, common_mocks, mock_zenoh_sample
    ):
        """Test error handling in zenoh_audio_message."""
        connector = SpeakKokoroTTSConnector(default_config)
        original_audio_status = connector.audio_status

        with patch(
            "actions.speak.connector.kokoro_tts.AudioStatus"
        ) as mock_audio_status_class:
            mock_audio_status_class.deserialize.side_effect = Exception(
                "Deserialization error"
            )

            connector.zenoh_audio_message(mock_zenoh_sample)

            assert connector.audio_status == original_audio_status

    @pytest.mark.parametrize(
        "code,expected_enabled",
        [
            (1, True),  # Enable TTS
            (0, False),  # Disable TTS
        ],
    )
    def test_zenoh_tts_status_request_enable_disable(
        self, default_config, common_mocks, mock_zenoh_sample, code, expected_enabled
    ):
        """Test TTS status request to enable/disable TTS."""
        mock_response_pub = Mock()
        connector = SpeakKokoroTTSConnector(default_config)
        connector._zenoh_tts_status_response_pub = mock_response_pub
        connector.tts_enabled = not expected_enabled

        tts_status = create_tts_status_mock(code)

        with (
            patch(
                "actions.speak.connector.kokoro_tts.TTSStatusRequest"
            ) as mock_request_class,
            patch("actions.speak.connector.kokoro_tts.TTSStatusResponse"),
        ):
            mock_request_class.deserialize.return_value = tts_status

            connector._zenoh_tts_status_request(mock_zenoh_sample)

            assert connector.tts_enabled is expected_enabled
            mock_response_pub.put.assert_called_once()

    def test_zenoh_tts_status_request_read(
        self, default_config, common_mocks, mock_zenoh_sample
    ):
        """Test TTS status request to read current status."""
        mock_response_pub = Mock()
        connector = SpeakKokoroTTSConnector(default_config)
        connector._zenoh_tts_status_response_pub = mock_response_pub
        connector.tts_enabled = True

        tts_status = create_tts_status_mock(2)  # Read status

        with (
            patch(
                "actions.speak.connector.kokoro_tts.TTSStatusRequest"
            ) as mock_request_class,
            patch("actions.speak.connector.kokoro_tts.TTSStatusResponse"),
        ):
            mock_request_class.deserialize.return_value = tts_status

            connector._zenoh_tts_status_request(mock_zenoh_sample)

            assert connector.tts_enabled is True
            mock_response_pub.put.assert_called_once()

    def test_zenoh_tts_status_request_null_publisher(
        self, default_config, common_mocks, mock_zenoh_sample
    ):
        """Test TTS status request when response publisher is None."""
        connector = SpeakKokoroTTSConnector(default_config)
        connector._zenoh_tts_status_response_pub = None

        tts_status = create_tts_status_mock(1)  # Enable TTS

        with (
            patch(
                "actions.speak.connector.kokoro_tts.TTSStatusRequest"
            ) as mock_request_class,
            patch("actions.speak.connector.kokoro_tts.TTSStatusResponse"),
        ):
            mock_request_class.deserialize.return_value = tts_status

            connector._zenoh_tts_status_request(mock_zenoh_sample)

            assert connector.tts_enabled is True

    def test_zenoh_tts_status_request_error_handling(
        self, default_config, common_mocks, mock_zenoh_sample
    ):
        """Test error handling in _zenoh_tts_status_request."""
        connector = SpeakKokoroTTSConnector(default_config)
        original_tts_enabled = connector.tts_enabled

        with patch(
            "actions.speak.connector.kokoro_tts.TTSStatusRequest"
        ) as mock_request_class:
            mock_request_class.deserialize.side_effect = Exception(
                "Deserialization error"
            )

            connector._zenoh_tts_status_request(mock_zenoh_sample)

            assert connector.tts_enabled == original_tts_enabled

    def test_stop(self, default_config, common_mocks):
        """Test stopping the connector."""
        connector = SpeakKokoroTTSConnector(default_config)

        connector.stop()

        common_mocks["zenoh_session"].close.assert_called_once()
        common_mocks["tts_instance"].stop.assert_called_once()

    def test_stop_no_session(self, default_config, common_mocks):
        """Test stopping the connector when session is None."""
        common_mocks["open_zenoh_session"].side_effect = Exception(
            "Failed to open session"
        )
        connector = SpeakKokoroTTSConnector(default_config)

        connector.stop()

        common_mocks["tts_instance"].stop.assert_called_once()

    def test_stop_no_tts(self, default_config, common_mocks):
        """Test stopping the connector when TTS is None."""
        connector = SpeakKokoroTTSConnector(default_config)
        connector.tts = None  # type: ignore

        connector.stop()

        common_mocks["zenoh_session"].close.assert_called_once()

    def test_last_voice_command_time_initialization(self, default_config, common_mocks):
        """Test that last_voice_command_time is initialized."""
        start_time = time.time()
        connector = SpeakKokoroTTSConnector(default_config)
        end_time = time.time()

        assert start_time <= connector.last_voice_command_time <= end_time

    @patch("actions.speak.connector.kokoro_tts.uuid4")
    def test_audio_status_initialization(
        self, mock_uuid4, default_config, common_mocks
    ):
        """Test that audio status is properly initialized."""
        mock_uuid4.return_value = "test-uuid"

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
