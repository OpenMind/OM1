from unittest.mock import Mock, patch

import pytest

from actions.speak.connector.base_elevenlabs_tts import (
    BaseElevenLabsTTSConfig,
    BaseElevenLabsTTSConnector,
)
from actions.speak.interface import SpeakInput
from zenoh_msgs import AudioStatus, TTSStatusRequest
from zenoh_msgs.idl.std_msgs import String


class ConcreteElevenLabsTTSConnector(BaseElevenLabsTTSConnector):
    """Concrete implementation for testing."""

    def _get_voice_id(self, output_interface: SpeakInput) -> str:
        return self.config.voice_id


@pytest.fixture
def mock_config():
    """Create a mock config with default values."""
    config = Mock(spec=BaseElevenLabsTTSConfig)
    config.api_key = "test_api_key"
    config.elevenlabs_api_key = "test_elevenlabs_key"
    config.voice_id = "test_voice_id"
    config.model_id = "eleven_flash_v2_5"
    config.output_format = "pcm_16000"
    config.silence_rate = 0
    config.enable_tts_interrupt = False
    return config


@pytest.fixture
def speak_input():
    """Create a SpeakInput instance for testing."""
    return SpeakInput(action="Hello, world!")


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
def test_init_with_full_config(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
):
    """Test initialization with full configuration."""
    mock_session = Mock()
    mock_pub = Mock()
    mock_open_zenoh_session.return_value = mock_session
    mock_session.declare_publisher.return_value = mock_pub
    mock_session.declare_subscriber.return_value = Mock()

    mock_tts_instance = Mock()
    mock_tts_provider.return_value = mock_tts_instance

    connector = ConcreteElevenLabsTTSConnector(mock_config)

    mock_open_zenoh_session.assert_called_once()
    assert mock_session.declare_publisher.call_count == 2
    mock_session.declare_publisher.assert_any_call("robot/status/audio")
    mock_session.declare_publisher.assert_any_call("om/tts/response")

    assert mock_session.declare_subscriber.call_count == 2
    mock_pub.put.assert_called_once()

    mock_tts_provider.assert_called_once_with(
        url="https://api.openmind.com/api/core/elevenlabs/tts",
        api_key="test_api_key",
        elevenlabs_api_key="test_elevenlabs_key",
        voice_id="test_voice_id",
        model_id="eleven_flash_v2_5",
        output_format="pcm_16000",
        enable_tts_interrupt=False,
    )

    mock_tts_instance.start.assert_called_once()
    mock_tts_instance.configure.assert_called_once()

    assert connector.silence_rate == 0
    assert connector.silence_counter == 0
    assert connector.tts_enabled is True
    assert connector.audio_topic == "robot/status/audio"

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
def test_zenoh_audio_message(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
):
    """Test Zenoh audio message handling."""
    mock_open_zenoh_session.return_value = Mock()

    connector = ConcreteElevenLabsTTSConnector(mock_config)

    mock_audio_status = Mock(spec=AudioStatus)
    mock_data = Mock()
    mock_data.payload.to_bytes.return_value = b"serialized_data"

    with patch.object(AudioStatus, "deserialize", return_value=mock_audio_status) as mock_deserialize:
        connector.zenoh_audio_message(mock_data)

    mock_deserialize.assert_called_once_with(b"serialized_data")
    assert connector.audio_status == mock_audio_status

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
@pytest.mark.asyncio
async def test_connect_normal_flow(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
    speak_input,
):
    """Test normal connect flow."""
    mock_tts_instance = Mock()
    mock_tts_instance.create_pending_message.return_value = {"text": "processed_message"}
    mock_tts_provider.return_value = mock_tts_instance

    mock_io_instance = Mock()
    mock_io_instance.llm_prompt = "Some prompt without voice input"
    mock_io_provider.return_value = mock_io_instance

    connector = ConcreteElevenLabsTTSConnector(mock_config)
    connector.audio_pub = None

    await connector.connect(speak_input)

    mock_tts_instance.create_pending_message.assert_called_once_with("Hello, world!", "test_voice_id")
    mock_tts_instance.add_pending_message.assert_called_once_with({"text": "processed_message"})

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
@pytest.mark.asyncio
async def test_connect_with_disabled_tts(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
    speak_input,
):
    """Test connect when TTS is disabled."""
    mock_tts_instance = Mock()
    mock_tts_provider.return_value = mock_tts_instance

    connector = ConcreteElevenLabsTTSConnector(mock_config)
    connector.tts_enabled = False
    connector.audio_pub = None

    await connector.connect(speak_input)

    mock_tts_instance.create_pending_message.assert_not_called()

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
@pytest.mark.asyncio
async def test_connect_with_silence_rate_skip(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
    speak_input,
):
    """Test connect flow with silence rate causing skip."""
    mock_config.silence_rate = 2

    mock_tts_instance = Mock()
    mock_tts_provider.return_value = mock_tts_instance

    mock_io_instance = Mock()
    mock_io_instance.llm_prompt = "Some prompt without voice"
    mock_io_provider.return_value = mock_io_instance

    connector = ConcreteElevenLabsTTSConnector(mock_config)
    connector.audio_pub = None

    await connector.connect(speak_input)

    assert connector.silence_counter == 1
    mock_tts_instance.create_pending_message.assert_not_called()

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
@pytest.mark.asyncio
async def test_connect_with_silence_rate_voice_input(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
    speak_input,
):
    """Test connect flow with silence rate but voice input present."""
    mock_config.silence_rate = 2

    mock_tts_instance = Mock()
    mock_tts_instance.create_pending_message.return_value = {"text": "processed_message"}
    mock_tts_provider.return_value = mock_tts_instance

    mock_io_instance = Mock()
    mock_io_instance.llm_prompt = "Some prompt with Voice: data"
    mock_io_provider.return_value = mock_io_instance

    connector = ConcreteElevenLabsTTSConnector(mock_config)
    connector.audio_pub = None

    await connector.connect(speak_input)

    assert connector.silence_counter == 0
    mock_tts_instance.create_pending_message.assert_called_once()

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
def test_zenoh_tts_status_request_read(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
):
    """Test TTS status request for reading current status."""
    mock_session = Mock()
    mock_pub = Mock()
    mock_open_zenoh_session.return_value = mock_session
    mock_session.declare_publisher.return_value = mock_pub

    connector = ConcreteElevenLabsTTSConnector(mock_config)

    mock_pub.reset_mock()

    mock_data = Mock()
    mock_tts_status = Mock(spec=TTSStatusRequest)
    mock_tts_status.code = 2
    mock_tts_status.request_id = String(data="test_request_id")
    mock_header = Mock()
    mock_header.frame_id = "test_frame_id"
    mock_tts_status.header = mock_header

    mock_data.payload.to_bytes.return_value = b"serialized_data"

    with patch.object(TTSStatusRequest, "deserialize", return_value=mock_tts_status):
        connector._zenoh_tts_status_request(mock_data)

    mock_pub.put.assert_called_once()

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
def test_zenoh_tts_status_request_enable(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
):
    """Test TTS status request for enabling TTS."""
    mock_session = Mock()
    mock_pub = Mock()
    mock_open_zenoh_session.return_value = mock_session
    mock_session.declare_publisher.return_value = mock_pub

    connector = ConcreteElevenLabsTTSConnector(mock_config)
    connector.tts_enabled = False

    # Reset the mock to ignore the initialization call
    mock_pub.reset_mock()

    mock_data = Mock()
    mock_tts_status = Mock(spec=TTSStatusRequest)
    mock_tts_status.code = 1
    mock_tts_status.request_id = String(data="test_request_id")
    mock_header = Mock()
    mock_header.frame_id = "test_frame_id"
    mock_tts_status.header = mock_header

    mock_data.payload.to_bytes.return_value = b"serialized_data"

    with patch.object(TTSStatusRequest, "deserialize", return_value=mock_tts_status):
        connector._zenoh_tts_status_request(mock_data)

    assert connector.tts_enabled is True
    mock_pub.put.assert_called_once()

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
def test_zenoh_tts_status_request_disable(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
):
    """Test TTS status request for disabling TTS."""
    mock_session = Mock()
    mock_pub = Mock()
    mock_open_zenoh_session.return_value = mock_session
    mock_session.declare_publisher.return_value = mock_pub

    connector = ConcreteElevenLabsTTSConnector(mock_config)

    mock_pub.reset_mock()

    mock_data = Mock()
    mock_tts_status = Mock(spec=TTSStatusRequest)
    mock_tts_status.code = 0
    mock_tts_status.request_id = String(data="test_request_id")
    mock_header = Mock()
    mock_header.frame_id = "test_frame_id"
    mock_tts_status.header = mock_header

    mock_data.payload.to_bytes.return_value = b"serialized_data"

    with patch.object(TTSStatusRequest, "deserialize", return_value=mock_tts_status):
        connector._zenoh_tts_status_request(mock_data)

    assert connector.tts_enabled is False
    mock_pub.put.assert_called_once()

    connector.stop()
