from unittest.mock import Mock, PropertyMock, patch

import pytest

from actions.speak.connector.elevenlabs_people_tts import (
    SpeakElevenLabsTTSConfig,
    SpeakElevenLabsTTSConnector,
)
from actions.speak.interface import SpeakInput


@pytest.fixture
def mock_config():
    """Create a mock config with default values."""
    config = Mock(spec=SpeakElevenLabsTTSConfig)
    config.api_key = "test_api_key"
    config.elevenlabs_api_key = "test_elevenlabs_key"
    config.voice_id = "default_voice_id"
    config.model_id = "eleven_flash_v2_5"
    config.output_format = "pcm_16000"
    config.silence_rate = 0
    config.enable_tts_interrupt = False
    config.voice_ids = {
        "John": "john_voice_id",
        "Jane": "jane_voice_id",
    }
    return config


@pytest.fixture
def mock_config_no_voice_ids():
    """Create a mock config without voice_ids mapping."""
    config = Mock(spec=SpeakElevenLabsTTSConfig)
    config.api_key = "test_api_key"
    config.elevenlabs_api_key = "test_elevenlabs_key"
    config.voice_id = "default_voice_id"
    config.model_id = "eleven_flash_v2_5"
    config.output_format = "pcm_16000"
    config.silence_rate = 0
    config.enable_tts_interrupt = False
    config.voice_ids = None
    return config


@pytest.fixture
def speak_input():
    """Create a SpeakInput instance for testing."""
    return SpeakInput(action="Hello, world!")


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
def test_init_with_voice_ids(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
):
    """Test initialization with voice_ids mapping."""
    mock_session = Mock()
    mock_pub = Mock()
    mock_open_zenoh_session.return_value = mock_session
    mock_session.declare_publisher.return_value = mock_pub
    mock_session.declare_subscriber.return_value = Mock()

    connector = SpeakElevenLabsTTSConnector(mock_config)

    mock_open_zenoh_session.assert_called_once()
    mock_tts_provider.assert_called_once_with(
        url="https://api.openmind.com/api/core/elevenlabs/tts",
        api_key="test_api_key",
        elevenlabs_api_key="test_elevenlabs_key",
        voice_id="default_voice_id",
        model_id="eleven_flash_v2_5",
        output_format="pcm_16000",
        enable_tts_interrupt=False,
    )

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
@pytest.mark.asyncio
async def test_get_voice_id_with_face_presence_match(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
    speak_input,
):
    """Test that _get_voice_id returns mapped voice when face presence matches."""
    mock_io_instance = Mock()
    mock_io_instance.get_input = Mock()
    type(mock_io_instance).tick_counter = PropertyMock(return_value=123)
    mock_io_provider.return_value = mock_io_instance

    connector = SpeakElevenLabsTTSConnector(mock_config)

    mock_face_presence = Mock()
    mock_face_presence.input = "In Camera View: 1 known (John). Closest: John."
    mock_face_presence.tick = 123

    mock_io_instance.get_input.return_value = mock_face_presence

    voice_id = connector._get_voice_id(speak_input)

    assert voice_id == "john_voice_id"

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
@pytest.mark.asyncio
async def test_get_voice_id_with_face_presence_no_match(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
    speak_input,
):
    """Test that _get_voice_id returns default voice when face presence doesn't match."""
    mock_io_instance = Mock()
    mock_io_instance.get_input = Mock()
    type(mock_io_instance).tick_counter = PropertyMock(return_value=123)
    mock_io_provider.return_value = mock_io_instance

    connector = SpeakElevenLabsTTSConnector(mock_config)

    mock_face_presence = Mock()
    mock_face_presence.input = "Unknown"
    mock_face_presence.tick = 123

    mock_io_instance.get_input.return_value = mock_face_presence

    voice_id = connector._get_voice_id(speak_input)

    assert voice_id == "default_voice_id"

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
@pytest.mark.asyncio
async def test_get_voice_id_without_face_presence(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
    speak_input,
):
    """Test that _get_voice_id returns default voice when no face presence input."""
    mock_io_instance = Mock()
    mock_io_instance.get_input = Mock(return_value=None)
    mock_io_provider.return_value = mock_io_instance

    connector = SpeakElevenLabsTTSConnector(mock_config)

    voice_id = connector._get_voice_id(speak_input)

    assert voice_id == "default_voice_id"

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
@pytest.mark.asyncio
async def test_get_voice_id_without_voice_ids_mapping(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config_no_voice_ids,
    speak_input,
):
    """Test that _get_voice_id returns default voice when voice_ids is None."""
    mock_io_instance = Mock()
    mock_io_instance.get_input = Mock()
    type(mock_io_instance).tick_counter = PropertyMock(return_value=123)
    mock_io_provider.return_value = mock_io_instance

    connector = SpeakElevenLabsTTSConnector(mock_config_no_voice_ids)

    mock_face_presence = Mock()
    mock_face_presence.input = "John"
    mock_face_presence.tick = 123

    mock_io_instance.get_input.return_value = mock_face_presence

    voice_id = connector._get_voice_id(speak_input)

    assert voice_id == "default_voice_id"

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
@pytest.mark.asyncio
async def test_get_voice_id_with_stale_face_presence(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
    speak_input,
):
    """Test that _get_voice_id ignores stale face presence data."""
    mock_io_instance = Mock()
    mock_io_instance.get_input = Mock()
    type(mock_io_instance).tick_counter = PropertyMock(return_value=123)
    mock_io_provider.return_value = mock_io_instance

    connector = SpeakElevenLabsTTSConnector(mock_config)

    mock_face_presence = Mock()
    mock_face_presence.input = "John"
    mock_face_presence.tick = 100

    mock_io_instance.get_input.return_value = mock_face_presence

    voice_id = connector._get_voice_id(speak_input)

    assert voice_id == "default_voice_id"

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
@pytest.mark.asyncio
async def test_get_voice_id_with_single_known_face_camera_format(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
    speak_input,
):
    """Test that _get_voice_id extracts name from camera format with single known face."""
    mock_io_instance = Mock()
    mock_io_instance.get_input = Mock()
    type(mock_io_instance).tick_counter = PropertyMock(return_value=123)
    mock_io_provider.return_value = mock_io_instance

    connector = SpeakElevenLabsTTSConnector(mock_config)

    mock_face_presence = Mock()
    mock_face_presence.input = "In Camera View: 1 known (John) and 1 unknown face. Closest: John."
    mock_face_presence.tick = 123

    mock_io_instance.get_input.return_value = mock_face_presence

    voice_id = connector._get_voice_id(speak_input)

    assert voice_id == "john_voice_id"

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
@pytest.mark.asyncio
async def test_get_voice_id_with_multiple_known_faces_picks_first(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
    speak_input,
):
    """Test that _get_voice_id picks the first name when multiple faces are detected."""
    mock_io_instance = Mock()
    mock_io_instance.get_input = Mock()
    type(mock_io_instance).tick_counter = PropertyMock(return_value=123)
    mock_io_provider.return_value = mock_io_instance

    connector = SpeakElevenLabsTTSConnector(mock_config)

    mock_face_presence = Mock()
    mock_face_presence.input = "In Camera View: 2 known (John and Jane) and 2 unknown faces. Closest: John."
    mock_face_presence.tick = 123

    mock_io_instance.get_input.return_value = mock_face_presence

    voice_id = connector._get_voice_id(speak_input)

    assert voice_id == "john_voice_id"

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
@pytest.mark.asyncio
async def test_get_voice_id_with_multiple_known_faces_no_match_for_first(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
    speak_input,
):
    """Test that _get_voice_id returns default when first name doesn't match voice_ids."""
    mock_io_instance = Mock()
    mock_io_instance.get_input = Mock()
    type(mock_io_instance).tick_counter = PropertyMock(return_value=123)
    mock_io_provider.return_value = mock_io_instance

    connector = SpeakElevenLabsTTSConnector(mock_config)

    mock_face_presence = Mock()
    mock_face_presence.input = "In Camera View: 2 known (shicai and Jane) and 2 unknown faces."
    mock_face_presence.tick = 123

    mock_io_instance.get_input.return_value = mock_face_presence

    voice_id = connector._get_voice_id(speak_input)

    assert voice_id == "default_voice_id"

    connector.stop()


@patch("actions.speak.connector.base_elevenlabs_tts.open_zenoh_session")
@patch("actions.speak.connector.base_elevenlabs_tts.ElevenLabsTTSProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.IOProvider")
@patch("actions.speak.connector.base_elevenlabs_tts.TeleopsConversationProvider")
@pytest.mark.asyncio
async def test_connect_uses_mapped_voice_id(
    mock_conversation_provider,
    mock_io_provider,
    mock_tts_provider,
    mock_open_zenoh_session,
    mock_config,
    speak_input,
):
    """Test that connect uses the mapped voice_id from face presence."""
    mock_tts_instance = Mock()
    mock_tts_instance.create_pending_message.return_value = {"text": "processed_message"}
    mock_tts_provider.return_value = mock_tts_instance

    mock_io_instance = Mock()
    mock_io_instance.llm_prompt = "Some prompt"
    mock_io_instance.get_input = Mock()
    type(mock_io_instance).tick_counter = PropertyMock(return_value=123)
    mock_io_provider.return_value = mock_io_instance

    connector = SpeakElevenLabsTTSConnector(mock_config)
    connector.audio_pub = None

    mock_face_presence = Mock()
    mock_face_presence.input = "In Camera View: 1 known (Jane). Closest: Jane."
    mock_face_presence.tick = 123

    mock_io_instance.get_input.return_value = mock_face_presence

    await connector.connect(speak_input)

    mock_tts_instance.create_pending_message.assert_called_once_with("Hello, world!", "jane_voice_id")

    connector.stop()
