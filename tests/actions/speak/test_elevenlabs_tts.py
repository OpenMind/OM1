# from unittest.mock import Mock, patch

# import pytest

# from actions.base import ActionConfig
# from actions.speak.connector.elevenlabs_tts import SpeakElevenLabsTTSConnector
# from actions.speak.interface import SpeakInput
# from zenoh_msgs import AudioStatus


# @pytest.fixture
# def mock_config():
#     """Create a mock ActionConfig with default values."""
#     config = Mock(spec=ActionConfig)
#     config.microphone_device_id = "test_mic_id"
#     config.microphone_name = "test_microphone"
#     config.api_key = "test_api_key"
#     config.elevenlabs_api_key = "test_elevenlabs_key"
#     config.voice_id = "test_voice_id"
#     config.model_id = "eleven_flash_v2_5"
#     config.output_format = "mp3_44100_128"
#     config.silence_rate = 0
#     config.auto_sleep_mode = False
#     return config


# @pytest.fixture
# def mock_minimal_config():
#     """Create a mock ActionConfig with minimal values (using defaults)."""
#     config = Mock(spec=ActionConfig)
#     return config


# @pytest.fixture
# def speak_input():
#     """Create a SpeakInput instance for testing."""
#     return SpeakInput(action="Hello, world!")


# @patch("actions.speak.connector.elevenlabs_tts.open_zenoh_session")
# @patch("actions.speak.connector.elevenlabs_tts.ASRRTSPProvider")
# @patch("actions.speak.connector.elevenlabs_tts.ElevenLabsTTSProvider")
# @patch("actions.speak.connector.elevenlabs_tts.IOProvider")
# def test_init_with_full_config(
#     mock_io_provider,
#     mock_tts_provider,
#     mock_asr_provider,
#     mock_open_zenoh_session,
#     mock_config,
# ):
#     """Test initialization with full configuration."""
#     mock_session = Mock()
#     mock_pub = Mock()
#     mock_open_zenoh_session.return_value = mock_session
#     mock_session.declare_publisher.return_value = mock_pub
#     mock_session.declare_subscriber.return_value = Mock()

#     connector = SpeakElevenLabsTTSConnector(mock_config)

#     mock_open_zenoh_session.assert_called_once()
#     assert mock_session.declare_publisher.call_count == 2
#     mock_session.declare_publisher.assert_any_call("robot/status/audio")
#     mock_session.declare_publisher.assert_any_call("om/tts/response")

#     assert mock_session.declare_subscriber.call_count == 2
#     mock_session.declare_subscriber.assert_any_call(
#         "robot/status/audio", connector.zenoh_audio_message
#     )
#     mock_session.declare_subscriber.assert_any_call(
#         "om/tts/request", connector._zenoh_tts_status_request
#     )
#     mock_pub.put.assert_called_once()

#     mock_asr_provider.assert_called_once_with(
#         ws_url="wss://api.openmind.org/api/core/google/asr?api_key=test_api_key"
#     )

#     mock_tts_provider.assert_called_once_with(
#         url="https://api.openmind.org/api/core/elevenlabs/tts",
#         api_key="test_api_key",
#         elevenlabs_api_key="test_elevenlabs_key",
#         voice_id="test_voice_id",
#         model_id="eleven_flash_v2_5",
#         output_format="mp3_44100_128",
#     )

#     connector.tts.start.assert_called_once()

#     assert connector.silence_rate == 0
#     assert connector.silence_counter == 0
#     assert connector.audio_topic == "robot/status/audio"
#     assert connector.session == mock_session
#     assert connector.audio_pub == mock_pub

#     connector.stop()

# @patch("actions.speak.connector.elevenlabs_tts.open_zenoh_session")
# @patch("actions.speak.connector.elevenlabs_tts.ASRRTSPProvider")
# @patch("actions.speak.connector.elevenlabs_tts.ElevenLabsTTSProvider")
# @patch("actions.speak.connector.elevenlabs_tts.IOProvider")
# def test_init_with_minimal_config(
#     mock_io_provider,
#     mock_tts_provider,
#     mock_asr_provider,
#     mock_open_zenoh_session,
#     mock_minimal_config,
# ):
#     """Test initialization with minimal configuration using defaults."""
#     mock_session = Mock()
#     mock_pub = Mock()
#     mock_open_zenoh_session.return_value = mock_session
#     mock_session.declare_publisher.return_value = mock_pub
#     mock_session.declare_subscriber.return_value = Mock()

#     connector = SpeakElevenLabsTTSConnector(mock_minimal_config)

#     mock_tts_provider.assert_called_once_with(
#         url="https://api.openmind.org/api/core/elevenlabs/tts",
#         api_key=None,
#         elevenlabs_api_key=None,
#         voice_id="JBFqnCBsd6RMkjVDRZzb",  # default
#         model_id="eleven_flash_v2_5",  # default
#         output_format="mp3_44100_128",  # default
#     )

#     mock_asr_provider.assert_called_once_with(
#         ws_url="wss://api.openmind.org/api/core/google/asr?api_key=None"
#     )

#     connector.stop()


# @patch("actions.speak.connector.elevenlabs_tts.open_zenoh_session")
# @patch("actions.speak.connector.elevenlabs_tts.ASRRTSPProvider")
# @patch("actions.speak.connector.elevenlabs_tts.ElevenLabsTTSProvider")
# @patch("actions.speak.connector.elevenlabs_tts.IOProvider")
# def test_zenoh_audio_message(
#     mock_io_provider,
#     mock_tts_provider,
#     mock_asr_provider,
#     mock_open_zenoh_session,
#     mock_config,
# ):
#     """Test Zenoh audio message handling."""
#     mock_open_zenoh_session.return_value = Mock()

#     connector = SpeakElevenLabsTTSConnector(mock_config)

#     mock_audio_status = Mock(spec=AudioStatus)
#     mock_data = Mock()
#     mock_data.payload.to_bytes.return_value = b"serialized_data"

#     with patch.object(
#         AudioStatus, "deserialize", return_value=mock_audio_status
#     ) as mock_deserialize:
#         connector.zenoh_audio_message(mock_data)

#     mock_deserialize.assert_called_once_with(b"serialized_data")
#     assert connector.audio_status == mock_audio_status

#     connector.stop()

# @patch("actions.speak.connector.elevenlabs_tts.open_zenoh_session")
# @patch("actions.speak.connector.elevenlabs_tts.ASRRTSPProvider")
# @patch("actions.speak.connector.elevenlabs_tts.ElevenLabsTTSProvider")
# @patch("actions.speak.connector.elevenlabs_tts.IOProvider")
# @pytest.mark.asyncio
# async def test_connect_normal_flow(
#     mock_io_provider,
#     mock_tts_provider,
#     mock_asr_provider,
#     mock_open_zenoh_session,
#     mock_config,
#     speak_input,
# ):
#     """Test normal connect flow without silence rate."""

#     connector = SpeakElevenLabsTTSConnector(mock_config)
#     # Set audio_pub to None so connect goes through full flow instead of returning early
#     connector.audio_pub = None
#     connector.tts.create_pending_message.return_value = "processed_message"

#     connector.io_provider.llm_prompt = "Some prompt without voice input"

#     await connector.connect(speak_input)

#     connector.tts.create_pending_message.assert_called_once_with("Hello, world!")
#     connector.tts.register_tts_state_callback.assert_called_once()
#     connector.tts.add_pending_message.assert_called_once_with("processed_message")

#     connector.stop()

# @patch("actions.speak.connector.elevenlabs_tts.open_zenoh_session")
# @patch("actions.speak.connector.elevenlabs_tts.ASRRTSPProvider")
# @patch("actions.speak.connector.elevenlabs_tts.ElevenLabsTTSProvider")
# @patch("actions.speak.connector.elevenlabs_tts.IOProvider")
# @pytest.mark.asyncio
# async def test_connect_with_silence_rate_skip(
#     mock_io_provider,
#     mock_tts_provider,
#     mock_asr_provider,
#     mock_open_zenoh_session,
#     mock_config,
#     speak_input,
# ):
#     """Test connect flow with silence rate causing skip."""
#     # Setup config with silence rate
#     mock_config.silence_rate = 2

#     connector = SpeakElevenLabsTTSConnector(mock_config)
#     connector.audio_pub = None

#     connector.io_provider.llm_prompt = "Some prompt without voice"

#     with patch("actions.speak.connector.elevenlabs_tts.logging") as mock_logging:
#         await connector.connect(speak_input)

#     mock_logging.info.assert_called_with(
#         "Skipping TTS due to silence_rate 2, counter 1"
#     )
#     assert connector.silence_counter == 1

#     connector.tts.create_pending_message.assert_not_called()

#     connector.stop()

# @patch("actions.speak.connector.elevenlabs_tts.open_zenoh_session")
# @patch("actions.speak.connector.elevenlabs_tts.ASRRTSPProvider")
# @patch("actions.speak.connector.elevenlabs_tts.ElevenLabsTTSProvider")
# @patch("actions.speak.connector.elevenlabs_tts.IOProvider")
# @pytest.mark.asyncio
# async def test_connect_with_silence_rate_voice_input(
#     mock_io_provider,
#     mock_tts_provider,
#     mock_asr_provider,
#     mock_open_zenoh_session,
#     mock_config,
#     speak_input,
# ):
#     """Test connect flow with silence rate but voice input present (should not skip)."""
#     mock_config.silence_rate = 2

#     connector = SpeakElevenLabsTTSConnector(mock_config)
#     connector.audio_pub = None
#     connector.tts.create_pending_message.return_value = "processed_message"

#     connector.io_provider.llm_prompt = "Some prompt with INPUT: Voice data"

#     await connector.connect(speak_input)

#     assert connector.silence_counter == 0
#     connector.tts.create_pending_message.assert_called_once_with("Hello, world!")
#     connector.tts.add_pending_message.assert_called_once_with("processed_message")

#     connector.stop()

# @patch("actions.speak.connector.elevenlabs_tts.open_zenoh_session")
# @patch("actions.speak.connector.elevenlabs_tts.ASRRTSPProvider")
# @patch("actions.speak.connector.elevenlabs_tts.ElevenLabsTTSProvider")
# @patch("actions.speak.connector.elevenlabs_tts.IOProvider")
# @pytest.mark.asyncio
# async def test_connect_with_silence_rate_counter_reached(
#     mock_io_provider,
#     mock_tts_provider,
#     mock_asr_provider,
#     mock_open_zenoh_session,
#     mock_config,
#     speak_input,
# ):
#     """Test connect flow when silence counter reaches the rate (should process)."""
#     mock_config.silence_rate = 2

#     connector = SpeakElevenLabsTTSConnector(mock_config)
#     connector.audio_pub = None
#     connector.tts.create_pending_message.return_value = "processed_message"

#     connector.io_provider.llm_prompt = "Some prompt without voice"

#     connector.silence_counter = 2

#     await connector.connect(speak_input)

#     assert connector.silence_counter == 0
#     connector.tts.create_pending_message.assert_called_once_with("Hello, world!")
#     connector.tts.add_pending_message.assert_called_once_with("processed_message")

#     connector.stop()

# def test_audio_status_initial_state():
#     """Test that the initial audio status is set correctly."""
#     with (
#         patch("actions.speak.connector.elevenlabs_tts.open_zenoh_session"),
#         patch("actions.speak.connector.elevenlabs_tts.ASRRTSPProvider"),
#         patch("actions.speak.connector.elevenlabs_tts.ElevenLabsTTSProvider"),
#         patch("actions.speak.connector.elevenlabs_tts.IOProvider"),
#     ):

#         config = Mock(spec=ActionConfig)
#         connector = SpeakElevenLabsTTSConnector(config)

#         assert connector.audio_status.status_mic == AudioStatus.STATUS_MIC.UNKNOWN.value
#         assert (
#             connector.audio_status.status_speaker
#             == AudioStatus.STATUS_SPEAKER.READY.value
#         )
#         assert connector.audio_status.sentence_to_speak.data == ""

#         connector.stop()

# @patch("actions.speak.connector.elevenlabs_tts.open_zenoh_session")
# @patch("actions.speak.connector.elevenlabs_tts.ASRRTSPProvider")
# @patch("actions.speak.connector.elevenlabs_tts.ElevenLabsTTSProvider")
# @patch("actions.speak.connector.elevenlabs_tts.IOProvider")
# @pytest.mark.asyncio
# async def test_connect_audio_status_update(
#     mock_io_provider,
#     mock_tts_provider,
#     mock_asr_provider,
#     mock_open_zenoh_session,
#     mock_config,
#     speak_input,
# ):
#     """Test that audio status is updated correctly during connect."""
#     mock_session = Mock()
#     mock_pub = Mock()
#     mock_open_zenoh_session.return_value = mock_session
#     mock_session.declare_publisher.return_value = mock_pub
#     mock_session.declare_subscriber.return_value = Mock()

#     connector = SpeakElevenLabsTTSConnector(mock_config)
#     connector.audio_pub = None
#     connector.tts.create_pending_message.return_value = "processed_message"

#     connector.io_provider.llm_prompt = "Some prompt"

#     initial_mic_status = AudioStatus.STATUS_MIC.ACTIVE.value
#     connector.audio_status.status_mic = initial_mic_status

#     await connector.connect(speak_input)

#     mock_pub.put.assert_called_once()

#     connector.stop()
