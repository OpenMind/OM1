import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from actions.speak.connector.riva_tts import (
    SpeakRivaTTSConfig,
    SpeakRivaTTSConnector,
)
from actions.speak.interface import SpeakInput


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
    mock_om1_speech.AudioInputStream = MagicMock()
    mock_om1_speech.AudioOutputStream = MagicMock()
    sys.modules["om1_speech"] = mock_om1_speech
    yield mock_om1_speech
    if "om1_speech" in sys.modules:
        del sys.modules["om1_speech"]


@pytest.fixture(autouse=True, scope="session")
def mock_om1_utils_module():
    """Mock the om1_utils module before any imports."""
    mock_om1_utils = MagicMock()
    sys.modules["om1_utils"] = mock_om1_utils
    sys.modules["om1_utils.ws"] = MagicMock()
    yield mock_om1_utils
    if "om1_utils" in sys.modules:
        del sys.modules["om1_utils"]
    if "om1_utils.ws" in sys.modules:
        del sys.modules["om1_utils.ws"]


@pytest.fixture
def default_config():
    """Create a default config for testing."""
    return SpeakRivaTTSConfig()


@pytest.fixture
def custom_config():
    """Create a custom config for testing."""
    return SpeakRivaTTSConfig(
        api_key="test_api_key",
        microphone_device_id=2,
        microphone_name="test_mic",
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
    return session


@pytest.fixture
def mock_zenoh_sample():
    """Create a mock Zenoh sample."""
    sample = Mock()
    sample.payload.to_bytes.return_value = b"test_data"
    return sample


@pytest.fixture(autouse=True)
def reset_mocks(mock_om1_speech_module, mock_zenoh_module, mock_om1_utils_module):
    """Reset all mock objects between tests."""
    mock_om1_speech_module.AudioInputStream.reset_mock()
    mock_om1_speech_module.AudioInputStream.return_value = MagicMock()
    mock_om1_speech_module.AudioOutputStream.reset_mock()
    mock_om1_speech_module.AudioOutputStream.return_value = MagicMock()
    mock_zenoh_module.reset_mock()
    mock_om1_utils_module.reset_mock()
    yield


@pytest.fixture
def common_mocks(mock_zenoh_session):
    """Provide commonly used mocks for connector tests."""
    with (
        patch("actions.speak.connector.riva_tts.open_zenoh_session") as mock_open_zenoh,
        patch("actions.speak.connector.riva_tts.ASRProvider") as mock_asr,
        patch("actions.speak.connector.riva_tts.RivaTTSProvider") as mock_tts,
    ):
        mock_open_zenoh.return_value = mock_zenoh_session

        mock_asr_instance = Mock()
        mock_asr_instance.audio_stream = Mock()
        mock_asr_instance.audio_stream.on_tts_state_change = Mock()
        mock_asr.return_value = mock_asr_instance

        mock_tts_instance = Mock()
        mock_tts.return_value = mock_tts_instance

        yield {
            "open_zenoh_session": mock_open_zenoh,
            "asr_provider": mock_asr,
            "asr_instance": mock_asr_instance,
            "tts_provider": mock_tts,
            "tts_instance": mock_tts_instance,
            "zenoh_session": mock_zenoh_session,
        }


def create_tts_status_mock(code, request_id="test_request_id", frame_id="test_frame"):
    """Helper to create TTS status mock objects."""
    header = Mock()
    header.frame_id = frame_id

    tts_status = Mock()
    tts_status.code = code
    tts_status.request_id = request_id
    tts_status.header = header

    return tts_status


class TestSpeakRivaTTSConfig:
    """Test the configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpeakRivaTTSConfig()

        assert config.api_key is None
        assert config.microphone_device_id is None
        assert config.microphone_name is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = SpeakRivaTTSConfig(
            api_key="my_key",
            microphone_device_id=3,
            microphone_name="my_mic",
        )

        assert config.api_key == "my_key"
        assert config.microphone_device_id == 3
        assert config.microphone_name == "my_mic"


class TestSpeakRivaTTSConnector:
    """Test the Riva TTS connector."""

    def test_init_with_default_config(self, default_config, common_mocks):
        """Test initialization with default configuration."""
        connector = SpeakRivaTTSConnector(default_config)

        common_mocks["open_zenoh_session"].assert_called_once()
        common_mocks["zenoh_session"].declare_subscriber.assert_called_once()
        common_mocks["zenoh_session"].declare_publisher.assert_called_once()

        common_mocks["asr_provider"].assert_called_once_with(
            ws_url="wss://api-asr.openmind.org",
            device_id=None,
            microphone_name=None,
        )
        common_mocks["tts_provider"].assert_called_once_with(
            url="https://api.openmind.org/api/core/riva/tts",
            api_key=None,
        )

        common_mocks["tts_instance"].start.assert_called_once()
        assert connector.tts_enabled is True
        assert connector.session == common_mocks["zenoh_session"]

    def test_init_with_custom_config(self, custom_config, common_mocks):
        """Test initialization with custom configuration."""
        SpeakRivaTTSConnector(custom_config)

        common_mocks["asr_provider"].assert_called_once_with(
            ws_url="wss://api-asr.openmind.org",
            device_id=2,
            microphone_name="test_mic",
        )
        common_mocks["tts_provider"].assert_called_once_with(
            url="https://api.openmind.org/api/core/riva/tts",
            api_key="test_api_key",
        )

    def test_init_zenoh_failure(self, default_config, common_mocks):
        """Test initialization when Zenoh session fails to open."""
        common_mocks["open_zenoh_session"].side_effect = Exception(
            "Zenoh connection failed"
        )

        connector = SpeakRivaTTSConnector(default_config)

        assert connector.session is None
        # ASR and TTS should still be initialized
        common_mocks["asr_provider"].assert_called_once()
        common_mocks["tts_provider"].assert_called_once()
        common_mocks["tts_instance"].start.assert_called_once()


class TestConnect:
    """Test the connect method."""

    @pytest.mark.asyncio
    async def test_connect_valid_input(self, default_config, common_mocks, speak_input):
        """Test connect method with valid input when TTS is enabled."""
        connector = SpeakRivaTTSConnector(default_config)

        await connector.connect(speak_input)

        common_mocks[
            "tts_instance"
        ].register_tts_state_callback.assert_called_once_with(
            common_mocks["asr_instance"].audio_stream.on_tts_state_change
        )
        common_mocks["tts_instance"].add_pending_message.assert_called_once_with(
            "Hello, world!"
        )

    @pytest.mark.asyncio
    async def test_connect_tts_disabled(
        self, default_config, common_mocks, speak_input
    ):
        """Test connect method when TTS is disabled."""
        connector = SpeakRivaTTSConnector(default_config)
        connector.tts_enabled = False

        await connector.connect(speak_input)

        common_mocks["tts_instance"].register_tts_state_callback.assert_not_called()
        common_mocks["tts_instance"].add_pending_message.assert_not_called()


class TestZenohTTSStatusRequest:
    """Test the _zenoh_tts_status_request method."""

    @pytest.mark.parametrize(
        "code,expected_enabled",
        [
            (1, True),  # Enable TTS
            (0, False),  # Disable TTS
        ],
    )
    def test_enable_disable_tts(
        self,
        default_config,
        common_mocks,
        mock_zenoh_sample,
        code,
        expected_enabled,
    ):
        """Test TTS status request to enable/disable TTS."""
        mock_response_pub = Mock()
        connector = SpeakRivaTTSConnector(default_config)
        connector._zenoh_tts_status_response_pub = mock_response_pub
        connector.tts_enabled = not expected_enabled

        tts_status = create_tts_status_mock(code)

        with (
            patch(
                "actions.speak.connector.riva_tts.TTSStatusRequest"
            ) as mock_request_class,
            patch("actions.speak.connector.riva_tts.TTSStatusResponse"),
        ):
            mock_request_class.deserialize.return_value = tts_status

            connector._zenoh_tts_status_request(mock_zenoh_sample)

            assert connector.tts_enabled is expected_enabled
            mock_response_pub.put.assert_called_once()

    def test_read_status(self, default_config, common_mocks, mock_zenoh_sample):
        """Test TTS status request to read current status (code=2)."""
        mock_response_pub = Mock()
        connector = SpeakRivaTTSConnector(default_config)
        connector._zenoh_tts_status_response_pub = mock_response_pub
        connector.tts_enabled = True

        tts_status = create_tts_status_mock(2)

        with (
            patch(
                "actions.speak.connector.riva_tts.TTSStatusRequest"
            ) as mock_request_class,
            patch("actions.speak.connector.riva_tts.TTSStatusResponse"),
        ):
            mock_request_class.deserialize.return_value = tts_status

            connector._zenoh_tts_status_request(mock_zenoh_sample)

            # tts_enabled should remain unchanged
            assert connector.tts_enabled is True
            mock_response_pub.put.assert_called_once()
