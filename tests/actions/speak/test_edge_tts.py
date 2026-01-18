from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from actions.speak.connector.edge_tts import SpeakEdgeTTSConfig, SpeakEdgeTTSConnector
from actions.speak.interface import SpeakInput


@pytest.fixture
def mock_config():
    return SpeakEdgeTTSConfig(voice="en-US-TestVoice", rate="+0%", volume="+0%")


@pytest.fixture
def connector(mock_config):
    # Mock Zenoh session to prevent network errors during init
    with patch("actions.speak.connector.edge_tts.open_zenoh_session") as MockZenoh:
        mock_session = MagicMock()
        MockZenoh.return_value = mock_session
        connector = SpeakEdgeTTSConnector(mock_config)
        yield connector


def test_config_defaults():
    """Test that default configuration values are correct."""
    config = SpeakEdgeTTSConfig()
    assert config.voice == "en-US-AriaNeural"
    assert config.rate == "+0%"
    assert config.volume == "+0%"


@pytest.mark.asyncio
async def test_init(connector):
    """Test initialization and config assignment."""
    assert connector.voice == "en-US-TestVoice"
    assert connector.tts_enabled is True
    # Ensure Zenoh client was initialized
    assert connector.session is not None


@pytest.mark.asyncio
async def test_connect_tts_disabled(connector):
    """Test connect method when TTS is disabled."""
    connector.tts_enabled = False

    with patch.object(
        connector, "_generate_and_play", new_callable=AsyncMock
    ) as mock_play:
        input_data = SpeakInput(action="Hello")
        await connector.connect(input_data)

        # Should NOT call generate_and_play
        mock_play.assert_not_called()


@pytest.mark.asyncio
async def test_connect_success(connector):
    """Test successful connect execution."""
    connector.tts_enabled = True

    # Mock dependencies
    with (
        patch("edge_tts.Communicate") as MockCommunicate,
        patch("pydub.AudioSegment.from_mp3") as MockAudioSegment,
        patch("pydub.playback.play"),
        patch("pathlib.Path.unlink") as MockUnlink,
    ):

        # Setup mocks
        mock_comm_instance = MockCommunicate.return_value
        mock_comm_instance.save = AsyncMock()

        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 1000
        MockAudioSegment.return_value = mock_audio

        # Run connect
        input_data = SpeakInput(action="Hello Zenoh")
        await connector.connect(input_data)

        # Verifications
        MockCommunicate.assert_called_with(
            text="Hello Zenoh", voice="en-US-TestVoice", rate="+0%", volume="+0%"
        )
        mock_comm_instance.save.assert_called_once()
        # Verify clean up was called
        MockUnlink.assert_called_once()


@pytest.mark.asyncio
async def test_stop(connector):
    """Test stop method closes session."""
    connector.stop()
    connector.session.close.assert_called_once()


class TestZenohTTSStatusRequest:
    """Tests for _zenoh_tts_status_request method."""

    @pytest.fixture
    def connector_with_publisher(self, mock_config):
        """Create connector with mocked Zenoh publisher."""
        with patch("actions.speak.connector.edge_tts.open_zenoh_session") as MockZenoh:
            mock_session = MagicMock()
            mock_publisher = MagicMock()
            mock_session.declare_publisher.return_value = mock_publisher
            MockZenoh.return_value = mock_session
            connector = SpeakEdgeTTSConnector(mock_config)
            connector._zenoh_tts_status_response_pub = mock_publisher
            yield connector

    def _create_mock_zenoh_sample(self, code: int, request_id: str = "test-123"):
        """Helper to create a mock Zenoh sample with TTSStatusRequest."""
        mock_sample = MagicMock()

        with patch("actions.speak.connector.edge_tts.TTSStatusRequest") as MockRequest:
            mock_request = MagicMock()
            mock_request.code = code
            mock_request.request_id = request_id
            mock_request.header.frame_id = "test-frame"
            MockRequest.deserialize.return_value = mock_request

        return mock_sample, mock_request

    def test_tts_status_request_disable(self, connector_with_publisher):
        """Test code=0 disables TTS."""
        connector = connector_with_publisher
        connector.tts_enabled = True

        with (
            patch("actions.speak.connector.edge_tts.TTSStatusRequest") as MockRequest,
            patch("actions.speak.connector.edge_tts.TTSStatusResponse") as MockResponse,
            patch("actions.speak.connector.edge_tts.prepare_header") as mock_header,
        ):

            mock_request = MagicMock()
            mock_request.code = 0
            mock_request.request_id = "test-123"
            mock_request.header.frame_id = "test-frame"
            MockRequest.deserialize.return_value = mock_request
            mock_header.return_value = "mocked-header"

            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test"

            connector._zenoh_tts_status_request(mock_sample)

            assert connector.tts_enabled is False
            MockResponse.assert_called_once()
            connector._zenoh_tts_status_response_pub.put.assert_called_once()

    def test_tts_status_request_enable(self, connector_with_publisher):
        """Test code=1 enables TTS."""
        connector = connector_with_publisher
        connector.tts_enabled = False

        with (
            patch("actions.speak.connector.edge_tts.TTSStatusRequest") as MockRequest,
            patch("actions.speak.connector.edge_tts.TTSStatusResponse") as MockResponse,
            patch("actions.speak.connector.edge_tts.prepare_header") as mock_header,
        ):

            mock_request = MagicMock()
            mock_request.code = 1
            mock_request.request_id = "test-456"
            mock_request.header.frame_id = "test-frame"
            MockRequest.deserialize.return_value = mock_request
            mock_header.return_value = "mocked-header"

            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test"

            connector._zenoh_tts_status_request(mock_sample)

            assert connector.tts_enabled is True
            MockResponse.assert_called_once()

    def test_tts_status_request_read_status(self, connector_with_publisher):
        """Test code=2 reads current status."""
        connector = connector_with_publisher
        connector.tts_enabled = True

        with (
            patch("actions.speak.connector.edge_tts.TTSStatusRequest") as MockRequest,
            patch("actions.speak.connector.edge_tts.TTSStatusResponse") as MockResponse,
            patch("actions.speak.connector.edge_tts.prepare_header") as mock_header,
        ):

            mock_request = MagicMock()
            mock_request.code = 2
            mock_request.request_id = "test-789"
            mock_request.header.frame_id = "test-frame"
            MockRequest.deserialize.return_value = mock_request
            mock_header.return_value = "mocked-header"

            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test"

            connector._zenoh_tts_status_request(mock_sample)

            # Status should not change
            assert connector.tts_enabled is True
            MockResponse.assert_called_once()

    def test_tts_status_request_unknown_code(self, connector_with_publisher):
        """Test unknown code is ignored."""
        connector = connector_with_publisher
        original_state = connector.tts_enabled

        with patch("actions.speak.connector.edge_tts.TTSStatusRequest") as MockRequest:
            mock_request = MagicMock()
            mock_request.code = 99  # Unknown code
            mock_request.request_id = "test-unknown"
            mock_request.header.frame_id = "test-frame"
            MockRequest.deserialize.return_value = mock_request

            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"test"

            connector._zenoh_tts_status_request(mock_sample)

            # State should not change
            assert connector.tts_enabled == original_state

    def test_tts_status_request_exception_handling(self, connector_with_publisher):
        """Test exception handling in status request."""
        connector = connector_with_publisher

        with patch("actions.speak.connector.edge_tts.TTSStatusRequest") as MockRequest:
            MockRequest.deserialize.side_effect = Exception("Deserialization error")

            mock_sample = MagicMock()
            mock_sample.payload.to_bytes.return_value = b"invalid"

            # Should not raise, just log error
            connector._zenoh_tts_status_request(mock_sample)


def test_init_zenoh_exception():
    """Test __init__ handles Zenoh connection errors gracefully."""
    config = SpeakEdgeTTSConfig()

    with patch("actions.speak.connector.edge_tts.open_zenoh_session") as MockZenoh:
        MockZenoh.side_effect = Exception("Zenoh connection failed")

        # Should not raise, just log error
        connector = SpeakEdgeTTSConnector(config)

        # Session should be None due to error
        assert connector.session is None


@pytest.mark.asyncio
async def test_connect_exception_handling():
    """Test connect handles _generate_and_play errors gracefully."""
    config = SpeakEdgeTTSConfig()

    with patch("actions.speak.connector.edge_tts.open_zenoh_session") as MockZenoh:
        MockZenoh.return_value = MagicMock()
        connector = SpeakEdgeTTSConnector(config)

    with patch.object(
        connector, "_generate_and_play", new_callable=AsyncMock
    ) as mock_play:
        mock_play.side_effect = Exception("TTS generation failed")

        input_data = SpeakInput(action="Test error handling")

        # Should not raise, just log error
        await connector.connect(input_data)


@pytest.mark.asyncio
async def test_generate_and_play_cleanup_error():
    """Test cleanup error handling in _generate_and_play."""
    config = SpeakEdgeTTSConfig()

    with patch("actions.speak.connector.edge_tts.open_zenoh_session") as MockZenoh:
        MockZenoh.return_value = MagicMock()
        connector = SpeakEdgeTTSConnector(config)

    with (
        patch("edge_tts.Communicate") as MockCommunicate,
        patch("pydub.AudioSegment.from_mp3") as MockAudioSegment,
        patch("pydub.playback.play"),
        patch("pathlib.Path.exists") as MockExists,
        patch("pathlib.Path.unlink") as MockUnlink,
    ):

        mock_comm = MockCommunicate.return_value
        mock_comm.save = AsyncMock()

        mock_audio = MagicMock()
        mock_audio.__len__.return_value = 1000
        MockAudioSegment.return_value = mock_audio

        # Simulate cleanup error
        MockExists.return_value = True
        MockUnlink.side_effect = Exception("Permission denied")

        # Should not raise, just log warning
        await connector._generate_and_play("Test cleanup error")
