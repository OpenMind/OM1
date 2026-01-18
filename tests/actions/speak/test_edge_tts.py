import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.actions.speak.connector.edge_tts import SpeakEdgeTTSConnector, SpeakEdgeTTSConfig
from actions.speak.interface import SpeakInput

@pytest.fixture
def mock_config():
    return SpeakEdgeTTSConfig(
        voice="en-US-TestVoice",
        rate="+0%",
        volume="+0%"
    )

@pytest.fixture
def connector(mock_config):
    # Mock Zenoh session to prevent network errors during init
    with patch("src.actions.speak.connector.edge_tts.open_zenoh_session") as MockZenoh:
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
    
    with patch.object(connector, '_generate_and_play', new_callable=AsyncMock) as mock_play:
        input_data = SpeakInput(action="Hello")
        await connector.connect(input_data)
        
        # Should NOT call generate_and_play
        mock_play.assert_not_called()

@pytest.mark.asyncio
async def test_connect_success(connector):
    """Test successful connect execution."""
    connector.tts_enabled = True
    
    # Mock dependencies
    with patch("edge_tts.Communicate") as MockCommunicate, \
         patch("pydub.AudioSegment.from_mp3") as MockAudioSegment, \
         patch("pydub.playback.play") as MockPlay, \
         patch("pathlib.Path.unlink") as MockUnlink:
        
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
        MockCommunicate.assert_called_with(text="Hello Zenoh", voice="en-US-TestVoice", rate="+0%", volume="+0%")
        mock_comm_instance.save.assert_called_once()
        # Verify clean up was called
        MockUnlink.assert_called_once()

@pytest.mark.asyncio
async def test_stop(connector):
    """Test stop method closes session."""
    connector.stop()
    connector.session.close.assert_called_once()
