"""Tests for the Emotion Unitree SDK connector."""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock the unitree module at module load time BEFORE any imports
mock_audio_client = MagicMock()
mock_unitree = MagicMock()
mock_unitree.unitree_sdk2py.g1.audio.g1_audio_client.AudioClient = mock_audio_client
sys.modules["unitree"] = mock_unitree
sys.modules["unitree.unitree_sdk2py"] = mock_unitree.unitree_sdk2py
sys.modules["unitree.unitree_sdk2py.g1"] = mock_unitree.unitree_sdk2py.g1
sys.modules["unitree.unitree_sdk2py.g1.audio"] = mock_unitree.unitree_sdk2py.g1.audio
sys.modules["unitree.unitree_sdk2py.g1.audio.g1_audio_client"] = (
    mock_unitree.unitree_sdk2py.g1.audio.g1_audio_client
)

from actions.emotion.connector.unitree_sdk import (  # noqa: E402
    EmotionUnitreeConfig,
    EmotionUnitreeConnector,
)
from actions.emotion.interface import EmotionAction, EmotionInput  # noqa: E402


@pytest.fixture
def default_config():
    """Create a default config for testing."""
    return EmotionUnitreeConfig()


@pytest.fixture
def ethernet_config():
    """Create a config with ethernet for testing."""
    return EmotionUnitreeConfig(unitree_ethernet="eth0")


@pytest.fixture
def emotion_input_happy():
    """Create an EmotionInput instance with happy action."""
    return EmotionInput(action=EmotionAction.HAPPY)


@pytest.fixture
def emotion_input_sad():
    """Create an EmotionInput instance with sad action."""
    return EmotionInput(action=EmotionAction.SAD)


@pytest.fixture
def emotion_input_mad():
    """Create an EmotionInput instance with mad action."""
    return EmotionInput(action=EmotionAction.MAD)


@pytest.fixture
def emotion_input_curious():
    """Create an EmotionInput instance with curious action."""
    return EmotionInput(action=EmotionAction.CURIOUS)


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset all mock objects between tests."""
    mock_audio_client.reset_mock()
    mock_audio_client.return_value = MagicMock()
    yield


class TestEmotionUnitreeConfig:
    """Tests for the EmotionUnitreeConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmotionUnitreeConfig()
        assert config.unitree_ethernet == ""

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EmotionUnitreeConfig(unitree_ethernet="eth0")
        assert config.unitree_ethernet == "eth0"


class TestEmotionUnitreeConnector:
    """Tests for the EmotionUnitreeConnector."""

    def test_init_without_ethernet(self, default_config):
        """Test initialization without ethernet configured."""
        connector = EmotionUnitreeConnector(default_config)

        assert connector.ao_client is None
        assert connector.unitree_ethernet == ""

    def test_init_with_ethernet(self, ethernet_config):
        """Test initialization with ethernet configured."""
        mock_client_instance = Mock()
        mock_audio_client.return_value = mock_client_instance

        connector = EmotionUnitreeConnector(ethernet_config)

        assert connector.ao_client is not None
        assert connector.unitree_ethernet == "eth0"
        # Verify SetTimeout, Init, and LedControl calls
        mock_client_instance.SetTimeout.assert_called_once_with(10.0)
        mock_client_instance.Init.assert_called_once()
        mock_client_instance.LedControl.assert_called_once_with(0, 255, 0)

    @pytest.mark.asyncio
    async def test_connect_without_client(self, default_config, emotion_input_happy):
        """Test connect when no audio client is available."""
        connector = EmotionUnitreeConnector(default_config)
        await connector.connect(emotion_input_happy)
        # Should not raise, just log error

    @pytest.mark.asyncio
    async def test_connect_happy(self, ethernet_config, emotion_input_happy):
        """Test connect with happy emotion."""
        mock_client_instance = Mock()
        mock_audio_client.return_value = mock_client_instance

        connector = EmotionUnitreeConnector(ethernet_config)
        await connector.connect(emotion_input_happy)

        mock_client_instance.LedControlNoReply.assert_called_with(0, 255, 0)

    @pytest.mark.asyncio
    async def test_connect_sad(self, ethernet_config, emotion_input_sad):
        """Test connect with sad emotion."""
        mock_client_instance = Mock()
        mock_audio_client.return_value = mock_client_instance

        connector = EmotionUnitreeConnector(ethernet_config)
        await connector.connect(emotion_input_sad)

        mock_client_instance.LedControlNoReply.assert_called_with(255, 255, 0)

    @pytest.mark.asyncio
    async def test_connect_mad(self, ethernet_config, emotion_input_mad):
        """Test connect with mad emotion."""
        mock_client_instance = Mock()
        mock_audio_client.return_value = mock_client_instance

        connector = EmotionUnitreeConnector(ethernet_config)
        await connector.connect(emotion_input_mad)

        mock_client_instance.LedControlNoReply.assert_called_with(255, 0, 0)

    @pytest.mark.asyncio
    async def test_connect_curious(self, ethernet_config, emotion_input_curious):
        """Test connect with curious emotion."""
        mock_client_instance = Mock()
        mock_audio_client.return_value = mock_client_instance

        connector = EmotionUnitreeConnector(ethernet_config)
        await connector.connect(emotion_input_curious)

        mock_client_instance.LedControlNoReply.assert_called_with(0, 0, 255)

    def test_tick(self, default_config):
        """Test tick method."""
        connector = EmotionUnitreeConnector(default_config)

        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()
            mock_sleep.assert_called_once_with(5)
