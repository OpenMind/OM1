"""Tests for the Emotion Unitree SDK connector."""

import sys
from unittest.mock import MagicMock, patch

import pytest


# Mock the unitree module before importing the connector
@pytest.fixture(autouse=True, scope="module")
def mock_unitree_module():
    """Mock the unitree SDK module before any imports."""
    mock_audio_client = MagicMock()
    mock_unitree = MagicMock()
    mock_unitree.unitree_sdk2py.g1.audio.g1_audio_client.AudioClient = mock_audio_client
    sys.modules["unitree"] = mock_unitree
    sys.modules["unitree.unitree_sdk2py"] = mock_unitree.unitree_sdk2py
    sys.modules["unitree.unitree_sdk2py.g1"] = mock_unitree.unitree_sdk2py.g1
    sys.modules["unitree.unitree_sdk2py.g1.audio"] = (
        mock_unitree.unitree_sdk2py.g1.audio
    )
    sys.modules["unitree.unitree_sdk2py.g1.audio.g1_audio_client"] = (
        mock_unitree.unitree_sdk2py.g1.audio.g1_audio_client
    )
    yield mock_audio_client
    # Cleanup
    for mod in list(sys.modules.keys()):
        if mod.startswith("unitree"):
            del sys.modules[mod]


class TestEmotionUnitreeConfig:
    """Tests for the EmotionUnitreeConfig."""

    def test_default_config(self, mock_unitree_module):
        """Test default configuration values."""
        from actions.emotion.connector.unitree_sdk import EmotionUnitreeConfig

        config = EmotionUnitreeConfig()
        assert config.unitree_ethernet == ""

    def test_custom_config(self, mock_unitree_module):
        """Test custom configuration values."""
        from actions.emotion.connector.unitree_sdk import EmotionUnitreeConfig

        config = EmotionUnitreeConfig(unitree_ethernet="eth0")
        assert config.unitree_ethernet == "eth0"


class TestEmotionUnitreeConnector:
    """Tests for the EmotionUnitreeConnector."""

    def test_init_without_ethernet(self, mock_unitree_module):
        """Test initialization without ethernet configured."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )

        config = EmotionUnitreeConfig(unitree_ethernet="")
        connector = EmotionUnitreeConnector(config)

        assert connector.ao_client is None
        assert connector.unitree_ethernet == ""

    def test_init_with_ethernet(self, mock_unitree_module):
        """Test initialization with ethernet configured."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )

        mock_client_instance = MagicMock()
        mock_unitree_module.return_value = mock_client_instance

        config = EmotionUnitreeConfig(unitree_ethernet="eth0")
        connector = EmotionUnitreeConnector(config)

        assert connector.ao_client is not None
        assert connector.unitree_ethernet == "eth0"

    @pytest.mark.asyncio
    async def test_connect_without_client(self, mock_unitree_module):
        """Test connect when no audio client is available."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )
        from actions.emotion.interface import EmotionAction, EmotionInput

        config = EmotionUnitreeConfig(unitree_ethernet="")
        connector = EmotionUnitreeConnector(config)

        emotion_input = EmotionInput(action=EmotionAction.HAPPY)
        await connector.connect(emotion_input)
        # Should not raise, just log error

    @pytest.mark.asyncio
    async def test_connect_happy(self, mock_unitree_module):
        """Test connect with happy emotion."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )
        from actions.emotion.interface import EmotionAction, EmotionInput

        mock_client_instance = MagicMock()
        mock_unitree_module.return_value = mock_client_instance

        config = EmotionUnitreeConfig(unitree_ethernet="eth0")
        connector = EmotionUnitreeConnector(config)

        emotion_input = EmotionInput(action=EmotionAction.HAPPY)
        await connector.connect(emotion_input)

        mock_client_instance.LedControlNoReply.assert_called_with(0, 255, 0)

    @pytest.mark.asyncio
    async def test_connect_sad(self, mock_unitree_module):
        """Test connect with sad emotion."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )
        from actions.emotion.interface import EmotionAction, EmotionInput

        mock_client_instance = MagicMock()
        mock_unitree_module.return_value = mock_client_instance

        config = EmotionUnitreeConfig(unitree_ethernet="eth0")
        connector = EmotionUnitreeConnector(config)

        emotion_input = EmotionInput(action=EmotionAction.SAD)
        await connector.connect(emotion_input)

        mock_client_instance.LedControlNoReply.assert_called_with(255, 255, 0)

    @pytest.mark.asyncio
    async def test_connect_mad(self, mock_unitree_module):
        """Test connect with mad emotion."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )
        from actions.emotion.interface import EmotionAction, EmotionInput

        mock_client_instance = MagicMock()
        mock_unitree_module.return_value = mock_client_instance

        config = EmotionUnitreeConfig(unitree_ethernet="eth0")
        connector = EmotionUnitreeConnector(config)

        emotion_input = EmotionInput(action=EmotionAction.MAD)
        await connector.connect(emotion_input)

        mock_client_instance.LedControlNoReply.assert_called_with(255, 0, 0)

    @pytest.mark.asyncio
    async def test_connect_curious(self, mock_unitree_module):
        """Test connect with curious emotion."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )
        from actions.emotion.interface import EmotionAction, EmotionInput

        mock_client_instance = MagicMock()
        mock_unitree_module.return_value = mock_client_instance

        config = EmotionUnitreeConfig(unitree_ethernet="eth0")
        connector = EmotionUnitreeConnector(config)

        emotion_input = EmotionInput(action=EmotionAction.CURIOUS)
        await connector.connect(emotion_input)

        mock_client_instance.LedControlNoReply.assert_called_with(0, 0, 255)

    def test_tick(self, mock_unitree_module):
        """Test tick method."""
        from actions.emotion.connector.unitree_sdk import (
            EmotionUnitreeConfig,
            EmotionUnitreeConnector,
        )

        config = EmotionUnitreeConfig(unitree_ethernet="")
        connector = EmotionUnitreeConnector(config)

        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()
            mock_sleep.assert_called_once_with(5)
