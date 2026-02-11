from unittest.mock import Mock, patch

import pytest

from actions.emotion.connector.unitree_sdk import (
    EmotionUnitreeConfig,
    EmotionUnitreeConnector,
)
from actions.emotion.interface import EmotionAction, EmotionInput


class TestEmotionUnitreeConfig:
    """Test EmotionUnitreeConfig configuration."""

    def test_default_config(self):
        """Test default configuration has empty ethernet."""
        config = EmotionUnitreeConfig()
        assert config.unitree_ethernet == ""

    def test_custom_ethernet(self):
        """Test custom ethernet adapter configuration."""
        config = EmotionUnitreeConfig(unitree_ethernet="eth0")
        assert config.unitree_ethernet == "eth0"


class TestEmotionUnitreeConnectorInit:
    """Test EmotionUnitreeConnector initialization."""

    def test_init_without_ethernet(self):
        """Test initialization without ethernet (no audio client)."""
        config = EmotionUnitreeConfig()
        connector = EmotionUnitreeConnector(config)
        assert connector.ao_client is None
        assert connector.unitree_ethernet == ""

    def test_init_with_ethernet(self):
        """Test initialization with ethernet creates AudioClient."""
        with patch("actions.emotion.connector.unitree_sdk.AudioClient") as mock_audio:
            mock_instance = Mock()
            mock_audio.return_value = mock_instance

            config = EmotionUnitreeConfig(unitree_ethernet="eth0")
            connector = EmotionUnitreeConnector(config)

            mock_audio.assert_called_once()
            mock_instance.SetTimeout.assert_called_once_with(10.0)
            mock_instance.Init.assert_called_once()
            mock_instance.LedControl.assert_called_once_with(0, 255, 0)
            assert connector.ao_client == mock_instance


class TestEmotionUnitreeConnectorConnect:
    """Test connect method for each emotion action."""

    @pytest.fixture
    def connector_no_client(self):
        """Create connector without audio client."""
        config = EmotionUnitreeConfig()
        return EmotionUnitreeConnector(config)

    @pytest.fixture
    def connector_with_client(self):
        """Create connector with mocked audio client."""
        with patch("actions.emotion.connector.unitree_sdk.AudioClient") as mock_audio:
            mock_instance = Mock()
            mock_audio.return_value = mock_instance

            config = EmotionUnitreeConfig(unitree_ethernet="eth0")
            connector = EmotionUnitreeConnector(config)
            return connector

    @pytest.mark.asyncio
    async def test_connect_no_client_logs_error(self, connector_no_client):
        """Test connect without audio client logs error."""
        emotion_input = EmotionInput(action=EmotionAction.HAPPY)
        with patch("actions.emotion.connector.unitree_sdk.logging") as mock_logging:
            await connector_no_client.connect(emotion_input)
            mock_logging.error.assert_called_with("No Unitree Emotion Client")

    @pytest.mark.asyncio
    async def test_connect_happy(self, connector_with_client):
        """Test happy emotion sets green LED."""
        emotion_input = EmotionInput(action=EmotionAction.HAPPY)
        await connector_with_client.connect(emotion_input)
        connector_with_client.ao_client.LedControlNoReply.assert_called_with(0, 255, 0)

    @pytest.mark.asyncio
    async def test_connect_sad(self, connector_with_client):
        """Test sad emotion sets yellow LED."""
        emotion_input = EmotionInput(action=EmotionAction.SAD)
        await connector_with_client.connect(emotion_input)
        connector_with_client.ao_client.LedControlNoReply.assert_called_with(
            255, 255, 0
        )

    @pytest.mark.asyncio
    async def test_connect_mad(self, connector_with_client):
        """Test mad emotion sets red LED."""
        emotion_input = EmotionInput(action=EmotionAction.MAD)
        await connector_with_client.connect(emotion_input)
        connector_with_client.ao_client.LedControlNoReply.assert_called_with(255, 0, 0)

    @pytest.mark.asyncio
    async def test_connect_curious(self, connector_with_client):
        """Test curious emotion sets blue LED."""
        emotion_input = EmotionInput(action=EmotionAction.CURIOUS)
        await connector_with_client.connect(emotion_input)
        connector_with_client.ao_client.LedControlNoReply.assert_called_with(0, 0, 255)

    @pytest.mark.asyncio
    async def test_connect_unknown_emotion(self, connector_with_client):
        """Test unknown emotion logs info."""
        emotion_input = EmotionInput(action="confused")  # type: ignore[arg-type]
        with patch("actions.emotion.connector.unitree_sdk.logging") as mock_logging:
            await connector_with_client.connect(emotion_input)
            mock_logging.info.assert_any_call("Unknown emotion: confused")

    @pytest.mark.asyncio
    async def test_connect_logs_action(self, connector_with_client):
        """Test connect logs the action being sent."""
        emotion_input = EmotionInput(action=EmotionAction.HAPPY)
        with patch("actions.emotion.connector.unitree_sdk.logging") as mock_logging:
            await connector_with_client.connect(emotion_input)
            mock_logging.info.assert_any_call("SendThisToUTClient: happy")


class TestEmotionUnitreeConnectorTick:
    """Test tick method."""

    def test_tick_calls_sleep(self):
        """Test tick calls sleep with 5 seconds."""
        config = EmotionUnitreeConfig()
        connector = EmotionUnitreeConnector(config)
        with patch.object(connector, "sleep") as mock_sleep:
            connector.tick()
            mock_sleep.assert_called_once_with(5)
