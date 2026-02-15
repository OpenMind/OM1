from unittest.mock import patch

import pytest

from actions.base import ActionConfig
from actions.speak.connector.ros2 import SpeakRos2Connector
from actions.speak.interface import SpeakInput


@pytest.fixture
def connector():
    """Create a SpeakRos2Connector with default config."""
    return SpeakRos2Connector(ActionConfig())


class TestSpeakRos2Connector:
    """Test the Speak ROS2 connector."""

    def test_init(self):
        """Test initialization of SpeakRos2Connector."""
        config = ActionConfig()
        connector = SpeakRos2Connector(config)
        assert connector.config == config

    @pytest.mark.asyncio
    async def test_connect(self, connector):
        """Test connect sends speak dict to ROS2."""
        speak_input = SpeakInput(action="Hello, world!")
        with patch("actions.speak.connector.ros2.logging") as mock_logging:
            await connector.connect(speak_input)
            mock_logging.info.assert_called_once_with(
                "SendThisToROS2: {'speak': 'Hello, world!'}"
            )

    @pytest.mark.asyncio
    async def test_connect_empty_text(self, connector):
        """Test connect with empty text."""
        speak_input = SpeakInput(action="")
        with patch("actions.speak.connector.ros2.logging") as mock_logging:
            await connector.connect(speak_input)
            mock_logging.info.assert_called_once_with("SendThisToROS2: {'speak': ''}")
