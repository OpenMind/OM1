from unittest.mock import Mock, patch

import pytest

from actions.base import ActionConfig
from actions.speak.connector.zenoh import SpeakZenohConnector
from actions.speak.interface import SpeakInput


@pytest.fixture
def mock_publisher():
    """Mock the ZenohPublisherProvider."""
    with patch("actions.speak.connector.zenoh.ZenohPublisherProvider") as mock_cls:
        mock_instance = Mock()
        mock_cls.return_value = mock_instance
        yield {"cls": mock_cls, "instance": mock_instance}


@pytest.fixture
def connector(mock_publisher):
    """Create a SpeakZenohConnector with mocked publisher."""
    return SpeakZenohConnector(ActionConfig())


class TestSpeakZenohConnector:
    """Test the Speak Zenoh connector."""

    def test_init_default_topic(self, mock_publisher):
        """Test initialization uses default 'speech' topic when not configured."""
        SpeakZenohConnector(ActionConfig())

        mock_publisher["cls"].assert_called_once_with("speech")
        mock_publisher["instance"].start.assert_called_once()

    def test_init_custom_topic(self, mock_publisher):
        """Test initialization uses custom topic from config."""
        config = ActionConfig(speak_topic="custom/topic")  # type: ignore[call-arg]
        SpeakZenohConnector(config)

        mock_publisher["cls"].assert_called_once_with("custom/topic")
        mock_publisher["instance"].start.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_adds_pending_message(self, connector, mock_publisher):
        """Test connect queues the speak message via publisher."""
        speak_input = SpeakInput(action="Hello from Zenoh!")
        await connector.connect(speak_input)
        mock_publisher["instance"].add_pending_message.assert_called_once_with(
            "Hello from Zenoh!"
        )

    @pytest.mark.asyncio
    async def test_connect_empty_message(self, connector, mock_publisher):
        """Test connect handles empty message."""
        speak_input = SpeakInput(action="")
        await connector.connect(speak_input)
        mock_publisher["instance"].add_pending_message.assert_called_once_with("")
