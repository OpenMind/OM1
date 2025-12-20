import os
from unittest.mock import MagicMock, patch

import pytest

from inputs.plugins.home_assistant import HomeAssistant, HomeAssistantConfig


@pytest.fixture
def ha_config():
    """Create test configuration."""
    return HomeAssistantConfig(
        ha_url="http://localhost:8123",
        ha_token="test_token_123",
        poll_interval=1.0,
    )


@pytest.fixture
def ha_plugin(ha_config):
    """Create HomeAssistant plugin instance."""
    return HomeAssistant(ha_config)


def test_home_assistant_init(ha_plugin):
    """Test HomeAssistant initialization."""
    assert ha_plugin.ha_url == "http://localhost:8123"
    assert ha_plugin.ha_token == "test_token_123"
    assert ha_plugin.poll_interval == 1.0
    assert ha_plugin.messages == []


@pytest.mark.asyncio
async def test_poll_with_new_command(ha_plugin):
    """Test polling with new voice command."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "state": "bring me water bottle from kitchen",
        "last_changed": "2024-12-20T10:00:00",
    }

    with patch("requests.get", return_value=mock_response):
        result = await ha_plugin._poll()

    assert result["command"] == "bring me water bottle from kitchen"
    assert "timestamp" in result
    assert result["command_id"] == "2024-12-20T10:00:00"


@pytest.mark.asyncio
async def test_poll_no_command(ha_plugin):
    """Test polling with no new command."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "state": "",
        "last_changed": "2024-12-20T09:00:00",
    }

    with patch("requests.get", return_value=mock_response):
        result = await ha_plugin._poll()

    assert result == {}


@pytest.mark.asyncio
async def test_raw_to_text(ha_plugin):
    """Test converting raw command to text message."""
    raw_input = {
        "command": "bring me coffee from kitchen",
        "timestamp": 1234567890.0,
        "command_id": "test_id",
    }

    message = await ha_plugin._raw_to_text(raw_input)

    assert message is not None
    assert "Voice command from Home Assistant" in message.message
    assert "bring me coffee from kitchen" in message.message
    assert message.timestamp == 1234567890.0


@pytest.mark.asyncio
async def test_formatted_latest_buffer(ha_plugin):
    """Test buffer formatting."""
    # Add test messages
    await ha_plugin.raw_to_text(
        {
            "command": "fetch water bottle",
            "timestamp": 1234567890.0,
            "command_id": "id1",
        }
    )

    result = ha_plugin.formatted_latest_buffer()

    assert result is not None
    assert "HomeAssistant INPUT" in result
    assert "fetch water bottle" in result
    assert len(ha_plugin.messages) == 0  # Buffer should be cleared


def test_env_variable_fallback():
    """Test that plugin uses environment variables as fallback."""
    os.environ["HOME_ASSISTANT_URL"] = "http://env-url:8123"
    os.environ["HOME_ASSISTANT_TOKEN"] = "env_token"

    config = HomeAssistantConfig()
    plugin = HomeAssistant(config)

    assert plugin.ha_url == "http://env-url:8123"
    assert plugin.ha_token == "env_token"

    # Cleanup
    del os.environ["HOME_ASSISTANT_URL"]
    del os.environ["HOME_ASSISTANT_TOKEN"]
