"""
Tests for Smart Assistant Input Plugin.
"""

import asyncio
import json
import time
from http.client import HTTPConnection
from queue import Queue
from unittest.mock import patch

import pytest


class TestSmartAssistantConfig:
    """Tests for SmartAssistantConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from inputs.plugins.smart_assistant import SmartAssistantConfig

        config = SmartAssistantConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8080
        assert config.webhook_path == "/webhook"

    def test_custom_config(self):
        """Test custom configuration values."""
        from inputs.plugins.smart_assistant import SmartAssistantConfig

        config = SmartAssistantConfig(
            host="127.0.0.1",
            port=9090,
            webhook_path="/api/webhook",
        )
        assert config.host == "127.0.0.1"
        assert config.port == 9090
        assert config.webhook_path == "/api/webhook"


class TestWebhookHandler:
    """Tests for WebhookHandler command extraction."""

    def test_extract_direct_command(self):
        """Test extracting direct command field."""
        from inputs.plugins.smart_assistant import WebhookHandler

        handler = WebhookHandler.__new__(WebhookHandler)
        data = {"command": "pay 0.001 ETH to 0x1234"}
        result = handler._extract_command(data)
        assert result == "pay 0.001 ETH to 0x1234"

    def test_extract_home_assistant_format(self):
        """Test extracting command from Home Assistant format."""
        from inputs.plugins.smart_assistant import WebhookHandler

        handler = WebhookHandler.__new__(WebhookHandler)

        # Simple action
        data = {"action": "turn_on_lights"}
        result = handler._extract_command(data)
        assert result == "turn_on_lights"

        # Action with data
        data = {
            "action": "pay",
            "data": {"amount": "0.001", "recipient": "0x1234"},
        }
        result = handler._extract_command(data)
        assert result == "pay 0.001 to 0x1234"

    def test_extract_alexa_format(self):
        """Test extracting command from Alexa format."""
        from inputs.plugins.smart_assistant import WebhookHandler

        handler = WebhookHandler.__new__(WebhookHandler)
        data = {
            "request": {
                "intent": {
                    "name": "PaymentIntent",
                    "slots": {
                        "amount": {"value": "0.001"},
                        "recipient": {"value": "friend"},
                    },
                }
            }
        }
        result = handler._extract_command(data)
        assert result is not None
        assert "PaymentIntent" in result
        assert "amount=0.001" in result
        assert "recipient=friend" in result

    def test_extract_ifttt_format(self):
        """Test extracting command from IFTTT format."""
        from inputs.plugins.smart_assistant import WebhookHandler

        handler = WebhookHandler.__new__(WebhookHandler)
        data = {"value1": "pay", "value2": "0.001 ETH", "value3": "to 0x1234"}
        result = handler._extract_command(data)
        assert result == "pay 0.001 ETH to 0x1234"

    def test_extract_message_field(self):
        """Test extracting command from message field."""
        from inputs.plugins.smart_assistant import WebhookHandler

        handler = WebhookHandler.__new__(WebhookHandler)
        data = {"message": "Hello, world!"}
        result = handler._extract_command(data)
        assert result == "Hello, world!"

    def test_extract_text_field(self):
        """Test extracting command from text field."""
        from inputs.plugins.smart_assistant import WebhookHandler

        handler = WebhookHandler.__new__(WebhookHandler)
        data = {"text": "Test text"}
        result = handler._extract_command(data)
        assert result == "Test text"

    def test_extract_no_command(self):
        """Test when no command is found."""
        from inputs.plugins.smart_assistant import WebhookHandler

        handler = WebhookHandler.__new__(WebhookHandler)
        data = {"unknown_field": "value"}
        result = handler._extract_command(data)
        assert result is None


class TestSmartAssistantInput:
    """Tests for SmartAssistantInput."""

    @pytest.fixture
    def mock_io_provider(self):
        """Mock IO provider."""
        with patch("inputs.plugins.smart_assistant.IOProvider") as mock:
            yield mock

    def test_init_creates_config(self, mock_io_provider):
        """Test initialization creates proper config."""
        from inputs.plugins.smart_assistant import (
            SmartAssistantConfig,
            SmartAssistantInput,
        )

        config = SmartAssistantConfig(port=18080)

        with patch.object(SmartAssistantInput, "_start_server"):
            input_plugin = SmartAssistantInput(config)
            assert input_plugin.config.port == 18080

    @pytest.mark.asyncio
    async def test_poll_returns_none_when_empty(self, mock_io_provider):
        """Test _poll returns None when queue is empty."""
        from inputs.plugins.smart_assistant import (
            SmartAssistantConfig,
            SmartAssistantInput,
        )

        config = SmartAssistantConfig(port=18081)

        with patch.object(SmartAssistantInput, "_start_server"):
            input_plugin = SmartAssistantInput(config)
            input_plugin.message_queue = Queue()

            result = await input_plugin._poll()
            assert result is None

    @pytest.mark.asyncio
    async def test_poll_returns_command_from_queue(self, mock_io_provider):
        """Test _poll returns command from queue."""
        from inputs.plugins.smart_assistant import (
            SmartAssistantConfig,
            SmartAssistantInput,
        )

        config = SmartAssistantConfig(port=18082)

        with patch.object(SmartAssistantInput, "_start_server"):
            input_plugin = SmartAssistantInput(config)
            input_plugin.message_queue = Queue()
            input_plugin.message_queue.put("test command")

            result = await input_plugin._poll()
            assert result == "test command"

    @pytest.mark.asyncio
    async def test_raw_to_text_creates_message(self, mock_io_provider):
        """Test _raw_to_text creates proper Message."""
        from inputs.plugins.smart_assistant import (
            SmartAssistantConfig,
            SmartAssistantInput,
        )

        config = SmartAssistantConfig(port=18083)

        with patch.object(SmartAssistantInput, "_start_server"):
            input_plugin = SmartAssistantInput(config)
            result = await input_plugin._raw_to_text("test command")

            assert result is not None
            assert result.message == "test command"
            assert result.timestamp > 0

    @pytest.mark.asyncio
    async def test_raw_to_text_returns_none_for_none_input(self, mock_io_provider):
        """Test _raw_to_text returns None for None input."""
        from inputs.plugins.smart_assistant import (
            SmartAssistantConfig,
            SmartAssistantInput,
        )

        config = SmartAssistantConfig(port=18084)

        with patch.object(SmartAssistantInput, "_start_server"):
            input_plugin = SmartAssistantInput(config)
            result = await input_plugin._raw_to_text(None)
            assert result is None

    def test_formatted_latest_buffer_returns_none_when_empty(self, mock_io_provider):
        """Test formatted_latest_buffer returns None when no messages."""
        from inputs.plugins.smart_assistant import (
            SmartAssistantConfig,
            SmartAssistantInput,
        )

        config = SmartAssistantConfig(port=18085)

        with patch.object(SmartAssistantInput, "_start_server"):
            input_plugin = SmartAssistantInput(config)
            result = input_plugin.formatted_latest_buffer()
            assert result is None

    def test_formatted_latest_buffer_returns_formatted_message(self, mock_io_provider):
        """Test formatted_latest_buffer returns formatted message."""
        from inputs.base import Message
        from inputs.plugins.smart_assistant import (
            SmartAssistantConfig,
            SmartAssistantInput,
        )

        config = SmartAssistantConfig(port=18086)

        with patch.object(SmartAssistantInput, "_start_server"):
            input_plugin = SmartAssistantInput(config)
            input_plugin.messages = [Message(timestamp=time.time(), message="test")]

            result = input_plugin.formatted_latest_buffer()

            assert result is not None
            assert "SmartAssistantInput INPUT" in result
            assert "test" in result
            assert len(input_plugin.messages) == 0  # Buffer cleared


class TestSmartAssistantIntegration:
    """Integration tests for Smart Assistant Input."""

    @pytest.fixture
    def server_port(self):
        """Return a unique port for each test."""
        import random

        return random.randint(19000, 19999)

    @pytest.mark.asyncio
    async def test_webhook_receives_command(self, server_port):
        """Test that webhook endpoint receives and queues commands."""
        from inputs.plugins.smart_assistant import (
            SmartAssistantConfig,
            SmartAssistantInput,
        )

        config = SmartAssistantConfig(host="127.0.0.1", port=server_port)

        with patch("inputs.plugins.smart_assistant.IOProvider"):
            input_plugin = SmartAssistantInput(config)

            # Wait for server to start
            await asyncio.sleep(0.5)

            # Send webhook request
            conn = None
            try:
                conn = HTTPConnection("127.0.0.1", server_port, timeout=5)
                payload = json.dumps({"command": "pay 0.001 ETH to 0x1234"})
                conn.request(
                    "POST",
                    "/webhook",
                    body=payload,
                    headers={"Content-Type": "application/json"},
                )
                response = conn.getresponse()
                assert response.status == 200

                # Check command was queued
                command = await input_plugin._poll()
                assert command == "pay 0.001 ETH to 0x1234"

            finally:
                if conn is not None:
                    conn.close()
                input_plugin.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
