import pytest

from actions.telegram.interface import Telegram, TelegramInput


class TestTelegramInput:
    """Test TelegramInput dataclass."""

    def test_default_action_empty(self):
        """Test default action is empty string."""
        telegram_input = TelegramInput()
        assert telegram_input.action == ""

    def test_custom_action(self):
        """Test custom action text."""
        telegram_input = TelegramInput(action="Hello, World!")
        assert telegram_input.action == "Hello, World!"

    def test_action_with_emojis(self):
        """Test action can contain emojis."""
        telegram_input = TelegramInput(action="Robot ready! 🤖✨")
        assert telegram_input.action == "Robot ready! 🤖✨"

    def test_action_multiline(self):
        """Test action can contain multiple lines."""
        multiline_text = "Line 1\nLine 2\nLine 3"
        telegram_input = TelegramInput(action=multiline_text)
        assert telegram_input.action == multiline_text
        assert "\n" in telegram_input.action


class TestTelegramInterface:
    """Test Telegram interface dataclass."""

    def test_interface_with_input_output(self):
        """Test interface has input and output of same type."""
        input_data = TelegramInput(action="Test message")
        output_data = TelegramInput(action="Test message")
        telegram = Telegram(input=input_data, output=output_data)

        assert telegram.input == input_data
        assert telegram.output == output_data
        assert telegram.input.action == "Test message"
        assert telegram.output.action == "Test message"

    def test_interface_input_output_independent(self):
        """Test input and output can have different values."""
        input_data = TelegramInput(action="Input message")
        output_data = TelegramInput(action="Output message")
        telegram = Telegram(input=input_data, output=output_data)

        assert telegram.input.action != telegram.output.action
        assert telegram.input.action == "Input message"
        assert telegram.output.action == "Output message"
