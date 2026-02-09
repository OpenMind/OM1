"""Tests for the Telegram action interface."""

from actions.telegram.interface import Telegram, TelegramInput


class TestTelegramInput:
    """Tests for the TelegramInput dataclass."""

    def test_telegram_input_default(self):
        """Test creating TelegramInput with default value."""
        telegram_input = TelegramInput()
        assert telegram_input.action == ""

    def test_telegram_input_with_message(self):
        """Test creating TelegramInput with a message."""
        telegram_input = TelegramInput(action="Hello from robot!")
        assert telegram_input.action == "Hello from robot!"

    def test_telegram_input_with_emoji(self):
        """Test creating TelegramInput with emoji."""
        telegram_input = TelegramInput(action="Hello! 🤖")
        assert telegram_input.action == "Hello! 🤖"


class TestTelegram:
    """Tests for the Telegram interface."""

    def test_telegram_creation(self):
        """Test creating Telegram with input and output."""
        telegram_input = TelegramInput(action="Test message")
        telegram = Telegram(input=telegram_input, output=telegram_input)
        assert telegram.input == telegram_input
        assert telegram.output == telegram_input

    def test_telegram_different_input_output(self):
        """Test creating Telegram with different input and output."""
        input_msg = TelegramInput(action="Input message")
        output_msg = TelegramInput(action="Output message")
        telegram = Telegram(input=input_msg, output=output_msg)
        assert telegram.input.action == "Input message"
        assert telegram.output.action == "Output message"
