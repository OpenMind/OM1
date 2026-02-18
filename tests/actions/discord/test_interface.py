"""Tests for the Discord action interface."""

from actions.discord.interface import Discord, DiscordInput


class TestDiscordInput:
    """Tests for the DiscordInput dataclass."""

    def test_discord_input_default(self):
        """Test creating DiscordInput with default value."""
        discord_input = DiscordInput()
        assert discord_input.action == ""

    def test_discord_input_with_message(self):
        """Test creating DiscordInput with message."""
        discord_input = DiscordInput(action="Hello Discord!")
        assert discord_input.action == "Hello Discord!"

    def test_discord_input_with_markdown(self):
        """Test creating DiscordInput with markdown formatting."""
        discord_input = DiscordInput(action="**Bold** and *italic*")
        assert discord_input.action == "**Bold** and *italic*"


class TestDiscord:
    """Tests for the Discord interface."""

    def test_discord_creation(self):
        """Test creating Discord with input and output."""
        discord_input = DiscordInput(action="Test message")
        discord = Discord(input=discord_input, output=discord_input)
        assert discord.input == discord_input
        assert discord.output == discord_input

    def test_discord_different_input_output(self):
        """Test creating Discord with different input and output."""
        input_msg = DiscordInput(action="Input message")
        output_msg = DiscordInput(action="Output message")
        discord = Discord(input=input_msg, output=output_msg)
        assert discord.input.action == "Input message"
        assert discord.output.action == "Output message"
