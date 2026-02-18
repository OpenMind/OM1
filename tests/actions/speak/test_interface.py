"""Tests for the Speak action interface."""

from actions.speak.interface import Speak, SpeakInput


class TestSpeakInput:
    """Tests for the SpeakInput dataclass."""

    def test_speak_input_creation(self):
        """Test creating SpeakInput with action."""
        speak_input = SpeakInput(action="Hello, how are you?")
        assert speak_input.action == "Hello, how are you?"

    def test_speak_input_empty(self):
        """Test creating SpeakInput with empty string."""
        speak_input = SpeakInput(action="")
        assert speak_input.action == ""

    def test_speak_input_long_text(self):
        """Test creating SpeakInput with long text."""
        long_text = "This is a very long sentence that the robot needs to speak."
        speak_input = SpeakInput(action=long_text)
        assert speak_input.action == long_text


class TestSpeak:
    """Tests for the Speak interface."""

    def test_speak_creation(self):
        """Test creating Speak with input and output."""
        speak_input = SpeakInput(action="Welcome!")
        speak = Speak(input=speak_input, output=speak_input)
        assert speak.input == speak_input
        assert speak.output == speak_input

    def test_speak_different_input_output(self):
        """Test creating Speak with different input and output."""
        input_speak = SpeakInput(action="Input text")
        output_speak = SpeakInput(action="Output text")
        speak = Speak(input=input_speak, output=output_speak)
        assert speak.input.action == "Input text"
        assert speak.output.action == "Output text"
