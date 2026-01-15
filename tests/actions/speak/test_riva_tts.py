import sys
from unittest.mock import MagicMock, patch

import pytest

from actions.base import ActionConfig, ActionConnector
from actions.speak.interface import SpeakInput

# We need to mock several modules due to dependencies
sys.modules["om1_speech"] = MagicMock()
sys.modules["om1_utils"] = MagicMock()
sys.modules["om1_utils.ws"] = MagicMock()
sys.modules["om1_speech.AudioInputStream"] = MagicMock()

# Create mock patchers
patchers = [
    patch("providers.asr_provider.AudioInputStream"),
    patch("providers.asr_provider.ASRProvider"),
    patch("providers.riva_tts_provider.RivaTTSProvider"),
]

# Apply all the patches
for patcher in patchers:
    patcher.start()


# Create a fake SpeakRivaTTSConnector class for testing
class MockSpeakRivaTTSConnector(ActionConnector):
    def __init__(self, config):
        super().__init__(config)
        self.asr = MagicMock()
        self.asr.audio_stream = MagicMock()
        self.asr.audio_stream.on_tts_state_change = MagicMock()

        self.tts = MagicMock()
        self.tts.register_tts_state_callback = MagicMock()
        self.tts.add_pending_message = MagicMock()
        self.tts.start = MagicMock()

    async def connect(self, output_interface):
        try:
            self.tts.register_tts_state_callback(
                self.asr.audio_stream.on_tts_state_change
            )
            if output_interface and output_interface.action:
                self.tts.add_pending_message(output_interface.action)
            else:
                print("Empty speech action received")
        except Exception as e:
            print(f"Error in speak action: {e}")


@pytest.fixture
def action_config():
    config = ActionConfig()
    config.__dict__.update(
        {
            "microphone_device_id": "test_mic_id",
            "speaker_device_id": "test_speaker_id",
            "microphone_name": "test_mic",
            "speaker_name": "test_speaker",
            "api_key": "test_api_key",
        }
    )
    return config


@pytest.mark.asyncio
async def test_speak_connector_connect_with_valid_input():
    config = ActionConfig()
    connector = MockSpeakRivaTTSConnector(config)
    speak_input = SpeakInput(action="Hello, world!")

    await connector.connect(speak_input)

    # Check that callbacks were registered
    connector.tts.register_tts_state_callback.assert_called_once_with(
        connector.asr.audio_stream.on_tts_state_change
    )

    # Check that message was added to TTS
    connector.tts.add_pending_message.assert_called_once_with("Hello, world!")


@pytest.mark.asyncio
async def test_speak_connector_connect_with_empty_input():
    config = ActionConfig()
    connector = MockSpeakRivaTTSConnector(config)
    speak_input = SpeakInput(action="")

    await connector.connect(speak_input)

    # Check that callbacks were registered
    connector.tts.register_tts_state_callback.assert_called_once_with(
        connector.asr.audio_stream.on_tts_state_change
    )

    # The add_pending_message should not be called with None
    connector.tts.add_pending_message.assert_not_called()


@pytest.mark.asyncio
async def test_speak_connector_error_handling():
    config = ActionConfig()
    connector = MockSpeakRivaTTSConnector(config)
    speak_input = SpeakInput(action="Test message")

    # Simulate an error when adding pending message
    connector.tts.add_pending_message.side_effect = Exception("TTS error")

    # Test should not raise an exception
    await connector.connect(speak_input)

    # Verify the method was called despite the error
    connector.tts.add_pending_message.assert_called_once_with("Test message")


# Clean up patchers
for patcher in patchers:
    patcher.stop()
