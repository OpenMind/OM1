import logging
import time
from typing import Optional

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.greeting_conversation.interface import GreetingConversationInput
from providers.context_provider import ContextProvider
from providers.elevenlabs_tts_provider import ElevenLabsTTSProvider
from providers.greeting_conversation_state_provider import (
    ConversationState,
    GreetingConversationStateMachineProvider,
)


class SpeakElevenLabsTTSConfig(ActionConfig):
    """
    Configuration for ElevenLabs TTS connector.

    Parameters
    ----------
    elevenlabs_api_key : Optional[str]
        ElevenLabs API key.
    voice_id : str
        ElevenLabs voice ID.
    model_id : str
        ElevenLabs model ID.
    output_format : str
        ElevenLabs output format.
    silence_rate : int
        Number of responses to skip before speaking.
    """

    elevenlabs_api_key: Optional[str] = Field(
        default=None,
        description="ElevenLabs API key",
    )
    voice_id: str = Field(
        default="JBFqnCBsd6RMkjVDRZzb",
        description="ElevenLabs voice ID",
    )
    model_id: str = Field(
        default="eleven_flash_v2_5",
        description="ElevenLabs model ID",
    )
    output_format: str = Field(
        default="mp3_44100_128",
        description="ElevenLabs output format",
    )
    silence_rate: int = Field(
        default=0,
        description="Number of responses to skip before speaking",
    )


class GreetingConversationConnector(
    ActionConnector[SpeakElevenLabsTTSConfig, GreetingConversationInput]
):
    """
    Connector that manages greeting conversations for the robot.
    """

    def __init__(self, config: SpeakElevenLabsTTSConfig):
        """
        Initialize the GreetingConversationConnector.

        Parameters
        ----------
        config : ActionConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        self.greeting_state_provider = GreetingConversationStateMachineProvider()
        self.context_provider = ContextProvider()

        # TODO: update the conversation state in the entry point
        self.greeting_state_provider.current_state = ConversationState.CONVERSING

        # OM API key
        api_key = getattr(self.config, "api_key", None)

        # Eleven Labs TTS configuration
        elevenlabs_api_key = self.config.elevenlabs_api_key
        voice_id = self.config.voice_id
        model_id = self.config.model_id
        output_format = self.config.output_format

        # TTS Setup
        self.tts = ElevenLabsTTSProvider(
            url="https://api.openmind.org/api/core/elevenlabs/tts",
            api_key=api_key,
            elevenlabs_api_key=elevenlabs_api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
        )
        self.tts.start()

        self.tts_triggered_time = time.time()
        self.tts_duration = 0.0  # Estimated TTS duration in seconds

    async def connect(self, output_interface: GreetingConversationInput) -> None:
        """
        Connects to the greeting conversation system and processes the input.

        Parameters
        ----------
        output_interface : GreetingConversationInput
            The output interface containing the greeting conversation data.
        """
        logging.info(f"Conversation State: {output_interface.conversation_state}")
        logging.info(f"Greeting Response: {output_interface.response}")
        logging.info(f"Confidence Score: {output_interface.confidence}")
        logging.info(f"Speech Clarity Score: {output_interface.speech_clarity}")

        llm_output = {
            "conversation_state": output_interface.conversation_state,
            "response": output_interface.response,
            "confidence": output_interface.confidence,
            "speech_clarity": output_interface.speech_clarity,
        }

        self.tts.add_pending_message(output_interface.response)

        # Estimate TTS duration based on text length (~100 words per minute speech rate)
        word_count = len(output_interface.response.split())
        self.tts_duration = (word_count / 100.0) * 60.0  # Convert to seconds
        self.tts_triggered_time = time.time()

        response = self.greeting_state_provider.process_conversation(llm_output)
        logging.info(f"Greeting Conversation Response: {response}")

        if response.get("current_state") == ConversationState.FINISHED:
            logging.info("Greeting conversation has finished.")
            self.context_provider.update_context(
                {"greeting_conversation_finished": True}
            )

    def tick(self) -> None:
        """
        Tick method for the connector.

        Periodically updates the conversation state even without LLM input.
        """
        logging.info("GreetingConversationConnector tick called")

        self.sleep(10)

        if time.time() - self.tts_triggered_time < self.tts_duration:
            logging.info(
                f"Skipping tick update due to recent TTS activity (remaining: {self.tts_duration - (time.time() - self.tts_triggered_time):.1f}s)."
            )
            return

        # Update state based on current factors (silence, time, etc.)
        state_update = self.greeting_state_provider.update_state_without_llm()

        # Check if conversation has finished
        if state_update.get("current_state") == ConversationState.FINISHED.value:
            logging.info("Greeting conversation has finished (detected in tick).")
            self.context_provider.update_context(
                {"greeting_conversation_finished": True}
            )

        # Log the updated state
        logging.info(
            f"State: {state_update.get('current_state')}, "
            f"Confidence: {state_update.get('confidence', {}).get('overall', 0):.2f}, "
            f"Silence: {state_update.get('silence_duration', 0):.1f}s"
        )
