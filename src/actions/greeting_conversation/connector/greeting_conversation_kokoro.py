import logging
import time

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.greeting_conversation.interface import GreetingConversationInput
from providers.context_provider import ContextProvider
from providers.greeting_conversation_state_provider import (
    ConversationState,
    GreetingConversationStateMachineProvider,
)
from providers.kokoro_tts_provider import KokoroTTSProvider


class SpeakKokoroTTSConfig(ActionConfig):
    """
    Configuration for Kokoro TTS connector.

    Parameters
    ----------
    voice_id : str
        Kokoro voice ID.
    model_id : str
        Kokoro model ID.
    output_format : str
        Kokoro output format.
    rate : int
        Audio sample rate in Hz.
    enable_tts_interrupt : bool
        Enable TTS interrupt when ASR detects speech during playback.
    silence_rate : int
        Number of responses to skip before speaking.
    """

    voice_id: str = Field(
        default="af_bella",
        description="Kokoro voice ID",
    )
    model_id: str = Field(
        default="kokoro",
        description="Kokoro model ID",
    )
    output_format: str = Field(
        default="pcm",
        description="Kokoro output format",
    )
    rate: int = Field(
        default=24000,
        description="Audio sample rate in Hz",
    )
    enable_tts_interrupt: bool = Field(
        default=False,
        description="Enable TTS interrupt when ASR detects speech during playback",
    )
    silence_rate: int = Field(
        default=0,
        description="Number of responses to skip before speaking",
    )


class GreetingConversationConnector(
    ActionConnector[SpeakKokoroTTSConfig, GreetingConversationInput]
):
    """
    Connector that manages greeting conversations for the robot.
    """

    def __init__(self, config: SpeakKokoroTTSConfig):
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

        # Kokoro TTS configuration
        voice_id = self.config.voice_id
        model_id = self.config.model_id
        output_format = self.config.output_format
        rate = self.config.rate
        enable_tts_interrupt = self.config.enable_tts_interrupt

        # TTS Setup
        self.tts = KokoroTTSProvider(
            url="http://127.0.0.1:8880/v1",
            api_key=api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            rate=rate,
            enable_tts_interrupt=enable_tts_interrupt,
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
