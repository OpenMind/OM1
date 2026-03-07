import asyncio
import json
import logging
import time
from uuid import uuid4

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.greeting_conversation_simplified.interface import (
    GreetingConversationSimplifiedInput,
)
from providers.context_provider import ContextProvider
from providers.greeting_conversation_state_provider import (
    ConversationState,
    GreetingConversationStateMachineProvider,
)
from providers.kokoro_tts_provider import KokoroTTSProvider
from providers.tts_text_utils import normalize_tts_text
from zenoh_msgs import (
    PersonGreetingStatus,
    String,
    open_zenoh_session,
    prepare_header,
)


class SpeakKokoroTTSConfig(ActionConfig):
    """
    Configuration for Kokoro TTS connector.

    Parameters
    ----------
    base_url : str
        Base URL for Kokoro TTS API.
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

    base_url: str = Field(
        default="http://127.0.0.1:8880/v1",
        description="Base URL for Kokoro TTS API",
    )
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
    ActionConnector[SpeakKokoroTTSConfig, GreetingConversationSimplifiedInput]
):
    """
    Simplified greeting conversation connector with Kokoro TTS.

    Uses a single 'response' field from the LLM and hardcodes
    conversation state values for the state machine.
    Applies TTS text normalization (e.g. month abbreviation expansion).
    """

    def __init__(self, config: SpeakKokoroTTSConfig):
        super().__init__(config)

        self.greeting_state_provider = GreetingConversationStateMachineProvider()
        self.greeting_state_provider.start_conversation()

        self.context_provider = ContextProvider()

        # Create Kokoro TTS provider
        api_key = getattr(self.config, "api_key", None)
        logging.info("Creating Kokoro TTS provider")
        self.tts = KokoroTTSProvider(
            url=self.config.base_url,
            api_key=api_key,
            voice_id=self.config.voice_id,
            model_id=self.config.model_id,
            output_format=self.config.output_format,
            rate=self.config.rate,
            enable_tts_interrupt=self.config.enable_tts_interrupt,
        )
        self.tts.start()

        self.tts_triggered_time = time.time()
        self.tts_duration = 0.0
        self.conversation_finished_sent = False
        self.pending_finished_update = False
        self.delayed_update_task = None

        self.person_greeting_topic = "om/person_greeting"
        try:
            self.session = open_zenoh_session()
            logging.info("Zenoh session opened for PersonGreetingStatus publishing")
        except Exception as e:
            logging.error(f"Error opening Zenoh session: {e}")
            self.session = None

        self.greeting_status = ConversationState.CONVERSING.value

    async def connect(
        self, output_interface: GreetingConversationSimplifiedInput
    ) -> None:
        """
        Process the greeting conversation response.

        Only reads 'response' from the LLM output and hardcodes
        conversation state values for the state machine.
        """
        logging.info(f"Greeting Response: {output_interface.response}")

        llm_output = {
            "conversation_state": ConversationState.CONVERSING.value,
            "response": output_interface.response,
            "confidence": 0.85,
            "speech_clarity": 0.85,
        }

        tts_text = normalize_tts_text(output_interface.response)
        self.tts.add_pending_message(tts_text)

        # Estimate TTS duration based on text length (~100 words per minute speech rate)
        word_count = len(output_interface.response.split())
        self.tts_duration = (
            word_count / 100.0
        ) * 60.0 + 5  # Convert to seconds and add buffer time
        self.tts_triggered_time = time.time()

        state_update = self.greeting_state_provider.process_conversation(llm_output)
        current_state = state_update.get("current_state", self.greeting_status)
        self.greeting_status = current_state
        self.publish_countdown_status(self.greeting_status)

        logging.info(f"Greeting Conversation Response: {state_update}")

        if (
            self.greeting_status == ConversationState.FINISHED.value
            and not self.conversation_finished_sent
        ):
            logging.info(
                f"Greeting conversation state is FINISHED. "
                f"Scheduling context update after TTS completes ({self.tts_duration:.1f}s)."
            )
            self.pending_finished_update = True
            self.conversation_finished_sent = True
            self.delayed_update_task = asyncio.create_task(
                self._delayed_context_update((word_count / 150.0) * 60.0)
            )

    async def _delayed_context_update(self, wait_duration: float) -> None:
        """Wait for TTS to finish, then update context to indicate conversation finished."""
        try:
            logging.info(
                f"Waiting {wait_duration:.1f}s for TTS to complete before updating context..."
            )
            await asyncio.sleep(wait_duration)

            if self.pending_finished_update:
                logging.info(
                    "TTS completed. Updating context: greeting_conversation_finished = True"
                )
                self.context_provider.update_context(
                    {"greeting_conversation_finished": True}
                )
                self.pending_finished_update = False
        except Exception as e:
            logging.error(f"Error in delayed context update: {e}")

    def tick(self) -> None:
        """Periodically update conversation state even without LLM input."""
        logging.info("GreetingConversationConnector tick called")
        self.sleep(10)

        if time.time() - self.tts_triggered_time < self.tts_duration:
            logging.info(
                f"Skipping tick update due to recent TTS activity "
                f"(remaining: {self.tts_duration - (time.time() - self.tts_triggered_time):.1f}s)."
            )
            return

        state_update = self.greeting_state_provider.update_state_without_llm()
        current_state = state_update.get("current_state", self.greeting_status)
        self.greeting_status = current_state
        self.publish_countdown_status(self.greeting_status)

        if (
            current_state == ConversationState.FINISHED.value
            and not self.conversation_finished_sent
        ):
            logging.info("Greeting conversation has finished (detected in tick).")
            self.context_provider.update_context(
                {"greeting_conversation_finished": True}
            )
            self.conversation_finished_sent = True

        logging.info(
            f"State: {current_state}, "
            f"Confidence: {state_update.get('confidence', {}).get('overall', 0):.2f}, "
            f"Silence: {state_update.get('silence_duration', 0):.1f}s"
        )

    def publish_countdown_status(self, current_state: str) -> None:
        """Publish countdown status to Zenoh based on current conversation state."""
        if current_state == ConversationState.CONVERSING.value:
            seconds_until_finished = 20
        elif current_state == ConversationState.CONCLUDING.value:
            seconds_until_finished = 10
        else:
            seconds_until_finished = 0

        if self.session:
            request_id = str(uuid4())
            message_text = json.dumps(
                {"seconds_until_finished": seconds_until_finished}
            )
            try:
                self.session.put(
                    self.person_greeting_topic,
                    PersonGreetingStatus(
                        header=prepare_header(request_id),
                        request_id=String(data=request_id),
                        status=PersonGreetingStatus.STATUS.CONVERSATION.value,
                        message=String(data=message_text),
                    ).serialize(),
                )
                logging.info(f"Published PersonGreetingStatus: {message_text}")
            except Exception as e:
                logging.error(f"Error publishing PersonGreetingStatus: {e}")

    def stop(self):
        """Stop the connector and clean up resources."""
        logging.info("Stopping Greeting Conversation action...")
        if self.session:
            self.session.close()
