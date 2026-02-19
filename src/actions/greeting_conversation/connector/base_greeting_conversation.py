import json
import logging
import time
from abc import abstractmethod
from typing import Any, Generic, TypeVar
from uuid import uuid4

from actions.base import ActionConfig, ActionConnector
from actions.greeting_conversation.interface import GreetingConversationInput
from providers.context_provider import ContextProvider
from providers.greeting_conversation_state_provider import (
    ConversationState,
    GreetingConversationStateMachineProvider,
)
from zenoh_msgs import PersonGreetingStatus, String, open_zenoh_session, prepare_header

ConfigT = TypeVar("ConfigT", bound=ActionConfig)


class BaseGreetingConversationConnector(
    ActionConnector[ConfigT, GreetingConversationInput], Generic[ConfigT]
):
    """
    Base connector that manages greeting conversations for the robot.

    This class provides common functionality for greeting conversation connectors
    with different TTS providers. Subclasses should implement the `create_tts_provider`
    method to instantiate their specific TTS provider.
    """

    def __init__(self, config: ConfigT):
        """
        Initialize the BaseGreetingConversationConnector.

        Parameters
        ----------
        config : ConfigT
            Configuration for the action connector.
        """
        super().__init__(config)

        self.greeting_state_provider = GreetingConversationStateMachineProvider()
        self.greeting_state_provider.start_conversation()

        self.context_provider = ContextProvider()

        # Create TTS provider
        self.tts = self.create_tts_provider()
        self.tts.start()

        self.tts_triggered_time = time.time()
        self.tts_duration = 0.0
        self.conversation_finished_sent = False

        self.person_greeting_topic = "om/person_greeting"
        try:
            self.session = open_zenoh_session()
            logging.info("Zenoh session opened for PersonGreetingStatus publishing")
        except Exception as e:
            logging.error(f"Error opening Zenoh session: {e}")
            self.session = None

        self.greeting_status = ConversationState.CONVERSING.value

    @abstractmethod
    def create_tts_provider(self) -> Any:
        """
        Create and return the TTS provider for this connector.

        This method must be implemented by subclasses to instantiate
        their specific TTS provider (e.g., ElevenLabsTTSProvider, KokoroTTSProvider).

        Returns
        -------
        Any
            The instantiated TTS provider with `start()` and `add_pending_message()` methods.
        """
        pass

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
            logging.info("Greeting conversation has finished.")
            self.context_provider.update_context(
                {"greeting_conversation_finished": True}
            )
            self.conversation_finished_sent = True

    def tick(self) -> None:
        """
        Tick method for the connector.

        Periodically updates the conversation state even without LLM input.
        """
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
        """
        Publish the countdown status to Zenoh based on the current conversation state.

        Parameters
        ----------
        current_state : str
            The current state of the conversation.
        """
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
        """
        Stop the connector and clean up resources.
        """
        logging.info("Stopping Greeting Conversation action...")

        if self.session:
            self.session.close()
