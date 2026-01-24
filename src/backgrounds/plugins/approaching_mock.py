import logging
import time

from backgrounds.base import Background, BackgroundConfig
from providers.context_provider import ContextProvider
from providers.greeting_conversation_state_provider import (
    ConversationState,
    GreetingConversationStateMachineProvider,
)


class ApproachingMock(Background[BackgroundConfig]):
    """
    Background task for simulating approaching behavior.
    """

    def __init__(self, config: BackgroundConfig):
        """
        Initialize D435 background task with configuration.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration object for the background task. The base configuration
            is used as D435 does not require additional parameters.
        """
        super().__init__(config)

        self.greeting_state_provider = GreetingConversationStateMachineProvider()

        logging.info("ApproachingMock background task initialized.")

    def run(self) -> None:
        """
        Run the approaching mock background task.
        Randomly decides whether to switch to greeting mode using context-aware transitions.
        """
        logging.info("ApproachingMock run executed.")

        if not self.sleep(10):
            logging.info("ApproachingMock: Sleep interrupted by stop signal")
            return

        logging.info("ApproachingMock: Triggering context for greeting mode")
        context_provider = ContextProvider()
        context_provider.update_context({"approaching_detected": True})

        self.greeting_state_provider.current_state = ConversationState.CONVERSING
        self.greeting_state_provider.previous_state = None
        self.greeting_state_provider.state_entry_time = time.time()
        self.greeting_state_provider.conversation_start_time = None
        self.greeting_state_provider.turn_count = 0
        self.greeting_state_provider.last_user_utterance = ""
        self.greeting_state_provider.confidence_history = []
