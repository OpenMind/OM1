import logging

from backgrounds.base import Background, BackgroundConfig
from providers.context_provider import ContextProvider
from providers.greeting_conversation_state_provider import (
    ConversationState,
    GreetingConversationStateMachineProvider,
)


class ApproachingPerson(Background[BackgroundConfig]):
    """
    Background task that approaches a person and triggers greeting mode.
    """

    def __init__(self, config: BackgroundConfig):
        """
        Initialize the ApproachingPerson background task.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration for the background task.
        """
        super().__init__(config)

        self.greeting_state_provider = GreetingConversationStateMachineProvider()

        logging.info("ApproachingPerson background task initialized.")

    def run(self) -> None:
        """
        Run the ApproachingPerson background task.
        """
        logging.info("ApproachingPerson run executed.")

        if not self.sleep(10):
            logging.info("ApproachingPerson: Sleep interrupted by stop signal")
            return

        context_provider = ContextProvider()
        context_provider.update_context({"approaching_detected": True})

        self.greeting_state_provider.reset_state(ConversationState.ENGAGING)
