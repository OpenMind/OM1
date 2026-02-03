import logging
import threading
from uuid import uuid4

import zenoh

from backgrounds.base import Background, BackgroundConfig
from providers.context_provider import ContextProvider
from providers.greeting_conversation_state_provider import (
    ConversationState,
    GreetingConversationStateMachineProvider,
)
from zenoh_msgs import PersonGreetingStatus, String, open_zenoh_session, prepare_header


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

        self.person_greeting_topic = "om/person_greeting"
        self._is_person_approached = False
        self._lock = threading.Lock()

        try:
            self.session = open_zenoh_session()
            self.session.declare_subscriber(
                self.person_greeting_topic, self._on_person_greeting
            )
        except Exception as e:
            logging.error(f"Error opening Zenoh session in ApproachingPerson: {e}")

        logging.info("ApproachingPerson background task initialized.")

    def _on_person_greeting(self, data: zenoh.Sample) -> None:
        """
        Callback function when a person greeting message is received.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received, which should have a 'payload' attribute.
        """
        logging.debug("Person greeting detected via Zenoh message.")

        person_greeting_status = PersonGreetingStatus.deserialize(
            data.payload.to_bytes()
        )

        if (
            person_greeting_status.status
            == PersonGreetingStatus.STATUS.APPROACHED.value
        ):
            with self._lock:
                logging.info("Person is approaching. Triggering greeting mode.")
                self._is_person_approached = True

                context_provider = ContextProvider()
                context_provider.update_context({"approaching_detected": True})

                self.greeting_state_provider.reset_state(ConversationState.ENGAGING)

    def run(self) -> None:
        """
        Run the ApproachingPerson background task.
        """
        with self._lock:
            if self._is_person_approached:
                logging.debug("Skipping SWITCH status - person already approached.")
                pass

        if self._is_person_approached:
            self.sleep(5)
            return

        if not self.session:
            logging.warning("No Zenoh session available in ApproachingPerson.")
            self.sleep(5)
            return

        request_id = str(uuid4())

        self.session.put(
            self.person_greeting_topic,
            PersonGreetingStatus(
                header=prepare_header(request_id),
                request_id=String(data=request_id),
                status=PersonGreetingStatus.STATUS.SWITCH.value,
            ).serialize(),
        )

        self.sleep(5)

    def stop(self) -> None:
        """
        Stop the ApproachingPerson background task.
        """
        logging.info("Stopping ApproachingPerson background task.")

        if self.session:
            self.session.close()
