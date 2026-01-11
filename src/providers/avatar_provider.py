import logging
from uuid import uuid4

import zenoh

from zenoh_msgs import (
    AvatarFaceRequest,
    AvatarFaceResponse,
    String,
    open_zenoh_session,
    prepare_header,
)

from .singleton import singleton


@singleton
class AvatarProvider:
    """
    Singleton provider for Avatar communication via Zenoh.
    """

    def __init__(self):
        """
        Initialize the AvatarProvider.
        """
        self.session = None
        # Face Publisher
        self.avatar_publisher = None
        # Health Check Publisher and Subscriber
        self.avatar_healthcheck_publisher = None
        self.avatar_subscriber = None
        self.running = False

        self._initialize_zenoh()

    def _initialize_zenoh(self):
        """
        Initialize Zenoh session, publishers, and subscriber.
        """
        try:
            self.session = open_zenoh_session()
            self.avatar_publisher = self.session.declare_publisher("om/avatar/request")
            self.avatar_healthcheck_publisher = self.session.declare_publisher(
                "om/avatar/response"
            )
            self.avatar_subscriber = self.session.declare_subscriber(
                "om/avatar/request", self._handle_avatar_request
            )
            self.running = True
            logging.info("AvatarProvider initialized with Zenoh on topics")
        except Exception as e:
            logging.error(f"Failed to initialize AvatarProvider Zenoh session: {e}")

    def _handle_avatar_request(self, sample: zenoh.Sample):
        """
        Handle incoming avatar requests from Zenoh subscriber.

        Processes health check requests (STATUS) and responds with system status.
        Face change requests (SWITCH_FACE) are ignored in this callback.

        Parameters
        ----------
        sample : zenoh.Sample
            The Zenoh sample containing the serialized AvatarFaceRequest message.
        """
        try:
            payload_bytes = sample.payload.to_bytes()
            if len(payload_bytes) == 0:
                logging.warning("Received empty avatar request payload")
                return

            request = AvatarFaceRequest.deserialize(payload_bytes)

            if request.code == AvatarFaceRequest.Code.STATUS.value:
                logging.debug("Received avatar health check request")

                response = AvatarFaceResponse(
                    header=prepare_header(str(uuid4())),
                    request_id=request.request_id,
                    code=AvatarFaceResponse.Code.ACTIVE.value,
                    message=String("Avatar system active"),
                )

                if self.avatar_healthcheck_publisher:
                    self.avatar_healthcheck_publisher.put(response.serialize())
                    logging.debug("Sent avatar active response")

        except Exception as e:
            logging.error(f"Error handling avatar request: {e}")

    def send_avatar_command(self, command: str) -> bool:
        """
        Send avatar command via Zenoh.

        Parameters
        ----------
        command : str

        Returns
        -------
        bool
            True if command was sent successfully, False otherwise
        """
        if not self.running or not self.avatar_publisher:
            logging.warning(
                f"AvatarProvider not running, cannot send command: {command}"
            )
            return False

        command = command.lower()

        try:
            request_id = str(uuid4())
            face_text = command

            face_msg = AvatarFaceRequest(
                header=prepare_header(request_id),
                request_id=String(request_id),
                code=AvatarFaceRequest.Code.SWITCH_FACE.value,
                face_text=String(face_text),
            )
            self.avatar_publisher.put(face_msg.serialize())
            logging.info(f"AvatarProvider sent command to Zenoh: {face_text}")
            return True

        except Exception as e:
            logging.error(f"Failed to send avatar command via Zenoh: {e}")
            return False

    def stop(self):
        """
        Stop the AvatarProvider and cleanup Zenoh session.
        """
        if not self.running:
            logging.info("AvatarProvider is not running")
            return

        self.running = False

        # Close publisher
        if self.avatar_publisher:
            try:
                self.avatar_publisher.undeclare()
            except Exception as e:
                logging.warning(f"Failed to undeclare avatar publisher: {e}")
            self.avatar_publisher = None

        # Close subscriber
        if self.avatar_subscriber:
            try:
                self.avatar_subscriber.close()  # ✅ Benar: subscriber pakai close()
            except Exception as e:
                logging.warning(f"Failed to close avatar subscriber: {e}")
            self.avatar_subscriber = None

        # Close healthcheck publisher
        if self.avatar_healthcheck_publisher:
            try:
                self.avatar_healthcheck_publisher.undeclare()
            except Exception as e:
                logging.warning(f"Failed to undeclare healthcheck publisher: {e}")
            self.avatar_healthcheck_publisher = None

        # Close session last
        if self.session:
            try:
                self.session.close()
            except Exception as e:
                logging.warning(f"Failed to close Zenoh session: {e}")
            self.session = None

        logging.info("AvatarProvider stopped and all Zenoh resources cleaned up")
