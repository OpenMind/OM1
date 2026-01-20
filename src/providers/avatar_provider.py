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
            request = AvatarFaceRequest.deserialize(sample.payload.to_bytes())

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

        if self.avatar_subscriber:
            self.avatar_subscriber.undeclare()
            self.avatar_subscriber = None

        if self.avatar_healthcheck_publisher:
            self.avatar_healthcheck_publisher.undeclare()
            self.avatar_healthcheck_publisher = None

        if self.session:
            self.session.close()

        logging.info("AvatarProvider stopped and Zenoh session closed")
