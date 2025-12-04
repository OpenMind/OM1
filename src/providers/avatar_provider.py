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
        self.session = None
        self.avatar_publisher = None
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

            # Publisher for avatar face switch requests
            self.avatar_publisher = self.session.declare_publisher("om/avatar/request")

            # Publisher for healthcheck responses
            self.avatar_healthcheck_publisher = self.session.declare_publisher(
                "om/avatar/response"
            )

            # Subscriber for healthcheck requests (same topic is intentional here)
            self.avatar_subscriber = self.session.declare_subscriber(
                "om/avatar/request", self._handle_avatar_request
            )

            self.running = True
            logging.info("AvatarProvider initialized with Zenoh successfully")

        except Exception as e:
            logging.error(f"Failed to initialize AvatarProvider Zenoh session: {e}")
            self.running = False

    def _handle_avatar_request(self, sample: zenoh.Sample):
        """
        Handle incoming avatar requests from Zenoh subscriber.
        """
        if not sample or not sample.payload:
            logging.warning("Received empty avatar payload")
            return

        try:
            payload_bytes = sample.payload.to_bytes()

            if not payload_bytes:
                logging.warning("Received avatar payload with empty bytes")
                return

            request = AvatarFaceRequest.deserialize(payload_bytes)

        except Exception as e:
            logging.error(f"Invalid avatar payload received: {e}")
            return

        if request.code == AvatarFaceRequest.Code.STATUS.value:
            logging.debug("Received avatar health check request")

            try:
                response = AvatarFaceResponse(
                    header=prepare_header(str(uuid4())),
                    request_id=request.request_id,
                    code=AvatarFaceResponse.Code.ACTIVE.value,
                    message=String("Avatar system active"),
                )

                if self.avatar_healthcheck_publisher:
                    self.avatar_healthcheck_publisher.put(response.serialize())
                    logging.debug("Sent avatar active healthcheck response")
                else:
                    logging.error("Healthcheck publisher is not available")

            except Exception as e:
                logging.error(f"Failed to send avatar healthcheck response: {e}")

    def send_avatar_command(self, command: str) -> bool:
        """
        Send avatar command via Zenoh.
        """
        if not self.running:
            logging.warning(
                f"AvatarProvider not running, cannot send command: {command}"
            )
            return False

        if not self.session or not self.avatar_publisher:
            logging.error("Zenoh session or avatar publisher not initialized")
            return False

        try:
            request_id = str(uuid4())
            face_text = command.lower()

            face_msg = AvatarFaceRequest(
                header=prepare_header(request_id),
                request_id=String(request_id),
                code=AvatarFaceRequest.Code.SWITCH_FACE.value,
                face_text=String(face_text),
            )

            self.avatar_publisher.put(face_msg.serialize())
            logging.info(f"Avatar command sent via Zenoh: {face_text}")
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

        try:
            if self.avatar_subscriber:
                self.avatar_subscriber.undeclare()
                self.avatar_subscriber = None
                logging.debug("Avatar subscriber undeclared")

            if self.avatar_healthcheck_publisher:
                self.avatar_healthcheck_publisher.undeclare()
                self.avatar_healthcheck_publisher = None
                logging.debug("Avatar healthcheck publisher undeclared")

            if self.avatar_publisher:
                self.avatar_publisher.undeclare()
                self.avatar_publisher = None
                logging.debug("Avatar publisher undeclared")

            if self.session:
                self.session.close()
                self.session = None
                logging.debug("Zenoh session closed")

            logging.info("AvatarProvider stopped and cleaned up successfully")

        except Exception as e:
            logging.error(f"Error while stopping AvatarProvider: {e}")
