import json
import logging
import time
from typing import Optional
from uuid import uuid4

import zenoh
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.fall_detection.interface import FallDetectionInput, FallSeverity
from providers.health_detection_provider import HealthDetectionProvider
from providers.io_provider import IOProvider
from zenoh_msgs import (
    AudioStatus,
    String,
    open_zenoh_session,
    prepare_header,
)


class FallDetectionEmergencyConfig(ActionConfig):
    """
    Configuration for Fall Detection Emergency Alert connector.

    Parameters
    ----------
    enable_emergency_calls : bool
        Whether to enable emergency service calls for high-severity falls.
    emergency_contact : Optional[str]
        Emergency contact number or identifier.
    alert_family_members : bool
        Whether to alert family members when a fall is detected.
    """

    enable_emergency_calls: bool = Field(
        default=False, description="Enable emergency service calls"
    )
    emergency_contact: Optional[str] = Field(
        default=None, description="Emergency contact number or identifier"
    )
    alert_family_members: bool = Field(
        default=True, description="Alert family members on fall detection"
    )


class FallDetectionEmergencyConnector(
    ActionConnector[FallDetectionEmergencyConfig, FallDetectionInput]
):
    """
    Connector that handles fall detection and triggers emergency alerts.
    """

    def __init__(self, config: FallDetectionEmergencyConfig):
        """
        Initialize the FallDetectionEmergencyConnector.

        Parameters
        ----------
        config : FallDetectionEmergencyConfig
            Configuration for the action connector.
        """
        super().__init__(config)

        self.io_provider = IOProvider()
        self.health_provider = HealthDetectionProvider()

        self.audio_topic = "robot/status/audio"
        self.session = None
        self.audio_pub = None

        self.audio_status = AudioStatus(
            header=prepare_header(str(uuid4())),
            status_mic=AudioStatus.STATUS_MIC.UNKNOWN.value,
            status_speaker=AudioStatus.STATUS_SPEAKER.READY.value,
            sentence_to_speak=String(""),
        )

        try:
            self.session = open_zenoh_session()
            self.audio_pub = self.session.declare_publisher(self.audio_topic)
            self.session.declare_subscriber(self.audio_topic, self.zenoh_audio_message)

            if self.audio_pub:
                self.audio_pub.put(self.audio_status.serialize())

            logging.info("Fall Detection Emergency Alert Zenoh client opened")
        except Exception as e:
            logging.error(f"Error opening Fall Detection Zenoh client: {e}")

    def zenoh_audio_message(self, data: zenoh.Sample):
        """
        Process an incoming audio status message.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample received.
        """
        self.audio_status = AudioStatus.deserialize(data.payload.to_bytes())

    async def connect(self, output_interface: FallDetectionInput) -> None:
        """
        Handle fall detection and trigger appropriate emergency response.

        Parameters
        ----------
        output_interface : FallDetectionInput
            The input protocol containing fall detection details.
        """
        # Record the fall event
        fall_event = self.health_provider.record_fall(
            severity=output_interface.severity.value,
            location=output_interface.location,
            person_name=output_interface.person_name,
            confidence=output_interface.confidence,
        )

        # Generate alert message based on severity
        alert_message = self._generate_alert_message(output_interface)

        # Log the fall detection
        logging.warning(
            f"FALL DETECTED: {output_interface.severity.value} severity, "
            f"confidence={output_interface.confidence:.2f}, "
            f"location={output_interface.location}, "
            f"person={output_interface.person_name}"
        )

        # For high severity, trigger immediate emergency response
        if output_interface.severity == FallSeverity.HIGH:
            if self.config.enable_emergency_calls:
                logging.critical(
                    f"EMERGENCY: High-severity fall detected. "
                    f"Contacting emergency services: {self.config.emergency_contact}"
                )
                # Here you would integrate with emergency service APIs

            if self.config.alert_family_members:
                logging.info("Alerting family members about the fall")

        # Update audio status to broadcast alert
        state = AudioStatus(
            header=prepare_header(str(uuid4())),
            status_mic=self.audio_status.status_mic,
            status_speaker=AudioStatus.STATUS_SPEAKER.ACTIVE.value,
            sentence_to_speak=String(json.dumps({"message": alert_message})),
        )

        if self.audio_pub:
            self.audio_pub.put(state.serialize())

        # Mark as responded
        self.health_provider.mark_fall_responded(fall_event)

    def _generate_alert_message(self, input_interface: FallDetectionInput) -> str:
        """
        Generate alert message based on fall detection input.

        Parameters
        ----------
        input_interface : FallDetectionInput
            The fall detection input.

        Returns
        -------
        str
            The alert message.
        """
        person = input_interface.person_name if input_interface.person_name else "Someone"
        location = f" in {input_interface.location}" if input_interface.location else ""

        if input_interface.severity == FallSeverity.HIGH:
            return (
                f"EMERGENCY ALERT: {person} has fallen{location}. "
                f"Immediate assistance is required. Please check on them right away."
            )
        elif input_interface.severity == FallSeverity.MEDIUM:
            return (
                f"Attention: Possible fall detected for {person}{location}. "
                f"Please check if assistance is needed."
            )
        else:  # LOW
            return (
                f"Notice: Unusual movement detected for {person}{location}. "
                f"Please verify if everything is okay."
            )

