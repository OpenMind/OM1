# src/inputs/plugins/posture_detection_input.py

import asyncio
import logging
import time
from collections import deque
from typing import Deque, Optional

import aiohttp
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class PostureDetectionInputConfig(SensorConfig):
    """
    Configuration for Posture Detection Input.

    Parameters
    ----------
    posture_detection_base_url : str
        Base URL for the posture detection HTTP service.
    poll_interval : float
        Polling interval in seconds.
    """

    posture_detection_base_url: str = Field(
        default="http://localhost:8080",
        description="Base URL for the posture detection HTTP service",
    )
    poll_interval: float = Field(
        default=1.0,
        description="Polling interval in seconds",
    )


class PostureDetectionInput(FuserInput[PostureDetectionInputConfig, Optional[str]]):
    """
    Input that polls the posture detection service for posture status.

    This input periodically queries the posture detection endpoint and provides
    structured posture detection information to the LLM, including:
    - Posture type (good, slumped, hunched, leaning, asymmetric, laying)
    - Severity level (mild, moderate, severe)
    - Duration in the current posture
    - Person name (if available from face recognition)
    - Recommendations for improvement

    The LLM can use this information to trigger the PostureDetection action
    when poor posture is detected.
    """

    def __init__(self, config: PostureDetectionInputConfig):
        """
        Initialize the PostureDetectionInput.

        Parameters
        ----------
        config : PostureDetectionInputConfig
            Configuration for the posture detection input.
        """
        super().__init__(config)

        self.io_provider = IOProvider()
        self.messages: Deque[Message] = deque(maxlen=50)

        self.base_url = config.posture_detection_base_url
        self.poll_interval = config.poll_interval
        self.status_url = f"{self.base_url}/posture/status"
        self.detection_url = f"{self.base_url}/posture/detect"

        self.descriptor_for_LLM = "PostureDetection"

        # Track previous state for change detection
        self._previous_posture_type: Optional[str] = None
        self._previous_severity: Optional[str] = None
        self._posture_start_time: Optional[float] = None

        logging.info(
            f"PostureDetectionInput initialized, polling {self.status_url} "
            f"every {self.poll_interval}s"
        )

    async def _poll(self) -> Optional[str]:
        """
        Poll the posture detection status endpoint.

        Returns
        -------
        Optional[str]
            Formatted posture detection message if there's a meaningful update, None otherwise.
        """
        await asyncio.sleep(self.poll_interval)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.status_url,
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as response:
                    if response.status != 200:
                        return None

                    data = await response.json()
                    return self._format_posture_detection(data)

        except aiohttp.ClientError as e:
            logging.debug(f"PostureDetectionInput: Poll failed: {e}")
            return None
        except Exception as e:
            logging.warning(f"PostureDetectionInput: Unexpected error: {e}")
            return None

    def _format_posture_detection(self, data: dict) -> Optional[str]:
        """
        Format the posture detection data into a structured message for the LLM.

        Only returns a message when there's a meaningful state change or
        periodically during poor posture detection.

        Parameters
        ----------
        data : dict
            Raw posture detection data from the endpoint.

        Returns
        -------
        Optional[str]
            Formatted posture detection message or None if no update needed.
        """
        posture_type = data.get("posture_type", "unknown")
        severity = data.get("severity", "mild")
        duration_seconds = data.get("duration_seconds", 0.0)
        person_name = data.get("person_name", "")
        recommendation = data.get("recommendation", "")
        confidence = data.get("confidence", 0.0)

        current_time = time.time()

        # Detect state changes
        posture_changed = self._previous_posture_type != posture_type
        severity_changed = self._previous_severity != severity

        # Update tracking
        if posture_changed:
            self._posture_start_time = current_time
            self._previous_posture_type = posture_type
            self._previous_severity = severity

        # Calculate duration if we have a start time
        duration_minutes = 0.0
        if self._posture_start_time is not None:
            duration_minutes = (current_time - self._posture_start_time) / 60.0
        elif duration_seconds > 0:
            duration_minutes = duration_seconds / 60.0

        # Only report poor posture or significant changes
        if posture_type == "good":
            # Only report when posture improves (was poor, now good)
            if posture_changed and self._previous_posture_type not in [
                None,
                "good",
                "unknown",
            ]:
                return f"POSTURE IMPROVED: Good posture detected. Keep it up!"
            return None

        # Report poor posture detection
        if posture_changed or severity_changed:
            message_parts = [
                f"POSTURE DETECTED: {posture_type.upper()}",
                f"Severity: {severity.upper()}",
            ]

            if duration_minutes > 0:
                message_parts.append(f"Duration: {duration_minutes:.1f} minutes")

            if person_name:
                message_parts.append(f"Person: {person_name}")

            if confidence > 0:
                message_parts.append(f"Confidence: {confidence:.0%}")

            if recommendation:
                message_parts.append(f"Recommendation: {recommendation}")

            # Format for LLM to trigger PostureDetection action
            formatted_message = "\n".join(message_parts)
            formatted_message += (
                f"\n\nUse PostureDetection action with: "
                f"posture_type='{posture_type}', severity='{severity}', "
                f"duration_minutes={duration_minutes:.1f}"
            )
            if person_name:
                formatted_message += f", person_name='{person_name}'"
            if recommendation:
                formatted_message += f", recommendation='{recommendation}'"

            return formatted_message

        # Periodic updates for persistent poor posture (every 5 minutes)
        if duration_minutes > 0 and duration_minutes % 5.0 < self.poll_interval / 60.0:
            return (
                f"POSTURE PERSISTENT: {posture_type.upper()} posture detected "
                f"for {duration_minutes:.1f} minutes. Severity: {severity.upper()}. "
                f"Use PostureDetection action to provide reminder."
            )

        return None

    async def _raw_to_text(self, raw_input: Optional[str]) -> Optional[Message]:
        """
        Process raw input to generate a timestamped message.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input string to be processed.

        Returns
        -------
        Optional[Message]
            A timestamped message containing the processed input.
        """
        if raw_input is None:
            return None

        return Message(timestamp=time.time(), message=raw_input)

    async def raw_to_text(self, raw_input: Optional[str]):
        """
        Convert raw input to text and update message buffer.

        Parameters
        ----------
        raw_input : Optional[str]
            Raw input to be processed, or None if no input is available.
        """
        if raw_input is None:
            return

        message = await self._raw_to_text(raw_input)
        if message is not None:
            self.messages.append(message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """
        Return the newest message as a formatted prompt block and clear history.

        Returns
        -------
        Optional[str]
            A formatted multi-line string ready for LLM consumption,
            or None if there are no messages.
        """
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = f"""
INPUT: {self.descriptor_for_LLM}
// START
{latest_message.message}
// END
"""

        self.io_provider.add_input(
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        self.messages.clear()
        return result
