# src/inputs/plugins/person_following_status.py

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


class PersonFollowingStatusConfig(SensorConfig):
    """
    Configuration for Person Following Status Input.

    Parameters
    ----------
    person_follow_base_url : str
        Base URL for the person-following HTTP service.
    poll_interval : float
        Polling interval in seconds.
    enroll_retry_interval : float
        Interval in seconds between re-enrollment attempts when not tracking.
    """

    person_follow_base_url: str = Field(
        default="http://localhost:8080",
        description="Base URL for the person-following HTTP service",
    )
    poll_interval: float = Field(
        default=0.5,
        description="Polling interval in seconds",
    )
    enroll_retry_interval: float = Field(
        default=3.0,
        description="Interval in seconds between re-enrollment attempts when not tracking",
    )


class PersonFollowingStatus(FuserInput[PersonFollowingStatusConfig, Optional[str]]):
    """
    Input that polls the person-following Docker container for tracking status.

    This input periodically queries the /status endpoint and provides
    human-readable status information to the LLM, including:
    - Whether a person is being tracked
    - Distance information (x, z coordinates)
    - Tracking state changes

    The LLM can use this information to:
    - Announce when tracking starts/stops
    - Provide distance-based feedback
    - Ask the user to come back into view if lost
    """

    def __init__(self, config: PersonFollowingStatusConfig):
        """
        Initialize the PersonFollowingStatus input.

        Parameters
        ----------
        config : PersonFollowingStatusConfig
            Configuration for the person following status input.
        """
        super().__init__(config)

        self.io_provider = IOProvider()
        self.messages: Deque[Message] = deque(maxlen=50)

        self.base_url = config.person_follow_base_url
        self.poll_interval = config.poll_interval
        self.enroll_retry_interval = config.enroll_retry_interval
        self.status_url = f"{self.base_url}/status"
        self.enroll_url = f"{self.base_url}/enroll"

        self.descriptor_for_LLM = "Person Following Status"

        # Track previous state for change detection
        self._previous_is_tracked: Optional[bool] = None
        self._lost_tracking_time: Optional[float] = None
        self._lost_tracking_announced: bool = False
        self._last_enroll_attempt: float = 0.0
        self._has_ever_tracked: bool = False

        logging.info(
            f"PersonFollowingStatus initialized, polling {self.status_url} "
            f"every {self.poll_interval}s, re-enroll every {self.enroll_retry_interval}s when not tracking"
        )

    async def _poll(self) -> Optional[str]:
        """
        Poll the person-following status endpoint.

        Also periodically calls /enroll when status is INACTIVE (no one enrolled yet).
        Does NOT re-enroll when status is SEARCHING (person enrolled but temporarily lost).

        Returns
        -------
        Optional[str]
            Formatted status message if there's a meaningful update, None otherwise.
        """
        await asyncio.sleep(self.poll_interval)

        try:
            async with aiohttp.ClientSession() as session:
                # First, get current status
                async with session.get(
                    self.status_url,
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as response:
                    if response.status != 200:
                        return None

                    data = await response.json()
                    is_tracked = data.get("is_tracked", False)
                    status = data.get("status", "UNKNOWN")
                    target_track_id = data.get("target_track_id")

                    # If tracking, remember we've successfully tracked
                    if is_tracked:
                        self._has_ever_tracked = True

                    # Only retry enrollment if INACTIVE (no one enrolled yet)
                    # Do NOT re-enroll if SEARCHING (person enrolled but temporarily out of frame)
                    if status == "INACTIVE" and target_track_id is None:
                        current_time = time.time()
                        time_since_last_enroll = (
                            current_time - self._last_enroll_attempt
                        )

                        if time_since_last_enroll >= self.enroll_retry_interval:
                            self._last_enroll_attempt = current_time
                            logging.info(
                                "PersonFollowingStatus: Status INACTIVE, attempting enrollment"
                            )
                            await self._try_enroll(session)

                    return self._format_status(data)

        except aiohttp.ClientError as e:
            logging.debug(f"PersonFollowingStatus: Poll failed: {e}")
            return None
        except Exception as e:
            logging.warning(f"PersonFollowingStatus: Unexpected error: {e}")
            return None

    async def _try_enroll(self, session: aiohttp.ClientSession) -> None:
        """
        Attempt to enroll a person for tracking.

        Parameters
        ----------
        session : aiohttp.ClientSession
            The HTTP session to use for the request.
        """
        try:
            async with session.post(
                self.enroll_url,
                timeout=aiohttp.ClientTimeout(total=3),
            ) as response:
                if response.status == 200:
                    logging.info("PersonFollowingStatus: Re-enrollment request sent")
                else:
                    logging.debug(
                        f"PersonFollowingStatus: Enroll returned status {response.status}"
                    )
        except Exception as e:
            logging.debug(f"PersonFollowingStatus: Enroll request failed: {e}")
            return None

    def _format_status(self, data: dict) -> Optional[str]:
        """
        Format the status data into a human-readable message for the LLM.

        Only returns a message when there's a meaningful state change or
        periodically during tracking.

        Status values from the Docker container:
        - INACTIVE: No one enrolled, waiting for enrollment
        - SEARCHING: Person enrolled but temporarily out of frame
        - TRACKING_ACTIVE: Actively tracking a person

        Parameters
        ----------
        data : dict
            Raw status data from the /status endpoint.

        Returns
        -------
        Optional[str]
            Formatted status message or None if no update needed.
        """
        is_tracked = data.get("is_tracked", False)
        x = data.get("x", 0.0)
        z = data.get("z", 0.0)
        status = data.get("status", "UNKNOWN")
        target_track_id = data.get("target_track_id")

        current_time = time.time()

        # Detect state changes
        tracking_just_started = (
            self._previous_is_tracked is not True and is_tracked is True
        )
        tracking_just_lost = self._previous_is_tracked is True and is_tracked is False

        self._previous_is_tracked = is_tracked

        if tracking_just_started:
            # Person was acquired - always report this
            self._lost_tracking_time = None
            self._lost_tracking_announced = False
            return f"TRACKING STARTED: Person detected and now following. Distance: {z:.1f}m ahead, {x:.1f}m to the side."

        if tracking_just_lost:
            # Person was lost - start timer but don't announce immediately
            self._lost_tracking_time = current_time
            self._lost_tracking_announced = False
            return None

        if not is_tracked:
            # Check if we should announce based on status
            if (
                self._lost_tracking_time is not None
                and not self._lost_tracking_announced
                and (current_time - self._lost_tracking_time) > 2.0
            ):
                self._lost_tracking_announced = True
                if status == "SEARCHING" and target_track_id is not None:
                    # Person was enrolled but went out of frame - they'll be re-acquired automatically
                    return "SEARCHING: Person went out of view. Looking for them to return."
                else:
                    # INACTIVE - no one enrolled
                    return "WAITING: No person detected yet. Please stand in front of me so I can see you."

            # Don't spam "not tracking" messages
            return None

        # Currently tracking - provide occasional updates
        # Only report significant distance changes or periodically
        return f"TRACKING: Following person at {z:.1f}m ahead, {x:.1f}m to the side. Status: {status}"

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
