import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .singleton import singleton


@dataclass
class FallEvent:
    """
    Represents a fall detection event.

    Attributes
    ----------
    timestamp : float
        Unix timestamp when the fall was detected.
    severity : str
        Severity level of the fall.
    location : str
        Location where fall occurred.
    person_name : str
        Name of person (if known).
    confidence : float
        Detection confidence score.
    responded : bool
        Whether the robot has responded to this fall.
    """

    timestamp: float
    severity: str
    location: str = ""
    person_name: str = ""
    confidence: float = 0.0
    responded: bool = False


@dataclass
class PostureRecord:
    """
    Represents a posture detection record.

    Attributes
    ----------
    timestamp : float
        Unix timestamp when posture was detected.
    posture_type : str
        Type of posture detected.
    severity : str
        Severity of posture issue.
    duration_minutes : float
        Duration in this posture.
    person_name : str
        Name of person (if known).
    """

    timestamp: float
    posture_type: str
    severity: str
    duration_minutes: float = 0.0
    person_name: str = ""


@singleton
class HealthDetectionProvider:
    """
    Singleton provider for managing health detection data and statistics.

    This provider maintains:
    - History of fall events
    - Posture tracking records
    - Health statistics and patterns
    - Alert thresholds and configurations
    """

    def __init__(self):
        """
        Initialize the HealthDetectionProvider.
        """
        self.fall_events: List[FallEvent] = []
        self.posture_records: List[PostureRecord] = []
        self.max_fall_history: int = 100
        self.max_posture_history: int = 500
        self.fall_alert_threshold: float = 0.7  # Confidence threshold for alerts
        self.posture_reminder_interval: float = 30.0  # Minutes between reminders

    def record_fall(
        self,
        severity: str,
        location: str = "",
        person_name: str = "",
        confidence: float = 0.0,
    ) -> FallEvent:
        """
        Record a fall detection event.

        Parameters
        ----------
        severity : str
            Severity level of the fall.
        location : str
            Location where fall occurred.
        person_name : str
            Name of person (if known).
        confidence : float
            Detection confidence score.

        Returns
        -------
        FallEvent
            The recorded fall event.
        """
        fall_event = FallEvent(
            timestamp=time.time(),
            severity=severity,
            location=location,
            person_name=person_name,
            confidence=confidence,
            responded=False,
        )

        self.fall_events.append(fall_event)

        # Keep only recent history
        if len(self.fall_events) > self.max_fall_history:
            self.fall_events = self.fall_events[-self.max_fall_history :]

        logging.info(
            f"Recorded fall event: {severity} severity, confidence={confidence:.2f}, "
            f"location={location}, person={person_name}"
        )

        return fall_event

    def record_posture(
        self,
        posture_type: str,
        severity: str,
        duration_minutes: float = 0.0,
        person_name: str = "",
    ) -> PostureRecord:
        """
        Record a posture detection.

        Parameters
        ----------
        posture_type : str
            Type of posture detected.
        severity : str
            Severity of posture issue.
        duration_minutes : float
            Duration in this posture.
        person_name : str
            Name of person (if known).

        Returns
        -------
        PostureRecord
            The recorded posture record.
        """
        posture_record = PostureRecord(
            timestamp=time.time(),
            posture_type=posture_type,
            severity=severity,
            duration_minutes=duration_minutes,
            person_name=person_name,
        )

        self.posture_records.append(posture_record)

        # Keep only recent history
        if len(self.posture_records) > self.max_posture_history:
            self.posture_records = self.posture_records[-self.max_posture_history :]

        return posture_record

    def get_recent_falls(self, hours: float = 24.0) -> List[FallEvent]:
        """
        Get fall events within the specified time window.

        Parameters
        ----------
        hours : float
            Number of hours to look back. Default is 24.

        Returns
        -------
        List[FallEvent]
            List of fall events in the time window.
        """
        cutoff_time = time.time() - (hours * 3600)
        return [event for event in self.fall_events if event.timestamp >= cutoff_time]

    def get_posture_statistics(
        self, person_name: Optional[str] = None, hours: float = 24.0
    ) -> Dict:
        """
        Get posture statistics for the specified time window.

        Parameters
        ----------
        person_name : Optional[str]
            Filter by person name. If None, includes all people.
        hours : float
            Number of hours to look back. Default is 24.

        Returns
        -------
        Dict
            Statistics including average posture quality, time in poor posture, etc.
        """
        cutoff_time = time.time() - (hours * 3600)
        relevant_records = [
            r
            for r in self.posture_records
            if r.timestamp >= cutoff_time
            and (person_name is None or r.person_name == person_name)
        ]

        if not relevant_records:
            return {
                "total_records": 0,
                "average_duration_minutes": 0.0,
                "poor_posture_percentage": 0.0,
            }

        total_duration = sum(r.duration_minutes for r in relevant_records)
        poor_posture_count = sum(
            1
            for r in relevant_records
            if r.posture_type not in ["good", "laying"]
        )

        return {
            "total_records": len(relevant_records),
            "average_duration_minutes": total_duration / len(relevant_records)
            if relevant_records
            else 0.0,
            "poor_posture_percentage": (poor_posture_count / len(relevant_records) * 100)
            if relevant_records
            else 0.0,
            "most_common_posture": max(
                set(r.posture_type for r in relevant_records),
                key=lambda x: sum(1 for r in relevant_records if r.posture_type == x),
            )
            if relevant_records
            else "unknown",
        }

    def should_alert_fall(self, confidence: float) -> bool:
        """
        Determine if a fall should trigger an alert based on confidence threshold.

        Parameters
        ----------
        confidence : float
            Detection confidence score.

        Returns
        -------
        bool
            True if alert should be triggered.
        """
        return confidence >= self.fall_alert_threshold

    def should_remind_posture(
        self, person_name: str, last_reminder_time: Optional[float] = None
    ) -> bool:
        """
        Determine if a posture reminder should be sent.

        Parameters
        ----------
        person_name : str
            Name of the person.
        last_reminder_time : Optional[float]
            Timestamp of last reminder for this person.

        Returns
        -------
        bool
            True if reminder should be sent.
        """
        if last_reminder_time is None or last_reminder_time == 0:
            return True

        time_since_reminder = (time.time() - last_reminder_time) / 60.0  # minutes
        return time_since_reminder >= self.posture_reminder_interval

    def mark_fall_responded(self, fall_event: FallEvent) -> None:
        """
        Mark a fall event as having been responded to.

        Parameters
        ----------
        fall_event : FallEvent
            The fall event to mark.
        """
        fall_event.responded = True
        logging.info(f"Marked fall event as responded: {fall_event.timestamp}")

