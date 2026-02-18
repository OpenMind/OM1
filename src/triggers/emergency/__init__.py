"""
Emergency Triggers

Multi-modal triggers for emergency detection:
- Voice keyword detection
- IMU fall detection
- Physical button press
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from actions.emergency_call.interface import (
    EmergencyCallInput,
    EmergencyLevel,
    EmergencyTriggerType,
)


@dataclass
class TriggerResult:
    """Result from trigger detection."""

    triggered: bool
    trigger_type: EmergencyTriggerType
    confidence: float
    data: Optional[dict] = None


class EmergencyTrigger(ABC):
    """Base class for emergency triggers."""

    @abstractmethod
    async def detect(self, data: Optional[dict] = None) -> TriggerResult:
        """Detect emergency condition."""
        pass

    @abstractmethod
    async def start_listening(self, callback: Callable[[TriggerResult], None]) -> None:
        """Start continuous listening/monitoring."""
        pass

    @abstractmethod
    async def stop_listening(self) -> None:
        """Stop listening."""
        pass


class VoiceKeywordTrigger(EmergencyTrigger):
    """
    Trigger based on voice keywords.

    Detects emergency keywords like "help", "emergency", "fall", etc.
    """

    EMERGENCY_KEYWORDS = ["help", "emergency", "fall", "fell", "hurt", "pain", "aid"]

    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self._listening = False
        self._callback: Optional[Callable[[TriggerResult], None]] = None

    async def detect(self, data: Optional[dict] = None) -> TriggerResult:
        """Detect emergency keywords in transcript."""
        if not data or "transcript" not in data:
            return TriggerResult(False, EmergencyTriggerType.VOICE_KEYWORD, 0.0)

        transcript = data["transcript"].lower()

        for keyword in self.EMERGENCY_KEYWORDS:
            if keyword in transcript:
                confidence = 1.0  # In real implementation, use ASR confidence
                return TriggerResult(
                    True,
                    EmergencyTriggerType.VOICE_KEYWORD,
                    confidence,
                    {"keyword": keyword, "transcript": transcript},
                )

        return TriggerResult(False, EmergencyTriggerType.VOICE_KEYWORD, 0.0)

    async def start_listening(self, callback: Callable[[TriggerResult], None]) -> None:
        """Start listening for voice keywords."""
        self._listening = True
        self._callback = callback
        logging.info("Voice keyword trigger started listening")

    async def stop_listening(self) -> None:
        """Stop listening."""
        self._listening = False
        self._callback = None
        logging.info("Voice keyword trigger stopped listening")


class FallDetectionTrigger(EmergencyTrigger):
    """
    Trigger based on IMU fall detection.

    Uses accelerometer/gyroscope data to detect falls.
    """

    FALL_ACCEL_THRESHOLD = 3.0  # g-force threshold
    IMPACT_THRESHOLD = 2.5  # g-force for impact detection
    INACTIVITY_TIMEOUT = 5.0  # seconds of inactivity after fall

    def __init__(
        self,
        accel_threshold: float = 3.0,
        impact_threshold: float = 2.5,
        inactivity_timeout: float = 5.0,
    ):
        self.accel_threshold = accel_threshold
        self.impact_threshold = impact_threshold
        self.inactivity_timeout = inactivity_timeout
        self._listening = False
        self._callback: Optional[Callable[[TriggerResult], None]] = None

        # State for fall detection
        self._free_fall_start: Optional[float] = None
        self._last_activity: float = time.time()
        self._imu_buffer: List[dict] = []

    async def detect(self, data: Optional[dict] = None) -> TriggerResult:
        """Detect fall from IMU data."""
        if not data or "accel" not in data:
            return TriggerResult(False, EmergencyTriggerType.FALL_DETECTION, 0.0)

        accel = np.array(data["accel"])  # [x, y, z] in g
        accel_magnitude = np.linalg.norm(accel)

        # Store in buffer (keep last 100 samples ~ 1 second at 100Hz)
        self._imu_buffer.append(
            {"timestamp": time.time(), "accel": accel_magnitude, "data": data}
        )
        if len(self._imu_buffer) > 100:
            self._imu_buffer.pop(0)

        # Fall detection logic:
        # 1. Free fall (low g) followed by high impact
        # 2. Inactivity after impact

        current_time = time.time()

        if accel_magnitude < 0.5:  # Free fall detection
            if self._free_fall_start is None:
                self._free_fall_start = current_time
                logging.debug("Potential free fall detected")
        elif accel_magnitude > self.impact_threshold:
            if self._free_fall_start is not None:
                # Impact after free fall = likely fall
                fall_duration = current_time - self._free_fall_start
                self._free_fall_start = None

                confidence = min(1.0, fall_duration / 1.0)  # More confident with longer free fall

                logging.warning(f"Fall detected! Impact: {accel_magnitude:.2f}g, Duration: {fall_duration:.2f}s")

                return TriggerResult(
                    True,
                    EmergencyTriggerType.FALL_DETECTION,
                    confidence,
                    {
                        "impact_g": float(accel_magnitude),
                        "fall_duration": fall_duration,
                        "timestamp": current_time,
                    },
                )

        # Check for inactivity after potential fall
        if accel_magnitude > 0.1:  # Some movement
            self._last_activity = current_time
        else:
            inactivity_duration = current_time - self._last_activity
            if inactivity_duration > self.inactivity_timeout and len(self._imu_buffer) > 50:
                # Inactivity + recent motion = might need help
                return TriggerResult(
                    True,
                    EmergencyTriggerType.FALL_DETECTION,
                    0.6,  # Lower confidence
                    {
                        "inactivity_duration": inactivity_duration,
                        "reason": "inactivity_after_motion",
                    },
                )

        return TriggerResult(False, EmergencyTriggerType.FALL_DETECTION, 0.0)

    async def start_listening(self, callback: Callable[[TriggerResult], None]) -> None:
        """Start monitoring IMU for falls."""
        self._listening = True
        self._callback = callback
        logging.info("Fall detection trigger started monitoring")

    async def stop_listening(self) -> None:
        """Stop monitoring."""
        self._listening = False
        self._callback = None
        logging.info("Fall detection trigger stopped monitoring")


class PhysicalButtonTrigger(EmergencyTrigger):
    """
    Trigger based on physical button presses.

    Supports single press, double press, and long press patterns.
    """

    DOUBLE_PRESS_WINDOW = 1.0  # seconds between presses
    LONG_PRESS_DURATION = 3.0  # seconds for long press

    def __init__(
        self,
        double_press_window: float = 1.0,
        long_press_duration: float = 3.0,
    ):
        self.double_press_window = double_press_window
        self.long_press_duration = long_press_duration
        self._listening = False
        self._callback: Optional[Callable[[TriggerResult], None]] = None
        self._press_times: List[float] = []
        self._press_start: Optional[float] = None

    async def detect(self, data: Optional[dict] = None) -> TriggerResult:
        """Detect button press patterns."""
        if not data:
            return TriggerResult(False, EmergencyTriggerType.PHYSICAL_BUTTON, 0.0)

        event = data.get("event")  # "press", "release"
        current_time = time.time()

        if event == "press":
            self._press_start = current_time
            self._press_times.append(current_time)

            # Clean old press times
            self._press_times = [t for t in self._press_times if current_time - t < self.DOUBLE_PRESS_WINDOW]

            # Check for double press
            if len(self._press_times) >= 2:
                logging.warning("Double button press detected - emergency triggered!")
                return TriggerResult(
                    True,
                    EmergencyTriggerType.PHYSICAL_BUTTON,
                    1.0,
                    {"pattern": "double_press", "press_count": len(self._press_times)},
                )

        elif event == "release" and self._press_start:
            press_duration = current_time - self._press_start
            self._press_start = None

            # Check for long press
            if press_duration >= self.LONG_PRESS_DURATION:
                logging.warning(f"Long button press detected ({press_duration:.1f}s) - emergency triggered!")
                return TriggerResult(
                    True,
                    EmergencyTriggerType.PHYSICAL_BUTTON,
                    1.0,
                    {"pattern": "long_press", "duration": press_duration},
                )

        return TriggerResult(False, EmergencyTriggerType.PHYSICAL_BUTTON, 0.0)

    async def start_listening(self, callback: Callable[[TriggerResult], None]) -> None:
        """Start listening for button events."""
        self._listening = True
        self._callback = callback
        logging.info("Physical button trigger started listening")

    async def stop_listening(self) -> None:
        """Stop listening."""
        self._listening = False
        self._callback = None
        logging.info("Physical button trigger stopped listening")


class EmergencyTriggerManager:
    """
    Manager for all emergency triggers.

    Coordinates multiple trigger sources and aggregates results.
    """

    def __init__(self):
        self.triggers: List[EmergencyTrigger] = [
            VoiceKeywordTrigger(),
            FallDetectionTrigger(),
            PhysicalButtonTrigger(),
        ]
        self._callbacks: List[Callable[[TriggerResult, EmergencyCallInput], None]] = []
        self._active = False

    def register_callback(
        self, callback: Callable[[TriggerResult, EmergencyCallInput], None]
    ) -> None:
        """Register callback for trigger events."""
        self._callbacks.append(callback)

    async def _on_trigger(self, result: TriggerResult) -> None:
        """Handle trigger event."""
        if not result.triggered:
            return

        # Create emergency input from trigger
        level = self._determine_emergency_level(result)

        emergency_input = EmergencyCallInput(
            trigger_type=result.trigger_type,
            emergency_level=level,
            location="unknown",  # Should be determined from context
            user_message=f"Emergency triggered by {result.trigger_type.value}",
            sensor_data=result.data,
            timestamp=time.time(),
        )

        for callback in self._callbacks:
            try:
                await callback(result, emergency_input)
            except Exception as e:
                logging.error(f"Trigger callback failed: {e}")

    def _determine_emergency_level(self, result: TriggerResult) -> EmergencyLevel:
        """Determine emergency level from trigger result."""
        if result.trigger_type == EmergencyTriggerType.VOICE_KEYWORD:
            return EmergencyLevel.MEDIUM
        elif result.trigger_type == EmergencyTriggerType.FALL_DETECTION:
            return EmergencyLevel.HIGH
        elif result.trigger_type == EmergencyTriggerType.PHYSICAL_BUTTON:
            return EmergencyLevel.CRITICAL
        elif result.trigger_type == EmergencyTriggerType.HEART_RATE_ALERT:
            return EmergencyLevel.HIGH
        else:
            return EmergencyLevel.MEDIUM

    async def start(self) -> None:
        """Start all triggers."""
        self._active = True
        logging.info("Emergency trigger manager started")

        for trigger in self.triggers:
            await trigger.start_listening(self._on_trigger)

    async def stop(self) -> None:
        """Stop all triggers."""
        self._active = False

        for trigger in self.triggers:
            await trigger.stop_listening()

        logging.info("Emergency trigger manager stopped")
