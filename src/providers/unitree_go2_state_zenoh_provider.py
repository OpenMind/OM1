"""Zenoh-based Unitree Go2 SportModeState provider.

Drop-in replacement for ``UnitreeGo2StateProvider`` over Zenoh.

``/sportmodestate`` is only published by the real Go2 firmware; the
sim launches don't synthesize it. When running against a sim, this
provider's fields stay at their defaults — consuming code should treat
``state_code is None`` and ``action_progress == 0`` as "no info, proceed".
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from queue import Empty, Full
from typing import Optional

from runtime.logging import LoggingConfig, get_logging_config, setup_logging
from zenoh_msgs import open_zenoh_session
from zenoh_msgs.idl.unitree_go import SportModeState

from .singleton import singleton

# Same code → state-name table the legacy provider exposes.
_STATE_MACHINE_CODES = {
    100: "Agile",
    1001: "Damping",
    1002: "Standing Lock",
    1004: "Crouch",
    1006: "Greeting/Stretching/Dancing/Bowing/Heart Shape/Happy",
    1007: "Sit",
    1008: "Front Jump",
    1009: "Lunge",
    1013: "Balance Standing",
    1015: "Regular Walking",
    1016: "Regular Running",
    1017: "Regular Endurance",
    1091: "Strike a Pose",
    2006: "Crouch",
    2007: "Dodge",
    2008: "Bound Run",
    2009: "Jump Run",
    2010: "Classic",
    2011: "Handstand",
    2012: "Front Flip",
    2013: "Back Flip",
    2014: "Left Flip",
    2016: "Cross Step",
    2017: "Upright",
    2019: "Towing",
}


def _state_zenoh_processor(
    topic: str,
    data_queue: mp.Queue,
    control_queue: mp.Queue,
    logging_config: LoggingConfig | None = None,
) -> None:
    setup_logging("unitree_go2_state_zenoh_processor", logging_config=logging_config)

    def on_sample(sample) -> None:  # type: ignore[no-untyped-def]
        try:
            msg = SportModeState.deserialize(sample.payload.to_bytes())
        except Exception:
            logging.exception("failed to decode SportModeState on %s", topic)
            return
        data = {
            "go2_sport_mode_state_msg": msg,
            "go2_state_code": msg.error_code,
            "go2_state": _STATE_MACHINE_CODES.get(msg.error_code, "unknown"),
            "go2_action_progress": msg.progress,
        }
        try:
            data_queue.put_nowait(data)
        except Full:
            try:
                data_queue.get_nowait()
                data_queue.put_nowait(data)
            except Empty:
                pass

    try:
        session = open_zenoh_session()
        session.declare_subscriber(topic, on_sample)
        logging.info("Zenoh sportmodestate subscriber on '%s' is live", topic)
    except Exception:
        logging.exception("failed to open Zenoh session for sportmodestate")
        return

    while True:
        try:
            if control_queue.get_nowait() == "STOP":
                break
        except Empty:
            pass
        time.sleep(0.1)


@singleton
class UnitreeGo2StateZenohProvider:
    """Drop-in for ``UnitreeGo2StateProvider`` over Zenoh."""

    def __init__(self, topic: str = "sportmodestate") -> None:
        self.topic = topic
        self.data_queue: mp.Queue = mp.Queue(maxsize=5)
        self.control_queue: mp.Queue = mp.Queue()

        self._reader_proc: Optional[mp.Process] = None
        self._processor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.go2_sport_mode_state_msg = None
        self.go2_state: Optional[str] = None
        self.go2_state_code: Optional[int] = None
        self.go2_action_progress: int = 0

        self.start()

    def start(self) -> None:
        """Start the reader process and processor thread (idempotent)."""
        if not self._reader_proc or not self._reader_proc.is_alive():
            self._reader_proc = mp.Process(
                target=_state_zenoh_processor,
                args=(self.topic, self.data_queue, self.control_queue, get_logging_config()),
                daemon=True,
            )
            self._reader_proc.start()
            logging.info("Unitree Go2 Zenoh state reader started.")

        if not self._processor_thread or not self._processor_thread.is_alive():
            self._processor_thread = threading.Thread(target=self._processor_loop, daemon=True)
            self._processor_thread.start()
            logging.info("Unitree Go2 Zenoh state processor started.")

    def stop(self) -> None:
        """Stop the reader process and processor thread."""
        self._stop_event.set()
        if self._reader_proc:
            self.control_queue.put("STOP")
            self._reader_proc.terminate()
            self._reader_proc.join(timeout=2)
        if self._processor_thread:
            self._processor_thread.join(timeout=2)

    def _processor_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                data = self.data_queue.get(timeout=0.5)
            except Empty:
                continue
            self.go2_sport_mode_state_msg = data.get("go2_sport_mode_state_msg")
            self.go2_state = data.get("go2_state")
            self.go2_state_code = data.get("go2_state_code")
            self.go2_action_progress = data.get("go2_action_progress")

    @property
    def state(self) -> Optional[str]:
        """Latest decoded high-level state string (e.g. ``"jointLock"``)."""
        return self.go2_state

    @property
    def state_code(self) -> Optional[int]:
        """Latest numeric state code from the Go2 state machine."""
        return self.go2_state_code

    @property
    def action_progress(self) -> int:
        """Progress percentage (0–100) of the currently-executing motion."""
        return self.go2_action_progress
