"""Zenoh-based Unitree Go2 odometry provider.

Drop-in replacement for ``UnitreeGo2OdomProvider``: subscribes to a Zenoh
keyexpression instead of a local CycloneDDS topic. Same ``position`` dict
surface — feeds ``OdomProviderBase.data_queue`` so the base class's
``process_odom()`` thread does the position/yaw decode unchanged.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading

from runtime.logging import LoggingConfig, get_logging_config, setup_logging
from zenoh_msgs import open_zenoh_session
from zenoh_msgs.idl.geometry_msgs import PoseStamped
from zenoh_msgs.idl.nav_msgs import Odometry

from .odom_provider_base import OdomProviderBase, RobotState
from .singleton import singleton

# Both message types are handled by ``OdomProviderBase.process_odom`` —
# it discriminates with ``hasattr(pose_data.pose, "pose")``.
_DECODERS = {
    "geometry_msgs/msg/PoseStamped": PoseStamped,
    "nav_msgs/msg/Odometry": Odometry,
}


def _go2_odom_zenoh_processor(
    topic: str,
    schema: str,
    data_queue: mp.Queue,
    logging_config: LoggingConfig | None = None,
) -> None:
    """Subprocess: subscribe Zenoh ``topic`` and feed pose messages to queue."""
    setup_logging("go2_odom_zenoh_processor", logging_config=logging_config)

    decoder = _DECODERS.get(schema)
    if decoder is None:
        logging.error("unknown odom schema %r — supported: %s", schema, list(_DECODERS))
        return

    def on_sample(sample) -> None:  # type: ignore[no-untyped-def]
        try:
            msg = decoder.deserialize(sample.payload.to_bytes())
        except Exception:
            logging.exception("failed to decode %s on %s", schema, topic)
            return
        data_queue.put(msg)

    try:
        session = open_zenoh_session()
    except Exception:
        logging.exception("failed to open Zenoh session for go2 odom")
        return

    session.declare_subscriber(topic, on_sample)
    logging.info("Zenoh odom subscriber on '%s' (%s) is live", topic, schema)

    # Block forever; the parent process tears the subprocess down.
    threading.Event().wait()


@singleton
class UnitreeGo2OdomZenohProvider(OdomProviderBase):
    """Reads Go2 odometry from a Zenoh keyexpression.

    Default ``topic="utlidar/robot_pose"`` (PoseStamped) — published by the
    Go2 firmware on the real robot, and by ``go2_remapping_node`` in the
    OM1-sim launch (which republishes ``/odom`` with a charging-aware
    z-offset so body_attitude classifies correctly).

    For setups that don't run ``go2_remapping_node``, point at ``odom``
    with ``schema="nav_msgs/msg/Odometry"``. Body_attitude will stay
    ``None`` since the raw odom z is ~0; consumers should treat that as
    "no info, proceed".
    """

    def __init__(
        self,
        topic: str = "utlidar/robot_pose",
        schema: str = "geometry_msgs/msg/PoseStamped",
    ) -> None:
        super().__init__()
        self.topic = topic
        self.schema = schema
        self.start()

    def start(self) -> None:
        """Start the background odom subscriber thread (idempotent)."""
        if self._odom_reader_thread and self._odom_reader_thread.is_alive():
            logging.warning("Go2 Zenoh Odom Provider is already running.")
            return

        logging.info(
            "Starting Unitree Go2 Zenoh Odom Provider on '%s' (%s)",
            self.topic,
            self.schema,
        )

        self._odom_reader_thread = mp.Process(
            target=_go2_odom_zenoh_processor,
            args=(self.topic, self.schema, self.data_queue, get_logging_config()),
            daemon=True,
        )
        self._odom_reader_thread.start()

        if self._odom_processor_thread and self._odom_processor_thread.is_alive():
            return
        self._odom_processor_thread = threading.Thread(target=self.process_odom, daemon=True)
        self._odom_processor_thread.start()

    def _update_body_state(self, pose) -> None:  # type: ignore[no-untyped-def]
        # Mirrors ``UnitreeGo2OdomProvider._update_body_state``: classify
        # sitting vs standing from body height (z, m → cm).
        self.body_height_cm = round(pose.position.z * 100.0)
        if self.body_height_cm > 24:
            self.body_attitude = RobotState.STANDING
        elif self.body_height_cm > 3:
            self.body_attitude = RobotState.SITTING
