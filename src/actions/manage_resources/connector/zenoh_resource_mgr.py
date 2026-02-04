import logging
from typing import Any, Dict, Optional

try:
    import zenoh
except ImportError:
    zenoh = None
    logging.error("Zenoh library not found. Please install it via 'pip install zenoh'.")

from actions.base import ActionConnector
from actions.manage_resources.interface import ManageResourcesInterface
from inputs.base import Message


class ZenohResourceManager(ActionConnector):
    """Manages resources using Zenoh."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.zenoh_config_path: Optional[str] = config.get("zenoh_config_path")
        self.zenoh_session: Optional[object] = (
            None  # Use object as fallback for pyright
        )
        self.qos_profiles: Dict[str, Dict[str, Any]] = self._define_qos_profiles()
        # Create interface instance for actual execution logic
        self.interface = ManageResourcesInterface(config)

    def _define_qos_profiles(self) -> Dict[str, Dict[str, Any]]:
        profiles = {
            "realtime": {
                "zenoh_priority": "realtime",
                "zenoh_congestion_control": "drop",
                "zenoh_reliability": "best_effort",
            },
            "high": {
                "zenoh_priority": "high",
                "zenoh_congestion_control": "block",
                "zenoh_reliability": "reliable",
            },
            "medium": {
                "zenoh_priority": "medium",
                "zenoh_congestion_control": "block",
                "zenoh_reliability": "reliable",
            },
            "low": {
                "zenoh_priority": "low",
                "zenoh_congestion_control": "drop",
                "zenoh_reliability": "reliable",
            },
        }
        return profiles

    async def _initialize_zenoh_session(self) -> bool:
        if self.zenoh_session is not None:
            logging.warning("[ZenohResourceManager] Session already initialized.")
            return True

        if zenoh is None:
            logging.error("[ZenohResourceManager] Zenoh library not available.")
            return False

        try:
            conf = zenoh.Config.default()  # type: ignore[attr-defined]
        except AttributeError:
            logging.error(
                "[ZenohResourceManager] Cannot access zenoh.Config.default()."
            )
            return False

        if self.zenoh_config_path:
            try:
                conf = zenoh.Config.from_file(self.zenoh_config_path)  # type: ignore[attr-defined]
                logging.info(
                    f"[ZenohResourceManager] Loaded Zenoh config from {self.zenoh_config_path}"
                )
            except FileNotFoundError:
                logging.error(
                    f"[ZenohResourceManager] Config file not found: {self.zenoh_config_path}"
                )
                try:
                    conf = zenoh.Config.default()  # type: ignore[attr-defined]
                except AttributeError:
                    logging.error(
                        "[ZenohResourceManager] Cannot access zenoh.Config.default()."
                    )
                    return False
            except Exception as e:
                logging.error(f"[ZenohResourceManager] Error loading config: {e}")
                try:
                    conf = zenoh.Config.default()  # type: ignore[attr-defined]
                except AttributeError:
                    logging.error(
                        "[ZenohResourceManager] Cannot access zenoh.Config.default()."
                    )
                    return False

        logging.info("[ZenohResourceManager] Opening Zenoh session...")
        try:
            self.zenoh_session = zenoh.open(conf)  # type: ignore[attr-defined]
            logging.info("[ZenohResourceManager] Zenoh session opened successfully.")
            return True
        except AttributeError:
            logging.error("[ZenohResourceManager] Cannot access zenoh.open().")
            return False
        except Exception as e:
            logging.error(f"[ZenohResourceManager] Failed to open Zenoh session: {e}")
            return False

    async def adjust_network_qos(
        self,
        target: str,
        priority: str,
        _reliability: str = "reliable",
        _durability: str = "volatile",
    ) -> bool:
        """Adjusts network QoS for a target resource."""
        if self.zenoh_session is None:
            success = await self._initialize_zenoh_session()
            if not success or self.zenoh_session is None:
                logging.error(
                    "[ZenohResourceManager] Cannot adjust QoS, Zenoh session is not open."
                )
                return False

        if priority not in self.qos_profiles:
            logging.error(
                f"[ZenohResourceManager] Invalid priority level: {priority}. Valid options: {list(self.qos_profiles.keys())}"
            )
            return False

        qos_profile = self.qos_profiles[priority]
        logging.info(
            f"[ZenohResourceManager] Attempting to apply QoS profile for '{target}' with priority '{priority}': {qos_profile}"
        )

        print(
            f"--- SIMULATION: Would attempt to adjust QoS for key '{target}' using profile {qos_profile} ---"
        )
        print(
            "--- NOTE: Actual dynamic QoS adjustment requires complex Zenoh admin space interaction or publisher re-creation. ---"
        )
        return True

    async def adjust_cpu_priority(self, task_name: str, priority: str) -> bool:
        """Adjusts CPU priority for a task."""
        valid_priorities = ["critical", "high", "normal", "low"]
        if priority not in valid_priorities:
            logging.error(
                f"[ManageResources] Invalid CPU priority level: {priority}. Valid options: {valid_priorities}"
            )
            return False

        logging.info(
            f"[ManageResources] Attempting to adjust CPU priority for task '{task_name}' to '{priority}'."
        )
        print(
            f"--- SIMULATION: Would attempt to adjust CPU priority for {task_name} to {priority} (OS-specific) ---"
        )
        return True

    async def execute(self, message: Message) -> bool:
        """Execute resource management command via interface."""
        return await self.interface.execute(message)
