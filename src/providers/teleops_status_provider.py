import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import requests

from .singleton import singleton


@dataclass
class BatteryStatus:
    """
    Data class to represent the battery status of a teleops system.
    """

    battery_level: float
    temperature: float
    voltage: float
    timestamp: str
    charging_status: bool = False

    def to_dict(self) -> dict:
        """
        Convert the BatteryStatus object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the BatteryStatus object.
        """
        return {
            "battery_level": self.battery_level,
            "charging_status": self.charging_status,
            "temperature": self.temperature,
            "voltage": self.voltage,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BatteryStatus":
        """
        Populate the BatteryStatus object from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing battery status information.
        """
        return cls(
            battery_level=data.get("battery_level", 0.0),
            charging_status=data.get("charging_status", False),
            temperature=data.get("temperature", 0.0),
            voltage=data.get("voltage", 0.0),
            timestamp=data.get("timestamp", str(time.time())),
        )


@dataclass
class CommandStatus:
    """
    Data class to represent the command status of a teleops system.
    """

    vx: float
    vy: float
    vyaw: float
    timestamp: str

    def to_dict(self) -> dict:
        """
        Convert the CommandStatus object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the CommandStatus object.
        """
        return {
            "vx": self.vx,
            "vy": self.vy,
            "vyaw": self.vyaw,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CommandStatus":
        """
        Populate the CommandStatus object from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing command status information.
        """
        return cls(
            vx=data.get("vx", 0.0),
            vy=data.get("vy", 0.0),
            vyaw=data.get("vyaw", 0.0),
            timestamp=data.get("timestamp", time.time()),
        )


class ActionType(Enum):
    """
    Enum for action type.
    """

    AI = "AI"
    TELEOPS = "TELEOPS"
    CONTROLLER = "CONTROLLER"


@dataclass
class ActionStatus:
    """
    Data class to represent the action status of a robot action system.
    """

    action: ActionType
    timestamp: float

    def to_dict(self) -> dict:
        """
        Convert the Action object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the Action object.
        """
        return {
            "action": self.action.value,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionStatus":
        """
        Populate the ActionStatus object from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing action status information.
        """
        return cls(
            action=ActionType(data.get("action", ActionType.AI.value)),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class TeleopsStatus:
    """
    Data class to represent the status of the teleops system.
    """

    update_time: str
    battery_status: BatteryStatus
    action_status: ActionStatus = field(
        default_factory=lambda: ActionStatus(ActionType.AI, time.time())
    )
    machine_name: str = "unknown"
    video_connected: bool = False

    def to_dict(self) -> dict:
        """
        Convert the TeleopsStatus object to a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the TeleopsStatus object.
        """
        return {
            "machine_name": self.machine_name,
            "update_time": self.update_time,
            "battery_status": self.battery_status.to_dict(),
            "action_status": self.action_status.to_dict(),
            "video_connected": self.video_connected,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TeleopsStatus":
        """
        Populate the TeleopsStatus object from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing teleops status information.
        """
        return cls(
            update_time=data.get("update_time", time.time()),
            battery_status=BatteryStatus.from_dict(data.get("battery_status", {})),
            action_status=ActionStatus.from_dict(data.get("action_status", {})),
            machine_name=data.get("machine_name", "unknown"),
            video_connected=data.get("video_connected", False),
        )


@singleton
class TeleopsStatusProvider:
    """
    Teleops Status provider reports the status of the machine.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openmind.org/api/core/teleops/status",
    ):
        """
        Initialize the TeleopsStatusProvider.

        Parameters
        ----------
        api_key : str
            API key for authentication. Default is None.
        base_url : str
            Base URL for the teleops status API. Default is
            "https://api.openmind.org/api/core/teleops/status".
        """
        self.api_key = api_key
        self.base_url = base_url
        self.executor = ThreadPoolExecutor(max_workers=1)

    def get_status(self) -> dict:
        """
        Get the status of the machine.
        """
        if self.api_key is None or self.api_key == "":
            logging.error("API key is missing. Cannot get status.")
            return {}

        try:
            api_key_id = self.api_key[9:25] if len(self.api_key) > 25 else self.api_key
            request = requests.get(
                f"{self.base_url}/{api_key_id}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            if request.status_code == 200:
                return request.json()
            else:
                logging.error(
                    f"Failed to get status: {request.status_code} - {request.text}"
                )
        except requests.exceptions.RequestException as e:
            logging.error(f"TeleopsStatusProvider: Error getting status: {e}")

        return {}

    def _share_status_worker(self, status: TeleopsStatus):
        """
        Worker function to share the status of the machine.
        This function runs in a separate thread to avoid blocking the main thread.

        Parameters
        ----------
        status : TeleopsStatus
            The status of the machine to be shared.
        """
        if self.api_key is None or self.api_key == "":
            logging.error("API key is missing. Cannot share status.")
            return

        try:
            request = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=status.to_dict(),
                timeout=10,
            )

            if request.status_code == 200:
                logging.debug(f"Status shared successfully: {request.json()}")
            else:
                logging.error(
                    f"Failed to share status: {request.status_code} - {request.text}"
                )
        except Exception as e:
            logging.error(f"Error sharing status: {str(e)}")

    def share_status(self, status: TeleopsStatus):
        """
        Share the status of the machine.
        This function submits the status sharing task to a thread pool executor
        to run in a separate thread.

        Parameters
        ----------
        status : TeleopsStatus
            The status of the machine to be shared.
        """
        self.executor.submit(self._share_status_worker, status)

    def stop(self):
        """
        Stop the TeleopsStatusProvider and clean up resources.
        """
        self.executor.shutdown(wait=True)
