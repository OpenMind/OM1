import logging
import time
from typing import Optional

from pydantic import Field

from backgrounds.base import Background, BackgroundConfig
from providers import TeleopsStatus
from providers.teleops_status_provider import TeleopsStatusProvider


class AgentTeleopsStatusConfig(BackgroundConfig):
    """
    Configuration for Teleops Status Background.

    Parameters
    ----------
    api_key : Optional[str], default=None
        OM API key for authentication.
    machine_name : str, default="agent_teleops_status_reporter"
        Machine name for teleops status reporting.
    battery_level : float, default=0.0
        Battery level percentage (0-100).
    voltage : float, default=0.0
        Battery voltage in volts.
    temperature : float, default=0.0
        Battery temperature in Celsius.
    """

    api_key: Optional[str] = Field(default=None, description="OM API key")
    machine_name: str = Field(
        default="agent_teleops_status_reporter",
        description="Machine name for teleops status",
    )
    battery_level: float = Field(
        default=0.0,
        description="Battery level percentage (0-100)",
    )
    voltage: float = Field(
        default=0.0,
        description="Battery voltage in volts",
    )
    temperature: float = Field(
        default=0.0,
        description="Battery temperature in Celsius",
    )


class AgentTeleopsStatusBackground(Background[AgentTeleopsStatusConfig]):
    """
    Background task for reporting teleops status.

    This background task initializes and manages a TeleopsStatusProvider
    instance that periodically retrieves and logs the teleops status of
    the machine. The provider communicates with a remote API to fetch
    the current status, including battery and action statuses.

    The teleops status information is essential for monitoring the
    operational state of the machine during remote operations.
    """

    def __init__(self, config: AgentTeleopsStatusConfig):
        """
        Initialize TeleopsStatus background task with configuration.

        Parameters
        ----------
        config : BackgroundConfig
            Configuration object for the background task. The base configuration
            is used as TeleopsStatus does not require additional parameters.
        """
        super().__init__(config)

        logging.warning("--------------------------------")
        logging.warning("Run Agent Teleops only if you don't have a robot connected.")
        logging.warning("--------------------------------")

        self.teleops_status_provider = TeleopsStatusProvider(
            api_key=self.config.api_key
        )
        logging.info("Initiated Teleops Status Provider in background")

    def run(self) -> None:
        """
        Run the teleops status background process.
        """
        current_time = str(time.time())
        self.teleops_status_provider.share_status(
            TeleopsStatus.from_dict(
                {
                    "machine_name": self.config.machine_name,
                    "update_time": current_time,
                    "battery_status": {
                        "battery_level": self.config.battery_level,
                        "voltage": self.config.voltage,
                        "temperature": self.config.temperature,
                        "timestamp": current_time,
                        "charging_status": False,
                    },
                }
            )
        )

        self.sleep(60)
