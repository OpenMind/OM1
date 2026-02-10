import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import Field

from inputs.base import Message, SensorConfig
from inputs.base.loop import FuserInput
from providers.io_provider import IOProvider


class AwairConfig(SensorConfig):
    """Configuration for AWAIR Element sensor.

    Parameters
    ----------
    mode : str
        API mode: "local" or "cloud".
    device_ip : Optional[str]
        IP address for local API mode.
    access_token : Optional[str]
        Bearer token for cloud API mode.
    device_id : Optional[str]
        Device ID for cloud API mode.
    device_type : str
        AWAIR device type identifier.
    poll_interval : float
        Polling interval in seconds.
    """

    mode: str = Field(default="local", description="API mode: local or cloud")
    device_ip: Optional[str] = Field(
        default=None, description="IP address for local API mode"
    )
    access_token: Optional[str] = Field(
        default=None, description="Bearer token for cloud API mode"
    )
    device_id: Optional[str] = Field(
        default=None, description="Device ID for cloud API mode"
    )
    device_type: str = Field(
        default="awair-element", description="AWAIR device type identifier"
    )
    poll_interval: float = Field(
        default=10.0, description="Polling interval in seconds"
    )


@dataclass
class AwairData:
    """
    Structured AWAIR sensor data.

    All fields directly from AWAIR Element API response.
    """

    timestamp: str
    score: int  # 0-100 AWAIR Score
    temp: float  # Temperature in Celsius
    humid: float  # Relative humidity %
    abs_humid: float  # Absolute humidity g/m³
    dew_point: float  # Dew point in Celsius
    co2: int  # CO2 in ppm
    co2_est: int  # Estimated CO2
    voc: int  # VOC in ppb
    voc_baseline: int  # VOC baseline
    pm25: int  # PM2.5 in µg/m³
    pm10_est: int  # Estimated PM10 in µg/m³


class AwairElement(FuserInput[AwairConfig, Dict[str, Any]]):
    """
    AWAIR Element Air Quality Monitor integration for OM1.

    This plugin connects to AWAIR Element devices via Local API or Cloud API
    and provides air quality data to the LLM for health-aware interactions.

    The robot can:
    - Warn about poor air quality
    - Suggest opening windows when CO2 is high
    - Alert about temperature/humidity extremes
    - Provide health recommendations based on air quality
    """

    def __init__(self, config: AwairConfig = AwairConfig()):
        super().__init__(config)

        self.io_provider = IOProvider()
        self.messages: List[Message] = []
        self.descriptor_for_LLM = "Indoor Air Quality (AWAIR Element)"

        # Configuration
        self.mode = config.mode
        self.device_ip = config.device_ip
        self.access_token = config.access_token
        self.device_id = config.device_id
        self.device_type = config.device_type
        self.poll_interval = config.poll_interval

        # State tracking
        self._previous_data: Optional[AwairData] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._consecutive_errors = 0
        self._max_errors = 5
        self._first_poll = True

        # Validate configuration
        if self.mode == "local" and not self.device_ip:
            logging.warning(
                "AwairElement: Local mode requires device_ip. "
                "Set device_ip in config (e.g., '192.168.0.17')"
            )
        elif self.mode == "cloud" and (not self.access_token or not self.device_id):
            logging.warning(
                "AwairElement: Cloud mode requires access_token and device_id."
            )

        logging.info(f"\033[94mAwairElement: Initialized ({self.mode} mode)\033[0m")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _fetch_local(self) -> Optional[Dict[str, Any]]:
        """
        Fetch air quality data from AWAIR Local API.

        Returns
        -------
        Optional[Dict[str, Any]]
            Raw sensor data or None if failed
        """
        if not self.device_ip:
            return None

        try:
            session = await self._get_session()
            url = f"http://{self.device_ip}/air-data/latest"

            async with session.get(url) as response:
                if response.status == 200:
                    self._consecutive_errors = 0
                    return await response.json()
                else:
                    logging.error(
                        f"AwairElement: Local API error HTTP {response.status}"
                    )
                    self._consecutive_errors += 1

        except aiohttp.ClientError as e:
            logging.error(f"AwairElement: Connection error: {e}")
            self._consecutive_errors += 1
        except Exception as e:
            logging.error(f"AwairElement: Unexpected error: {e}")
            self._consecutive_errors += 1

        return None

    async def _fetch_cloud(self) -> Optional[Dict[str, Any]]:
        """
        Fetch air quality data from AWAIR Cloud API.

        Returns
        -------
        Optional[Dict[str, Any]]
            Raw sensor data or None if failed
        """
        if not self.access_token or not self.device_id:
            return None

        try:
            session = await self._get_session()
            url = (
                f"https://developer-apis.awair.is/v1/users/self/devices/"
                f"{self.device_type}/{self.device_id}/air-data/latest"
            )
            headers = {"Authorization": f"Bearer {self.access_token}"}

            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    self._consecutive_errors = 0
                    data = await response.json()
                    # Cloud API returns data in a different format
                    if "data" in data and len(data["data"]) > 0:
                        return data["data"][0]
                    return data
                elif response.status == 401:
                    logging.error("AwairElement: Cloud API authentication failed")
                else:
                    logging.error(
                        f"AwairElement: Cloud API error HTTP {response.status}"
                    )
                self._consecutive_errors += 1

        except aiohttp.ClientError as e:
            logging.error(f"AwairElement: Cloud API connection error: {e}")
            self._consecutive_errors += 1
        except Exception as e:
            logging.error(f"AwairElement: Cloud API unexpected error: {e}")
            self._consecutive_errors += 1

        return None

    def _parse_data(self, raw: Dict[str, Any]) -> AwairData:
        """Parse raw API response into structured AwairData."""
        return AwairData(
            timestamp=raw.get("timestamp", ""),
            score=raw.get("score", 0),
            temp=raw.get("temp", 0.0),
            humid=raw.get("humid", 0.0),
            abs_humid=raw.get("abs_humid", 0.0),
            dew_point=raw.get("dew_point", 0.0),
            co2=raw.get("co2", 0),
            co2_est=raw.get("co2_est", 0),
            voc=raw.get("voc", 0),
            voc_baseline=raw.get("voc_baseline", 0),
            pm25=raw.get("pm25", 0),
            pm10_est=raw.get("pm10_est", 0),
        )

    def _get_score_description(self, score: int) -> str:
        """Get human-readable description of AWAIR score."""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 40:
            return "Poor"
        else:
            return "Unhealthy"

    def _has_significant_change(
        self, current: AwairData, previous: Optional[AwairData]
    ) -> bool:
        """Check if there's a significant change worth reporting."""
        if previous is None:
            return True

        # Significant change thresholds
        if abs(current.score - previous.score) >= 10:
            return True
        if abs(current.temp - previous.temp) >= 2.0:
            return True
        if abs(current.humid - previous.humid) >= 10:
            return True
        if abs(current.co2 - previous.co2) >= 200:
            return True
        if abs(current.voc - previous.voc) >= 500:
            return True
        if abs(current.pm25 - previous.pm25) >= 10:
            return True

        return False

    async def _poll(self) -> Dict[str, Any]:
        """
        Poll for air quality data from AWAIR device.

        Returns
        -------
        Dict[str, Any]
            Raw sensor data, empty dict if unavailable
        """
        # First call: fetch immediately. Subsequent calls: wait first.
        if self._first_poll:
            self._first_poll = False
        else:
            await asyncio.sleep(self.poll_interval)

        # Check for too many consecutive errors
        if self._consecutive_errors >= self._max_errors:
            logging.error(
                f"AwairElement: Too many errors ({self._consecutive_errors}), "
                "reducing poll frequency"
            )
            await asyncio.sleep(60)  # Wait a minute before retrying
            self._consecutive_errors = 0

        # Fetch based on mode
        if self.mode == "local":
            raw_data = await self._fetch_local()
        else:
            raw_data = await self._fetch_cloud()

        return raw_data if raw_data is not None else {}

    async def _raw_to_text(self, raw_input: Dict[str, Any]) -> Optional[Message]:
        """
        Convert raw AWAIR data to human-readable message.

        Parameters
        ----------
        raw_input : Dict[str, Any]
            Raw sensor data from API

        Returns
        -------
        Optional[Message]
            Formatted message for LLM
        """
        if not raw_input:
            return None

        data = self._parse_data(raw_input)

        # Build message with clear labels
        message = f"""Room Temperature: {data.temp:.1f}°C
Humidity: {data.humid:.0f}%
Air Quality Score: {data.score}/100"""
        return Message(timestamp=time.time(), message=message)

    async def raw_to_text(self, raw_input: Dict[str, Any]):
        """Update message buffer with new data."""
        pending_message = await self._raw_to_text(raw_input)

        if pending_message is not None:
            self.messages.append(pending_message)

    def formatted_latest_buffer(self) -> Optional[str]:
        """Format and clear the latest buffer contents."""
        if len(self.messages) == 0:
            return None

        latest_message = self.messages[-1]

        result = (
            f"\nINPUT: {self.descriptor_for_LLM}\n// START\n"
            f"{latest_message.message}\n// END\n"
        )

        self.io_provider.add_input(
            self.__class__.__name__, latest_message.message, latest_message.timestamp
        )
        self.messages = []

        return result

    async def cleanup(self):
        """Clean up resources."""
        if self._session and not self._session.closed:
            await self._session.close()
