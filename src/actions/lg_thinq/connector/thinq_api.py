import asyncio
import base64
import logging
import re
import uuid
from typing import Any, Dict, Optional

import aiohttp
from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.lg_thinq.interface import LGThinQInput
from providers.io_provider import IOProvider


class LGThinQConfig(ActionConfig):
    """
    Configuration for LG ThinQ connector.

    Parameters
    ----------
    pat_token : Optional[str]
        Personal Access Token from https://connect-pat.lgthinq.com
    country_code : str
        Country code for region detection (e.g., "TR", "US", "KR").
    device_id : Optional[str]
        LG device ID. Auto-detected if not provided.
    """

    pat_token: Optional[str] = Field(
        default=None,
        description="Personal Access Token from https://connect-pat.lgthinq.com",
    )
    country_code: str = Field(
        default="TR",
        description="Country code for region detection",
    )
    device_id: Optional[str] = Field(
        default=None,
        description="LG device ID. Auto-detected if not provided",
    )


# Region mapping based on country code
REGION_MAP = {
    "KIC": [
        "AU",
        "BD",
        "CN",
        "HK",
        "ID",
        "IN",
        "JP",
        "KH",
        "KR",
        "LA",
        "LK",
        "MM",
        "MY",
        "NP",
        "NZ",
        "PH",
        "SG",
        "TH",
        "TW",
        "VN",
    ],
    "AIC": [
        "AG",
        "AR",
        "AW",
        "BB",
        "BO",
        "BR",
        "BS",
        "BZ",
        "CA",
        "CL",
        "CO",
        "CR",
        "CU",
        "DM",
        "DO",
        "EC",
        "GD",
        "GT",
        "GY",
        "HN",
        "HT",
        "JM",
        "KN",
        "LC",
        "MX",
        "NI",
        "PA",
        "PE",
        "PR",
        "PY",
        "SR",
        "SV",
        "TT",
        "US",
        "UY",
        "VC",
        "VE",
    ],
}

# LG ThinQ Connect public API key (shared across all ThinQ Connect integrations)
# User authentication is done via Personal Access Token (pat_token) in config
THINQ_API_KEY = "v6GFvkweNo7DK7yD3ylIZ9w52aKBU0eJ7wLXkSR3"


def get_region_from_country(country_code: str) -> str:
    """Get the ThinQ API region from country code."""
    country_code = country_code.upper()
    for region, countries in REGION_MAP.items():
        if country_code in countries:
            return region.lower()
    return "eic"  # Default to Europe/Middle East/Africa


def generate_message_id() -> str:
    """Generate a URL-safe base64 message ID from UUID."""
    uid = uuid.uuid4()
    return base64.urlsafe_b64encode(uid.bytes).decode("utf-8").rstrip("=")[:22]


def extract_temperature(text: str) -> Optional[float]:
    """Extract temperature value from text like 'set temperature 24' or '24 degrees'."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:degrees?|celsius|c)?", text.lower())
    if match:
        temp = float(match.group(1))
        if 16 <= temp <= 30:
            return temp
    return None


class LGThinQConnector(ActionConnector[LGThinQConfig, LGThinQInput]):
    """
    Connector for controlling LG devices via ThinQ Connect API.
    """

    # Class-level: Last executed action (persists across instances)
    _last_action: Optional[str] = None

    def __init__(self, config: LGThinQConfig):
        super().__init__(config)

        self.io_provider = IOProvider()

        # Configuration
        self.pat_token = config.pat_token
        self.country_code = config.country_code
        self.device_id = config.device_id

        # Determine region and base URL
        self.region = get_region_from_country(self.country_code)
        self.base_url = f"https://api-{self.region}.lgthinq.com"

        # Client ID
        self.client_id = f"thinq-open-{uuid.uuid4().hex[:16]}"

        # Session for async requests
        self._session: Optional[aiohttp.ClientSession] = None

        if not self.pat_token:
            logging.error(
                "LGThinQConnector: No pat_token provided. "
                "Get one from https://connect-pat.lgthinq.com"
            )

        logging.info(f"\033[94mLGThinQConnector: Initialized ({self.region})\033[0m")

    def _get_headers(self) -> Dict[str, str]:
        """Generate headers for ThinQ API requests."""
        return {
            "Authorization": f"Bearer {self.pat_token}",
            "x-country": self.country_code,
            "x-client-id": self.client_id,
            "x-message-id": generate_message_id(),
            "x-api-key": THINQ_API_KEY,
            "x-service-phase": "OP",
            "Content-Type": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _api_request(
        self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Make a request to the ThinQ API."""
        url = f"{self.base_url}/{endpoint}"
        session = await self._get_session()

        try:
            if method == "GET":
                async with session.get(
                    url, headers=self._get_headers(), timeout=10
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("response")
                    else:
                        text = await resp.text()
                        logging.error(f"LGThinQ: API error {resp.status}: {text}")
            elif method == "POST":
                async with session.post(
                    url, headers=self._get_headers(), json=data, timeout=10
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result.get("response")
                    else:
                        text = await resp.text()
                        logging.error(
                            f"\033[91mLGThinQ: API error {resp.status}\033[0m"
                        )
        except asyncio.TimeoutError:
            logging.error("\033[91mLGThinQ: Request timed out\033[0m")
        except aiohttp.ClientError:
            logging.error("\033[91mLGThinQ: Connection error\033[0m")
        except Exception as e:
            logging.error(f"\033[91mLGThinQ: Error: {e}\033[0m")

        return None

    async def _discover_device(self) -> Optional[str]:
        """Discover the first AC device if device_id not configured."""
        if self.device_id:
            return self.device_id

        devices = await self._api_request("GET", "devices")
        if devices:
            for device in devices:
                if (
                    device.get("deviceInfo", {}).get("deviceType")
                    == "DEVICE_AIR_CONDITIONER"
                ):
                    self.device_id = device.get("deviceId")
                    alias = device.get("deviceInfo", {}).get("alias", "Unknown")
                    logging.info(
                        f"LGThinQ: Auto-discovered AC: {alias} ({self.device_id[:16]}...)"
                    )
                    return self.device_id

        logging.error("LGThinQ: No AC device found")
        return None

    async def _control_device(self, command: Dict[str, Any]) -> bool:
        """Send a control command to the device."""
        device_id = await self._discover_device()
        if not device_id:
            return False

        result = await self._api_request(
            "POST", f"devices/{device_id}/control", command
        )

        if result is not None:
            logging.info(f"\033[96mLGThinQ: Command sent → {command}\033[0m")
            return True
        return False

    def _parse_action(self, action_text: str) -> tuple[str, Optional[float]]:
        """Parse action text and return (action_type, optional_value)."""
        text = action_text.lower().strip()

        if text in ["idle", "nothing", "do nothing", "no action"]:
            return ("idle", None)

        if (
            any(x in text for x in ["turn on", "power on", "switch on", "start"])
            and "power save" not in text
        ):
            return ("power_on", None)
        if (
            any(x in text for x in ["turn off", "power off", "switch off", "stop"])
            and "power save" not in text
        ):
            return ("power_off", None)

        if "power save" in text or "energy sav" in text:
            if any(x in text for x in ["on", "enable", "start"]):
                return ("power_save_on", None)
            if any(x in text for x in ["off", "disable", "stop"]):
                return ("power_save_off", None)

        if any(x in text for x in ["swing", "oscillat", "rotate"]):
            if any(x in text for x in ["off", "stop", "disable"]):
                return ("swing_off", None)
            if any(x in text for x in ["horizontal", "left", "right"]):
                return ("swing_horizontal", None)
            if any(x in text for x in ["vertical", "up", "down"]):
                return ("swing_vertical", None)
            return ("swing_on", None)

        if any(x in text for x in ["cool", "cooling"]):
            return ("mode_cool", None)
        if any(x in text for x in ["heat", "heating", "warm"]):
            return ("mode_heat", None)
        if "auto" in text and "fan" not in text:
            return ("mode_auto", None)
        if any(x in text for x in ["fan only", "fan mode", "ventilat"]):
            return ("mode_fan", None)
        if any(x in text for x in ["dry", "dehumid"]):
            return ("mode_dry", None)

        temp = extract_temperature(text)
        if temp is not None:
            return ("temperature", temp)

        if "fan" in text:
            if any(x in text for x in ["low", "slow", "quiet"]):
                return ("fan_low", None)
            if any(x in text for x in ["mid", "medium", "normal"]):
                return ("fan_mid", None)
            if any(x in text for x in ["high", "strong", "max", "fast"]):
                return ("fan_high", None)
            if "auto" in text:
                return ("fan_auto", None)

        logging.warning(f"LGThinQ: Could not parse action: {text}")
        return ("idle", None)

    def _get_command(
        self, action_type: str, value: Optional[float]
    ) -> Optional[Dict[str, Any]]:
        """Get API command for action type."""
        commands = {
            "power_on": {"operation": {"airConOperationMode": "POWER_ON"}},
            "power_off": {"operation": {"airConOperationMode": "POWER_OFF"}},
            "mode_cool": {"airConJobMode": {"currentJobMode": "COOL"}},
            "mode_heat": {"airConJobMode": {"currentJobMode": "HEAT"}},
            "mode_auto": {"airConJobMode": {"currentJobMode": "AUTO"}},
            "mode_fan": {"airConJobMode": {"currentJobMode": "FAN"}},
            "mode_dry": {"airConJobMode": {"currentJobMode": "AIR_DRY"}},
            "fan_low": {"airFlow": {"windStrength": "LOW"}},
            "fan_mid": {"airFlow": {"windStrength": "MID"}},
            "fan_high": {"airFlow": {"windStrength": "HIGH"}},
            "fan_auto": {"airFlow": {"windStrength": "AUTO"}},
            "power_save_on": {"powerSave": {"powerSaveEnabled": True}},
            "power_save_off": {"powerSave": {"powerSaveEnabled": False}},
            "swing_on": {
                "windDirection": {"rotateLeftRight": True, "rotateUpDown": True}
            },
            "swing_off": {
                "windDirection": {"rotateLeftRight": False, "rotateUpDown": False}
            },
            "swing_horizontal": {
                "windDirection": {"rotateLeftRight": True, "rotateUpDown": False}
            },
            "swing_vertical": {
                "windDirection": {"rotateLeftRight": False, "rotateUpDown": True}
            },
        }

        if action_type == "temperature" and value:
            return {"temperature": {"targetTemperature": max(18, min(30, value))}}

        return commands.get(action_type)

    async def connect(self, output_interface: LGThinQInput) -> None:
        """
        Execute ThinQ actions based on LLM decision.
        Skips if same action was already executed.
        """
        action_text = output_interface.action

        if not action_text:
            return

        # Normalize action text
        normalized = action_text.lower().strip()

        # If idle, do nothing
        if normalized in ["idle", "nothing", "do nothing", "no action"]:
            logging.info("\033[93mLGThinQ: LLM decided → idle\033[0m")
            return

        # Skip "turn off" if it's the very first action (no sensor data yet)
        if LGThinQConnector._last_action is None and "off" in normalized:
            logging.info("\033[90mLGThinQ: Skipping first-cycle turn off\033[0m")
            return

        # If same action as last time, skip entirely
        if normalized == LGThinQConnector._last_action:
            logging.info("\033[90mLGThinQ: Same action, skipping\033[0m")
            return

        # New action - execute it
        logging.info(f"\033[92mLGThinQ: LLM decided → {normalized}\033[0m")

        # Split by 'and' for multiple commands
        commands = [cmd.strip() for cmd in normalized.replace(" and ", "|").split("|")]

        for cmd in commands:
            if not cmd:
                continue

            action_type, value = self._parse_action(cmd)

            if action_type == "idle":
                continue

            # Mode changes need power on first
            if action_type in [
                "mode_cool",
                "mode_heat",
                "mode_auto",
                "mode_fan",
                "mode_dry",
            ]:
                power_cmd = self._get_command("power_on", None)
                if power_cmd:
                    await self._control_device(power_cmd)
                    await asyncio.sleep(0.5)

            api_command = self._get_command(action_type, value)
            if api_command:
                await self._control_device(api_command)
                await asyncio.sleep(0.5)

        # Save as last action
        LGThinQConnector._last_action = normalized

    def __del__(self):
        """Cleanup session on destruction."""
        if self._session and not self._session.closed:
            try:
                asyncio.get_event_loop().create_task(self._session.close())
            except RuntimeError:
                pass
