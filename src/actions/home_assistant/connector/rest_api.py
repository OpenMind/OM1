import logging
from typing import Dict, List, Optional, Tuple

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.home_assistant.interface import HomeAssistantInput
from providers.home_assistant_provider import HomeAssistantProvider

logger = logging.getLogger(__name__)

# Universal commands supported by all domains
UNIVERSAL_COMMANDS = {"on", "off", "toggle"}

# All supported commands grouped by domain
DOMAIN_COMMANDS: Dict[str, List[str]] = {
    "light": ["set_brightness", "set_color", "set_color_temp"],
    "climate": ["set_temperature", "set_hvac_mode", "set_fan_mode", "set_preset"],
    "lock": ["lock", "unlock"],
    "cover": ["open", "close", "stop", "set_position"],
    "media_player": [
        "play",
        "pause",
        "media_stop",
        "volume_set",
        "volume_mute",
        "volume_unmute",
        "select_source",
    ],
    "fan": ["set_percentage", "oscillate", "stop_oscillate"],
    "vacuum": ["start", "stop", "vacuum_pause", "return_to_base"],
    "scene": ["activate"],
    "alarm_control_panel": ["arm_home", "arm_away", "arm_night", "disarm"],
    "script": ["run"],
}

# Legacy "set" command is still supported for backward compatibility
LEGACY_COMMANDS = {"set"}

ALL_COMMANDS = (
    UNIVERSAL_COMMANDS
    | LEGACY_COMMANDS
    | {cmd for cmds in DOMAIN_COMMANDS.values() for cmd in cmds}
)


def _parse_rgb_color(hex_color: str) -> list:
    """
    Parse a hex color string into an RGB list.

    Parameters
    ----------
    hex_color : str
        Hex color string (e.g. "#FF0000" or "FF0000").

    Returns
    -------
    list
        [R, G, B] values as integers (0-255).

    Raises
    ------
    ValueError
        If the hex string is invalid.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(
            f"Invalid color hex '{hex_color}'. Expected 6-character hex (e.g. '#FF0000')."
        )
    return [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]


class HomeAssistantConfig(ActionConfig):
    """
    Configuration for the Home Assistant REST API connector.

    Parameters
    ----------
    base_url : str
        Base URL of the Home Assistant instance.
    token_env : str
        Environment variable name for the HA access token.
    token : str
        Direct access token (used if env var is not set).
    devices : Dict[str, str]
        Mapping of device aliases to Home Assistant entity IDs.
        Example: {"living_room_light": "light.living_room"}
    timeout_seconds : int
        HTTP request timeout in seconds.
    verify_ssl : bool
        Whether to verify SSL certificates.
    """

    base_url: str = Field(
        default="http://homeassistant.local:8123",
        description="Base URL of the Home Assistant instance",
    )
    token_env: str = Field(
        default="HOME_ASSISTANT_TOKEN",
        description="Environment variable name for the HA access token",
    )
    token: str = Field(
        default="",
        description="Direct access token (fallback if env var not set)",
    )
    devices: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of device aliases to HA entity IDs",
    )
    timeout_seconds: int = Field(
        default=10,
        description="HTTP request timeout in seconds",
    )
    verify_ssl: bool = Field(
        default=True,
        description="Whether to verify SSL certificates",
    )


class HomeAssistantRESTConnector(
    ActionConnector[HomeAssistantConfig, HomeAssistantInput]
):
    """
    Connector that controls Home Assistant devices via the REST API.
    """

    def __init__(self, config: HomeAssistantConfig):
        """
        Initialize the HomeAssistantRESTConnector.

        Parameters
        ----------
        config : HomeAssistantConfig
            Configuration for the connector.
        """
        super().__init__(config)
        self.provider = HomeAssistantProvider(
            base_url=config.base_url,
            token=config.token,
            token_env=config.token_env,
            timeout_seconds=config.timeout_seconds,
            verify_ssl=config.verify_ssl,
        )

    def _resolve_entity(self, device: str) -> str:
        """
        Resolve a device alias to a Home Assistant entity ID.

        Parameters
        ----------
        device : str
            The device alias from the LLM output.

        Returns
        -------
        str
            The corresponding Home Assistant entity ID.

        Raises
        ------
        ValueError
            If the device alias is not found in the config.
        """
        entity_id = self.config.devices.get(device)
        if entity_id is None:
            raise ValueError(
                f"Unknown device alias '{device}'. "
                f"Available devices: {list(self.config.devices.keys())}"
            )
        return entity_id

    def _require_value(self, command: str, value: Optional[float]) -> float:
        """
        Validate that a numeric value is provided.

        Parameters
        ----------
        command : str
            The command name (for error messages).
        value : Optional[float]
            The value to validate.

        Returns
        -------
        float
            The validated value.

        Raises
        ------
        ValueError
            If value is None.
        """
        if value is None:
            raise ValueError(f"The '{command}' command requires a numeric value.")
        return value

    def _require_mode(self, command: str, mode: Optional[str]) -> str:
        """
        Validate that a string mode is provided.

        Parameters
        ----------
        command : str
            The command name (for error messages).
        mode : Optional[str]
            The mode to validate.

        Returns
        -------
        str
            The validated mode string.

        Raises
        ------
        ValueError
            If mode is None or empty.
        """
        if not mode:
            raise ValueError(f"The '{command}' command requires a mode string.")
        return mode

    def _get_service_call(
        self,
        domain: str,
        command: str,
        value: Optional[float],
        mode: Optional[str],
    ) -> Tuple[str, dict]:
        """
        Determine the HA service name and extra data for a command.

        Parameters
        ----------
        domain : str
            The HA domain (e.g. "light", "climate", "lock").
        command : str
            The command to execute.
        value : Optional[float]
            Numeric parameter for the command.
        mode : Optional[str]
            String parameter for the command.

        Returns
        -------
        Tuple[str, dict]
            (service_name, extra_data_dict)

        Raises
        ------
        ValueError
            If the command is invalid or required parameters are missing.
        """
        # --- Universal commands ---
        if command == "on":
            return ("turn_on", {})
        if command == "off":
            return ("turn_off", {})
        if command == "toggle":
            return ("toggle", {})

        # --- Legacy "set" command (backward compatible) ---
        if command == "set":
            v = self._require_value(command, value)
            if domain == "light":
                return ("turn_on", {"brightness_pct": v})
            elif domain == "climate":
                return ("set_temperature", {"temperature": v})
            else:
                return ("set_value", {"value": v})

        # --- Light commands ---
        if command == "set_brightness":
            v = self._require_value(command, value)
            return ("turn_on", {"brightness_pct": v})

        if command == "set_color":
            m = self._require_mode(command, mode)
            rgb = _parse_rgb_color(m)
            return ("turn_on", {"rgb_color": rgb})

        if command == "set_color_temp":
            v = self._require_value(command, value)
            return ("turn_on", {"color_temp_kelvin": v})

        # --- Climate commands ---
        if command == "set_temperature":
            v = self._require_value(command, value)
            return ("set_temperature", {"temperature": v})

        if command == "set_hvac_mode":
            m = self._require_mode(command, mode)
            return ("set_hvac_mode", {"hvac_mode": m})

        if command == "set_fan_mode":
            m = self._require_mode(command, mode)
            return ("set_fan_mode", {"fan_mode": m})

        if command == "set_preset":
            m = self._require_mode(command, mode)
            return ("set_preset_mode", {"preset_mode": m})

        # --- Lock commands ---
        if command == "lock":
            return ("lock", {})

        if command == "unlock":
            return ("unlock", {})

        # --- Cover commands ---
        if command == "open":
            return ("open_cover", {})

        if command == "close":
            return ("close_cover", {})

        if command == "stop":
            if domain == "cover":
                return ("stop_cover", {})
            elif domain == "vacuum":
                return ("stop", {})
            elif domain == "media_player":
                return ("media_stop", {})
            else:
                return ("stop", {})

        if command == "set_position":
            v = self._require_value(command, value)
            return ("set_cover_position", {"position": v})

        # --- Media player commands ---
        if command == "play":
            return ("media_play", {})

        if command == "pause":
            return ("media_pause", {})

        if command == "media_stop":
            return ("media_stop", {})

        if command == "volume_set":
            v = self._require_value(command, value)
            return ("volume_set", {"volume_level": v / 100})

        if command == "volume_mute":
            return ("volume_mute", {"is_volume_muted": True})

        if command == "volume_unmute":
            return ("volume_mute", {"is_volume_muted": False})

        if command == "select_source":
            m = self._require_mode(command, mode)
            return ("select_source", {"source": m})

        # --- Fan commands ---
        if command == "set_percentage":
            v = self._require_value(command, value)
            return ("set_percentage", {"percentage": v})

        if command == "oscillate":
            return ("oscillate", {"oscillating": True})

        if command == "stop_oscillate":
            return ("oscillate", {"oscillating": False})

        # --- Vacuum commands ---
        if command == "start":
            return ("start", {})

        if command == "vacuum_pause":
            return ("pause", {})

        if command == "return_to_base":
            return ("return_to_base", {})

        # --- Scene commands ---
        if command == "activate":
            return ("turn_on", {})

        # --- Alarm commands ---
        if command == "arm_home":
            return ("alarm_arm_home", {})

        if command == "arm_away":
            return ("alarm_arm_away", {})

        if command == "arm_night":
            return ("alarm_arm_night", {})

        if command == "disarm":
            return ("alarm_disarm", {})

        # --- Script commands ---
        if command == "run":
            return ("turn_on", {})

        raise ValueError(
            f"Unknown command '{command}' for domain '{domain}'. "
            f"Supported commands: {ALL_COMMANDS}"
        )

    async def connect(self, output_interface: HomeAssistantInput) -> None:
        """
        Execute a Home Assistant command.

        Parameters
        ----------
        output_interface : HomeAssistantInput
            The command details from the LLM.
        """
        device = output_interface.device
        command = output_interface.command.lower().strip()

        try:
            entity_id = self._resolve_entity(device)
            domain = entity_id.split(".", 1)[0]
            service, extra_data = self._get_service_call(
                domain, command, output_interface.value, output_interface.mode
            )

            logger.info(
                f"HomeAssistant: {domain}.{service} on {entity_id} "
                f"(alias={device}, extra={extra_data})"
            )

            await self.provider.call_service(
                domain=domain,
                service=service,
                entity_id=entity_id,
                **extra_data,
            )
        except (ValueError, RuntimeError) as e:
            logger.error(f"HomeAssistant action failed: {e}")
