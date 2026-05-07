import logging
from typing import Optional

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.bittle.interface import BittleInput
from providers.bittle_ble_provider import (
    DEFAULT_BITTLE_DEVICE_NAME,
    NUS_RX_CHARACTERISTIC_UUID,
    NUS_TX_CHARACTERISTIC_UUID,
    bittle_settings_from_config,
    get_bittle_ble_provider,
)


class BittleConfiguredBLEConfig(ActionConfig):
    """
    Configuration for a single Petoi Bittle BLE command action.
    """

    command: str = Field(description="Petoi ASCII token to send, such as kwkF, kbalance, or ksit")
    device_address: Optional[str] = Field(default=None, description="Bittle BLE address or identifier")
    device_name: Optional[str] = Field(default=DEFAULT_BITTLE_DEVICE_NAME, description="Advertised Bittle BLE name")
    tx_characteristic_uuid: str = Field(default=NUS_TX_CHARACTERISTIC_UUID, description="NUS TX notify UUID")
    rx_characteristic_uuid: str = Field(default=NUS_RX_CHARACTERISTIC_UUID, description="NUS RX write UUID")
    connect_timeout: float = Field(default=10.0, description="BLE connect or scan timeout in seconds")
    write_with_response: bool = Field(default=True, description="Write BLE commands with response")
    command_suffix: str = Field(default="", description="Optional suffix appended to each ASCII command")
    simulate: bool = Field(default=False, description="Log commands without opening BLE")


class BittleConfiguredBLEConnector(ActionConnector[BittleConfiguredBLEConfig, BittleInput]):
    """
    BLE connector for one configured Petoi Bittle ASCII token.
    """

    def __init__(self, config: BittleConfiguredBLEConfig):
        super().__init__(config)
        if not self.config.command:
            raise ValueError("Bittle configured action requires config.command")
        self.provider = get_bittle_ble_provider(bittle_settings_from_config(config))

    async def connect(self, output_interface: BittleInput) -> None:
        logging.info("Bittle configured command: %s", self.config.command)
        await self.provider.send_command(self.config.command)

    def stop(self) -> None:
        """
        Connection cleanup is left to process teardown because providers can be shared.
        """
        pass
