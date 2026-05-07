import logging
from typing import Optional

from pydantic import Field

from actions.base import ActionConfig, ActionConnector
from actions.move_bittle.interface import BittleMoveAction, BittleMoveInput
from providers.bittle_ble_provider import (
    DEFAULT_BITTLE_DEVICE_NAME,
    NUS_RX_CHARACTERISTIC_UUID,
    NUS_TX_CHARACTERISTIC_UUID,
    bittle_settings_from_config,
    get_bittle_ble_provider,
)


class BittleBLEConfig(ActionConfig):
    """
    Configuration for Petoi Bittle BLE action connectors.

    Parameters
    ----------
    device_address : Optional[str]
        BLE MAC address or platform-specific BLE identifier. If omitted, the connector scans by device_name.
    device_name : Optional[str]
        Advertised BLE device name to scan for.
    command_suffix : str
        Optional command suffix. The Petoi controller sends ASCII tokens as-is, so the default is empty.
    simulate : bool
        If true, commands are logged and captured without opening BLE.
    """

    device_address: Optional[str] = Field(default=None, description="Bittle BLE address or identifier")
    device_name: Optional[str] = Field(default=DEFAULT_BITTLE_DEVICE_NAME, description="Advertised Bittle BLE name")
    tx_characteristic_uuid: str = Field(default=NUS_TX_CHARACTERISTIC_UUID, description="NUS TX notify UUID")
    rx_characteristic_uuid: str = Field(default=NUS_RX_CHARACTERISTIC_UUID, description="NUS RX write UUID")
    connect_timeout: float = Field(default=10.0, description="BLE connect or scan timeout in seconds")
    write_with_response: bool = Field(default=True, description="Write BLE commands with response")
    command_suffix: str = Field(default="", description="Optional suffix appended to each ASCII command")
    simulate: bool = Field(default=False, description="Log commands without opening BLE")


BITTLE_MOVE_COMMANDS: dict[BittleMoveAction, str] = {
    BittleMoveAction.WALK_FORWARD: "kwkF",
    BittleMoveAction.WALK_LEFT: "kwkL",
    BittleMoveAction.WALK_RIGHT: "kwkR",
    BittleMoveAction.WALK_BACKWARD: "kbk",
    BittleMoveAction.CRAWL_FORWARD: "kcrF",
    BittleMoveAction.CRAWL_LEFT: "kcrL",
    BittleMoveAction.CRAWL_RIGHT: "kcrR",
    BittleMoveAction.TROT_FORWARD: "ktrF",
    BittleMoveAction.TROT_LEFT: "ktrL",
    BittleMoveAction.TROT_RIGHT: "ktrR",
    BittleMoveAction.STAND_STILL: "kbalance",
    BittleMoveAction.BALANCE: "kbalance",
    BittleMoveAction.BUTT_UP: "kbuttUp",
    BittleMoveAction.CHECK_AROUND: "kck",
    BittleMoveAction.STRETCH: "kstr",
    BittleMoveAction.GREETING: "khi",
    BittleMoveAction.PEE_POSE: "kpee",
    BittleMoveAction.PUSH_UP: "kpu",
    BittleMoveAction.REST: "krest",
    BittleMoveAction.STEP_IN_PLACE: "kstp",
    BittleMoveAction.BACK_FLIP: "kbf",
    BittleMoveAction.SIT: "ksit",
    BittleMoveAction.BUNNY_JUMP: "kbdF",
    BittleMoveAction.VIBRATE: "kvt",
}


class BittleBLEMoveConnector(ActionConnector[BittleBLEConfig, BittleMoveInput]):
    """
    BLE connector for Petoi Bittle movement, gait, posture, and skill commands.
    """

    def __init__(self, config: BittleBLEConfig):
        super().__init__(config)
        self.provider = get_bittle_ble_provider(bittle_settings_from_config(config))

    async def connect(self, output_interface: BittleMoveInput) -> None:
        action = BittleMoveAction(output_interface.action)
        command = BITTLE_MOVE_COMMANDS[action]
        logging.info("Bittle move command: %s -> %s", action.value, command)
        await self.provider.send_command(command)

    def stop(self) -> None:
        """
        Connection cleanup is left to process teardown because providers can be shared.
        """
        pass
