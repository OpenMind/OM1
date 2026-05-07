import logging
from typing import Optional

from pydantic import Field

from actions.adjust_bittle_calibration.interface import BittleCalibrationAdjustmentInput
from actions.base import ActionConfig, ActionConnector
from actions.calibrate_bittle.connector.ble import (
    MAX_CALIBRATION_DEGREES,
    MIN_CALIBRATION_DEGREES,
    VALID_CALIBRATION_SERVOS,
)
from providers.bittle_ble_provider import (
    DEFAULT_BITTLE_DEVICE_NAME,
    NUS_RX_CHARACTERISTIC_UUID,
    NUS_TX_CHARACTERISTIC_UUID,
    bittle_settings_from_config,
    get_bittle_ble_provider,
)


class BittleBLECalibrationAdjustmentConfig(ActionConfig):
    """
    Configuration for Petoi Bittle BLE calibration adjustment connector.
    """

    device_address: Optional[str] = Field(default=None, description="Bittle BLE address or identifier")
    device_name: Optional[str] = Field(default=DEFAULT_BITTLE_DEVICE_NAME, description="Advertised Bittle BLE name")
    tx_characteristic_uuid: str = Field(default=NUS_TX_CHARACTERISTIC_UUID, description="NUS TX notify UUID")
    rx_characteristic_uuid: str = Field(default=NUS_RX_CHARACTERISTIC_UUID, description="NUS RX write UUID")
    connect_timeout: float = Field(default=10.0, description="BLE connect or scan timeout in seconds")
    write_with_response: bool = Field(default=True, description="Write BLE commands with response")
    command_suffix: str = Field(default="", description="Optional suffix appended to each ASCII command")
    simulate: bool = Field(default=False, description="Log commands without opening BLE")


class BittleBLECalibrationAdjustmentConnector(
    ActionConnector[BittleBLECalibrationAdjustmentConfig, BittleCalibrationAdjustmentInput]
):
    """
    BLE connector for Petoi Bittle c<N> <deg> calibration adjustment commands.
    """

    def __init__(self, config: BittleBLECalibrationAdjustmentConfig):
        super().__init__(config)
        self.provider = get_bittle_ble_provider(bittle_settings_from_config(config))

    async def connect(self, output_interface: BittleCalibrationAdjustmentInput) -> None:
        servo_index = int(output_interface.servo_index)
        degrees = int(output_interface.degrees)
        if servo_index not in VALID_CALIBRATION_SERVOS:
            raise ValueError(
                "Bittle calibration servo_index must be one of "
                f"{sorted(VALID_CALIBRATION_SERVOS)}, got {servo_index}"
            )
        if degrees < MIN_CALIBRATION_DEGREES or degrees > MAX_CALIBRATION_DEGREES:
            raise ValueError(
                "Bittle calibration degrees must be between "
                f"{MIN_CALIBRATION_DEGREES} and {MAX_CALIBRATION_DEGREES}, got {degrees}"
            )

        command = f"c{servo_index} {degrees}"
        logging.info("Bittle calibration adjustment command: %s", command)
        await self.provider.send_command(command)

    def stop(self) -> None:
        """
        Connection cleanup is left to process teardown because providers can be shared.
        """
        pass
