import pytest

from actions.adjust_bittle_calibration.connector.ble import (
    BittleBLECalibrationAdjustmentConfig,
    BittleBLECalibrationAdjustmentConnector,
)
from actions.adjust_bittle_calibration.interface import BittleCalibrationAdjustmentInput
from actions.bittle.connector.ble import BittleConfiguredBLEConfig, BittleConfiguredBLEConnector
from actions.bittle.interface import BittleInput
from actions.calibrate_bittle.connector.ble import BittleBLECalibrationConfig, BittleBLECalibrationConnector
from actions.calibrate_bittle.interface import BittleCalibrationInput, BittleCalibrationOperation
from actions.move_bittle.connector.ble import BITTLE_MOVE_COMMANDS, BittleBLEConfig, BittleBLEMoveConnector
from actions.move_bittle.interface import BittleMoveInput
from providers.bittle_ble_provider import reset_bittle_ble_providers


@pytest.fixture(autouse=True)
def reset_bittle_providers():
    reset_bittle_ble_providers()
    yield
    reset_bittle_ble_providers()


@pytest.mark.asyncio
async def test_bittle_move_connector_exposes_protocol_commands():
    connector = BittleBLEMoveConnector(BittleBLEConfig(simulate=True))

    for action in BITTLE_MOVE_COMMANDS:
        await connector.connect(BittleMoveInput(action=action))

    assert list(connector.provider.sent_commands) == list(BITTLE_MOVE_COMMANDS.values())


@pytest.mark.asyncio
async def test_bittle_configured_connector_sends_command_from_config():
    connector = BittleConfiguredBLEConnector(BittleConfiguredBLEConfig(command="kwkF", simulate=True))

    await connector.connect(BittleInput())

    assert list(connector.provider.sent_commands) == ["kwkF"]


@pytest.mark.asyncio
async def test_bittle_calibration_connector_maps_session_adjust_and_save():
    connector = BittleBLECalibrationConnector(BittleBLECalibrationConfig(simulate=True))

    await connector.connect(BittleCalibrationInput(operation=BittleCalibrationOperation.ENTER))
    await connector.connect(
        BittleCalibrationInput(
            operation=BittleCalibrationOperation.ADJUST,
            servo_index=8,
            degrees=-3,
        )
    )
    await connector.connect(BittleCalibrationInput(operation=BittleCalibrationOperation.SAVE))

    assert list(connector.provider.sent_commands) == ["c", "c8 -3", "s"]


@pytest.mark.asyncio
async def test_bittle_calibration_adjustment_connector_uses_servo_and_degrees_only():
    connector = BittleBLECalibrationAdjustmentConnector(BittleBLECalibrationAdjustmentConfig(simulate=True))

    await connector.connect(BittleCalibrationAdjustmentInput(servo_index=8, degrees=-3))

    assert list(connector.provider.sent_commands) == ["c8 -3"]


@pytest.mark.asyncio
async def test_bittle_calibration_connector_validates_servo_and_degrees():
    connector = BittleBLECalibrationConnector(BittleBLECalibrationConfig(simulate=True))

    with pytest.raises(ValueError, match="servo_index"):
        await connector.connect(
            BittleCalibrationInput(
                operation=BittleCalibrationOperation.ADJUST,
                servo_index=7,
                degrees=0,
            )
        )

    with pytest.raises(ValueError, match="degrees"):
        await connector.connect(
            BittleCalibrationInput(
                operation=BittleCalibrationOperation.ADJUST,
                servo_index=8,
                degrees=10,
            )
        )
