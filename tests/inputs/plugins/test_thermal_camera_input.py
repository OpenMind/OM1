from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import serial

from inputs.base import Message
from inputs.plugins.thermal_camera_input import (
    ThermalCameraConfig,
    ThermalCameraInput,
    ThermalReading,
    _MockThermalConnector,
    _SerialThermalConnector,
    _USBThermalConnector,
)


def test_thermal_reading_max_temp():
    """Test max_temp property."""
    reading = ThermalReading([20.0, 36.5, 25.0], 3, 1)
    assert reading.max_temp == 36.5


def test_thermal_reading_min_temp():
    """Test min_temp property."""
    reading = ThermalReading([20.0, 36.5, 25.0], 3, 1)
    assert reading.min_temp == 20.0


def test_thermal_reading_empty_frame():
    """Test max/min temp with empty frame."""
    reading = ThermalReading([], 0, 0)
    assert reading.max_temp == 0.0
    assert reading.min_temp == 0.0


def test_thermal_reading_get_zone_max_left():
    """Test get_zone_max returns correct max for left zone."""
    reading = ThermalReading([10.0, 20.0, 30.0], 3, 1)
    assert reading.get_zone_max("left") == 10.0


def test_thermal_reading_get_zone_max_center():
    """Test get_zone_max returns correct max for center zone."""
    reading = ThermalReading([10.0, 20.0, 30.0], 3, 1)
    assert reading.get_zone_max("center") == 20.0


def test_thermal_reading_get_zone_max_right():
    """Test get_zone_max returns correct max for right zone."""
    reading = ThermalReading([10.0, 20.0, 30.0], 3, 1)
    assert reading.get_zone_max("right") == 30.0


def test_thermal_reading_get_zone_max_empty():
    """Test get_zone_max returns 0.0 for empty frame."""
    reading = ThermalReading([], 0, 0)
    assert reading.get_zone_max("left") == 0.0


@pytest.mark.asyncio
async def test_mock_connector_returns_reading():
    """Test mock connector returns ThermalReading."""
    connector = _MockThermalConnector(scenario="clear")
    result = await connector.read()
    assert result is not None
    assert isinstance(result, ThermalReading)


@pytest.mark.asyncio
async def test_mock_connector_human_scenario():
    """Test mock connector human scenario has human-range temperatures."""
    connector = _MockThermalConnector(scenario="human")
    result = await connector.read()
    assert result is not None
    assert result.max_temp >= 34.0


@pytest.mark.asyncio
async def test_mock_connector_alert_scenario():
    """Test mock connector alert scenario has high temperature."""
    connector = _MockThermalConnector(scenario="alert")
    result = await connector.read()
    assert result is not None
    assert result.max_temp >= 60.0


def test_mock_connector_stop():
    """Test mock connector stop does not raise."""
    connector = _MockThermalConnector()
    connector.stop()


def test_serial_connector_init_success():
    """Test serial connector initializes successfully."""
    mock_serial = MagicMock()
    with patch(
        "inputs.plugins.thermal_camera_input._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialThermalConnector("/dev/ttyUSB0", 9600, 1.0)
        assert connector._ser == mock_serial


def test_serial_connector_init_failure():
    """Test serial connector handles connection failure gracefully."""
    with patch(
        "inputs.plugins.thermal_camera_input._serial.Serial",
        side_effect=serial.SerialException("Port not found"),
    ):
        connector = _SerialThermalConnector("/dev/ttyUSB0", 9600, 1.0)
        assert connector._ser is None


@pytest.mark.asyncio
async def test_serial_connector_read_valid():
    """Test serial connector parses valid THERMAL line."""
    mock_serial = MagicMock()
    mock_serial.readline.return_value = b"THERMAL:20.0,21.0,22.0,23.0,24.0,25.0,26.0,27.0,28.0,29.0,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0,41.0,42.0,43.0,44.0,45.0,46.0,47.0,48.0,49.0,50.0,51.0,52.0,53.0,54.0,55.0,56.0,57.0,58.0,59.0,60.0,61.0,62.0,63.0,64.0,65.0,66.0,67.0,68.0,69.0,70.0,71.0,72.0,73.0,74.0,75.0,76.0,77.0,78.0,79.0,80.0,81.0,82.0,83.0,8,8\n"
    with patch(
        "inputs.plugins.thermal_camera_input._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialThermalConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
        assert result is not None
        assert result.width == 8
        assert result.height == 8


@pytest.mark.asyncio
async def test_serial_connector_read_invalid_prefix():
    """Test serial connector ignores lines without THERMAL prefix."""
    mock_serial = MagicMock()
    mock_serial.readline.return_value = b"INVALID:data\n"
    with patch(
        "inputs.plugins.thermal_camera_input._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialThermalConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
        assert result is None


@pytest.mark.asyncio
async def test_serial_connector_read_no_connection():
    """Test serial connector read returns None when not connected."""
    with patch(
        "inputs.plugins.thermal_camera_input._serial.Serial",
        side_effect=serial.SerialException("Port not found"),
    ):
        connector = _SerialThermalConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
        assert result is None


@pytest.mark.asyncio
async def test_serial_connector_read_exception():
    """Test serial connector handles read exception gracefully."""
    mock_serial = MagicMock()
    mock_serial.readline.side_effect = Exception("Read error")
    with patch(
        "inputs.plugins.thermal_camera_input._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialThermalConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
        assert result is None


def test_serial_connector_stop():
    """Test serial connector stop closes port."""
    mock_serial = MagicMock()
    mock_serial.is_open = True
    with patch(
        "inputs.plugins.thermal_camera_input._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialThermalConnector("/dev/ttyUSB0", 9600, 1.0)
        connector.stop()
        mock_serial.close.assert_called_once()


def test_usb_connector_init_success():
    """Test USB connector initializes successfully."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    with patch(
        "inputs.plugins.thermal_camera_input._cv2.VideoCapture",
        return_value=mock_cap,
    ):
        connector = _USBThermalConnector(camera_index=0)
        assert connector._ready is True


def test_usb_connector_init_failure():
    """Test USB connector handles camera not found."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    with patch(
        "inputs.plugins.thermal_camera_input._cv2.VideoCapture",
        return_value=mock_cap,
    ):
        connector = _USBThermalConnector(camera_index=0)
        assert connector._ready is False


@pytest.mark.asyncio
async def test_usb_connector_read_success():
    """Test USB connector reads and converts frame."""
    import numpy as np

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_frame = np.zeros((24, 32, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, mock_frame)
    with patch(
        "inputs.plugins.thermal_camera_input._cv2.VideoCapture",
        return_value=mock_cap,
    ):
        connector = _USBThermalConnector(camera_index=0)
        result = await connector.read()
        assert result is not None
        assert result.width == 32
        assert result.height == 24


@pytest.mark.asyncio
async def test_usb_connector_read_failure():
    """Test USB connector handles read failure."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)
    with patch(
        "inputs.plugins.thermal_camera_input._cv2.VideoCapture",
        return_value=mock_cap,
    ):
        connector = _USBThermalConnector(camera_index=0)
        result = await connector.read()
        assert result is None


def test_usb_connector_stop():
    """Test USB connector releases camera on stop."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    with patch(
        "inputs.plugins.thermal_camera_input._cv2.VideoCapture",
        return_value=mock_cap,
    ):
        connector = _USBThermalConnector(camera_index=0)
        connector.stop()
        mock_cap.release.assert_called_once()


def test_initialization_mock_connector():
    """Test ThermalCameraInput initializes with mock connector."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(connector="mock")
        sensor = ThermalCameraInput(config=config)
        assert isinstance(sensor._connector, _MockThermalConnector)
        assert sensor.messages == []
        assert sensor.descriptor_for_LLM == "Thermal Camera"


def test_initialization_unknown_connector_falls_back_to_mock():
    """Test unknown connector falls back to mock."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(connector="unknown_hw")
        sensor = ThermalCameraInput(config=config)
        assert isinstance(sensor._connector, _MockThermalConnector)


def test_initialization_serial_connector():
    """Test ThermalCameraInput initializes with serial connector."""
    mock_serial = MagicMock()
    with (
        patch("inputs.plugins.thermal_camera_input.IOProvider"),
        patch(
            "inputs.plugins.thermal_camera_input._serial.Serial",
            return_value=mock_serial,
        ),
    ):
        config = ThermalCameraConfig(connector="serial", port="/dev/ttyUSB0")
        sensor = ThermalCameraInput(config=config)
        assert isinstance(sensor._connector, _SerialThermalConnector)


def test_initialization_usb_connector():
    """Test ThermalCameraInput initializes with USB connector."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    with (
        patch("inputs.plugins.thermal_camera_input.IOProvider"),
        patch(
            "inputs.plugins.thermal_camera_input._cv2.VideoCapture",
            return_value=mock_cap,
        ),
    ):
        config = ThermalCameraConfig(connector="usb", camera_index=0)
        sensor = ThermalCameraInput(config=config)
        assert isinstance(sensor._connector, _USBThermalConnector)


@pytest.mark.asyncio
async def test_poll_returns_reading():
    """Test _poll returns thermal reading from connector."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(connector="mock", mock_scenario="human")
        sensor = ThermalCameraInput(config=config)
        with patch(
            "inputs.plugins.thermal_camera_input.asyncio.sleep", new=AsyncMock()
        ):
            result = await sensor._poll()
        assert result is not None
        assert isinstance(result, ThermalReading)


@pytest.mark.asyncio
async def test_poll_returns_none_when_connector_fails():
    """Test _poll returns None when connector returns None."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(connector="mock")
        sensor = ThermalCameraInput(config=config)
        sensor._connector = MagicMock()
        sensor._connector.read = AsyncMock(return_value=None)
        with patch(
            "inputs.plugins.thermal_camera_input.asyncio.sleep", new=AsyncMock()
        ):
            result = await sensor._poll()
        assert result is None


def test_classify_alert():
    """Test _classify returns alert for high temperature."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(connector="mock", alert_temp_threshold=60.0)
        sensor = ThermalCameraInput(config=config)
        reading = ThermalReading([25.0, 25.0, 75.0], 3, 1)
        category, peak, zone = sensor._classify(reading)
        assert category == "alert"
        assert peak == 75.0


def test_classify_human():
    """Test _classify returns human for body-temperature range."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(
            connector="mock",
            human_temp_min=34.0,
            human_temp_max=39.0,
        )
        sensor = ThermalCameraInput(config=config)
        reading = ThermalReading([22.0, 36.5, 23.0], 3, 1)
        category, peak, zone = sensor._classify(reading)
        assert category == "human"
        assert peak == 36.5


def test_classify_clear():
    """Test _classify returns clear for ambient temperatures."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(connector="mock")
        sensor = ThermalCameraInput(config=config)
        reading = ThermalReading([22.0, 23.0, 24.0], 3, 1)
        category, peak, zone = sensor._classify(reading)
        assert category == "clear"


@pytest.mark.asyncio
async def test_raw_to_text_alert():
    """Test _raw_to_text returns alert message for high temperature."""
    with (
        patch("inputs.plugins.thermal_camera_input.IOProvider"),
        patch("inputs.plugins.thermal_camera_input.time.time", return_value=1000.0),
    ):
        config = ThermalCameraConfig(connector="mock", alert_temp_threshold=60.0)
        sensor = ThermalCameraInput(config=config)
        reading = ThermalReading([25.0, 25.0, 75.0], 3, 1)
        result = await sensor._raw_to_text(reading)
        assert result is not None
        assert "THERMAL ALERT" in result.message
        assert "75.0" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_human():
    """Test _raw_to_text returns human presence message."""
    with (
        patch("inputs.plugins.thermal_camera_input.IOProvider"),
        patch("inputs.plugins.thermal_camera_input.time.time", return_value=1000.0),
    ):
        config = ThermalCameraConfig(
            connector="mock",
            human_temp_min=34.0,
            human_temp_max=39.0,
        )
        sensor = ThermalCameraInput(config=config)
        reading = ThermalReading([22.0, 36.5, 23.0], 3, 1)
        result = await sensor._raw_to_text(reading)
        assert result is not None
        assert "Human-like heat signature" in result.message
        assert "36.5" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_clear():
    """Test _raw_to_text returns clear message for ambient temperatures."""
    with (
        patch("inputs.plugins.thermal_camera_input.IOProvider"),
        patch("inputs.plugins.thermal_camera_input.time.time", return_value=1000.0),
    ):
        config = ThermalCameraConfig(connector="mock")
        sensor = ThermalCameraInput(config=config)
        reading = ThermalReading([22.0, 23.0, 24.0], 3, 1)
        result = await sensor._raw_to_text(reading)
        assert result is not None
        assert "No significant heat signatures" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_none_input():
    """Test _raw_to_text returns None for None input."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(connector="mock")
        sensor = ThermalCameraInput(config=config)
        result = await sensor._raw_to_text(None)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_cooldown_suppresses_repeated_alerts():
    """Test cooldown suppresses repeated alert messages."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(
            connector="mock", cooldown=5.0, alert_temp_threshold=60.0
        )
        sensor = ThermalCameraInput(config=config)
        reading = ThermalReading([25.0, 25.0, 75.0], 3, 1)
        sensor._last_alert_time = 1000.0
        sensor.config.cooldown = 5.0
        with patch(
            "inputs.plugins.thermal_camera_input.time.time", return_value=1001.0
        ):
            result = await sensor._raw_to_text(reading)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_cooldown_allows_after_expiry():
    """Test alert is allowed again after cooldown expires."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(
            connector="mock", cooldown=5.0, alert_temp_threshold=60.0
        )
        sensor = ThermalCameraInput(config=config)
        reading = ThermalReading([25.0, 25.0, 75.0], 3, 1)
        sensor._last_alert_time = 1000.0
        sensor.config.cooldown = 5.0
        with patch(
            "inputs.plugins.thermal_camera_input.time.time", return_value=1006.0
        ):
            result = await sensor._raw_to_text(reading)
        assert result is not None


@pytest.mark.asyncio
async def test_raw_to_text_updates_messages():
    """Test raw_to_text appends to messages buffer."""
    with (
        patch("inputs.plugins.thermal_camera_input.IOProvider"),
        patch("inputs.plugins.thermal_camera_input.time.time", return_value=1000.0),
    ):
        config = ThermalCameraConfig(connector="mock")
        sensor = ThermalCameraInput(config=config)
        reading = ThermalReading([22.0, 23.0, 24.0], 3, 1)
        await sensor.raw_to_text(reading)
        assert len(sensor.messages) == 1


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer returns formatted string and clears buffer."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(connector="mock")
        sensor = ThermalCameraInput(config=config)
        sensor.io_provider = MagicMock()
        sensor.messages = [
            Message(
                timestamp=1000.0,
                message="Thermal camera: No significant heat signatures detected. Max temperature: 24.0°C.",
            )
        ]
        result = sensor.formatted_latest_buffer()
        assert result is not None
        assert "Thermal Camera" in result
        assert "No significant heat signatures" in result
        sensor.io_provider.add_input.assert_called_once()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer returns None when buffer is empty."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(connector="mock")
        sensor = ThermalCameraInput(config=config)
        result = sensor.formatted_latest_buffer()
        assert result is None


def test_stop_calls_connector_stop():
    """Test stop calls connector stop method."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(connector="mock")
        sensor = ThermalCameraInput(config=config)
        sensor._connector = MagicMock()
        sensor.stop()
        sensor._connector.stop.assert_called_once()
        assert sensor.messages == []


@pytest.mark.asyncio
async def test_mlx90640_connector_init_library_unavailable():
    """Test MLX90640 connector handles missing library."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._MLX90640_AVAILABLE
    mod._MLX90640_AVAILABLE = False
    from inputs.plugins.thermal_camera_input import _MLX90640Connector

    connector = _MLX90640Connector(i2c_address=0x33, refresh_rate=4)
    assert connector._ready is False
    mod._MLX90640_AVAILABLE = original


@pytest.mark.asyncio
async def test_mlx90640_connector_init_success():
    """Test MLX90640 connector initializes with mocked hardware."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._MLX90640_AVAILABLE
    original_lib = mod._mlx90640_lib
    mock_lib = MagicMock()
    mock_sensor = MagicMock()
    mock_lib.MLX90640.return_value = mock_sensor
    mock_lib.RefreshRate.REFRESH_4_HZ = 4
    mod._MLX90640_AVAILABLE = True
    mod._mlx90640_lib = mock_lib
    mock_i2c = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "board": MagicMock(),
            "busio": MagicMock(I2C=MagicMock(return_value=mock_i2c)),
        },
    ):
        from inputs.plugins.thermal_camera_input import _MLX90640Connector

        connector = _MLX90640Connector(i2c_address=0x33, refresh_rate=4)
        assert connector._ready is True
    mod._MLX90640_AVAILABLE = original
    mod._mlx90640_lib = original_lib


@pytest.mark.asyncio
async def test_mlx90640_connector_read_success():
    """Test MLX90640 connector reads frame successfully."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._MLX90640_AVAILABLE
    original_lib = mod._mlx90640_lib
    mock_lib = MagicMock()
    mock_sensor = MagicMock()
    mock_lib.MLX90640.return_value = mock_sensor
    mock_lib.RefreshRate.REFRESH_4_HZ = 4
    mod._MLX90640_AVAILABLE = True
    mod._mlx90640_lib = mock_lib
    mock_i2c = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "board": MagicMock(),
            "busio": MagicMock(I2C=MagicMock(return_value=mock_i2c)),
        },
    ):
        from inputs.plugins.thermal_camera_input import _MLX90640Connector

        connector = _MLX90640Connector(i2c_address=0x33, refresh_rate=4)
        result = await connector.read()
        assert result is not None
    mod._MLX90640_AVAILABLE = original
    mod._mlx90640_lib = original_lib


@pytest.mark.asyncio
async def test_mlx90640_connector_read_not_ready():
    """Test MLX90640 connector returns None when not ready."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._MLX90640_AVAILABLE
    mod._MLX90640_AVAILABLE = False
    from inputs.plugins.thermal_camera_input import _MLX90640Connector

    connector = _MLX90640Connector(i2c_address=0x33, refresh_rate=4)
    result = await connector.read()
    assert result is None
    mod._MLX90640_AVAILABLE = original


def test_mlx90640_connector_stop():
    """Test MLX90640 connector stop."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._MLX90640_AVAILABLE
    mod._MLX90640_AVAILABLE = False
    from inputs.plugins.thermal_camera_input import _MLX90640Connector

    connector = _MLX90640Connector(i2c_address=0x33, refresh_rate=4)
    connector.stop()
    assert connector._ready is False
    mod._MLX90640_AVAILABLE = original


@pytest.mark.asyncio
async def test_amg8833_connector_init_library_unavailable():
    """Test AMG8833 connector handles missing library."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._AMG8833_AVAILABLE
    mod._AMG8833_AVAILABLE = False
    from inputs.plugins.thermal_camera_input import _AMG8833Connector

    connector = _AMG8833Connector()
    assert connector._ready is False
    mod._AMG8833_AVAILABLE = original


@pytest.mark.asyncio
async def test_amg8833_connector_init_success():
    """Test AMG8833 connector initializes with mocked hardware."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._AMG8833_AVAILABLE
    original_lib = mod._amg88xx_lib
    mock_lib = MagicMock()
    mock_sensor = MagicMock()
    mock_lib.AMG88XX.return_value = mock_sensor
    mod._AMG8833_AVAILABLE = True
    mod._amg88xx_lib = mock_lib
    mock_i2c = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "board": MagicMock(),
            "busio": MagicMock(I2C=MagicMock(return_value=mock_i2c)),
        },
    ):
        from inputs.plugins.thermal_camera_input import _AMG8833Connector

        connector = _AMG8833Connector()
        assert connector._ready is True
    mod._AMG8833_AVAILABLE = original
    mod._amg88xx_lib = original_lib


@pytest.mark.asyncio
async def test_amg8833_connector_read_success():
    """Test AMG8833 connector reads pixels successfully."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._AMG8833_AVAILABLE
    original_lib = mod._amg88xx_lib
    mock_lib = MagicMock()
    mock_sensor = MagicMock()
    mock_sensor.pixels = [[22.0] * 8 for _ in range(8)]
    mock_lib.AMG88XX.return_value = mock_sensor
    mod._AMG8833_AVAILABLE = True
    mod._amg88xx_lib = mock_lib
    mock_i2c = MagicMock()
    with patch.dict(
        "sys.modules",
        {
            "board": MagicMock(),
            "busio": MagicMock(I2C=MagicMock(return_value=mock_i2c)),
        },
    ):
        from inputs.plugins.thermal_camera_input import _AMG8833Connector

        connector = _AMG8833Connector()
        result = await connector.read()
        assert result is not None
        assert len(result.frame) == 64
    mod._AMG8833_AVAILABLE = original
    mod._amg88xx_lib = original_lib


@pytest.mark.asyncio
async def test_amg8833_connector_read_not_ready():
    """Test AMG8833 connector returns None when not ready."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._AMG8833_AVAILABLE
    mod._AMG8833_AVAILABLE = False
    from inputs.plugins.thermal_camera_input import _AMG8833Connector

    connector = _AMG8833Connector()
    result = await connector.read()
    assert result is None
    mod._AMG8833_AVAILABLE = original


def test_amg8833_connector_stop():
    """Test AMG8833 connector stop."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._AMG8833_AVAILABLE
    mod._AMG8833_AVAILABLE = False
    from inputs.plugins.thermal_camera_input import _AMG8833Connector

    connector = _AMG8833Connector()
    connector.stop()
    assert connector._ready is False
    mod._AMG8833_AVAILABLE = original


def test_initialization_amg8833_connector():
    """Test ThermalCameraInput initializes with amg8833 connector."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._AMG8833_AVAILABLE
    mod._AMG8833_AVAILABLE = False
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        from inputs.plugins.thermal_camera_input import _AMG8833Connector

        config = ThermalCameraConfig(connector="amg8833")
        sensor = ThermalCameraInput(config=config)
        assert isinstance(sensor._connector, _AMG8833Connector)
    mod._AMG8833_AVAILABLE = original


def test_initialization_mlx90640_connector():
    """Test ThermalCameraInput initializes with mlx90640 connector."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._MLX90640_AVAILABLE
    mod._MLX90640_AVAILABLE = False
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        from inputs.plugins.thermal_camera_input import _MLX90640Connector

        config = ThermalCameraConfig(connector="mlx90640")
        sensor = ThermalCameraInput(config=config)
        assert isinstance(sensor._connector, _MLX90640Connector)
    mod._MLX90640_AVAILABLE = original


@pytest.mark.asyncio
async def test_raw_to_text_human_cooldown_suppresses():
    """Test cooldown suppresses repeated human detection messages."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(
            connector="mock",
            cooldown=5.0,
            human_temp_min=34.0,
            human_temp_max=39.0,
        )
        sensor = ThermalCameraInput(config=config)
        reading = ThermalReading([22.0, 36.5, 23.0], 3, 1)
        sensor._last_alert_time = 1000.0
        sensor.config.cooldown = 5.0
        with patch(
            "inputs.plugins.thermal_camera_input.time.time", return_value=1001.0
        ):
            result = await sensor._raw_to_text(reading)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_human_cooldown_allows_after_expiry():
    """Test human detection allowed again after cooldown expires."""
    with patch("inputs.plugins.thermal_camera_input.IOProvider"):
        config = ThermalCameraConfig(
            connector="mock",
            cooldown=5.0,
            human_temp_min=34.0,
            human_temp_max=39.0,
        )
        sensor = ThermalCameraInput(config=config)
        reading = ThermalReading([22.0, 36.5, 23.0], 3, 1)
        sensor._last_alert_time = 1000.0
        sensor.config.cooldown = 5.0
        with patch(
            "inputs.plugins.thermal_camera_input.time.time", return_value=1006.0
        ):
            result = await sensor._raw_to_text(reading)
        assert result is not None


@pytest.mark.asyncio
async def test_serial_connector_read_too_few_values():
    """Test serial connector returns None for lines with too few values."""
    mock_serial = MagicMock()
    mock_serial.readline.return_value = b"THERMAL:20.0,8\n"
    with patch(
        "inputs.plugins.thermal_camera_input._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialThermalConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
        assert result is None


@pytest.mark.asyncio
async def test_serial_connector_read_pixel_count_mismatch():
    """Test serial connector returns None when pixel count mismatches dimensions."""
    mock_serial = MagicMock()
    mock_serial.readline.return_value = b"THERMAL:20.0,21.0,22.0,4,4\n"
    with patch(
        "inputs.plugins.thermal_camera_input._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialThermalConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
        assert result is None


@pytest.mark.asyncio
async def test_usb_connector_read_exception():
    """Test USB connector handles read exception gracefully."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = Exception("Read error")
    with patch(
        "inputs.plugins.thermal_camera_input._cv2.VideoCapture",
        return_value=mock_cap,
    ):
        connector = _USBThermalConnector(camera_index=0)
        result = await connector.read()
        assert result is None


def test_thermal_reading_get_zone_idx_out_of_bounds():
    """Test get_zone_max handles idx beyond frame length gracefully."""
    reading = ThermalReading([10.0, 20.0], 3, 2)
    result = reading.get_zone_max("left")
    assert result == 10.0


@pytest.mark.asyncio
async def test_mlx90640_connector_init_exception():
    """Test MLX90640 connector handles init exception gracefully."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._MLX90640_AVAILABLE
    original_lib = mod._mlx90640_lib
    mock_lib = MagicMock()
    mock_lib.MLX90640.side_effect = Exception("I2C error")
    mod._MLX90640_AVAILABLE = True
    mod._mlx90640_lib = mock_lib
    with patch.dict(
        "sys.modules", {"board": MagicMock(), "busio": MagicMock(I2C=MagicMock())}
    ):
        from inputs.plugins.thermal_camera_input import _MLX90640Connector

        connector = _MLX90640Connector(i2c_address=0x33, refresh_rate=4)
        assert connector._ready is False
    mod._MLX90640_AVAILABLE = original
    mod._mlx90640_lib = original_lib


@pytest.mark.asyncio
async def test_mlx90640_connector_read_exception():
    """Test MLX90640 connector handles read exception gracefully."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._MLX90640_AVAILABLE
    original_lib = mod._mlx90640_lib
    mock_lib = MagicMock()
    mock_sensor = MagicMock()
    mock_sensor.getFrame.side_effect = Exception("Read error")
    mock_lib.MLX90640.return_value = mock_sensor
    mock_lib.RefreshRate.REFRESH_4_HZ = 4
    mod._MLX90640_AVAILABLE = True
    mod._mlx90640_lib = mock_lib
    with patch.dict(
        "sys.modules", {"board": MagicMock(), "busio": MagicMock(I2C=MagicMock())}
    ):
        from inputs.plugins.thermal_camera_input import _MLX90640Connector

        connector = _MLX90640Connector(i2c_address=0x33, refresh_rate=4)
        result = await connector.read()
        assert result is None
    mod._MLX90640_AVAILABLE = original
    mod._mlx90640_lib = original_lib


@pytest.mark.asyncio
async def test_amg8833_connector_init_exception():
    """Test AMG8833 connector handles init exception gracefully."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._AMG8833_AVAILABLE
    original_lib = mod._amg88xx_lib
    mock_lib = MagicMock()
    mock_lib.AMG88XX.side_effect = Exception("I2C error")
    mod._AMG8833_AVAILABLE = True
    mod._amg88xx_lib = mock_lib
    with patch.dict(
        "sys.modules", {"board": MagicMock(), "busio": MagicMock(I2C=MagicMock())}
    ):
        from inputs.plugins.thermal_camera_input import _AMG8833Connector

        connector = _AMG8833Connector()
        assert connector._ready is False
    mod._AMG8833_AVAILABLE = original
    mod._amg88xx_lib = original_lib


@pytest.mark.asyncio
async def test_amg8833_connector_read_exception():
    """Test AMG8833 connector handles read exception gracefully."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._AMG8833_AVAILABLE
    original_lib = mod._amg88xx_lib
    mock_lib = MagicMock()
    mock_sensor = MagicMock()
    type(mock_sensor).pixels = property(
        lambda self: (_ for _ in ()).throw(Exception("Read error"))
    )
    mock_lib.AMG88XX.return_value = mock_sensor
    mod._AMG8833_AVAILABLE = True
    mod._amg88xx_lib = mock_lib
    with patch.dict(
        "sys.modules", {"board": MagicMock(), "busio": MagicMock(I2C=MagicMock())}
    ):
        from inputs.plugins.thermal_camera_input import _AMG8833Connector

        connector = _AMG8833Connector()
        result = await connector.read()
        assert result is None
    mod._AMG8833_AVAILABLE = original
    mod._amg88xx_lib = original_lib


def test_usb_connector_init_cv2_unavailable():
    """Test USB connector handles missing cv2 library."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._CV2_AVAILABLE
    mod._CV2_AVAILABLE = False
    connector = _USBThermalConnector(camera_index=0)
    assert connector._ready is False
    mod._CV2_AVAILABLE = original


def test_usb_connector_init_videocapture_exception():
    """Test USB connector handles VideoCapture exception."""
    with patch(
        "inputs.plugins.thermal_camera_input._cv2.VideoCapture",
        side_effect=Exception("Camera error"),
    ):
        connector = _USBThermalConnector(camera_index=0)
        assert connector._ready is False


def test_serial_connector_init_serial_unavailable():
    """Test serial connector handles missing pyserial library."""
    import inputs.plugins.thermal_camera_input as mod

    original = mod._SERIAL_AVAILABLE
    mod._SERIAL_AVAILABLE = False
    connector = _SerialThermalConnector("/dev/ttyUSB0", 9600, 1.0)
    assert connector._ser is None
    mod._SERIAL_AVAILABLE = original


@pytest.mark.asyncio
async def test_serial_connector_read_parse_error():
    """Test serial connector handles parse error gracefully."""
    mock_serial = MagicMock()
    mock_serial.readline.return_value = b"THERMAL:not,a,number,x,y\n"
    with patch(
        "inputs.plugins.thermal_camera_input._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialThermalConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
        assert result is None
