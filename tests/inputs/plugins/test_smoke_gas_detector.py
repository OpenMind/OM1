from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import serial

from inputs.base import Message
from inputs.plugins.smoke_gas_detector import (
    SmokeGasDetector,
    SmokeGasDetectorConfig,
    SmokeGasReading,
    _MockSmokeConnector,
    _SerialSmokeConnector,
)


def test_smoke_gas_reading_properties():
    """Test SmokeGasReading stores values correctly."""
    reading = SmokeGasReading(smoke_ppm=450.0, gas_ppm=320.0, sensor_type="test")
    assert reading.smoke_ppm == 450.0
    assert reading.gas_ppm == 320.0
    assert reading.sensor_type == "test"


def test_smoke_gas_reading_default_sensor_type():
    """Test SmokeGasReading default sensor type."""
    reading = SmokeGasReading(smoke_ppm=50.0, gas_ppm=40.0)
    assert reading.sensor_type == "unknown"


@pytest.mark.asyncio
async def test_mock_connector_normal_scenario():
    """Test mock connector normal scenario returns low ppm."""
    connector = _MockSmokeConnector(scenario="normal")
    result = await connector.read()
    assert result is not None
    assert result.smoke_ppm < 300.0
    assert result.gas_ppm < 300.0


@pytest.mark.asyncio
async def test_mock_connector_warning_scenario():
    """Test mock connector warning scenario returns warning-level ppm."""
    connector = _MockSmokeConnector(scenario="warning")
    result = await connector.read()
    assert result is not None
    assert result.smoke_ppm >= 300.0


@pytest.mark.asyncio
async def test_mock_connector_danger_scenario():
    """Test mock connector danger scenario returns high ppm."""
    connector = _MockSmokeConnector(scenario="danger")
    result = await connector.read()
    assert result is not None
    assert result.smoke_ppm >= 600.0


def test_mock_connector_stop():
    """Test mock connector stop does not raise."""
    connector = _MockSmokeConnector()
    connector.stop()


def test_serial_connector_init_success():
    """Test serial connector initializes successfully."""
    mock_serial = MagicMock()
    with patch(
        "inputs.plugins.smoke_gas_detector._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialSmokeConnector("/dev/ttyUSB0", 9600, 1.0)
        assert connector._ser == mock_serial


def test_serial_connector_init_failure():
    """Test serial connector handles connection failure gracefully."""
    with patch(
        "inputs.plugins.smoke_gas_detector._serial.Serial",
        side_effect=serial.SerialException("Port not found"),
    ):
        connector = _SerialSmokeConnector("/dev/ttyUSB0", 9600, 1.0)
        assert connector._ser is None


def test_serial_connector_init_serial_unavailable():
    """Test serial connector handles missing pyserial library."""
    import inputs.plugins.smoke_gas_detector as mod

    original = mod._SERIAL_AVAILABLE
    mod._SERIAL_AVAILABLE = False
    connector = _SerialSmokeConnector("/dev/ttyUSB0", 9600, 1.0)
    assert connector._ser is None
    mod._SERIAL_AVAILABLE = original


@pytest.mark.asyncio
async def test_serial_connector_read_valid():
    """Test serial connector parses valid SMOKE line."""
    mock_serial = MagicMock()
    mock_serial.readline.return_value = b"SMOKE:450,GAS:320\n"
    with patch(
        "inputs.plugins.smoke_gas_detector._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialSmokeConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
        assert result is not None
        assert result.smoke_ppm == 450.0
        assert result.gas_ppm == 320.0


@pytest.mark.asyncio
async def test_serial_connector_read_invalid_prefix():
    """Test serial connector ignores lines without SMOKE prefix."""
    mock_serial = MagicMock()
    mock_serial.readline.return_value = b"INVALID:data\n"
    with patch(
        "inputs.plugins.smoke_gas_detector._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialSmokeConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
        assert result is None


@pytest.mark.asyncio
async def test_serial_connector_read_no_connection():
    """Test serial connector read returns None when not connected."""
    with patch(
        "inputs.plugins.smoke_gas_detector._serial.Serial",
        side_effect=serial.SerialException("Port not found"),
    ):
        connector = _SerialSmokeConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
        assert result is None


@pytest.mark.asyncio
async def test_serial_connector_read_exception():
    """Test serial connector handles read exception gracefully."""
    mock_serial = MagicMock()
    mock_serial.readline.side_effect = Exception("Read error")
    with patch(
        "inputs.plugins.smoke_gas_detector._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialSmokeConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
        assert result is None


@pytest.mark.asyncio
async def test_serial_connector_read_parse_error():
    """Test serial connector handles parse error gracefully."""
    mock_serial = MagicMock()
    mock_serial.readline.return_value = b"SMOKE:not_a_number,GAS:bad\n"
    with patch(
        "inputs.plugins.smoke_gas_detector._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialSmokeConnector("/dev/ttyUSB0", 9600, 1.0)
        result = await connector.read()
        assert result is None


def test_serial_connector_stop():
    """Test serial connector stop closes port."""
    mock_serial = MagicMock()
    mock_serial.is_open = True
    with patch(
        "inputs.plugins.smoke_gas_detector._serial.Serial",
        return_value=mock_serial,
    ):
        connector = _SerialSmokeConnector("/dev/ttyUSB0", 9600, 1.0)
        connector.stop()
        mock_serial.close.assert_called_once()


def test_initialization_mock_connector():
    """Test SmokeGasDetector initializes with mock connector."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(connector="mock")
        sensor = SmokeGasDetector(config=config)
        assert isinstance(sensor._connector, _MockSmokeConnector)
        assert sensor.messages == []
        assert sensor.descriptor_for_LLM == "Smoke and Gas Detector"


def test_initialization_unknown_connector_falls_back_to_mock():
    """Test unknown connector falls back to mock."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(connector="unknown_hw")
        sensor = SmokeGasDetector(config=config)
        assert isinstance(sensor._connector, _MockSmokeConnector)


def test_initialization_serial_connector():
    """Test SmokeGasDetector initializes with serial connector."""
    mock_serial = MagicMock()
    with (
        patch("inputs.plugins.smoke_gas_detector.IOProvider"),
        patch(
            "inputs.plugins.smoke_gas_detector._serial.Serial",
            return_value=mock_serial,
        ),
    ):
        config = SmokeGasDetectorConfig(connector="serial", port="/dev/ttyUSB0")
        sensor = SmokeGasDetector(config=config)
        assert isinstance(sensor._connector, _SerialSmokeConnector)


def test_initialization_ens160_connector():
    """Test SmokeGasDetector initializes with ENS160 connector."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _ENS160Connector

    original = mod._ENS160_AVAILABLE
    mod._ENS160_AVAILABLE = False
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(connector="i2c_ens160")
        sensor = SmokeGasDetector(config=config)
        assert isinstance(sensor._connector, _ENS160Connector)
    mod._ENS160_AVAILABLE = original


def test_initialization_sgp30_connector():
    """Test SmokeGasDetector initializes with SGP30 connector."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _SGP30Connector

    original = mod._SGP30_AVAILABLE
    mod._SGP30_AVAILABLE = False
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(connector="i2c_sgp30")
        sensor = SmokeGasDetector(config=config)
        assert isinstance(sensor._connector, _SGP30Connector)
    mod._SGP30_AVAILABLE = original


@pytest.mark.asyncio
async def test_poll_returns_reading():
    """Test _poll returns smoke/gas reading from connector."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(connector="mock", mock_scenario="normal")
        sensor = SmokeGasDetector(config=config)
        with patch("inputs.plugins.smoke_gas_detector.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()
        assert result is not None
        assert isinstance(result, SmokeGasReading)


@pytest.mark.asyncio
async def test_poll_returns_none_when_connector_fails():
    """Test _poll returns None when connector returns None."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(connector="mock")
        sensor = SmokeGasDetector(config=config)
        sensor._connector = MagicMock()
        sensor._connector.read = AsyncMock(return_value=None)
        with patch("inputs.plugins.smoke_gas_detector.asyncio.sleep", new=AsyncMock()):
            result = await sensor._poll()
        assert result is None


def test_classify_danger():
    """Test _classify returns danger for critical ppm."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(connector="mock", smoke_danger_threshold=600)
        sensor = SmokeGasDetector(config=config)
        reading = SmokeGasReading(smoke_ppm=750.0, gas_ppm=40.0)
        assert sensor._classify(reading) == "danger"


def test_classify_danger_via_gas():
    """Test _classify returns danger when gas exceeds threshold."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(connector="mock", gas_danger_threshold=600)
        sensor = SmokeGasDetector(config=config)
        reading = SmokeGasReading(smoke_ppm=50.0, gas_ppm=700.0)
        assert sensor._classify(reading) == "danger"


def test_classify_warning():
    """Test _classify returns warning for elevated ppm."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(
            connector="mock",
            smoke_warning_threshold=300,
            smoke_danger_threshold=600,
        )
        sensor = SmokeGasDetector(config=config)
        reading = SmokeGasReading(smoke_ppm=400.0, gas_ppm=40.0)
        assert sensor._classify(reading) == "warning"


def test_classify_normal():
    """Test _classify returns normal for low ppm."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(connector="mock")
        sensor = SmokeGasDetector(config=config)
        reading = SmokeGasReading(smoke_ppm=50.0, gas_ppm=40.0)
        assert sensor._classify(reading) == "normal"


@pytest.mark.asyncio
async def test_raw_to_text_danger():
    """Test _raw_to_text returns danger alert message."""
    with (
        patch("inputs.plugins.smoke_gas_detector.IOProvider"),
        patch("inputs.plugins.smoke_gas_detector.time.time", return_value=1000.0),
    ):
        config = SmokeGasDetectorConfig(connector="mock", smoke_danger_threshold=600)
        sensor = SmokeGasDetector(config=config)
        reading = SmokeGasReading(smoke_ppm=750.0, gas_ppm=700.0)
        result = await sensor._raw_to_text(reading)
        assert result is not None
        assert "SMOKE ALERT" in result.message
        assert "750" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_warning():
    """Test _raw_to_text returns warning message."""
    with (
        patch("inputs.plugins.smoke_gas_detector.IOProvider"),
        patch("inputs.plugins.smoke_gas_detector.time.time", return_value=1000.0),
    ):
        config = SmokeGasDetectorConfig(
            connector="mock",
            smoke_warning_threshold=300,
            smoke_danger_threshold=600,
        )
        sensor = SmokeGasDetector(config=config)
        reading = SmokeGasReading(smoke_ppm=400.0, gas_ppm=40.0)
        result = await sensor._raw_to_text(reading)
        assert result is not None
        assert "SMOKE WARNING" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_normal():
    """Test _raw_to_text returns normal message for low ppm."""
    with (
        patch("inputs.plugins.smoke_gas_detector.IOProvider"),
        patch("inputs.plugins.smoke_gas_detector.time.time", return_value=1000.0),
    ):
        config = SmokeGasDetectorConfig(connector="mock")
        sensor = SmokeGasDetector(config=config)
        reading = SmokeGasReading(smoke_ppm=50.0, gas_ppm=40.0)
        result = await sensor._raw_to_text(reading)
        assert result is not None
        assert "Air quality normal" in result.message


@pytest.mark.asyncio
async def test_raw_to_text_none_input():
    """Test _raw_to_text returns None for None input."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(connector="mock")
        sensor = SmokeGasDetector(config=config)
        result = await sensor._raw_to_text(None)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_cooldown_suppresses_danger():
    """Test cooldown suppresses repeated danger alerts."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(
            connector="mock", cooldown=5.0, smoke_danger_threshold=600
        )
        sensor = SmokeGasDetector(config=config)
        reading = SmokeGasReading(smoke_ppm=750.0, gas_ppm=700.0)
        sensor._last_alert_time = 1000.0
        with patch("inputs.plugins.smoke_gas_detector.time.time", return_value=1001.0):
            result = await sensor._raw_to_text(reading)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_cooldown_allows_after_expiry():
    """Test danger alert allowed again after cooldown expires."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(
            connector="mock", cooldown=5.0, smoke_danger_threshold=600
        )
        sensor = SmokeGasDetector(config=config)
        reading = SmokeGasReading(smoke_ppm=750.0, gas_ppm=700.0)
        sensor._last_alert_time = 1000.0
        with patch("inputs.plugins.smoke_gas_detector.time.time", return_value=1006.0):
            result = await sensor._raw_to_text(reading)
        assert result is not None


@pytest.mark.asyncio
async def test_raw_to_text_cooldown_suppresses_warning():
    """Test cooldown suppresses repeated warning messages."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(
            connector="mock",
            cooldown=5.0,
            smoke_warning_threshold=300,
            smoke_danger_threshold=600,
        )
        sensor = SmokeGasDetector(config=config)
        reading = SmokeGasReading(smoke_ppm=400.0, gas_ppm=40.0)
        sensor._last_alert_time = 1000.0
        with patch("inputs.plugins.smoke_gas_detector.time.time", return_value=1001.0):
            result = await sensor._raw_to_text(reading)
        assert result is None


@pytest.mark.asyncio
async def test_raw_to_text_updates_messages():
    """Test raw_to_text appends to messages buffer."""
    with (
        patch("inputs.plugins.smoke_gas_detector.IOProvider"),
        patch("inputs.plugins.smoke_gas_detector.time.time", return_value=1000.0),
    ):
        config = SmokeGasDetectorConfig(connector="mock")
        sensor = SmokeGasDetector(config=config)
        reading = SmokeGasReading(smoke_ppm=50.0, gas_ppm=40.0)
        await sensor.raw_to_text(reading)
        assert len(sensor.messages) == 1


def test_formatted_latest_buffer_with_messages():
    """Test formatted_latest_buffer returns formatted string and clears buffer."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(connector="mock")
        sensor = SmokeGasDetector(config=config)
        sensor.io_provider = MagicMock()
        sensor.messages = [
            Message(
                timestamp=1000.0,
                message="Smoke/gas detector: Air quality normal. Smoke: 50 ppm, Gas: 40 ppm.",
            )
        ]
        result = sensor.formatted_latest_buffer()
        assert result is not None
        assert "Smoke and Gas Detector" in result
        assert "Air quality normal" in result
        sensor.io_provider.add_input.assert_called_once()
        assert len(sensor.messages) == 0


def test_formatted_latest_buffer_empty():
    """Test formatted_latest_buffer returns None when buffer is empty."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(connector="mock")
        sensor = SmokeGasDetector(config=config)
        result = sensor.formatted_latest_buffer()
        assert result is None


def test_stop_calls_connector_stop():
    """Test stop calls connector stop method."""
    with patch("inputs.plugins.smoke_gas_detector.IOProvider"):
        config = SmokeGasDetectorConfig(connector="mock")
        sensor = SmokeGasDetector(config=config)
        sensor._connector = MagicMock()
        sensor.stop()
        sensor._connector.stop.assert_called_once()
        assert sensor.messages == []


def test_ens160_connector_init_library_unavailable():
    """Test ENS160 connector handles missing library."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _ENS160Connector

    original = mod._ENS160_AVAILABLE
    mod._ENS160_AVAILABLE = False
    connector = _ENS160Connector()
    assert connector._ready is False
    mod._ENS160_AVAILABLE = original


@pytest.mark.asyncio
async def test_ens160_connector_read_not_ready():
    """Test ENS160 connector returns None when not ready."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _ENS160Connector

    original = mod._ENS160_AVAILABLE
    mod._ENS160_AVAILABLE = False
    connector = _ENS160Connector()
    result = await connector.read()
    assert result is None
    mod._ENS160_AVAILABLE = original


@pytest.mark.asyncio
async def test_ens160_connector_init_success():
    """Test ENS160 connector initializes with mocked hardware."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _ENS160Connector

    original = mod._ENS160_AVAILABLE
    original_lib = mod._ens160_lib
    mock_lib = MagicMock()
    mock_sensor = MagicMock()
    mock_lib.ENS160.return_value = mock_sensor
    mod._ENS160_AVAILABLE = True
    mod._ens160_lib = mock_lib
    with patch.dict(
        "sys.modules",
        {"board": MagicMock(), "busio": MagicMock(I2C=MagicMock())},
    ):
        connector = _ENS160Connector()
        assert connector._ready is True
    mod._ENS160_AVAILABLE = original
    mod._ens160_lib = original_lib


@pytest.mark.asyncio
async def test_ens160_connector_read_success():
    """Test ENS160 connector reads successfully."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _ENS160Connector

    original = mod._ENS160_AVAILABLE
    original_lib = mod._ens160_lib
    mock_lib = MagicMock()
    mock_sensor = MagicMock()
    mock_sensor.TVOC = 120
    mock_sensor.eCO2 = 450
    mock_lib.ENS160.return_value = mock_sensor
    mod._ENS160_AVAILABLE = True
    mod._ens160_lib = mock_lib
    with patch.dict(
        "sys.modules",
        {"board": MagicMock(), "busio": MagicMock(I2C=MagicMock())},
    ):
        connector = _ENS160Connector()
        result = await connector.read()
        assert result is not None
        assert result.smoke_ppm == 120.0
    mod._ENS160_AVAILABLE = original
    mod._ens160_lib = original_lib


@pytest.mark.asyncio
async def test_ens160_connector_read_exception():
    """Test ENS160 connector handles read exception."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _ENS160Connector

    original = mod._ENS160_AVAILABLE
    original_lib = mod._ens160_lib
    mock_lib = MagicMock()
    mock_sensor = MagicMock()
    type(mock_sensor).TVOC = property(
        lambda self: (_ for _ in ()).throw(Exception("Read error"))
    )
    mock_lib.ENS160.return_value = mock_sensor
    mod._ENS160_AVAILABLE = True
    mod._ens160_lib = mock_lib
    with patch.dict(
        "sys.modules",
        {"board": MagicMock(), "busio": MagicMock(I2C=MagicMock())},
    ):
        connector = _ENS160Connector()
        result = await connector.read()
        assert result is None
    mod._ENS160_AVAILABLE = original
    mod._ens160_lib = original_lib


def test_ens160_connector_stop():
    """Test ENS160 connector stop."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _ENS160Connector

    original = mod._ENS160_AVAILABLE
    mod._ENS160_AVAILABLE = False
    connector = _ENS160Connector()
    connector.stop()
    assert connector._ready is False
    mod._ENS160_AVAILABLE = original


def test_sgp30_connector_init_library_unavailable():
    """Test SGP30 connector handles missing library."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _SGP30Connector

    original = mod._SGP30_AVAILABLE
    mod._SGP30_AVAILABLE = False
    connector = _SGP30Connector()
    assert connector._ready is False
    mod._SGP30_AVAILABLE = original


@pytest.mark.asyncio
async def test_sgp30_connector_read_not_ready():
    """Test SGP30 connector returns None when not ready."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _SGP30Connector

    original = mod._SGP30_AVAILABLE
    mod._SGP30_AVAILABLE = False
    connector = _SGP30Connector()
    result = await connector.read()
    assert result is None
    mod._SGP30_AVAILABLE = original


@pytest.mark.asyncio
async def test_sgp30_connector_init_success():
    """Test SGP30 connector initializes with mocked hardware."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _SGP30Connector

    original = mod._SGP30_AVAILABLE
    original_lib = mod._sgp30_lib
    mock_lib = MagicMock()
    mock_sensor = MagicMock()
    mock_lib.Adafruit_SGP30.return_value = mock_sensor
    mod._SGP30_AVAILABLE = True
    mod._sgp30_lib = mock_lib
    with patch.dict(
        "sys.modules",
        {"board": MagicMock(), "busio": MagicMock(I2C=MagicMock())},
    ):
        connector = _SGP30Connector()
        assert connector._ready is True
    mod._SGP30_AVAILABLE = original
    mod._sgp30_lib = original_lib


@pytest.mark.asyncio
async def test_sgp30_connector_read_success():
    """Test SGP30 connector reads successfully."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _SGP30Connector

    original = mod._SGP30_AVAILABLE
    original_lib = mod._sgp30_lib
    mock_lib = MagicMock()
    mock_sensor = MagicMock()
    mock_sensor.TVOC = 80
    mock_sensor.eCO2 = 400
    mock_lib.Adafruit_SGP30.return_value = mock_sensor
    mod._SGP30_AVAILABLE = True
    mod._sgp30_lib = mock_lib
    with patch.dict(
        "sys.modules",
        {"board": MagicMock(), "busio": MagicMock(I2C=MagicMock())},
    ):
        connector = _SGP30Connector()
        result = await connector.read()
        assert result is not None
        assert result.smoke_ppm == 80.0
    mod._SGP30_AVAILABLE = original
    mod._sgp30_lib = original_lib


@pytest.mark.asyncio
async def test_sgp30_connector_read_exception():
    """Test SGP30 connector handles read exception."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _SGP30Connector

    original = mod._SGP30_AVAILABLE
    original_lib = mod._sgp30_lib
    mock_lib = MagicMock()
    mock_sensor = MagicMock()
    type(mock_sensor).TVOC = property(
        lambda self: (_ for _ in ()).throw(Exception("Read error"))
    )
    mock_lib.Adafruit_SGP30.return_value = mock_sensor
    mod._SGP30_AVAILABLE = True
    mod._sgp30_lib = mock_lib
    with patch.dict(
        "sys.modules",
        {"board": MagicMock(), "busio": MagicMock(I2C=MagicMock())},
    ):
        connector = _SGP30Connector()
        result = await connector.read()
        assert result is None
    mod._SGP30_AVAILABLE = original
    mod._sgp30_lib = original_lib


def test_sgp30_connector_stop():
    """Test SGP30 connector stop."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _SGP30Connector

    original = mod._SGP30_AVAILABLE
    mod._SGP30_AVAILABLE = False
    connector = _SGP30Connector()
    connector.stop()
    assert connector._ready is False
    mod._SGP30_AVAILABLE = original


@pytest.mark.asyncio
async def test_ens160_connector_init_exception():
    """Test ENS160 connector handles init exception gracefully."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _ENS160Connector

    original = mod._ENS160_AVAILABLE
    original_lib = mod._ens160_lib
    mock_lib = MagicMock()
    mock_lib.ENS160.side_effect = Exception("I2C error")
    mod._ENS160_AVAILABLE = True
    mod._ens160_lib = mock_lib
    with patch.dict(
        "sys.modules",
        {"board": MagicMock(), "busio": MagicMock(I2C=MagicMock())},
    ):
        connector = _ENS160Connector()
        assert connector._ready is False
    mod._ENS160_AVAILABLE = original
    mod._ens160_lib = original_lib


@pytest.mark.asyncio
async def test_sgp30_connector_init_exception():
    """Test SGP30 connector handles init exception gracefully."""
    import inputs.plugins.smoke_gas_detector as mod
    from inputs.plugins.smoke_gas_detector import _SGP30Connector

    original = mod._SGP30_AVAILABLE
    original_lib = mod._sgp30_lib
    mock_lib = MagicMock()
    mock_lib.Adafruit_SGP30.side_effect = Exception("I2C error")
    mod._SGP30_AVAILABLE = True
    mod._sgp30_lib = mock_lib
    with patch.dict(
        "sys.modules",
        {"board": MagicMock(), "busio": MagicMock(I2C=MagicMock())},
    ):
        connector = _SGP30Connector()
        assert connector._ready is False
    mod._SGP30_AVAILABLE = original
    mod._sgp30_lib = original_lib
