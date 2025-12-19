"""Tests for BME280 sensor plugin."""

import pytest

from inputs.plugins.bme280 import BME280Config, BME280Input


def test_bme280_config_default():
    """Test BME280Config with default values."""
    config = BME280Config()
    assert config.i2c_address == 0x76
    assert config.sampling_rate == 1.0


def test_bme280_config_custom():
    """Test BME280Config with custom values."""
    config = BME280Config(i2c_address=0x77, sampling_rate=2.0)
    assert config.i2c_address == 0x77
    assert config.sampling_rate == 2.0


def test_bme280_initialization():
    """Test BME280Input initialization."""
    config = BME280Config()
    sensor = BME280Input(config)
    assert sensor.config == config
    assert isinstance(sensor, BME280Input)


def test_bme280_mock_data():
    """Test BME280Input mock data generation."""
    config = BME280Config()
    sensor = BME280Input(config)
    data = sensor._mock_data()

    # Check all required fields exist
    assert "temperature" in data
    assert "humidity" in data
    assert "pressure" in data
    assert "timestamp" in data
    assert data["mock"] is True

    # Check reasonable ranges
    assert 20 <= data["temperature"] <= 30
    assert 40 <= data["humidity"] <= 60
    assert 1000 <= data["pressure"] <= 1020


def test_bme280_read_sensor():
    """Test BME280Input sensor reading (mock mode)."""
    config = BME280Config()
    sensor = BME280Input(config)
    data = sensor._read_sensor()

    # Should return mock data when no hardware
    assert "temperature" in data
    assert "humidity" in data
    assert "pressure" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_bme280_raw_to_text():
    """Test raw data to text conversion."""
    config = BME280Config()
    sensor = BME280Input(config)

    raw_data = {
        "temperature": 25.5,
        "humidity": 45.0,
        "pressure": 1013.25,
        "timestamp": 1703001234.56,
        "mock": True,
    }

    message = await sensor._raw_to_text(raw_data)
    assert message is not None
    assert "25.5°C" in message.message
    assert "45.0%" in message.message
    assert "1013.25 hPa" in message.message
    assert "(mock data)" in message.message
    assert message.timestamp == 1703001234.56


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
