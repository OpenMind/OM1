"""Test BME280 plugin following OM1 style."""

import pytest

from inputs.plugins.bme280 import BME280Config, BME280Input


def test_om1_style_initialization():
    """Test that BME280 follows OM1 input plugin style."""
    config = BME280Config()
    sensor = BME280Input(config)

    # Check that it has required methods
    assert hasattr(sensor, "read") or hasattr(sensor, "_read_sensor")
    assert hasattr(sensor, "_listen_loop")
    assert hasattr(sensor, "_raw_to_text")


@pytest.mark.asyncio
async def test_om1_style_listen():
    """Test async listen pattern."""
    config = BME280Config(sampling_rate=0.1)
    sensor = BME280Input(config)

    # Test that listen yields data
    count = 0
    async for data in sensor.listen():
        assert "temperature" in data
        assert "humidity" in data
        assert "pressure" in data
        count += 1
        if count >= 2:  # Test 2 readings
            break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
