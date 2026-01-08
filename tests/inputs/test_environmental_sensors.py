import time

import pytest

from inputs.plugins.environmental_sensors import (
    EnvironmentalSensors,
    EnvironmentalSensorsConfig,
)


class TestEnvironmentalSensorsConfig:
    """Tests for EnvironmentalSensorsConfig."""

    def test_config_defaults(self):
        """Test configuration defaults."""
        config = EnvironmentalSensorsConfig()
        assert config.use_zenoh is False
        assert config.update_interval == 1.0
        assert config.temperature_topic is None

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = EnvironmentalSensorsConfig(
            use_zenoh=True,
            temperature_topic="sensors/temperature",
            update_interval=2.0,
        )
        assert config.use_zenoh is True
        assert config.temperature_topic == "sensors/temperature"
        assert config.update_interval == 2.0


class TestEnvironmentalSensors:
    """Tests for EnvironmentalSensors input plugin."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test initializing EnvironmentalSensors."""
        config = EnvironmentalSensorsConfig(use_zenoh=False)
        sensor = EnvironmentalSensors(config)
        assert sensor.descriptor_for_LLM == "Environmental Sensors"
        assert sensor.use_zenoh is False

    @pytest.mark.asyncio
    async def test_generate_mock_data(self):
        """Test generating mock sensor data."""
        config = EnvironmentalSensorsConfig(use_zenoh=False)
        sensor = EnvironmentalSensors(config)
        data = sensor._generate_mock_data()
        assert "temperature" in data
        assert "humidity" in data
        assert "air_quality" in data
        assert "timestamp" in data
        assert isinstance(data["temperature"], float)
        assert isinstance(data["humidity"], float)

    @pytest.mark.asyncio
    async def test_poll_without_zenoh(self):
        """Test polling without Zenoh (mock data)."""
        config = EnvironmentalSensorsConfig(use_zenoh=False, update_interval=0.1)
        sensor = EnvironmentalSensors(config)
        data = await sensor._poll()
        assert data is not None
        assert "temperature" in data
        assert "humidity" in data
        assert "air_quality" in data

    @pytest.mark.asyncio
    async def test_raw_to_text(self):
        """Test converting raw data to text message."""
        config = EnvironmentalSensorsConfig(use_zenoh=False)
        sensor = EnvironmentalSensors(config)
        raw_data = {
            "temperature": 25.5,
            "humidity": 60.0,
            "air_quality": 100.0,
            "timestamp": time.time(),
        }
        message = await sensor._raw_to_text(raw_data)
        assert message is not None
        assert "Temperature: 25.5°C" in message.message
        assert "Humidity: 60.0%" in message.message
        assert "Air Quality Index: 100.0" in message.message

    @pytest.mark.asyncio
    async def test_raw_to_text_with_none_values(self):
        """Test converting raw data with None values."""
        config = EnvironmentalSensorsConfig(use_zenoh=False)
        sensor = EnvironmentalSensors(config)
        raw_data = {"temperature": 25.5, "timestamp": time.time()}
        message = await sensor._raw_to_text(raw_data)
        assert message is not None
        assert "Temperature: 25.5°C" in message.message
        assert "Humidity" not in message.message

    @pytest.mark.asyncio
    async def test_raw_to_text_with_empty_data(self):
        """Test converting empty raw data."""
        config = EnvironmentalSensorsConfig(use_zenoh=False)
        sensor = EnvironmentalSensors(config)
        message = await sensor._raw_to_text(None)
        assert message is None

    @pytest.mark.asyncio
    async def test_formatted_latest_buffer(self):
        """Test formatting latest buffer."""
        config = EnvironmentalSensorsConfig(use_zenoh=False)
        sensor = EnvironmentalSensors(config)
        raw_data = {
            "temperature": 25.5,
            "humidity": 60.0,
            "timestamp": time.time(),
        }
        await sensor.raw_to_text(raw_data)
        formatted = sensor.formatted_latest_buffer()
        assert formatted is not None
        assert "Environmental Sensors" in formatted
        assert "Temperature" in formatted

    @pytest.mark.asyncio
    async def test_formatted_latest_buffer_empty(self):
        """Test formatting empty buffer."""
        config = EnvironmentalSensorsConfig(use_zenoh=False)
        sensor = EnvironmentalSensors(config)
        formatted = sensor.formatted_latest_buffer()
        assert formatted is None
