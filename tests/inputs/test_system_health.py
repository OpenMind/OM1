import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch
from inputs.plugins.system_health_input import SystemHealthInput
from inputs.base import SensorConfig

@pytest.mark.asyncio
async def test_system_health_initialization():
    """Test that the sensor initializes with default config."""
    config = SensorConfig(input_name="TestHealth", interval=1.0)
    sensor = SystemHealthInput(config)
    assert sensor.descriptor_for_LLM == "TestHealth"
    assert sensor.interval == 1.0

@pytest.mark.asyncio
async def test_system_health_poll():
    """Test data collection logic by mocking psutil."""
    config = SensorConfig()
    sensor = SystemHealthInput(config)
    
    # Mock psutil to verify data extraction
    with patch('psutil.cpu_percent', return_value=50.0), \
         patch('psutil.virtual_memory') as mock_mem, \
         patch('psutil.disk_usage') as mock_disk, \
         patch('psutil.sensors_battery') as mock_bat:
        
        mock_mem.return_value.percent = 60.0
        mock_disk.return_value.percent = 70.0
        mock_bat.return_value.percent = 80.0
        mock_bat.return_value.power_plugged = True

        sensor.interval = 0
        
        result_json = await sensor._poll()
        
        assert result_json is not None
        data = json.loads(result_json)
        
        assert data["cpu_usage_percent"] == 50.0
        assert data["ram_usage_percent"] == 60.0
        assert data["disk_usage_percent"] == 70.0
        assert data["battery_level"] == 80.0
        assert data["power_status"] == "Charging"