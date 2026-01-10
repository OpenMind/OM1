"""
Unit tests for the Gps input plugin in inputs/plugins/gps.py.

Tests cover:
- Initialization
- Message processing (coordinate formatting)
- Buffer management handling
- Integration with IO provider
"""

from unittest.mock import patch

import pytest

from inputs.base import SensorConfig
from inputs.plugins.gps import Gps


class TestGpsInitialization:
    """Tests for Gps class initialization."""

    def test_initialization(self):
        """Test that GPS is initialized correctly with mocks."""
        with patch("inputs.plugins.gps.GpsProvider") as MockGpsProvider, \
             patch("inputs.plugins.gps.IOProvider") as MockIOProvider:
            
            config = SensorConfig(name="gps")
            gps_input = Gps(config)
            
            assert gps_input.descriptor_for_LLM == "GPS Location"
            assert gps_input.gps is MockGpsProvider.return_value
            assert gps_input.io_provider is MockIOProvider.return_value
            assert gps_input.messages == []


@pytest.mark.asyncio
class TestGpsRawToText:
    """Tests for raw_to_text method logic."""

    @pytest.fixture
    def gps_input(self):
        with patch("inputs.plugins.gps.GpsProvider"), \
             patch("inputs.plugins.gps.IOProvider"):
            return Gps(SensorConfig(name="gps"))

    async def test_raw_to_text_valid_data_north_east(self, gps_input):
        """Test conversion of valid GPS data (North, East)."""
        raw_data = {
            "gps_lat": 40.0,
            "gps_lon": 29.0,
            "gps_alt": 100.0,
            "gps_qua": 1  # Quality indicator > 0
        }
        
        message = await gps_input._raw_to_text(raw_data)
        
        assert message is not None
        assert "40.0 North" in message.message
        assert "29.0 East" in message.message
        assert "100.0m altitude" in message.message

    async def test_raw_to_text_valid_data_south_west(self, gps_input):
        """Test conversion of valid GPS data (South, West)."""
        raw_data = {
            "gps_lat": -40.0,
            "gps_lon": -29.0,
            "gps_alt": 100.0,
            "gps_qua": 1
        }
        
        message = await gps_input._raw_to_text(raw_data)
        
        # Original code multiplies by -1.0 for output
        assert message is not None
        assert "40.0 South" in message.message
        assert "29.0 West" in message.message

    async def test_raw_to_text_invalid_quality(self, gps_input):
        """Test that data with quality <= 0 is ignored."""
        raw_data = {
            "gps_lat": 40.0,
            "gps_lon": 29.0,
            "gps_alt": 100.0,
            "gps_qua": 0  # Invalid quality
        }
        
        message = await gps_input._raw_to_text(raw_data)
        assert message is None

    async def test_raw_to_text_none_input(self, gps_input):
        """Test handling of None input."""
        message = await gps_input._raw_to_text(None)
        assert message is None


@pytest.mark.asyncio
class TestGpsBufferProcessing:
    """Tests for buffer and update logic."""

    @pytest.fixture
    def gps_input(self):
        with patch("inputs.plugins.gps.GpsProvider"), \
             patch("inputs.plugins.gps.IOProvider"):
            return Gps(SensorConfig(name="gps"))

    async def test_raw_to_text_updates_buffer(self, gps_input):
        """Test that valid messages are appended to buffer."""
        raw_data = {
            "gps_lat": 40.0,
            "gps_lon": 29.0,
            "gps_alt": 100.0,
            "gps_qua": 1
        }
        
        await gps_input.raw_to_text(raw_data)
        assert len(gps_input.messages) == 1
        assert "North" in gps_input.messages[0].message

    async def test_raw_to_text_ignores_none(self, gps_input):
        """Test that None messages are not appended."""
        await gps_input.raw_to_text(None)
        assert len(gps_input.messages) == 0

    def test_formatted_latest_buffer_returns_formatted_string(self, gps_input):
        """Test formatting of the latest message."""
        raw_data = {
            "gps_lat": 40.0,
            "gps_lon": 29.0,
            "gps_alt": 100.0,
            "gps_qua": 1
        }
        
        # Manually verify loop-less behavior for test
        loop = __import__("asyncio").new_event_loop()
        loop.run_until_complete(gps_input.raw_to_text(raw_data))
        
        result = gps_input.formatted_latest_buffer()
        
        assert result is not None
        assert "INPUT: GPS Location" in result
        assert "// START" in result
        assert "40.0 North" in result
        assert "// END" in result
        
        # Verify buffer is cleared
        assert len(gps_input.messages) == 0
        
        # Verify IO provider called
        gps_input.io_provider.add_input.assert_called_once()

    def test_formatted_latest_buffer_empty(self, gps_input):
        """Test formatted_latest_buffer with empty messages."""
        result = gps_input.formatted_latest_buffer()
        assert result is None
