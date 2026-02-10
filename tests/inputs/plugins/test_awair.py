import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.plugins.awair import AwairConfig

# Sample AWAIR data for testing
SAMPLE_AWAIR_DATA = {
    "timestamp": "2025-12-03T13:58:59.121Z",
    "score": 74,
    "dew_point": 8.60,
    "temp": 27.19,
    "humid": 31.02,
    "abs_humid": 8.05,
    "co2": 822,
    "co2_est": 526,
    "co2_est_baseline": 36188,
    "voc": 1848,
    "voc_baseline": 39534,
    "voc_h2_raw": 25,
    "voc_ethanol_raw": 36,
    "pm25": 2,
    "pm10_est": 3,
}


@pytest.fixture
def mock_local_config():
    """Create a mock configuration for Local API mode."""
    return AwairConfig(
        mode="local",
        device_ip="192.168.0.17",
        poll_interval=1.0,
    )


@pytest.fixture
def mock_cloud_config():
    """Create a mock configuration for Cloud API mode."""
    return AwairConfig(
        mode="cloud",
        access_token="test_token_12345",
        device_id="12345",
        device_type="awair-element",
        poll_interval=1.0,
    )


@pytest.fixture
def awair_local_plugin(mock_local_config):
    """Create an AwairElement instance in local mode."""
    from inputs.plugins import awair as awair_module

    with patch.object(awair_module, "IOProvider"):
        return awair_module.AwairElement(mock_local_config)


@pytest.fixture
def awair_cloud_plugin(mock_cloud_config):
    """Create an AwairElement instance in cloud mode."""
    from inputs.plugins import awair as awair_module

    with patch.object(awair_module, "IOProvider"):
        return awair_module.AwairElement(mock_cloud_config)


class TestAwairConfiguration:
    """Tests for AWAIR plugin configuration."""

    def test_init_local_mode(self, awair_local_plugin):
        """Test initialization in local mode."""
        assert awair_local_plugin.mode == "local"
        assert awair_local_plugin.device_ip == "192.168.0.17"
        assert awair_local_plugin.poll_interval == 1.0

    def test_init_cloud_mode(self, mock_cloud_config):
        """Test initialization in cloud mode."""
        with patch("inputs.plugins.awair.IOProvider"):
            from inputs.plugins.awair import AwairElement

            plugin = AwairElement(mock_cloud_config)

            assert plugin.mode == "cloud"
            assert plugin.access_token == "test_token_12345"
            assert plugin.device_id == "12345"

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        with patch("inputs.plugins.awair.IOProvider"):
            from inputs.plugins.awair import AwairElement

            plugin = AwairElement(AwairConfig())

            assert plugin.mode == "local"
            assert plugin.device_ip is None
            assert plugin.poll_interval == 10.0

    def test_descriptor_for_llm(self, awair_local_plugin):
        """Test that the LLM descriptor is set correctly."""
        assert (
            awair_local_plugin.descriptor_for_LLM
            == "Indoor Air Quality (AWAIR Element)"
        )


class TestDataParsing:
    """Tests for AWAIR data parsing."""

    def test_parse_data(self, awair_local_plugin):
        """Test parsing raw API response."""
        data = awair_local_plugin._parse_data(SAMPLE_AWAIR_DATA)

        assert data.score == 74
        assert data.temp == 27.19
        assert data.humid == 31.02
        assert data.co2 == 822
        assert data.voc == 1848
        assert data.pm25 == 2

    def test_parse_data_missing_fields(self, awair_local_plugin):
        """Test parsing with missing fields uses defaults."""
        incomplete_data = {"score": 50, "temp": 22.0}
        data = awair_local_plugin._parse_data(incomplete_data)

        assert data.score == 50
        assert data.temp == 22.0
        assert data.humid == 0.0
        assert data.co2 == 0
        assert data.voc == 0
        assert data.pm25 == 0

    def test_parse_data_empty(self, awair_local_plugin):
        """Test parsing empty data."""
        data = awair_local_plugin._parse_data({})

        assert data.score == 0
        assert data.temp == 0.0


class TestScoreDescription:
    """Tests for AWAIR score description."""

    def test_score_excellent(self, awair_local_plugin):
        """Test excellent score description."""
        assert awair_local_plugin._get_score_description(95) == "Excellent"

    def test_score_good(self, awair_local_plugin):
        """Test good score description."""
        assert awair_local_plugin._get_score_description(85) == "Good"

    def test_score_fair(self, awair_local_plugin):
        """Test fair score description."""
        assert awair_local_plugin._get_score_description(65) == "Fair"

    def test_score_poor(self, awair_local_plugin):
        """Test poor score description."""
        assert awair_local_plugin._get_score_description(45) == "Poor"

    def test_score_unhealthy(self, awair_local_plugin):
        """Test unhealthy score description."""
        assert awair_local_plugin._get_score_description(30) == "Unhealthy"


class TestSignificantChangeDetection:
    """Tests for detecting significant changes in air quality."""

    def test_first_reading_is_significant(self, awair_local_plugin):
        """Test that first reading is always significant."""
        from inputs.plugins.awair import AwairData

        data = AwairData(
            timestamp="",
            score=74,
            temp=27.0,
            humid=31.0,
            abs_humid=0,
            dew_point=0,
            co2=822,
            co2_est=0,
            voc=1848,
            voc_baseline=0,
            pm25=2,
            pm10_est=0,
        )

        assert awair_local_plugin._has_significant_change(data, None) is True

    def test_no_change(self, awair_local_plugin):
        """Test no significant change for similar readings."""
        from inputs.plugins.awair import AwairData

        data1 = AwairData(
            timestamp="",
            score=74,
            temp=27.0,
            humid=31.0,
            abs_humid=0,
            dew_point=0,
            co2=822,
            co2_est=0,
            voc=1848,
            voc_baseline=0,
            pm25=2,
            pm10_est=0,
        )

        data2 = AwairData(
            timestamp="",
            score=75,
            temp=27.1,
            humid=31.5,
            abs_humid=0,
            dew_point=0,
            co2=830,
            co2_est=0,
            voc=1850,
            voc_baseline=0,
            pm25=2,
            pm10_est=0,
        )

        assert awair_local_plugin._has_significant_change(data2, data1) is False

    def test_score_change_significant(self, awair_local_plugin):
        """Test score change of 10+ is significant."""
        from inputs.plugins.awair import AwairData

        data1 = AwairData(
            timestamp="",
            score=74,
            temp=27.0,
            humid=31.0,
            abs_humid=0,
            dew_point=0,
            co2=822,
            co2_est=0,
            voc=1848,
            voc_baseline=0,
            pm25=2,
            pm10_est=0,
        )

        data2 = AwairData(
            timestamp="",
            score=60,
            temp=27.0,
            humid=31.0,
            abs_humid=0,
            dew_point=0,
            co2=822,
            co2_est=0,
            voc=1848,
            voc_baseline=0,
            pm25=2,
            pm10_est=0,
        )

        assert awair_local_plugin._has_significant_change(data2, data1) is True

    def test_temp_change_significant(self, awair_local_plugin):
        """Test temperature change of 2°C+ is significant."""
        from inputs.plugins.awair import AwairData

        data1 = AwairData(
            timestamp="",
            score=74,
            temp=22.0,
            humid=31.0,
            abs_humid=0,
            dew_point=0,
            co2=822,
            co2_est=0,
            voc=1848,
            voc_baseline=0,
            pm25=2,
            pm10_est=0,
        )

        data2 = AwairData(
            timestamp="",
            score=74,
            temp=25.0,
            humid=31.0,
            abs_humid=0,
            dew_point=0,
            co2=822,
            co2_est=0,
            voc=1848,
            voc_baseline=0,
            pm25=2,
            pm10_est=0,
        )

        assert awair_local_plugin._has_significant_change(data2, data1) is True

    def test_co2_change_significant(self, awair_local_plugin):
        """Test CO2 change of 200+ ppm is significant."""
        from inputs.plugins.awair import AwairData

        data1 = AwairData(
            timestamp="",
            score=74,
            temp=22.0,
            humid=31.0,
            abs_humid=0,
            dew_point=0,
            co2=600,
            co2_est=0,
            voc=1848,
            voc_baseline=0,
            pm25=2,
            pm10_est=0,
        )

        data2 = AwairData(
            timestamp="",
            score=74,
            temp=22.0,
            humid=31.0,
            abs_humid=0,
            dew_point=0,
            co2=850,
            co2_est=0,
            voc=1848,
            voc_baseline=0,
            pm25=2,
            pm10_est=0,
        )

        assert awair_local_plugin._has_significant_change(data2, data1) is True


class TestMessageFormatting:
    """Tests for message formatting."""

    @pytest.mark.asyncio
    async def test_raw_to_text(self, awair_local_plugin):
        """Test converting raw data to text message."""
        message = await awair_local_plugin._raw_to_text(SAMPLE_AWAIR_DATA)

        assert message is not None
        assert "Room Temperature:" in message.message
        assert "27.2" in message.message or "27.1" in message.message  # Temp
        assert "Humidity:" in message.message
        assert "Air Quality Score:" in message.message

    @pytest.mark.asyncio
    async def test_raw_to_text_empty_input(self, awair_local_plugin):
        """Test that empty input returns None."""
        message = await awair_local_plugin._raw_to_text({})
        assert message is None

    @pytest.mark.asyncio
    async def test_raw_to_text_none_input(self, awair_local_plugin):
        """Test that None input returns None."""
        message = await awair_local_plugin._raw_to_text(None)
        assert message is None

    @pytest.mark.asyncio
    async def test_buffer_update(self, awair_local_plugin):
        """Test that the message buffer is updated correctly."""
        await awair_local_plugin.raw_to_text(SAMPLE_AWAIR_DATA)
        assert len(awair_local_plugin.messages) == 1

    def test_formatted_latest_buffer(self, awair_local_plugin):
        """Test formatting the latest buffer for LLM."""
        from inputs.plugins.awair import Message

        awair_local_plugin.messages = [
            Message(
                timestamp=time.time(),
                message="Room Temperature: 22.0°C\nHumidity: 45%\nAir Quality Score: 74/100",
            )
        ]

        result = awair_local_plugin.formatted_latest_buffer()

        assert result is not None
        assert "INPUT: Indoor Air Quality (AWAIR Element)" in result
        assert "Room Temperature:" in result
        assert len(awair_local_plugin.messages) == 0  # Buffer cleared

    def test_formatted_latest_buffer_empty(self, awair_local_plugin):
        """Test that empty buffer returns None."""
        awair_local_plugin.messages = []
        result = awair_local_plugin.formatted_latest_buffer()
        assert result is None


class TestAPIFetching:
    """Tests for API data fetching."""

    @pytest.mark.asyncio
    async def test_fetch_local_success(self, awair_local_plugin):
        """Test successful local API fetch."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=SAMPLE_AWAIR_DATA)

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        mock_context.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_context)

        with patch.object(
            awair_local_plugin, "_get_session", AsyncMock(return_value=mock_session)
        ):
            result = await awair_local_plugin._fetch_local()

            assert result is not None
            assert result["score"] == 74

    @pytest.mark.asyncio
    async def test_fetch_local_no_ip(self):
        """Test local fetch without IP configured."""
        with patch("inputs.plugins.awair.IOProvider"):
            from inputs.plugins.awair import AwairElement

            plugin = AwairElement(AwairConfig(mode="local"))

            result = await plugin._fetch_local()
            assert result is None

    @pytest.mark.asyncio
    async def test_fetch_cloud_success(self, awair_cloud_plugin):
        """Test successful cloud API fetch."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": [SAMPLE_AWAIR_DATA]})

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        mock_context.__aexit__.return_value = None

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_context)

        with patch.object(
            awair_cloud_plugin, "_get_session", AsyncMock(return_value=mock_session)
        ):
            result = await awair_cloud_plugin._fetch_cloud()

            assert result is not None
            assert result["score"] == 74
