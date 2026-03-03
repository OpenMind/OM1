from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inputs.plugins.air_quality.connector.aqicn import AqicnConnector
from inputs.plugins.air_quality.connector.base import AirQualityData


class TestAqicnConnectorInit:
    """Tests for AqicnConnector initialization."""

    def test_default_values(self):
        connector = AqicnConnector({})
        assert connector.api_key == ""
        assert connector.latitude == -6.2088
        assert connector.longitude == 106.8456

    def test_custom_values(self):
        config = {
            "api_key": "test_token",
            "latitude": -6.9667,
            "longitude": 110.4167,
        }
        connector = AqicnConnector(config)
        assert connector.api_key == "test_token"
        assert connector.latitude == -6.9667
        assert connector.longitude == 110.4167


class TestAqicnConnectorConnect:
    """Tests for connect/disconnect."""

    @pytest.mark.asyncio
    async def test_connect_with_api_key(self):
        connector = AqicnConnector({"api_key": "valid_token"})
        result = await connector.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_connect_without_api_key(self):
        connector = AqicnConnector({})
        result = await connector.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect_is_safe(self):
        connector = AqicnConnector({"api_key": "token"})
        await connector.disconnect()  # should not raise


class TestAqicnConnectorRead:
    """Tests for read() and _parse()."""

    @pytest.fixture
    def connector(self):
        return AqicnConnector({"api_key": "test_token"})

    @pytest.mark.asyncio
    async def test_read_returns_none_without_api_key(self):
        connector = AqicnConnector({})
        result = await connector.read()
        assert result is None

    @pytest.mark.asyncio
    async def test_read_success(self, connector):
        mock_payload = {
            "status": "ok",
            "data": {
                "aqi": 78,
                "city": {"name": "Semarang"},
                "iaqi": {
                    "pm25": {"v": 22.5},
                    "pm10": {"v": 45.0},
                    "co": {"v": 0.5},
                    "no2": {"v": 12.0},
                    "t": {"v": 31.0},
                    "h": {"v": 80.0},
                },
            },
        }

        with patch(
            "inputs.plugins.air_quality.connector.aqicn.aiohttp.ClientSession"
        ) as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_payload)

            mock_get = MagicMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_get)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            result = await connector.read()

        assert isinstance(result, AirQualityData)
        assert result.aqi == 78
        assert result.pm25 == 22.5
        assert result.pm10 == 45.0
        assert result.co == 0.5
        assert result.temperature == 31.0
        assert result.humidity == 80.0
        assert result.location == "Semarang"
        assert result.source == "aqicn"

    @pytest.mark.asyncio
    async def test_read_http_error(self, connector):
        with patch(
            "inputs.plugins.air_quality.connector.aqicn.aiohttp.ClientSession"
        ) as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_response.text = AsyncMock(return_value="Unauthorized")

            mock_get = MagicMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_get)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            result = await connector.read()

        assert result is None

    @pytest.mark.asyncio
    async def test_read_api_status_error(self, connector):
        mock_payload = {"status": "error", "data": "Invalid token"}

        with patch(
            "inputs.plugins.air_quality.connector.aqicn.aiohttp.ClientSession"
        ) as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_payload)

            mock_get = MagicMock()
            mock_get.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_get)
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)

            mock_session.return_value = mock_session_instance

            result = await connector.read()

        assert result is None

    @pytest.mark.asyncio
    async def test_read_timeout(self, connector):
        import asyncio

        with patch(
            "inputs.plugins.air_quality.connector.aqicn.aiohttp.ClientSession"
        ) as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.__aenter__ = AsyncMock(
                side_effect=asyncio.TimeoutError
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance

            result = await connector.read()

        assert result is None

    def test_parse_missing_iaqi_fields(self, connector):
        """Test _parse handles missing iaqi fields gracefully."""
        payload = {
            "data": {
                "aqi": 50,
                "city": {"name": "Test City"},
                "iaqi": {},
            }
        }
        result = connector._parse(payload)
        assert result.aqi == 50
        assert result.pm25 is None
        assert result.pm10 is None
        assert result.location == "Test City"
        assert result.source == "aqicn"

    def test_parse_aqi_dash(self, connector):
        """Test _parse handles AQI '-' (unavailable) gracefully."""
        payload = {
            "data": {
                "aqi": "-",
                "city": {"name": "Unknown"},
                "iaqi": {},
            }
        }
        result = connector._parse(payload)
        assert result.aqi is None


class TestAqicnConnectorExceptions:
    """Cover aiohttp.ClientError and generic Exception handlers."""

    @pytest.fixture
    def connector(self):
        return AqicnConnector({"api_key": "test_token"})

    @pytest.mark.asyncio
    async def test_read_client_error(self, connector):
        import aiohttp

        with patch(
            "inputs.plugins.air_quality.connector.aqicn.aiohttp.ClientSession"
        ) as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.__aenter__ = AsyncMock(
                side_effect=aiohttp.ClientError("network fail")
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance
            result = await connector.read()
        assert result is None

    @pytest.mark.asyncio
    async def test_read_unexpected_exception(self, connector):
        with patch(
            "inputs.plugins.air_quality.connector.aqicn.aiohttp.ClientSession"
        ) as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.__aenter__ = AsyncMock(
                side_effect=RuntimeError("unexpected")
            )
            mock_session_instance.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = mock_session_instance
            result = await connector.read()
        assert result is None
