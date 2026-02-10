import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from actions.remember_location.connector.unitree_g1_location import (
    UnitreeG1RememberLocationConfig,
    UnitreeG1RememberLocationConnector,
)
from actions.remember_location.connector.unitree_go2_location import (
    UnitreeGo2RememberLocationConfig,
    UnitreeGo2RememberLocationConnector,
)
from actions.remember_location.interface import RememberLocationInput


def create_aiohttp_mock(status=200, text="OK"):
    """Create aiohttp ClientSession mock with proper async context managers."""
    mock_response = Mock()
    mock_response.status = status
    mock_response.text = AsyncMock(return_value=text)

    mock_post_cm = Mock()
    mock_post_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = Mock()
    mock_session.post = Mock(return_value=mock_post_cm)

    mock_session_cm = Mock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=None)

    return mock_session_cm, mock_session


class TestUnitreeG1RememberLocationConfig:
    """Test UnitreeG1RememberLocationConfig."""

    def test_default_config(self):
        config = UnitreeG1RememberLocationConfig()
        assert config.base_url == "http://localhost:5000/maps/locations/add/slam"
        assert config.timeout == 5
        assert config.map_name == "map"

    def test_custom_config(self):
        config = UnitreeG1RememberLocationConfig(
            base_url="http://custom:8080",
            timeout=10,
            map_name="custom_map",
        )
        assert config.base_url == "http://custom:8080"
        assert config.timeout == 10
        assert config.map_name == "custom_map"


class TestUnitreeG1RememberLocationConnector:
    """Test UnitreeG1RememberLocationConnector."""

    @pytest.fixture
    def g1_connector(self):
        with patch(
            "actions.remember_location.connector.unitree_g1_location.ElevenLabsTTSProvider"
        ) as mock_tts:
            mock_tts_instance = Mock()
            mock_tts.return_value = mock_tts_instance
            config = UnitreeG1RememberLocationConfig()
            connector = UnitreeG1RememberLocationConnector(config)
            yield connector, mock_tts_instance

    def test_init(self, g1_connector):
        connector, mock_tts = g1_connector
        assert connector.base_url == "http://localhost:5000/maps/locations/add/slam"
        assert connector.timeout == 5
        assert connector.map_name == "map"

    @pytest.mark.asyncio
    async def test_connect_success(self, g1_connector):
        connector, mock_tts = g1_connector
        mock_session_cm, mock_session = create_aiohttp_mock(status=200, text="saved")

        with (
            patch(
                "actions.remember_location.connector.unitree_g1_location.aiohttp.ClientSession",
                return_value=mock_session_cm,
            ),
            patch(
                "actions.remember_location.connector.unitree_g1_location.logging"
            ) as mock_logging,
        ):
            loc_input = RememberLocationInput(action="kitchen")
            await connector.connect(loc_input)

            mock_session.post.assert_called_once()
            call_kwargs = mock_session.post.call_args[1]
            assert call_kwargs["json"]["label"] == "kitchen"
            assert call_kwargs["json"]["map_name"] == "map"
            mock_logging.info.assert_called()
            mock_tts.add_pending_message.assert_called_once_with(
                "Location kitchen remembered !"
            )

    @pytest.mark.asyncio
    async def test_connect_api_error(self, g1_connector):
        connector, mock_tts = g1_connector
        mock_session_cm, mock_session = create_aiohttp_mock(
            status=500, text="Internal Server Error"
        )

        with (
            patch(
                "actions.remember_location.connector.unitree_g1_location.aiohttp.ClientSession",
                return_value=mock_session_cm,
            ),
            patch(
                "actions.remember_location.connector.unitree_g1_location.logging"
            ) as mock_logging,
        ):
            loc_input = RememberLocationInput(action="office")
            await connector.connect(loc_input)

            mock_logging.error.assert_called()
            mock_tts.add_pending_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_no_base_url(self, g1_connector):
        connector, mock_tts = g1_connector
        connector.base_url = ""

        with patch(
            "actions.remember_location.connector.unitree_g1_location.logging"
        ) as mock_logging:
            loc_input = RememberLocationInput(action="somewhere")
            await connector.connect(loc_input)
            mock_logging.error.assert_called_with(
                "RememberLocationG1 connector missing 'base_url' in config"
            )

    @pytest.mark.asyncio
    async def test_connect_timeout_error(self, g1_connector):
        connector, mock_tts = g1_connector

        with (
            patch(
                "actions.remember_location.connector.unitree_g1_location.aiohttp.ClientSession"
            ) as mock_cls,
            patch(
                "actions.remember_location.connector.unitree_g1_location.logging"
            ) as mock_logging,
        ):
            mock_cm = Mock()
            mock_cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_cm

            loc_input = RememberLocationInput(action="garden")
            await connector.connect(loc_input)
            mock_logging.error.assert_called_with(
                "RememberLocationG1 API request timed out"
            )


class TestUnitreeGo2RememberLocationConfig:
    """Test UnitreeGo2RememberLocationConfig."""

    def test_default_config(self):
        config = UnitreeGo2RememberLocationConfig()
        assert config.base_url == "http://localhost:5000/maps/locations/add/slam"
        assert config.timeout == 5
        assert config.map_name == "map"

    def test_custom_config(self):
        config = UnitreeGo2RememberLocationConfig(
            base_url="http://go2:9090",
            timeout=20,
            map_name="go2_map",
        )
        assert config.base_url == "http://go2:9090"
        assert config.timeout == 20
        assert config.map_name == "go2_map"


class TestUnitreeGo2RememberLocationConnector:
    """Test UnitreeGo2RememberLocationConnector."""

    @pytest.fixture
    def go2_connector(self):
        with patch(
            "actions.remember_location.connector.unitree_go2_location.ElevenLabsTTSProvider"
        ) as mock_tts:
            mock_tts_instance = Mock()
            mock_tts.return_value = mock_tts_instance
            config = UnitreeGo2RememberLocationConfig()
            connector = UnitreeGo2RememberLocationConnector(config)
            yield connector, mock_tts_instance

    def test_init(self, go2_connector):
        connector, mock_tts = go2_connector
        assert connector.base_url == "http://localhost:5000/maps/locations/add/slam"
        assert connector.timeout == 5
        assert connector.map_name == "map"

    @pytest.mark.asyncio
    async def test_connect_success(self, go2_connector):
        connector, mock_tts = go2_connector
        mock_session_cm, mock_session = create_aiohttp_mock(status=200, text="saved")

        with (
            patch(
                "actions.remember_location.connector.unitree_go2_location.aiohttp.ClientSession",
                return_value=mock_session_cm,
            ),
            patch(
                "actions.remember_location.connector.unitree_go2_location.logging"
            ) as mock_logging,
        ):
            loc_input = RememberLocationInput(action="charging_station")
            await connector.connect(loc_input)

            mock_session.post.assert_called_once()
            call_kwargs = mock_session.post.call_args[1]
            assert call_kwargs["json"]["label"] == "charging_station"
            mock_logging.info.assert_called()
            mock_tts.add_pending_message.assert_called_once_with(
                "Location charging_station remembered for Go2. Woof! Woof!"
            )

    @pytest.mark.asyncio
    async def test_connect_api_error(self, go2_connector):
        connector, mock_tts = go2_connector
        mock_session_cm, mock_session = create_aiohttp_mock(
            status=404, text="Not Found"
        )

        with (
            patch(
                "actions.remember_location.connector.unitree_go2_location.aiohttp.ClientSession",
                return_value=mock_session_cm,
            ),
            patch(
                "actions.remember_location.connector.unitree_go2_location.logging"
            ) as mock_logging,
        ):
            loc_input = RememberLocationInput(action="hall")
            await connector.connect(loc_input)
            mock_logging.error.assert_called()

    @pytest.mark.asyncio
    async def test_connect_no_base_url(self, go2_connector):
        connector, mock_tts = go2_connector
        connector.base_url = ""

        with patch(
            "actions.remember_location.connector.unitree_go2_location.logging"
        ) as mock_logging:
            loc_input = RememberLocationInput(action="somewhere")
            await connector.connect(loc_input)
            mock_logging.error.assert_called_with(
                "RememberLocationGo2 connector missing 'base_url' in config"
            )

    @pytest.mark.asyncio
    async def test_connect_exception(self, go2_connector):
        connector, mock_tts = go2_connector

        with (
            patch(
                "actions.remember_location.connector.unitree_go2_location.aiohttp.ClientSession"
            ) as mock_cls,
            patch(
                "actions.remember_location.connector.unitree_go2_location.logging"
            ) as mock_logging,
        ):
            mock_cm = Mock()
            mock_cm.__aenter__ = AsyncMock(
                side_effect=ConnectionError("Connection refused")
            )
            mock_cm.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_cm

            loc_input = RememberLocationInput(action="unknown")
            await connector.connect(loc_input)
            mock_logging.error.assert_called()
