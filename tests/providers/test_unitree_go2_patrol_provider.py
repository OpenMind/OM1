from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientError

from providers.unitree_go2_patrol_provider import UnitreeGo2PatrolProvider


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton instances between tests."""
    UnitreeGo2PatrolProvider.reset()  # type: ignore
    yield
    UnitreeGo2PatrolProvider.reset()  # type: ignore


class TestUnitreeGo2PatrolProviderInitialization:
    """Tests for UnitreeGo2PatrolProvider initialization."""

    def test_initialization_default_params(self):
        """Test initialization with default parameters."""
        provider = UnitreeGo2PatrolProvider()

        assert provider.patrol_base_url == "http://localhost:5000"
        assert provider.face_presence_base_url == "http://localhost:6793"
        assert provider.patrol_image_report_base_url == "https://api.openmind.com"
        assert provider.api_key == ""

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        provider = UnitreeGo2PatrolProvider(
            api_key="test_key",
            patrol_base_url="http://robot.local:8000",
            face_presence_base_url="http://robot.local:9000",
            patrol_image_report_base_url="https://custom.api.com",
        )

        assert provider.patrol_base_url == "http://robot.local:8000"
        assert provider.face_presence_base_url == "http://robot.local:9000"
        assert provider.patrol_image_report_base_url == "https://custom.api.com"
        assert provider.api_key == "test_key"

    def test_singleton_pattern(self):
        """Test that UnitreeGo2PatrolProvider follows singleton pattern."""
        provider1 = UnitreeGo2PatrolProvider(api_key="key1")
        provider2 = UnitreeGo2PatrolProvider(api_key="key2")

        assert provider1 is provider2
        assert provider1.api_key == "key1"


class TestUnitreeGo2PatrolProviderStartPatrol:
    """Tests for start_patrol method."""

    @pytest.mark.asyncio
    async def test_start_patrol_success(self):
        """Test successful patrol start."""
        provider = UnitreeGo2PatrolProvider()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await provider.start_patrol()

        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://localhost:5000/patrol/start"
        mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_patrol_client_error(self):
        """Test patrol start with client error."""
        provider = UnitreeGo2PatrolProvider()

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Connection failed"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(ClientError, match="Connection failed"):
                await provider.start_patrol()


class TestUnitreeGo2PatrolProviderStopPatrol:
    """Tests for stop_patrol method."""

    @pytest.mark.asyncio
    async def test_stop_patrol_success(self):
        """Test successful patrol stop."""
        provider = UnitreeGo2PatrolProvider()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await provider.stop_patrol()

        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://localhost:5000/patrol/stop"
        mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_patrol_client_error(self):
        """Test patrol stop with client error."""
        provider = UnitreeGo2PatrolProvider()

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Connection timeout"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(ClientError, match="Connection timeout"):
                await provider.stop_patrol()


class TestUnitreeGo2PatrolProviderPausePatrol:
    """Tests for pause_patrol method."""

    @pytest.mark.asyncio
    async def test_pause_patrol_success(self):
        """Test successful patrol pause."""
        provider = UnitreeGo2PatrolProvider()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await provider.pause_patrol()

        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://localhost:5000/patrol/pause"
        mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_pause_patrol_client_error(self):
        """Test patrol pause with client error."""
        provider = UnitreeGo2PatrolProvider()

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Network error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(ClientError, match="Network error"):
                await provider.pause_patrol()


class TestUnitreeGo2PatrolProviderResumePatrol:
    """Tests for resume_patrol method."""

    @pytest.mark.asyncio
    async def test_resume_patrol_success(self):
        """Test successful patrol resume."""
        provider = UnitreeGo2PatrolProvider()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await provider.resume_patrol()

        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://localhost:5000/patrol/resume"
        mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_patrol_client_error(self):
        """Test patrol resume with client error."""
        provider = UnitreeGo2PatrolProvider()

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Service unavailable"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(ClientError, match="Service unavailable"):
                await provider.resume_patrol()


class TestUnitreeGo2PatrolProviderGetReport:
    """Tests for get_report method."""

    @pytest.mark.asyncio
    async def test_get_report_success_default_params(self):
        """Test successful get report with default parameters."""
        provider = UnitreeGo2PatrolProvider()

        expected_report = {
            "frame_b64": "base64_encoded_image",
            "unknown_captures": [
                {"track_id": 1, "unknown_duration": 3},
                {"track_id": 2, "unknown_duration": 5},
            ],
        }

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=expected_report)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            report = await provider.get_report()

        assert report == expected_report
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://localhost:6793/who"
        assert call_args[1]["json"] == {"recent_sec": 3}
        assert call_args[1]["headers"]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_get_report_success_custom_recent_sec(self):
        """Test successful get report with custom recent_sec parameter."""
        provider = UnitreeGo2PatrolProvider()

        expected_report = {"frame_b64": "", "unknown_captures": []}

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=expected_report)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            report = await provider.get_report(recent_sec=10)

        assert report == expected_report
        call_args = mock_session.post.call_args
        assert call_args[1]["json"] == {"recent_sec": 10}

    @pytest.mark.asyncio
    async def test_get_report_client_error(self):
        """Test get report with client error."""
        provider = UnitreeGo2PatrolProvider()

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Connection refused"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(ClientError, match="Connection refused"):
                await provider.get_report()


class TestUnitreeGo2PatrolProviderUploadPatrolImage:
    """Tests for upload_patrol_image method."""

    @pytest.mark.asyncio
    async def test_upload_patrol_image_success(self):
        """Test successful patrol image upload."""
        provider = UnitreeGo2PatrolProvider(api_key="test_api_key")

        image_base64 = "aGVsbG8gd29ybGQ="  # "hello world" in base64
        description = "Test patrol image"

        expected_response = {"upload_id": "12345", "status": "success"}

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=expected_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await provider.upload_patrol_image(image_base64, description)

        assert result == expected_response
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "https://api.openmind.com/api/core/patrol/upload"
        assert "Authorization" in call_args[1]["headers"]
        assert call_args[1]["headers"]["Authorization"] == "Bearer test_api_key"

    @pytest.mark.asyncio
    async def test_upload_patrol_image_no_api_key(self):
        """Test patrol image upload without API key."""
        provider = UnitreeGo2PatrolProvider()  # No API key

        image_base64 = "aGVsbG8gd29ybGQ="
        description = "Test patrol image"

        result = await provider.upload_patrol_image(image_base64, description)

        assert result == {}

    @pytest.mark.asyncio
    async def test_upload_patrol_image_empty_description(self):
        """Test patrol image upload with empty description."""
        provider = UnitreeGo2PatrolProvider(api_key="test_api_key")

        image_base64 = "aGVsbG8gd29ybGQ="

        expected_response = {"upload_id": "12345"}

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value=expected_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await provider.upload_patrol_image(image_base64)

        assert result == expected_response

    @pytest.mark.asyncio
    async def test_upload_patrol_image_client_error(self):
        """Test patrol image upload with client error."""
        provider = UnitreeGo2PatrolProvider(api_key="test_api_key")

        image_base64 = "aGVsbG8gd29ybGQ="

        mock_session = MagicMock()
        mock_session.post = MagicMock(side_effect=ClientError("Upload failed"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            with pytest.raises(ClientError, match="Upload failed"):
                await provider.upload_patrol_image(image_base64, "Test")

    @pytest.mark.asyncio
    async def test_upload_patrol_image_invalid_base64(self):
        """Test patrol image upload with invalid base64 data."""
        provider = UnitreeGo2PatrolProvider(api_key="test_api_key")

        invalid_base64 = "not_valid_base64!!!"

        with pytest.raises(Exception):
            await provider.upload_patrol_image(invalid_base64, "Test")
