from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hooks.unitree_go2_patrol_hook import (
    UnitreeGo2PatrolHookContext,
    start_unitree_go2_patrol_hook,
    stop_unitree_go2_patrol_hook,
)


class TestUnitreeGo2PatrolHookContext:
    """Tests for UnitreeGo2PatrolHookContext."""

    def test_default_context(self):
        """Test default context values."""
        context = UnitreeGo2PatrolHookContext()

        assert context.patrol_base_url == "http://localhost:5000"
        assert context.face_presence_base_url == "http://127.0.0.1:6793"
        assert context.patrol_image_report_base_url == "https://api.openmind.com"
        assert context.api_key == ""

    def test_custom_context(self):
        """Test custom context values."""
        context = UnitreeGo2PatrolHookContext(
            patrol_base_url="http://robot.local:8000",
            face_presence_base_url="http://robot.local:9000",
            patrol_image_report_base_url="https://custom.api.com",
            api_key="test_key",
        )

        assert context.patrol_base_url == "http://robot.local:8000"
        assert context.face_presence_base_url == "http://robot.local:9000"
        assert context.patrol_image_report_base_url == "https://custom.api.com"
        assert context.api_key == "test_key"


class TestStartUnitreeGo2PatrolHook:
    """Tests for start_unitree_go2_patrol_hook function."""

    @pytest.mark.asyncio
    async def test_start_patrol_success_default_context(self):
        """Test successful patrol start with default context."""
        context = {}

        mock_provider = MagicMock()
        mock_provider.start_patrol = AsyncMock()

        with patch(
            "hooks.unitree_go2_patrol_hook.UnitreeGo2PatrolProvider",
            return_value=mock_provider,
        ) as mock_provider_class:
            result = await start_unitree_go2_patrol_hook(context)

        assert result is True
        mock_provider_class.assert_called_once_with(
            api_key="",
            patrol_base_url="http://localhost:5000",
            face_presence_base_url="http://127.0.0.1:6793",
            patrol_image_report_base_url="https://api.openmind.com",
        )
        mock_provider.start_patrol.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_patrol_success_custom_context(self):
        """Test successful patrol start with custom context."""
        context = {
            "patrol_base_url": "http://robot.local:8000",
            "face_presence_base_url": "http://robot.local:9000",
            "patrol_image_report_base_url": "https://custom.api.com",
            "api_key": "test_key",
        }

        mock_provider = MagicMock()
        mock_provider.start_patrol = AsyncMock()

        with patch(
            "hooks.unitree_go2_patrol_hook.UnitreeGo2PatrolProvider",
            return_value=mock_provider,
        ) as mock_provider_class:
            result = await start_unitree_go2_patrol_hook(context)

        assert result is True
        mock_provider_class.assert_called_once_with(
            api_key="test_key",
            patrol_base_url="http://robot.local:8000",
            face_presence_base_url="http://robot.local:9000",
            patrol_image_report_base_url="https://custom.api.com",
        )
        mock_provider.start_patrol.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_patrol_provider_error(self):
        """Test patrol start with provider error."""
        context = {}

        mock_provider = MagicMock()
        mock_provider.start_patrol = AsyncMock(side_effect=Exception("Connection failed"))

        with patch(
            "hooks.unitree_go2_patrol_hook.UnitreeGo2PatrolProvider",
            return_value=mock_provider,
        ):
            result = await start_unitree_go2_patrol_hook(context)

        assert result is False

    @pytest.mark.asyncio
    async def test_start_patrol_logs_success(self, caplog):
        """Test that successful patrol start logs correct messages."""
        context = {}

        mock_provider = MagicMock()
        mock_provider.start_patrol = AsyncMock()

        with caplog.at_level("INFO"):
            with patch(
                "hooks.unitree_go2_patrol_hook.UnitreeGo2PatrolProvider",
                return_value=mock_provider,
            ):
                await start_unitree_go2_patrol_hook(context)

        assert "Starting Unitree Go2 patrol with context:" in caplog.text
        assert "Unitree Go2 patrol started successfully" in caplog.text

    @pytest.mark.asyncio
    async def test_start_patrol_logs_error(self, caplog):
        """Test that patrol start error logs exception."""
        context = {}

        mock_provider = MagicMock()
        mock_provider.start_patrol = AsyncMock(side_effect=Exception("Test error"))

        with caplog.at_level("ERROR"):
            with patch(
                "hooks.unitree_go2_patrol_hook.UnitreeGo2PatrolProvider",
                return_value=mock_provider,
            ):
                await start_unitree_go2_patrol_hook(context)

        assert "Error in starting Unitree Go2 patrol" in caplog.text


class TestStopUnitreeGo2PatrolHook:
    """Tests for stop_unitree_go2_patrol_hook function."""

    @pytest.mark.asyncio
    async def test_stop_patrol_success_default_context(self):
        """Test successful patrol stop with default context."""
        context = {}

        mock_provider = MagicMock()
        mock_provider.stop_patrol = AsyncMock()

        with patch(
            "hooks.unitree_go2_patrol_hook.UnitreeGo2PatrolProvider",
            return_value=mock_provider,
        ) as mock_provider_class:
            result = await stop_unitree_go2_patrol_hook(context)

        assert result is True
        mock_provider_class.assert_called_once_with(
            api_key="",
            patrol_base_url="http://localhost:5000",
            face_presence_base_url="http://127.0.0.1:6793",
            patrol_image_report_base_url="https://api.openmind.com",
        )
        mock_provider.stop_patrol.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_patrol_success_custom_context(self):
        """Test successful patrol stop with custom context."""
        context = {
            "patrol_base_url": "http://robot.local:8000",
            "face_presence_base_url": "http://robot.local:9000",
            "patrol_image_report_base_url": "https://custom.api.com",
            "api_key": "test_key",
        }

        mock_provider = MagicMock()
        mock_provider.stop_patrol = AsyncMock()

        with patch(
            "hooks.unitree_go2_patrol_hook.UnitreeGo2PatrolProvider",
            return_value=mock_provider,
        ) as mock_provider_class:
            result = await stop_unitree_go2_patrol_hook(context)

        assert result is True
        mock_provider_class.assert_called_once_with(
            api_key="test_key",
            patrol_base_url="http://robot.local:8000",
            face_presence_base_url="http://robot.local:9000",
            patrol_image_report_base_url="https://custom.api.com",
        )
        mock_provider.stop_patrol.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_patrol_provider_error(self):
        """Test patrol stop with provider error."""
        context = {}

        mock_provider = MagicMock()
        mock_provider.stop_patrol = AsyncMock(side_effect=Exception("Connection failed"))

        with patch(
            "hooks.unitree_go2_patrol_hook.UnitreeGo2PatrolProvider",
            return_value=mock_provider,
        ):
            result = await stop_unitree_go2_patrol_hook(context)

        assert result is False

    @pytest.mark.asyncio
    async def test_stop_patrol_logs_success(self, caplog):
        """Test that successful patrol stop logs correct messages."""
        context = {}

        mock_provider = MagicMock()
        mock_provider.stop_patrol = AsyncMock()

        with caplog.at_level("INFO"):
            with patch(
                "hooks.unitree_go2_patrol_hook.UnitreeGo2PatrolProvider",
                return_value=mock_provider,
            ):
                await stop_unitree_go2_patrol_hook(context)

        assert "Stopping Unitree Go2 patrol with context:" in caplog.text
        assert "Unitree Go2 patrol stopped successfully" in caplog.text

    @pytest.mark.asyncio
    async def test_stop_patrol_logs_error(self, caplog):
        """Test that patrol stop error logs exception."""
        context = {}

        mock_provider = MagicMock()
        mock_provider.stop_patrol = AsyncMock(side_effect=Exception("Test error"))

        with caplog.at_level("ERROR"):
            with patch(
                "hooks.unitree_go2_patrol_hook.UnitreeGo2PatrolProvider",
                return_value=mock_provider,
            ):
                await stop_unitree_go2_patrol_hook(context)

        assert "Error in stopping Unitree Go2 patrol" in caplog.text
