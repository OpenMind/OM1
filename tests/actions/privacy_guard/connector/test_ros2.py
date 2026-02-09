"""Tests for the Privacy Guard ROS2 connector."""

import time
from unittest.mock import MagicMock, patch

import pytest

from actions.privacy_guard.interface import PrivacyAction, PrivacyGuardInput


class TestPrivacyGuardROS2Connector:
    """Tests for PrivacyGuardROS2Connector with mocked Zenoh."""

    @pytest.fixture
    def mock_zenoh(self):
        """Patch open_zenoh_session so no real Zenoh session is created."""
        with patch(
            "actions.privacy_guard.connector.ros2.open_zenoh_session"
        ) as mock_open:
            mock_session = MagicMock()
            mock_pub = MagicMock()
            mock_session.declare_publisher.return_value = mock_pub
            mock_open.return_value = mock_session
            yield {
                "session": mock_session,
                "publisher": mock_pub,
                "open_session": mock_open,
            }

    @pytest.fixture
    def connector(self, mock_zenoh):
        from actions.privacy_guard.connector.ros2 import (
            PrivacyGuardConfig,
            PrivacyGuardROS2Connector,
        )

        config = PrivacyGuardConfig(
            look_away_angle=90,
            resume_delay_seconds=2.0,
        )
        conn = PrivacyGuardROS2Connector(config)
        return conn

    @pytest.mark.asyncio
    async def test_look_away_publishes(self, connector, mock_zenoh):
        inp = PrivacyGuardInput(action=PrivacyAction.LOOK_AWAY)
        await connector.connect(inp)

        assert connector._is_looking_away is True
        # head_tilt_pub.put and stream_ctrl_pub.put should both be called
        assert mock_zenoh["publisher"].put.call_count == 2

    @pytest.mark.asyncio
    async def test_resume_after_cooldown(self, connector, mock_zenoh):
        # First look away
        inp_look = PrivacyGuardInput(action=PrivacyAction.LOOK_AWAY)
        await connector.connect(inp_look)
        assert connector._is_looking_away is True

        # Force the timestamp into the past so cooldown has elapsed
        connector._last_look_away_time = time.time() - 10.0

        # Now resume
        mock_zenoh["publisher"].put.reset_mock()
        inp_resume = PrivacyGuardInput(action=PrivacyAction.RESUME)
        await connector.connect(inp_resume)

        assert connector._is_looking_away is False
        assert mock_zenoh["publisher"].put.call_count == 2

    @pytest.mark.asyncio
    async def test_resume_blocked_during_cooldown(self, connector, mock_zenoh):
        # Look away
        inp_look = PrivacyGuardInput(action=PrivacyAction.LOOK_AWAY)
        await connector.connect(inp_look)

        # Immediately try to resume (within cooldown)
        mock_zenoh["publisher"].put.reset_mock()
        inp_resume = PrivacyGuardInput(action=PrivacyAction.RESUME)
        await connector.connect(inp_resume)

        # Should still be looking away — resume was blocked
        assert connector._is_looking_away is True
        assert mock_zenoh["publisher"].put.call_count == 0

    @pytest.mark.asyncio
    async def test_resume_when_not_looking_away(self, connector, mock_zenoh):
        """Resume is a no-op when already in neutral position."""
        mock_zenoh["publisher"].put.reset_mock()
        inp = PrivacyGuardInput(action=PrivacyAction.RESUME)
        await connector.connect(inp)

        assert connector._is_looking_away is False
        assert mock_zenoh["publisher"].put.call_count == 0

    @pytest.mark.asyncio
    async def test_stop_stream_publishes(self, connector, mock_zenoh):
        inp = PrivacyGuardInput(action=PrivacyAction.STOP_STREAM)
        await connector.connect(inp)

        assert connector._is_looking_away is True
        assert mock_zenoh["publisher"].put.call_count == 2

    @pytest.mark.asyncio
    async def test_idle_no_publish(self, connector, mock_zenoh):
        mock_zenoh["publisher"].put.reset_mock()
        inp = PrivacyGuardInput(action=PrivacyAction.IDLE)
        await connector.connect(inp)

        assert mock_zenoh["publisher"].put.call_count == 0

    def test_stop_closes_session(self, connector, mock_zenoh):
        connector.stop()
        mock_zenoh["session"].close.assert_called_once()

    def test_zenoh_session_failure_graceful(self):
        """Connector should not crash if Zenoh session fails to open."""
        with patch(
            "actions.privacy_guard.connector.ros2.open_zenoh_session",
            side_effect=Exception("Connection refused"),
        ):
            from actions.privacy_guard.connector.ros2 import (
                PrivacyGuardConfig,
                PrivacyGuardROS2Connector,
            )

            config = PrivacyGuardConfig()
            conn = PrivacyGuardROS2Connector(config)
            assert conn.session is None
            assert conn.head_tilt_pub is None
