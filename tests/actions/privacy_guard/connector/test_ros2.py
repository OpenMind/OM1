"""Tests for the Privacy Guard ROS2 connector.

The zenoh_msgs module is mocked at the sys.modules level because it may
not be importable in all environments (e.g., missing C extensions or
Python version incompatibilities with pycdr2).
"""

import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from actions.privacy_guard.interface import PrivacyAction, PrivacyGuardInput

# ---------------------------------------------------------------------------
# Mock the zenoh_msgs package so the ros2 connector can be imported even
# when zenoh / pycdr2 are not available or incompatible.
# ---------------------------------------------------------------------------
_mock_zenoh_msgs = MagicMock()
_mock_vector3_cls = MagicMock()
_mock_camera_status_cls = MagicMock()
_mock_camera_status_cls.STATUS.ENABLED.value = 1
_mock_camera_status_cls.STATUS.DISABLED.value = 0
_mock_zenoh_msgs.Vector3 = _mock_vector3_cls
_mock_zenoh_msgs.CameraStatus = _mock_camera_status_cls
_mock_zenoh_msgs.prepare_header = MagicMock(return_value=MagicMock())
_mock_open_session = MagicMock()
_mock_zenoh_msgs.open_zenoh_session = _mock_open_session

# Inject mocks before importing the connector
_original_modules = {}
for mod_name in ["zenoh_msgs", "zenoh_msgs.idl", "zenoh_msgs.session", "zenoh"]:
    _original_modules[mod_name] = sys.modules.get(mod_name)
    sys.modules[mod_name] = _mock_zenoh_msgs

# Now we can safely import
from actions.privacy_guard.connector.ros2 import (  # noqa: E402
    PrivacyGuardConfig,
    PrivacyGuardROS2Connector,
)


class TestPrivacyGuardROS2Connector:
    """Tests for PrivacyGuardROS2Connector with mocked Zenoh."""

    @pytest.fixture(autouse=True)
    def setup_mock_session(self):
        """Reset the mock session before each test."""
        mock_session = MagicMock()
        mock_pub = MagicMock()
        mock_session.declare_publisher.return_value = mock_pub
        _mock_open_session.reset_mock()
        _mock_open_session.return_value = mock_session
        _mock_open_session.side_effect = None
        self.mock_session = mock_session
        self.mock_pub = mock_pub

    @pytest.fixture
    def connector(self):
        config = PrivacyGuardConfig(
            look_away_angle=90,
            resume_delay_seconds=2.0,
        )
        conn = PrivacyGuardROS2Connector(config)
        return conn

    @pytest.mark.asyncio
    async def test_look_away_publishes(self, connector):
        inp = PrivacyGuardInput(action=PrivacyAction.LOOK_AWAY)
        await connector.connect(inp)

        assert connector._is_looking_away is True
        # head_tilt_pub.put and stream_ctrl_pub.put should both be called
        assert self.mock_pub.put.call_count == 2

    @pytest.mark.asyncio
    async def test_resume_after_cooldown(self, connector):
        # First look away
        inp_look = PrivacyGuardInput(action=PrivacyAction.LOOK_AWAY)
        await connector.connect(inp_look)
        assert connector._is_looking_away is True

        # Force the timestamp into the past so cooldown has elapsed
        connector._last_look_away_time = time.time() - 10.0

        # Now resume
        self.mock_pub.put.reset_mock()
        inp_resume = PrivacyGuardInput(action=PrivacyAction.RESUME)
        await connector.connect(inp_resume)

        assert connector._is_looking_away is False
        assert self.mock_pub.put.call_count == 2

    @pytest.mark.asyncio
    async def test_resume_blocked_during_cooldown(self, connector):
        # Look away
        inp_look = PrivacyGuardInput(action=PrivacyAction.LOOK_AWAY)
        await connector.connect(inp_look)

        # Immediately try to resume (within cooldown)
        self.mock_pub.put.reset_mock()
        inp_resume = PrivacyGuardInput(action=PrivacyAction.RESUME)
        await connector.connect(inp_resume)

        # Should still be looking away — resume was blocked
        assert connector._is_looking_away is True
        assert self.mock_pub.put.call_count == 0

    @pytest.mark.asyncio
    async def test_resume_when_not_looking_away(self, connector):
        """Resume is a no-op when already in neutral position."""
        self.mock_pub.put.reset_mock()
        inp = PrivacyGuardInput(action=PrivacyAction.RESUME)
        await connector.connect(inp)

        assert connector._is_looking_away is False
        assert self.mock_pub.put.call_count == 0

    @pytest.mark.asyncio
    async def test_stop_stream_publishes(self, connector):
        inp = PrivacyGuardInput(action=PrivacyAction.STOP_STREAM)
        await connector.connect(inp)

        assert connector._is_looking_away is True
        assert self.mock_pub.put.call_count == 2

    @pytest.mark.asyncio
    async def test_idle_no_publish(self, connector):
        self.mock_pub.put.reset_mock()
        inp = PrivacyGuardInput(action=PrivacyAction.IDLE)
        await connector.connect(inp)

        assert self.mock_pub.put.call_count == 0

    def test_stop_closes_session(self, connector):
        connector.stop()
        self.mock_session.close.assert_called_once()

    def test_zenoh_session_failure_graceful(self):
        """Connector should not crash if Zenoh session fails to open."""
        _mock_open_session.side_effect = Exception("Connection refused")
        config = PrivacyGuardConfig()
        conn = PrivacyGuardROS2Connector(config)
        assert conn.session is None
        assert conn.head_tilt_pub is None
