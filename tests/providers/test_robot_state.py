"""Tests for robot state data structures."""

from src.providers.robot_state import BatteryStatus, Position, RobotState


class TestRobotState:
    """Test suite for RobotState and related dataclasses."""

    def test_position_defaults(self):
        """Test Position default values."""
        pos = Position()
        assert pos.x == 0.0
        assert pos.y == 0.0
        assert pos.yaw == 0.0

    def test_position_custom(self):
        """Test Position with custom values."""
        pos = Position(x=1.0, y=2.0, yaw=90.0)
        assert pos.x == 1.0
        assert pos.y == 2.0
        assert pos.yaw == 90.0

    def test_battery_status_defaults(self):
        """Test BatteryStatus default values."""
        bat = BatteryStatus()
        assert bat.percentage == 100.0
        assert bat.voltage == 0.0
        assert bat.temperature == 0.0
        assert bat.charging is False

    def test_battery_status_custom(self):
        """Test BatteryStatus with custom values."""
        bat = BatteryStatus(
            percentage=50.0, voltage=12.5, temperature=25.0, charging=True
        )
        assert bat.percentage == 50.0
        assert bat.voltage == 12.5
        assert bat.temperature == 25.0
        assert bat.charging is True

    def test_robot_state_defaults(self):
        """Test RobotState default values."""
        state = RobotState()
        assert state.timestamp > 0
        assert state.position.x == 0.0
        assert state.is_moving is False
        assert state.body_state == "unknown"
        assert state.battery.percentage == 100.0
        assert state.is_localized is False
        assert state.localization_pose is None
        assert state.safe_paths == []
        assert state.obstacles_nearby is False

    def test_robot_state_custom(self):
        """Test RobotState with custom values and to_dict conversion."""
        pos = Position(x=1.5, y=2.5, yaw=45.0)
        bat = BatteryStatus(
            percentage=75.0, voltage=11.0, temperature=30.0, charging=False
        )
        state = RobotState(
            position=pos,
            is_moving=True,
            body_state="standing",
            battery=bat,
            is_localized=True,
            localization_pose={"x": 10.0, "y": 20.0, "z": 0.0},
            safe_paths=["move forwards", "turn left"],
            obstacles_nearby=True,
        )
        d = state.to_dict()
        assert d["position"]["x"] == 1.5
        assert d["position"]["y"] == 2.5
        assert d["position"]["yaw"] == 45.0
        assert d["is_moving"] is True
        assert d["body_state"] == "standing"
        assert d["battery"]["percentage"] == 75.0
        assert d["is_localized"] is True
        assert d["localization_pose"] == {"x": 10.0, "y": 20.0, "z": 0.0}
        assert d["safe_paths"] == ["move forwards", "turn left"]
        assert d["obstacles_nearby"] is True
