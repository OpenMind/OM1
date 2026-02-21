from zenoh_msgs.idl.geometry_msgs import (
    Point,
    Pose,
    PoseWithCovariance,
    Quaternion,
    Twist,
    TwistWithCovariance,
    Vector3,
)
from zenoh_msgs.idl.nav_msgs import (
    AMCLPose,
    GoalID,
    GoalInfo,
    GoalStatus,
    LidarLocalization,
    Nav2Status,
    Odometry,
    Time,
)
from zenoh_msgs.idl.std_msgs import Header, String
from zenoh_msgs.idl.std_msgs import Time as StdTime


class TestOdometry:
    """Tests for the Odometry dataclass."""

    def test_create_odometry(self):
        """Test Odometry creation."""
        stamp = StdTime(sec=1, nanosec=0)
        header = Header(stamp=stamp, frame_id="odom")
        position = Point(x=1.0, y=2.0, z=0.0)
        orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        pose = Pose(position=position, orientation=orientation)
        pose_cov = PoseWithCovariance(pose=pose, covariance=[0.0] * 36)  # type: ignore
        linear = Vector3(x=1.0, y=0.0, z=0.0)
        angular = Vector3(x=0.0, y=0.0, z=0.0)
        twist = Twist(linear=linear, angular=angular)
        twist_cov = TwistWithCovariance(twist=twist, covariance=[0.0] * 36)  # type: ignore
        odom = Odometry(
            header=header,
            child_frame_id=String(data="base_link"),
            pose=pose_cov,
            twist=twist_cov,
        )
        assert odom.header.frame_id == "odom"
        assert odom.child_frame_id.data == "base_link"
        assert odom.pose.pose.position.x == 1.0


class TestAMCLPose:
    """Tests for the AMCLPose dataclass."""

    def test_create_amclpose(self):
        """Test AMCLPose creation."""
        stamp = StdTime(sec=1, nanosec=0)
        header = Header(stamp=stamp, frame_id="map")
        position = Point(x=1.0, y=2.0, z=0.0)
        orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        pose = Pose(position=position, orientation=orientation)
        amcl = AMCLPose(header=header, pose=pose, covariance=[0.0] * 36)  # type: ignore
        assert amcl.header.frame_id == "map"
        assert amcl.pose.position.x == 1.0


class TestLidarLocalization:
    """Tests for the LidarLocalization dataclass."""

    def test_create_lidar_localization(self):
        """Test LidarLocalization creation."""
        stamp = StdTime(sec=1, nanosec=0)
        header = Header(stamp=stamp, frame_id="map")
        position = Point(x=1.0, y=2.0, z=0.0)
        orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        pose = Pose(position=position, orientation=orientation)
        ll = LidarLocalization(
            header=header,
            pose=pose,
            match_score=95,
            quality_percent=0.95,
            num_points=1000,
        )
        assert ll.match_score == 95
        assert ll.quality_percent == 0.95
        assert ll.num_points == 1000


class TestNavTime:
    """Tests for the nav_msgs Time dataclass."""

    def test_create_time(self):
        """Test nav_msgs Time creation."""
        t = Time(sec=100, nanosec=500)
        assert t.sec == 100
        assert t.nanosec == 500


class TestGoalID:
    """Tests for the GoalID dataclass."""

    def test_create_goal_id(self):
        """Test GoalID creation with uuid."""
        uuid = [0] * 16  # type: ignore
        goal_id = GoalID(uuid=uuid)  # type: ignore
        assert goal_id is not None


class TestGoalInfo:
    """Tests for the GoalInfo dataclass."""

    def test_create_goal_info(self):
        """Test GoalInfo creation."""
        uuid = [0] * 16  # type: ignore
        goal_id = GoalID(uuid=uuid)  # type: ignore
        stamp = Time(sec=1, nanosec=0)
        goal_info = GoalInfo(goal_id=goal_id, stamp=stamp)
        assert goal_info.stamp.sec == 1


class TestGoalStatus:
    """Tests for the GoalStatus dataclass."""

    def test_create_goal_status(self):
        """Test GoalStatus creation."""
        uuid = [0] * 16  # type: ignore
        goal_id = GoalID(uuid=uuid)  # type: ignore
        stamp = Time(sec=1, nanosec=0)
        goal_info = GoalInfo(goal_id=goal_id, stamp=stamp)
        goal_status = GoalStatus(goal_info=goal_info, status=1)
        assert goal_status.status == 1


class TestNav2Status:
    """Tests for the Nav2Status dataclass."""

    def test_create_nav2status_empty(self):
        """Test Nav2Status creation with empty status list."""
        nav2 = Nav2Status(status_list=[])  # type: ignore
        assert nav2.status_list == []

    def test_create_nav2status_with_items(self):
        """Test Nav2Status creation with status list items."""
        uuid = [0] * 16  # type: ignore
        goal_id = GoalID(uuid=uuid)  # type: ignore
        stamp = Time(sec=1, nanosec=0)
        goal_info = GoalInfo(goal_id=goal_id, stamp=stamp)
        goal_status = GoalStatus(goal_info=goal_info, status=2)
        nav2 = Nav2Status(status_list=[goal_status])  # type: ignore
        assert nav2 is not None
