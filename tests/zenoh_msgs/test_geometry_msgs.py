from zenoh_msgs.idl.geometry_msgs import (
    Accel,
    AccelWithCovariance,
    AccelWithCovarianceStamped,
    Point,
    Point32,
    Pose,
    PoseStamped,
    PoseWithCovariance,
    PoseWithCovarianceStamped,
    Quaternion,
    Twist,
    TwistWithCovariance,
    TwistWithCovarianceStamped,
    Vector3,
)
from zenoh_msgs.idl.std_msgs import Header, Time


class TestPoint:
    """Tests for the Point dataclass."""

    def test_create_point(self):
        """Test Point creation."""
        p = Point(x=1.0, y=2.0, z=3.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0

    def test_point_zero(self):
        """Test Point with zero values."""
        p = Point(x=0.0, y=0.0, z=0.0)
        assert p.x == 0.0
        assert p.y == 0.0
        assert p.z == 0.0


class TestPoint32:
    """Tests for the Point32 dataclass."""

    def test_create_point32(self):
        """Test Point32 creation."""
        p = Point32(x=1.0, y=2.0, z=3.0)
        assert p.x == 1.0
        assert p.y == 2.0
        assert p.z == 3.0


class TestQuaternion:
    """Tests for the Quaternion dataclass."""

    def test_create_quaternion(self):
        """Test Quaternion creation."""
        q = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0
        assert q.w == 1.0


class TestPose:
    """Tests for the Pose dataclass."""

    def test_create_pose(self):
        """Test Pose creation with position and orientation."""
        position = Point(x=1.0, y=2.0, z=3.0)
        orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        pose = Pose(position=position, orientation=orientation)
        assert pose.position.x == 1.0
        assert pose.orientation.w == 1.0


class TestPoseStamped:
    """Tests for the PoseStamped dataclass."""

    def test_create_posestamped(self):
        """Test PoseStamped creation."""
        stamp = Time(sec=1, nanosec=0)
        header = Header(stamp=stamp, frame_id="map")
        position = Point(x=1.0, y=2.0, z=0.0)
        orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        pose = Pose(position=position, orientation=orientation)
        ps = PoseStamped(header=header, pose=pose)
        assert ps.header.frame_id == "map"
        assert ps.pose.position.x == 1.0


class TestPoseWithCovariance:
    """Tests for the PoseWithCovariance dataclass."""

    def test_create_pose_with_covariance(self):
        """Test PoseWithCovariance creation."""
        position = Point(x=0.0, y=0.0, z=0.0)
        orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        pose = Pose(position=position, orientation=orientation)
        covariance = [0.0] * 36
        pwc = PoseWithCovariance(pose=pose, covariance=covariance)  # type: ignore
        assert pwc.pose.position.x == 0.0


class TestPoseWithCovarianceStamped:
    """Tests for the PoseWithCovarianceStamped dataclass."""

    def test_create_pose_with_covariance_stamped(self):
        """Test PoseWithCovarianceStamped creation."""
        stamp = Time(sec=1, nanosec=0)
        header = Header(stamp=stamp, frame_id="odom")
        position = Point(x=0.0, y=0.0, z=0.0)
        orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        pose = Pose(position=position, orientation=orientation)
        covariance = [0.0] * 36
        pwc = PoseWithCovariance(pose=pose, covariance=covariance)  # type: ignore
        pwcs = PoseWithCovarianceStamped(header=header, pose=pwc)
        assert pwcs.header.frame_id == "odom"


class TestVector3:
    """Tests for the Vector3 dataclass."""

    def test_create_vector3(self):
        """Test Vector3 creation."""
        v = Vector3(x=1.0, y=2.0, z=3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0


class TestTwist:
    """Tests for the Twist dataclass."""

    def test_create_twist(self):
        """Test Twist creation with linear and angular."""
        linear = Vector3(x=1.0, y=0.0, z=0.0)
        angular = Vector3(x=0.0, y=0.0, z=0.5)
        twist = Twist(linear=linear, angular=angular)
        assert twist.linear.x == 1.0
        assert twist.angular.z == 0.5


class TestTwistWithCovariance:
    """Tests for the TwistWithCovariance dataclass."""

    def test_create_twist_with_covariance(self):
        """Test TwistWithCovariance creation."""
        linear = Vector3(x=1.0, y=0.0, z=0.0)
        angular = Vector3(x=0.0, y=0.0, z=0.0)
        twist = Twist(linear=linear, angular=angular)
        covariance = [0.0] * 36
        twc = TwistWithCovariance(twist=twist, covariance=covariance)  # type: ignore
        assert twc.twist.linear.x == 1.0


class TestTwistWithCovarianceStamped:
    """Tests for the TwistWithCovarianceStamped dataclass."""

    def test_create_twist_with_covariance_stamped(self):
        """Test TwistWithCovarianceStamped creation."""
        stamp = Time(sec=1, nanosec=0)
        header = Header(stamp=stamp, frame_id="base_link")
        linear = Vector3(x=1.0, y=0.0, z=0.0)
        angular = Vector3(x=0.0, y=0.0, z=0.0)
        twist = Twist(linear=linear, angular=angular)
        covariance = [0.0] * 36
        twc = TwistWithCovariance(twist=twist, covariance=covariance)  # type: ignore
        twcs = TwistWithCovarianceStamped(header=header, twist=twc)
        assert twcs.header.frame_id == "base_link"


class TestAccel:
    """Tests for the Accel dataclass."""

    def test_create_accel(self):
        """Test Accel creation with linear and angular."""
        linear = Vector3(x=1.0, y=0.0, z=0.0)
        angular = Vector3(x=0.0, y=0.0, z=0.0)
        accel = Accel(linear=linear, angular=angular)
        assert accel.linear.x == 1.0
        assert accel.angular.z == 0.0


class TestAccelWithCovariance:
    """Tests for the AccelWithCovariance dataclass."""

    def test_create_accel_with_covariance(self):
        """Test AccelWithCovariance creation."""
        linear = Vector3(x=0.0, y=0.0, z=0.0)
        angular = Vector3(x=0.0, y=0.0, z=0.0)
        accel = Accel(linear=linear, angular=angular)
        covariance = [0.0] * 36
        awc = AccelWithCovariance(accel=accel, covariance=covariance)  # type: ignore
        assert awc.accel.linear.x == 0.0


class TestAccelWithCovarianceStamped:
    """Tests for the AccelWithCovarianceStamped dataclass."""

    def test_create_accel_with_covariance_stamped(self):
        """Test AccelWithCovarianceStamped creation."""
        stamp = Time(sec=1, nanosec=0)
        header = Header(stamp=stamp, frame_id="base_link")
        linear = Vector3(x=0.0, y=0.0, z=0.0)
        angular = Vector3(x=0.0, y=0.0, z=0.0)
        accel = Accel(linear=linear, angular=angular)
        covariance = [0.0] * 36
        awc = AccelWithCovariance(accel=accel, covariance=covariance)  # type: ignore
        awcs = AccelWithCovarianceStamped(header=header, accel=awc)
        assert awcs.header.frame_id == "base_link"
