from zenoh_msgs.idl.geometry_msgs import Quaternion, Vector3
from zenoh_msgs.idl.sensor_msgs import (
    IMU,
    BatteryState,
    CameraInfo,
    Detection,
    DockStatus,
    HazardDetection,
    HazardDetectionVector,
    Image,
    LaserScan,
    NavSatFix,
    NavSatStatus,
    PointField,
    RegionOfInterest,
)
from zenoh_msgs.idl.std_msgs import Header, String, Time


def make_header(frame_id: str = "map") -> Header:
    """Helper to create a Header."""
    return Header(stamp=Time(sec=1, nanosec=0), frame_id=frame_id)


class TestRegionOfInterest:
    """Tests for the RegionOfInterest dataclass."""

    def test_create_roi(self):
        """Test RegionOfInterest creation."""
        roi = RegionOfInterest(
            x_offset=0, y_offset=0, height=480, width=640, do_rectify=False
        )
        assert roi.width == 640
        assert roi.height == 480
        assert roi.do_rectify is False


class TestCameraInfo:
    """Tests for the CameraInfo dataclass."""

    def test_create_camera_info(self):
        """Test CameraInfo creation."""
        roi = RegionOfInterest(
            x_offset=0, y_offset=0, height=480, width=640, do_rectify=False
        )
        ci = CameraInfo(
            header=make_header(),
            height=480,
            width=640,
            distortion_model="plumb_bob",
            d=[0.0] * 5,  # type: ignore
            k=[1.0] * 9,  # type: ignore
            r=[1.0] * 9,  # type: ignore
            p=[1.0] * 12,  # type: ignore
            binning_x=0,
            binning_y=0,
            roi=roi,
        )
        assert ci.width == 640
        assert ci.height == 480
        assert ci.distortion_model == "plumb_bob"


class TestImage:
    """Tests for the Image dataclass."""

    def test_create_image(self):
        """Test Image creation."""
        img = Image(
            header=make_header(),
            height=480,
            width=640,
            encoding="rgb8",
            is_bigendian=0,
            step=1920,
            data=[],  # type: ignore
        )
        assert img.encoding == "rgb8"
        assert img.width == 640
        assert img.height == 480


class TestIMU:
    """Tests for the IMU dataclass."""

    def test_create_imu(self):
        """Test IMU creation."""
        orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        angular_velocity = Vector3(x=0.0, y=0.0, z=0.0)
        linear_acceleration = Vector3(x=0.0, y=0.0, z=9.8)
        imu = IMU(
            header=make_header(),
            orientation=orientation,
            orientation_covariance=[0.0] * 9,  # type: ignore
            angular_velocity=angular_velocity,
            angular_velocity_covariance=[0.0] * 9,  # type: ignore
            linear_acceleration=linear_acceleration,
            linear_acceleration_covariance=[0.0] * 9,  # type: ignore
        )
        assert imu.linear_acceleration.z == 9.8
        assert imu.header.frame_id == "map"


class TestDetection:
    """Tests for the Detection dataclass."""

    def test_create_detection(self):
        """Test Detection creation."""
        orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        angular_velocity = Vector3(x=0.0, y=0.0, z=0.0)
        linear_acceleration = Vector3(x=0.0, y=0.0, z=0.0)
        detection = Detection(
            header=make_header(),
            orientation=orientation,
            orientation_covariance=[0.0] * 9,  # type: ignore
            angular_velocity=angular_velocity,
            angular_velocity_covariance=[0.0] * 9,  # type: ignore
            linear_acceleration=linear_acceleration,
            linear_acceleration_covariance=[0.0] * 9,  # type: ignore
        )
        assert detection.header.frame_id == "map"


class TestHazardDetection:
    """Tests for the HazardDetection dataclass."""

    def test_create_hazard_detection(self):
        """Test HazardDetection creation."""
        hd = HazardDetection(
            header=make_header(), type=HazardDetection.TYPE.CLIFF.value
        )
        assert hd.type == HazardDetection.TYPE.CLIFF.value

    def test_hazard_type_enum_values(self):
        """Test HazardDetection TYPE enum values."""
        assert HazardDetection.TYPE.BACKUP_LIMIT.value == 0
        assert HazardDetection.TYPE.BUMP.value == 1
        assert HazardDetection.TYPE.CLIFF.value == 2
        assert HazardDetection.TYPE.STALL.value == 3
        assert HazardDetection.TYPE.WHEEL_DROP.value == 4
        assert HazardDetection.TYPE.OBJECT_PROXIMITY.value == 5


class TestHazardDetectionVector:
    """Tests for the HazardDetectionVector dataclass."""

    def test_create_hazard_detection_vector_empty(self):
        """Test HazardDetectionVector with empty detections."""
        hdv = HazardDetectionVector(header=make_header(), detections=[])  # type: ignore
        assert hdv.detections == []

    def test_create_hazard_detection_vector_with_items(self):
        """Test HazardDetectionVector with detections."""
        hd = HazardDetection(header=make_header(), type=HazardDetection.TYPE.BUMP.value)
        hdv = HazardDetectionVector(header=make_header(), detections=[hd])  # type: ignore
        assert hdv is not None


class TestNavSatStatus:
    """Tests for the NavSatStatus dataclass."""

    def test_create_navsat_status(self):
        """Test NavSatStatus creation."""
        status = NavSatStatus(
            status=NavSatStatus.STATUS.FIX.value,
            service=NavSatStatus.SERVICE.GPS.value,
        )
        assert status.status == NavSatStatus.STATUS.FIX.value
        assert status.service == NavSatStatus.SERVICE.GPS.value

    def test_navsat_status_enum_values(self):
        """Test NavSatStatus STATUS enum values."""
        assert NavSatStatus.STATUS.NO_FIX.value == -1
        assert NavSatStatus.STATUS.FIX.value == 0
        assert NavSatStatus.STATUS.SBAS_FIX.value == 1
        assert NavSatStatus.STATUS.GBAS_FIX.value == 2


class TestNavSatFix:
    """Tests for the NavSatFix dataclass."""

    def test_create_navsat_fix(self):
        """Test NavSatFix creation."""
        nav_status = NavSatStatus(
            status=NavSatStatus.STATUS.FIX.value,
            service=NavSatStatus.SERVICE.GPS.value,
        )
        fix = NavSatFix(
            header=make_header(),
            status=nav_status,
            latitude=1.0,
            longitude=2.0,
            altitude=3.0,
            position_covariance=[0.0] * 9,  # type: ignore
            position_covariance_type=NavSatFix.POSITION_COVARIANCE_TYPE.UNKNOWN.value,
        )
        assert fix.latitude == 1.0
        assert fix.longitude == 2.0
        assert fix.altitude == 3.0


class TestPointField:
    """Tests for the PointField dataclass."""

    def test_create_point_field(self):
        """Test PointField creation."""
        pf = PointField(
            name="x",
            offset=0,
            datatype=PointField.DATA_TYPE.FLOAT32.value,
            count=1,
        )
        assert pf.name == "x"
        assert pf.offset == 0
        assert pf.count == 1

    def test_point_field_data_type_enum(self):
        """Test PointField DATA_TYPE enum values."""
        assert PointField.DATA_TYPE.FLOAT32.value == 7
        assert PointField.DATA_TYPE.FLOAT64.value == 8


class TestDockStatus:
    """Tests for the DockStatus dataclass."""

    def test_create_dock_status_docked(self):
        """Test DockStatus when docked."""
        ds = DockStatus(header=make_header(), docker_visible=True, is_docked=True)
        assert ds.docker_visible is True
        assert ds.is_docked is True

    def test_create_dock_status_not_docked(self):
        """Test DockStatus when not docked."""
        ds = DockStatus(header=make_header(), docker_visible=False, is_docked=False)
        assert ds.docker_visible is False
        assert ds.is_docked is False


class TestBatteryState:
    """Tests for the BatteryState dataclass."""

    def test_create_battery_state(self):
        """Test BatteryState creation."""
        bs = BatteryState(
            header=make_header(),
            voltage=12.5,
            temperature=25.0,
            current=-1.5,
            charge=10.0,
            capacity=15.0,
            design_capacity=15.0,
            percentage=0.75,
            power_supply_status=2,
            power_supply_health=1,
            power_supply_technology=3,
            present=True,
            cell_voltage=[],
            cell_temperature=[],
            location=String(data="slot_0"),
            serial_number=String(data="SN12345"),
        )
        assert bs.voltage == 12.5
        assert bs.percentage == 0.75
        assert bs.present is True
        assert bs.location.data == "slot_0"


class TestLaserScan:
    """Tests for the LaserScan dataclass."""

    def test_create_laser_scan(self):
        """Test LaserScan creation."""
        ls = LaserScan(
            header=make_header(),
            angle_min=-1.57,
            angle_max=1.57,
            angle_increment=0.01,
            time_increment=0.0,
            scan_time=0.1,
            range_min=0.1,
            range_max=10.0,
            ranges=[1.0, 2.0, 3.0],
            intensities=[],
        )
        assert ls.angle_min == -1.57
        assert ls.angle_max == 1.57
        assert ls.range_max == 10.0
        assert len(ls.ranges) == 3
