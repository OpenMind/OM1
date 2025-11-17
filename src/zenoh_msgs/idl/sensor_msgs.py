from dataclasses import dataclass
from enum import Enum
from typing import List

from pycdr2 import IdlStruct
from pycdr2.types import array, float32, float64, int8, sequence, uint8, uint16, uint32

from .geometry_msgs import Point32, Quaternion, Vector3
from .std_msgs import Header, String


@dataclass
class RegionOfInterest(IdlStruct, typename="RegionOfInterest"):
    x_offset: uint32
    y_offset: uint32
    height: uint32
    width: uint32
    do_rectify: bool


@dataclass
class CameraInfo(IdlStruct, typename="CameraInfo"):
    header: Header
    height: uint32
    width: uint32
    distortion_model: str
    d: sequence[float64]
    k: array[float64, 9]
    r: array[float64, 9]
    p: array[float64, 12]
    binning_x: uint32
    binning_y: uint32
    roi: RegionOfInterest


@dataclass
class Image(IdlStruct, typename="Image"):
    header: Header
    height: uint32
    width: uint32
    encoding: str
    is_bigendian: uint8
    step: uint32
    data: sequence[uint8]


@dataclass
class IMU(IdlStruct, typename="IMU"):
    header: Header
    orientation: Quaternion
    orientation_covariance: array[float64, 9]
    angular_velocity: Vector3
    angular_velocity_covariance: array[float64, 9]
    linear_acceleration: Vector3
    linear_acceleration_covariance: array[float64, 9]


@dataclass
class Detection(IdlStruct, typename="Detection"):
    header: Header
    orientation: Quaternion
    orientation_covariance: array[float64, 9]
    angular_velocity: Vector3
    angular_velocity_covariance: array[float64, 9]
    linear_acceleration: Vector3
    linear_acceleration_covariance: array[float64, 9]


@dataclass
class HazardDetection(IdlStruct, typename="HazardDetection"):
    header: Header

    class TYPE(Enum):
        BACKUP_LIMIT = 0
        BUMP = 1  # The robot has bumped against an obstacle
        CLIFF = 2  # The robot detected a cliff
        STALL = 3  # The wheels of the robot are stalled against an obstacle
        WHEEL_DROP = 4  # The wheels of the robot are fully dropped
        OBJECT_PROXIMITY = 5  # The robot detects an obstacle in close proximity

    type: uint8


@dataclass
class HazardDetectionVector(IdlStruct, typename="HazardDetectionVector"):
    header: Header
    detections: sequence[HazardDetection]


@dataclass
class NavSatStatus(IdlStruct, typename="NavSatStatus"):

    class STATUS(Enum):
        NO_FIX = -1  # unable to fix position
        FIX = 0  # unaugmented fix
        SBAS_FIX = 1  # with satellite-based augmentation
        GBAS_FIX = 2  # with ground-based augmentation

    status: int8

    class SERVICE(Enum):
        GPS = 1
        GLONASS = 2
        COMPASS = 4  # includes BeiDou
        GALILEO = 8

    service: uint16


@dataclass
class NavSatFix(IdlStruct, typename="NavSatFix"):
    header: Header
    status: NavSatStatus
    latitude: float64
    longitude: float64
    altitude: float64
    position_covariance: array[float64, 9]

    class POSITION_COVARIANCE_TYPE(Enum):
        UNKNOWN = 0
        APPROXIMATED = 1
        DIAGONAL_KNOWN = 2
        KNOWN = 3

    position_covariance_type: uint8


@dataclass
class PointField(IdlStruct, typename="PointField"):
    name: str
    offset: uint32

    class DATA_TYPE(Enum):
        INT8 = 1
        UINT8 = 2
        INT16 = 3
        UINT16 = 4
        INT32 = 5
        UINT32 = 6
        FLOAT32 = 7
        FLOAT64 = 8

    datatype: uint8
    count: uint32


@dataclass
class PointCloud(IdlStruct, typename="PointCloud"):
    header: Header
    points: sequence[Point32]
    channels: sequence[PointField]


@dataclass
class PointCloud2(IdlStruct, typename="PointCloud2"):
    header: Header
    height: uint32
    width: uint32
    fields: sequence[PointField]
    is_bigendian: bool
    point_step: uint32
    row_step: uint32
    data: sequence[uint8]
    is_dense: bool


@dataclass
class BatteryState(IdlStruct, typename="BatteryState"):
    header: Header
    voltage: float32  # Voltage in Volts (Mandatory)
    temperature: float32
    current: float32  # Negative when discharging (A)  (If unmeasured NaN)
    charge: float32  # Current charge in Ah  (If unmeasured NaN)
    capacity: float32  # Capacity in Ah (last full capacity)  (If unmeasured NaN)
    design_capacity: float32  # Capacity in Ah (design capacity)  (If unmeasured NaN)
    percentage: float32  # Charge percentage on 0 to 1 range  (If unmeasured NaN)
    power_supply_status: uint8  # The charging status as reported. Values defined above
    power_supply_health: uint8  # The battery health metric. Values defined above
    power_supply_technology: uint8  # The battery chemistry. Values defined above
    present: bool  # True if the battery is present
    cell_voltage: List[
        float32
    ]  # An array of individual cell voltages for each cell in the pack
    # If individual voltages unknown but number of cells known set each to NaN
    cell_temperature: List[
        float32
    ]  # An array of individual cell voltages for each cell in the pack
    # If individual voltages unknown but number of cells known set each to NaN
    location: (
        String  # The location into which the battery is inserted. (slot number or plug)
    )
    serial_number: String  # The best approximation of the battery serial number


@dataclass
class LaserScan(IdlStruct, typename="LaserScan"):
    header: Header
    angle_min: float32
    angle_max: float32
    angle_increment: float32
    time_increment: float32
    scan_time: float32
    range_min: float32
    range_max: float32
    ranges: List[float32]
    intensities: List[float32]


@dataclass
class DockStatus(IdlStruct, typename="DockStatus"):
    header: Header
    docker_visible: bool
    is_docked: bool


@dataclass
class Paths(IdlStruct, typename="Paths"):
    header: Header
    paths: List[uint32]
    blocked_by_obstacle_idx: List[uint32]
    blocked_by_hazard_idx: List[uint32]
