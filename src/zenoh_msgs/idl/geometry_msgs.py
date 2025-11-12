from dataclasses import dataclass

from pycdr2 import IdlStruct
from pycdr2.types import array, float32, float64

from .std_msgs import Header


@dataclass
class Point(IdlStruct, typename="Point"):
    x: float64
    y: float64
    z: float64


@dataclass
class Point32(IdlStruct, typename="Point32"):
    x: float32
    y: float32
    z: float32


@dataclass
class Quaternion(IdlStruct, typename="Quaternion"):
    x: float64
    y: float64
    z: float64
    w: float64


@dataclass
class Pose(IdlStruct, typename="Pose"):
    position: Point
    orientation: Quaternion


@dataclass
class PoseStamped(IdlStruct, typename="PoseStamped"):
    header: Header
    pose: Pose


@dataclass
class PoseWithCovariance(IdlStruct, typename="PoseWithCovariance"):
    pose: Pose
    covariance: array[float64, 36]


@dataclass
class PoseWithCovarianceStamped(IdlStruct, typename="PoseWithCovarianceStamped"):
    header: Header
    pose: PoseWithCovariance


@dataclass
class Vector3(IdlStruct, typename="Vector3"):
    x: float64
    y: float64
    z: float64


@dataclass
class Twist(IdlStruct, typename="Twist"):
    linear: Vector3
    angular: Vector3


@dataclass
class TwistWithCovariance(IdlStruct, typename="TwistWithCovariance"):
    twist: Twist
    covariance: array[float64, 36]


@dataclass
class TwistWithCovarianceStamped(IdlStruct, typename="TwistWithCovarianceStamped"):
    header: Header
    twist: TwistWithCovariance


@dataclass
class Accel(IdlStruct, typename="Accel"):
    linear: Vector3
    angular: Vector3


@dataclass
class AccelWithCovariance(IdlStruct, typename="AccelWithCovariance"):
    accel: Accel
    covariance: array[float64, 36]


@dataclass
class AccelWithCovarianceStamped(IdlStruct, typename="AccelWithCovarianceStamped"):
    header: Header
    accel: AccelWithCovariance
