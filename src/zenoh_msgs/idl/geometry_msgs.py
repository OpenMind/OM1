from dataclasses import dataclass

from pycdr2 import IdlStruct
from pycdr2.types import array, float32, float64

from .std_msgs import Header


@dataclass
class Point(IdlStruct, typename="Point"):
    """Point message."""

    x: float64
    y: float64
    z: float64


@dataclass
class Point32(IdlStruct, typename="Point32"):
    """Point32 message."""

    x: float32
    y: float32
    z: float32


@dataclass
class Quaternion(IdlStruct, typename="Quaternion"):
    """Quaternion message."""

    x: float64
    y: float64
    z: float64
    w: float64


@dataclass
class Pose(IdlStruct, typename="Pose"):
    """Pose message."""

    position: Point
    orientation: Quaternion


@dataclass
class PoseStamped(IdlStruct, typename="PoseStamped"):
    """PoseStamped message."""

    header: Header
    pose: Pose


@dataclass
class PoseWithCovariance(IdlStruct, typename="PoseWithCovariance"):
    """PoseWithCovariance message."""

    pose: Pose
    covariance: array[float64, 36]


@dataclass
class PoseWithCovarianceStamped(IdlStruct, typename="PoseWithCovarianceStamped"):
    """PoseWithCovarianceStamped message."""

    header: Header
    pose: PoseWithCovariance


@dataclass
class Vector3(IdlStruct, typename="Vector3"):
    """Vector3 message."""

    x: float64
    y: float64
    z: float64


@dataclass
class Twist(IdlStruct, typename="Twist"):
    """Twist message."""

    linear: Vector3
    angular: Vector3


@dataclass
class TwistWithCovariance(IdlStruct, typename="TwistWithCovariance"):
    """TwistWithCovariance message."""

    twist: Twist
    covariance: array[float64, 36]


@dataclass
class TwistWithCovarianceStamped(IdlStruct, typename="TwistWithCovarianceStamped"):
    """TwistWithCovarianceStamped message."""

    header: Header
    twist: TwistWithCovariance


@dataclass
class Accel(IdlStruct, typename="Accel"):
    """Accel message."""

    linear: Vector3
    angular: Vector3


@dataclass
class AccelWithCovariance(IdlStruct, typename="AccelWithCovariance"):
    """AccelWithCovariance message."""

    accel: Accel
    covariance: array[float64, 36]


@dataclass
class AccelWithCovarianceStamped(IdlStruct, typename="AccelWithCovarianceStamped"):
    """AccelWithCovarianceStamped message."""

    header: Header
    accel: AccelWithCovariance
