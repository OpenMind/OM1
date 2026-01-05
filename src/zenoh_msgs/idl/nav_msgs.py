from dataclasses import dataclass

from pycdr2 import IdlStruct
from pycdr2.types import array, float32, float64, int32, sequence, uint8

from .geometry_msgs import Pose, PoseWithCovariance, TwistWithCovariance
from .std_msgs import Header, String


@dataclass
class Odometry(IdlStruct, typename="Odometry"):
    """Odometry message."""

    header: Header
    child_frame_id: String
    pose: PoseWithCovariance
    twist: TwistWithCovariance


@dataclass
class AMCLPose(IdlStruct, typename="AMCLPose"):
    """AMCLPose message."""

    header: Header
    pose: Pose
    covariance: array[float64, 36]


@dataclass
class LidarLocalization(IdlStruct, typename="LidarLocalization"):
    """LidarLocalization message."""

    header: Header
    pose: Pose
    match_score: int32
    quality_percent: float32
    num_points: int32


@dataclass
class Time(IdlStruct, typename="Time"):
    """Time message."""

    sec: int32
    nanosec: int32


@dataclass
class GoalID(IdlStruct, typename="GoalID"):
    """GoalID message."""

    uuid: array[uint8, 16]


@dataclass
class GoalInfo(IdlStruct, typename="GoalInfo"):
    """GoalInfo message."""

    goal_id: GoalID
    stamp: Time


@dataclass
class GoalStatus(IdlStruct, typename="GoalStatus"):
    """GoalStatus message."""

    goal_info: GoalInfo
    status: int32


@dataclass
class Nav2Status(IdlStruct, typename="Nav2Status"):
    """Nav2Status message."""

    status_list: sequence[GoalStatus]
