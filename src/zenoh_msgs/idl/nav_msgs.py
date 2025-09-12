from dataclasses import dataclass

from pycdr2 import IdlStruct
from pycdr2.types import array, float64, int32, sequence, uint8

from .geometry_msgs import Pose, PoseWithCovariance, TwistWithCovariance
from .std_msgs import Header, String


@dataclass
class Odometry(IdlStruct, typename="Odometry"):
    header: Header
    child_frame_id: String
    pose: PoseWithCovariance
    twist: TwistWithCovariance


@dataclass
class AMCLPose(IdlStruct, typename="AMCLPose"):
    header: Header
    pose: Pose
    covariance: array[float64, 36]


@dataclass
class Time(IdlStruct, typename="Time"):
    sec: int32
    nanosec: int32


@dataclass
class GoalID(IdlStruct, typename="GoalID"):
    uuid: array[uint8, 16]


@dataclass
class GoalInfo(IdlStruct, typename="GoalInfo"):
    goal_id: GoalID
    stamp: Time


@dataclass
class GoalStatus(IdlStruct, typename="GoalStatus"):
    goal_info: GoalInfo
    status: int32


@dataclass
class Nav2Status(IdlStruct, typename="Nav2Status"):
    status_list: sequence[GoalStatus]
