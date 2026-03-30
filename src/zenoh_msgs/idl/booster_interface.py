from dataclasses import dataclass

from pycdr2 import IdlStruct
from pycdr2.types import float32, int16, int32, int64


@dataclass
class Odometer(IdlStruct, typename="Odometer"):
    """
    Odometer message for Booster robot odometry data.

    Simple odometry message containing position and orientation.
    """

    x: float32 = 0.0
    y: float32 = 0.0
    theta: float32 = 0.0


@dataclass
class BoosterApiReqMsg(IdlStruct, typename="BoosterApiReqMsg"):
    """
    Booster API Request Message for RPC service.

    Used in /booster_rpc_service to send commands to the robot.
    """

    api_id: int64 = 0
    body: str = ""


@dataclass
class BoosterApiRespMsg(IdlStruct, typename="BoosterApiRespMsg"):
    """
    Booster API Response Message for RPC service.

    Response from /booster_rpc_service after processing a request.
    """

    status: int64 = 0
    body: str = ""


@dataclass
class RpcServiceRequest(IdlStruct, typename="RpcServiceRequest"):
    """
    Wrapper for RPC service request.
    """

    msg: BoosterApiReqMsg


@dataclass
class RpcServiceResponse(IdlStruct, typename="RpcServiceResponse"):
    """
    Wrapper for RPC service response.
    """

    msg: BoosterApiRespMsg


@dataclass
class RemoteControllerState(IdlStruct, typename="RemoteControllerState"):
    """
    RemoteControllerState message for Booster robot control.

    This message represents the state of a remote controller used to control
    the Booster robot's movement.
    """

    event: int32 = 1536
    lx: float32 = 0.0
    ly: float32 = 0.0
    rx: float32 = 0.0
    ry: float32 = 0.0
    a: bool = False
    b: bool = False
    x: bool = False
    y: bool = False
    lb: bool = False
    rb: bool = False
    lt: bool = False
    rt: bool = False
    ls: bool = False
    rs: bool = False
    back: bool = False
    start: bool = False
    hat_c: bool = False
    hat_u: bool = False
    hat_d: bool = False
    hat_l: bool = False
    hat_r: bool = False
    hat_lu: bool = False
    hat_ld: bool = False
    hat_ru: bool = False
    hat_rd: bool = False
    hat_pos: int16 = 0
