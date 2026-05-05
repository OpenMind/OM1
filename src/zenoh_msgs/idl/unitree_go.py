from dataclasses import dataclass

from pycdr2 import IdlStruct
from pycdr2.types import array, float32, int8, int16, int32, uint8, uint16, uint32


@dataclass
class TimeSpec(IdlStruct, typename="TimeSpec"):
    """TimeSpec message."""

    sec: int32
    nanosec: uint32


@dataclass
class IMUState(IdlStruct, typename="IMUState"):
    """IMUState message."""

    quaternion: array[float32, 4]
    gyroscope: array[float32, 3]
    accelerometer: array[float32, 3]
    rpy: array[float32, 3]
    temperature: int8


@dataclass
class MotorState(IdlStruct, typename="MotorState"):
    """MotorState message."""

    mode: uint8
    q: float32
    dq: float32
    ddq: float32
    tau_est: float32
    q_raw: float32
    dq_raw: float32
    ddq_raw: float32
    temperature: int8
    lost: uint32
    reserve: array[uint32, 2]


@dataclass
class BmsState(IdlStruct, typename="BmsState"):
    """BmsState message."""

    version_high: uint8
    version_low: uint8
    status: uint8
    soc: uint8
    current: int32
    cycle: uint16
    bq_ntc: array[int8, 2]
    mcu_ntc: array[int8, 2]
    cell_vol: array[uint16, 15]


@dataclass
class LowState(IdlStruct, typename="LowState"):
    """LowState message."""

    head: array[uint8, 2]
    level_flag: uint8
    frame_reserve: uint8
    sn: array[uint32, 2]
    version: array[uint32, 2]
    bandwidth: uint16
    imu_state: IMUState
    motor_state: array[MotorState, 20]
    bms_state: BmsState
    foot_force: array[int16, 4]
    foot_force_est: array[int16, 4]
    tick: uint32
    wireless_remote: array[uint8, 40]
    bit_flag: uint8
    adc_reel: float32
    temperature_ntc1: int8
    temperature_ntc2: int8
    power_v: float32
    power_a: float32
    fan_frequency: array[uint16, 4]
    reserve: uint32
    crc: uint32


@dataclass
class SportModeState(IdlStruct, typename="SportModeState"):
    """SportModeState message."""

    stamp: TimeSpec
    error_code: uint32
    imu_state: IMUState
    mode: uint8
    progress: float32
    gait_type: uint8
    foot_raise_height: float32
    position: array[float32, 3]
    body_height: float32
    velocity: array[float32, 3]
    yaw_speed: float32
    range_obstacle: array[float32, 4]
    foot_force: array[int16, 4]
    foot_position_body: array[float32, 12]
    foot_speed_body: array[float32, 12]
