import sys
import time

import unitree_legged_const as go2

from unitree.unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
)
from unitree.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


class Custom:
    def __init__(self):
        self.low_state = None

    # Public methods
    def Init(self):
        # create subscriber #
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        print("FR_0 motor state: ", msg.motor_state[go2.LegID["FR_0"]])
        print("IMU state: ", msg.imu_state)
        print("Battery state: voltage: ", msg.power_v, "current: ", msg.power_a)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()

    while True:
        time.sleep(1)
