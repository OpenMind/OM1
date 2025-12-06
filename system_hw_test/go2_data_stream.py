import sys
import time
import logging
import unitree_legged_const as go2

from unitree.unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
)
from unitree.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


class Custom:
    def __init__(self):
        self.low_state = None
        self.logger = logging.getLogger(self.__class__.__name__)

    # Public methods
    def Init(self):
        # create subscriber #
        self.logger.info("Initializing lowstate subscriber")
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateMessageHandler, 10)
        self.logger.info("Lowstate subscriber initialized successfully")

    def LowStateMessageHandler(self, msg: LowState_):
        self.low_state = msg
        self.logger.info("FR_0 motor state: %s", msg.motor_state[go2.LegID["FR_0"]])
        self.logger.info("IMU state: %s", msg.imu_state)
        self.logger.info("Battery state: voltage: %s, current: %s", msg.power_v, msg.power_a)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        logger.info("Initializing channel factory with config: %s", sys.argv[1])
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        logger.info("Initializing channel factory with default config")
        ChannelFactoryInitialize(0)

    logger.info("Creating Custom instance")
    custom = Custom()
    custom.Init()

    logger.info("Starting main loop")
    while True:
        time.sleep(1)
