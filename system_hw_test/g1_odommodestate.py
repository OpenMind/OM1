import sys
import time

sys.path.insert(0, "../src")

try:
    from unitree.unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
except ImportError:
    print("Please run this script from inside /system_hw_test")

from unitree.unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
)


class Custom:
    def __init__(self):
        self.low_state = None

    # Public methods
    def Init(self):
        print("Initiating subscriber")
        self.subscriber = ChannelSubscriber("rt/odommodestate", SportModeState_)
        self.subscriber.Init(self.MessageHandler, 10)
        print("Initiated subscriber")
        print("Waiting for messages on topic: rt/odommodestate")
        print("Press Ctrl+C to exit")

    def MessageHandler(self, msg: SportModeState_):
        print("\n[Message Received]")
        print(f"message info: {msg}")
        print(f"Timestamp: {time.time()}\n")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        print(f"Initializing channel factory with network interface: {sys.argv[1]}")
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        print("Initializing channel factory with default network interface")
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
