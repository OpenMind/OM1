import math
import struct
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

sys.path.insert(0, "../src")

try:
    from unitree.unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
except ImportError:
    print("Please run this script from inside /system_hw_test")

from unitree.unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelSubscriber,
)

# header: 'unitree.unitree_sdk2py.idl.std_msgs.msg.dds_.Header_'
# height: types.uint32
# width: types.uint32
# fields: types.sequence['unitree.unitree_sdk2py.idl.sensor_msgs.msg.dds_.PointField_']
# is_bigendian: bool
# point_step: types.uint32
# row_step: types.uint32
# data: types.sequence[types.uint8]
# is_dense: bool


# fields=[
# PointField_(name='x', offset=0, datatype=7, count=1),
# PointField_(name='y', offset=4, datatype=7, count=1),
# PointField_(name='z', offset=8, datatype=7, count=1),
# PointField_(name='intensity', offset=12, datatype=7, count=1)], is_bigendian=False, point_step=16

# fields=[
#       PointField_(name='x', offset=0, datatype=7, count=1),
#       PointField_(name='y', offset=4, datatype=7, count=1),
#       PointField_(name='z', offset=8, datatype=7, count=1),
#       PointField_(name='intensity', offset=16, datatype=7, count=1),
#       PointField_(name='ring', offset=20, datatype=4, count=1),
#       PointField_(name='time', offset=24, datatype=7, count=1)
# ]


fig, ax = plt.subplots(1)
line = ax.plot([], [], ".", color="green")[0]
ax.set_xlim(-180, 180)
ax.set_ylim(-0.40, 1.5)


class Custom:
    def __init__(self):
        self.low_state = None

    # Public methods
    def Init(self):
        # create subscriber #
        # /utlidar/cloud - this is noisy
        # /utlidar/cloud_deskewed - much better behaved?
        self.subscriber = ChannelSubscriber("rt/utlidar/cloud_deskewed", PointCloud2_)

        # self.subscriber = ChannelSubscriber("rt/utlidar/cloud", PointCloud2_)
        self.subscriber.Init(self.MessageHandler, 10)

    def MessageHandler(self, msg: PointCloud2_):
        self.cloud = msg
        print("Cloud: ", msg)
        # data = data.astype(np.uint8)
        data = self.cloud.data
        # print("Data: ", data)
        width = self.cloud.width

        print("Height: ", self.cloud.height)
        print("Width: ", width)
        print("Point Step: ", self.cloud.point_step)
        print("Row Step: ", self.cloud.row_step)
        print("len(Data): ", len(data))

        sliced = bytes(data)

        print("len(sliced): ", len(sliced))

        unpack_size = int(width * self.cloud.point_step / 4)
        print("unpack_size: ", unpack_size)

        format_string = str(unpack_size) + "f"
        print("format unpack_size: ", format_string)

        float_array2 = np.array(struct.unpack(format_string, sliced))
        # '>4f' means big-endian, 4 floats
        # arr = float_array2.reshape((width, 4))

        arr = float_array2.reshape((width, 4))

        # #print("Size:", size(arr))

        # print("Array:", arr[0])
        # # print("Array:", arr[1])
        # # print("Array:", arr[2])

        x = arr[:, 0]
        y = arr[:, 1]
        z = arr[:, 2]
        intensity = arr[:, 3]

        print("min(x): ", min(x))
        print("max(x): ", max(x))

        print("min(y): ", min(y))
        print("max(y): ", max(y))

        print("min(z): ", min(z))
        print("max(z): ", max(z))

        print("min(intensity): ", min(intensity))
        print("max(intensity): ", max(intensity))

        print("len(x): ", len(x))
        print("len(y): ", len(y))
        print("len(z): ", len(z))
        print("len(intensity): ", len(intensity))

        # this gives us 361 bins, from -180.0 to 180
        ls = np.full(361, 20.0)
        angles = np.linspace(-180, 180, 361)
        # print("angles: ", angles)
        # # convert each point to a ray

        # if width != 1437:
        #     return

        for p in arr:
            rad = math.atan2(p[1], p[0])
            deg = rad * (180 / math.pi)
            z = p[2]
            distance = math.sqrt(p[0] * p[0] + p[1] * p[1] + z * z)
            shift = int(deg) + 180
            if shift < 0:
                shift = 0
            elif shift > 360:
                shift = 360

            print("deg: ", deg)
            print("distance: ", distance)

            # if z < 0.25:
            #     continue
            # if z > 0.50:
            #     continue
            # if p[2] > 1.0:
            #     continue
            if distance > 1.5:
                continue
            if ls[shift] > z:
                ls[shift] = z
            # runs from -180 to +180

        # print("ls: ", ls)
        global line
        line.set_data(angles, ls)

        # global colors
        # colors = cmap.to_rgba(z)

        # global scatter
        # #scatter.set_data(x, y, z)
        # #scatter = ax.scatter(x, y, z, c=colors, marker="o")

        """
        The x,y,z data are a spatial map composed of about 1435 points
        Each point is a LIDAR return from a particular place in the space around the sensor.
        The data are already correctly compensated for the angle/orientatiuon of the sensor
        Occasionally, there are significant data glitches
        """


if __name__ == "__main__":

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()

    while True:
        time.sleep(1)
        ani = FuncAnimation(fig, lambda _: None)
        plt.show()
